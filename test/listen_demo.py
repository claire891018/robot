import sys
from pathlib import Path

HERE = Path(__file__).resolve()
# 依序嘗試：.../robot、.../repo 根、.../更上一層
for up in [HERE.parent, HERE.parents[1], HERE.parents[2]]:
    src_dir = up / "src"
    if src_dir.exists():
        sys.path.insert(0, str(up))  
        break

from src.listener import Listener


import logging
import queue
import threading
from typing import List, Dict

import numpy as np
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import matplotlib.pyplot as plt

# sound_window_len = 5000  #
# sound_window_buffer = None

st.set_page_config(
    page_title="Listener Demo",
    page_icon="https://api.dicebear.com/9.x/thumbs/svg?",
    layout="wide",
)

logger = logging.getLogger(__name__)

def init_state():
    if "listener" not in st.session_state:
        st.session_state.listener = None
    if "listener_running" not in st.session_state:
        st.session_state.listener_running = False
    if "events" not in st.session_state:
        st.session_state.events = []  # type: List[Dict]
    if "lock" not in st.session_state:
        st.session_state.lock = threading.Lock()
    if "sound_window_buffer" not in st.session_state:
        st.session_state.sound_window_buffer = None  # pydub.AudioSegment
    if "sound_window_len" not in st.session_state:
        st.session_state.sound_window_len = 5000  # 5 秒滾動窗

def on_evt(evt: Dict):
    with st.session_state.lock:
        st.session_state.events.append(evt)

def resample_to_16k(mono_i16: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return mono_i16.astype(np.int16, copy=False)
    x = mono_i16.astype(np.float32)
    n_in = x.shape[-1]
    n_out = int(round(n_in * 16000 / sr))
    if n_out <= 0 or n_in <= 1:
        return np.zeros(0, dtype=np.int16)
    xp = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, xp, x).astype(np.int16)

def render_header():
    icon = "https://api.dicebear.com/9.x/thumbs/svg?"
    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:.5rem;"><img src="{icon}" height="28">Listener Demo</h2>',
        unsafe_allow_html=True,
    )
    st.caption("點擊「START」，開始說話。")

def render_events():
    st.subheader("即時輸出")
    with st.session_state.lock:
        evts = list(st.session_state.events)[-200:]
    if not evts:
        st.info("（尚無輸出）")
        return
    for evt in evts:
        t = evt.get("type")
        if t == "utterance":
            txt = (evt.get("text") or "").strip()
            conf = evt.get("confidence", 0.0)
            meta = evt.get("meta", {}) or {}
            st.write(f"**你：** {txt}")
            st.caption(f"conf={conf:.2f} · len={meta.get('audio_len_sec','?')}s · note={meta.get('note','')}")
            st.divider()
        elif t == "error":
            st.error(f"{evt.get('error')} — {evt.get('detail','')}")
        elif t == "info":
            st.info(evt.get("msg",""))
        else:
            st.write(evt)

def main():
    init_state()
    render_header()

    fig_place = st.empty()
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2})
    
    ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,           
        media_stream_constraints={"audio": True},  
        async_processing=True,
        rtc_configuration={                    # ← 官方建議：傳「純 dict」
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )


    if ctx.state.playing and ctx.audio_receiver:
        if not st.session_state.listener_running:
            st.session_state.listener = Listener(on_utterance=on_evt, source="external")
            st.session_state.listener.start()
            st.session_state.listener_running = True

        try:
            audio_frames = ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            audio_frames = []

        sound_chunk = pydub.AudioSegment.empty()
        for af in audio_frames:
            arr = af.to_ndarray()

            if arr.ndim == 2:
                mono = arr.mean(axis=0).astype(np.int16)
                chs = arr.shape[0]
            else:
                mono = arr.astype(np.int16)
                chs = 1
            sr = af.sample_rate
            mono_16k = resample_to_16k(mono, sr)
            if mono_16k.size > 0 and st.session_state.listener is not None:
                st.session_state.listener.append_pcm(mono_16k.tobytes())

            sound = pydub.AudioSegment(
                data=arr.tobytes(),
                sample_width=af.format.bytes,
                frame_rate=sr,
                channels=chs,
            )
            sound_chunk += sound

        if len(sound_chunk) > 0:
            if st.session_state.sound_window_buffer is None:
                st.session_state.sound_window_buffer = pydub.AudioSegment.silent(
                    duration=st.session_state.sound_window_len
                )
            st.session_state.sound_window_buffer += sound_chunk
            if len(st.session_state.sound_window_buffer) > st.session_state.sound_window_len:
                st.session_state.sound_window_buffer = st.session_state.sound_window_buffer[-st.session_state.sound_window_len:]

        swb = st.session_state.sound_window_buffer
        if swb and len(swb) > 0:
            swb = swb.set_channels(1)
            sample = np.array(swb.get_array_of_samples())

            ax_time.cla()
            times = (np.arange(-len(sample), 0)) / swb.frame_rate
            ax_time.plot(times, sample)
            ax_time.set_xlabel("Time")
            ax_time.set_ylabel("Magnitude")

            spec = np.fft.fft(sample)
            freq = np.fft.fftfreq(sample.shape[0], 1.0 / swb.frame_rate)
            freq = freq[: int(freq.shape[0] / 2)]
            spec = spec[: int(spec.shape[0] / 2)]
            spec[0] = spec[0] / 2

            ax_freq.cla()
            ax_freq.plot(freq, np.abs(spec))
            ax_freq.set_xlabel("Frequency")
            ax_freq.set_yscale("log")
            ax_freq.set_ylabel("Magnitude")

            fig_place.pyplot(fig)
    else:
        if st.session_state.listener_running and st.session_state.listener is not None:
            try:
                st.session_state.listener.stop()
            finally:
                st.session_state.listener = None
                st.session_state.listener_running = False

    st.divider()
    render_events()

if __name__ == "__main__":
    main()
