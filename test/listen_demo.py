import sys, json, asyncio, logging, queue, threading, time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import matplotlib.pyplot as plt
import websockets

ASR_WS_URL = st.secrets.get("ASR_WS_URL", "ws://140.116.158.98:9999/asr")

st.set_page_config(
    page_title="Listener Demo",
    page_icon="https://api.dicebear.com/9.x/thumbs/svg?",
    layout="wide",
)

logger = logging.getLogger(__name__)

def init_state():
    if "events" not in st.session_state:
        st.session_state.events: List[Dict] = []
    if "lock" not in st.session_state:
        st.session_state.lock = threading.Lock()
    if "sound_window_buffer" not in st.session_state:
        st.session_state.sound_window_buffer = None
    if "sound_window_len" not in st.session_state:
        st.session_state.sound_window_len = 5000
    if "send_q" not in st.session_state:
        st.session_state.send_q = queue.Queue(maxsize=64)
    if "recv_q" not in st.session_state:
        st.session_state.recv_q = queue.Queue()
    if "ws_thread" not in st.session_state:
        st.session_state.ws_thread = None
    if "ws_running" not in st.session_state:
        st.session_state.ws_running = False
    if "last_plot" not in st.session_state:
        st.session_state.last_plot = 0.0

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

async def asr_loop_async(send_q: queue.Queue, recv_q: queue.Queue, url: str):
    async with websockets.connect(url, max_size=2**23) as ws:
        await ws.send(json.dumps({"type": "start", "sr": 16000, "lang": "zh"}))
        async def reader():
            try:
                async for msg in ws:
                    try:
                        evt = json.loads(msg)
                    except Exception:
                        evt = {"type": "error", "error": "bad_json", "detail": msg}
                    recv_q.put(evt)
            except Exception as e:
                recv_q.put({"type": "error", "error": "ws_reader", "detail": str(e)})
        reader_task = asyncio.create_task(reader())
        try:
            while True:
                kind, payload = await asyncio.to_thread(send_q.get)
                if kind == "audio":
                    await ws.send(payload)
                elif kind == "end":
                    await ws.send(json.dumps({"type": "end"}))
                    break
        finally:
            await reader_task

def ws_worker(send_q: queue.Queue, recv_q: queue.Queue, url: str):
    asyncio.run(asr_loop_async(send_q, recv_q, url))

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
        audio_receiver_size=2048,
        media_stream_constraints={"audio": True},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    if ctx.state.playing and ctx.audio_receiver:
        if not st.session_state.ws_running:
            t = threading.Thread(
                target=ws_worker,
                args=(st.session_state.send_q, st.session_state.recv_q, ASR_WS_URL),
                daemon=True,
            )
            t.start()
            st.session_state.ws_thread = t
            st.session_state.ws_running = True

        audio_frames = []
        while True:
            try:
                audio_frames.extend(ctx.audio_receiver.get_frames(timeout=0))
            except queue.Empty:
                break

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
            if mono_16k.size > 0:
                try:
                    st.session_state.send_q.put_nowait(("audio", mono_16k.tobytes()))
                except queue.Full:
                    try:
                        _ = st.session_state.send_q.get_nowait()
                    except queue.Empty:
                        pass
                    st.session_state.send_q.put_nowait(("audio", mono_16k.tobytes()))

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
            now = time.time()
            if now - st.session_state.last_plot >= 0.25:
                fig_place.pyplot(fig)
                st.session_state.last_plot = now

        while True:
            try:
                evt = st.session_state.recv_q.get_nowait()
            except queue.Empty:
                break
            on_evt(evt)

    else:
        if st.session_state.ws_running:
            try:
                st.session_state.send_q.put_nowait(("end", None))
            except Exception:
                pass
            st.session_state.ws_running = False
            st.session_state.ws_thread = None

    st.divider()
    render_events()

if __name__ == "__main__":
    main()
