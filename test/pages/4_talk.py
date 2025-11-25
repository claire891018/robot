import sys, json, asyncio, logging, queue, threading, time, io, base64
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import numpy as np
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import matplotlib.pyplot as plt
import websockets
from PIL import Image

try:
    BRAIN_WS_URL = st.secrets.get("BRAIN_WS_URL", "ws://140.116.158.98:9999/brain/ws/chat")
except Exception:
    BRAIN_WS_URL = "ws://140.116.158.98:9999/brain/ws/chat"

st.set_page_config(
    page_title="對話機器人 Demo",
    page_icon="https://api.dicebear.com/9.x/thumbs/svg?",
    layout="wide",
)

logger = logging.getLogger(__name__)

def _init_state():
    ss = st.session_state
    ss.setdefault("listen_events", [])
    ss.setdefault("listen_lock", threading.Lock())
    ss.setdefault("sound_window_len", 5000)
    ss.setdefault("listen_send_q", asyncio.Queue(maxsize=64))
    ss.setdefault("listen_recv_q", asyncio.Queue())
    ss.setdefault("listen_ws_task", None)
    ss.setdefault("tts_last_audio", None)


def on_evt(evt: Dict):
    evt["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with st.session_state.listen_lock:
        st.session_state.listen_events.append(evt)

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


async def brain_loop_async(send_q: asyncio.Queue, recv_q: asyncio.Queue, url: str):
    async with websockets.connect(
        url,
        max_size=2**23,
        ping_interval=15,     # 重要：避免 keepalive timeout
        ping_timeout=15,
    ) as ws:

        async def reader():
            try:
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)):
                        b = bytes(msg)
                        if len(b) >= 4 and b[:4] == b"TTS0":
                            await recv_q.put({"type": "tts_audio", "audio": b[4:]})
                        else:
                            await recv_q.put({"type": "error", "error": "unknown_binary"})
                    else:
                        try:
                            evt = json.loads(msg)
                        except:
                            evt = {"type": "error", "error": "bad_json", "detail": msg}
                        await recv_q.put(evt)
            except Exception as e:
                await recv_q.put({"type": "error", "error": "ws_reader", "detail": str(e)})

        reader_task = asyncio.create_task(reader())

        try:
            while True:
                kind, payload = await send_q.get()
                if kind == "audio":
                    await ws.send(b"AUD0" + payload)
                elif kind == "end":
                    await ws.send(json.dumps({"type": "end"}))
                    break
        finally:
            reader_task.cancel()

def render_header():
    icon = "https://api.dicebear.com/9.x/thumbs/svg?"
    st.markdown(
        f"""
        <h2 style="display:flex;align-items:center;gap:.5rem;">
          <img src="{icon}" width="28" height="28" style="border-radius:20%; display:block;" />
          對話機器人 Demo
        </h2>
        """,
        unsafe_allow_html=True,
    )
    st.caption("點擊 START，對著麥克風說話，機器人會聽、回、而且開口說。")

def render_events(container):
    with container.container():
        with st.session_state.listen_lock:
            utterances = [
                e
                for e in st.session_state.listen_events
                if e.get("type") in ("utterance", "reply", "error")
            ]
            recent = utterances[-20:]
            recent.reverse()

        if not recent:
            st.info("等待對話中...")
        else:
            for evt in recent:
                t = evt.get("type")
                timestamp = evt.get("timestamp", "—")

                if t == "utterance":
                    txt = (evt.get("text") or "").strip()
                    conf = evt.get("confidence", 0.0)
                    st.markdown(f"**你：** {txt}")
                    st.caption(f"信心度: {conf:.2f} | 時間: {timestamp}")
                    st.divider()

                elif t == "reply":
                    txt = (evt.get("text") or "").strip()
                    st.markdown(f"**機器人：** {txt}")
                    st.caption(f"時間: {timestamp}")
                    st.divider()

                elif t == "error":
                    st.error(f"[錯誤] {evt.get('error')} - {evt.get('detail')}")


def main():
    _init_state()
    render_header()

    ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        media_stream_constraints={"audio": True},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("即時音訊波形")
        fig_place = st.empty()
    with col2:
        st.subheader("對話內容")
        with st.container(height=550):
            events_container = st.empty()

    st.subheader("機器人回覆語音")
    tts_player = st.empty()

    fig, (ax_time, ax_freq) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"hspace": 0.5}
    )
    sound_window_buffer = None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 啟動 WebSocket（不使用 thread）
    if st.session_state.listen_ws_task is None:
        st.session_state.listen_ws_task = loop.create_task(
            brain_loop_async(
                st.session_state.listen_send_q,
                st.session_state.listen_recv_q,
                BRAIN_WS_URL,
            )
        )

    def run_loop():
        try:
            loop.run_forever()
        except RuntimeError:
            pass

    threading.Thread(target=run_loop, daemon=True).start()

    while True:
        if ctx.state.playing and ctx.audio_receiver:

            try:
                audio_frames = ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                break

            sound_chunk = pydub.AudioSegment.empty()

            for af in audio_frames:
                arr = af.to_ndarray()
                if arr.ndim == 2:
                    mono = arr.mean(axis=0).astype(np.int16)
                else:
                    mono = arr.astype(np.int16)

                sr = af.sample_rate
                mono_16k = resample_to_16k(mono, sr)

                if mono_16k.size > 0:
                    try:
                        st.session_state.listen_send_q.put_nowait(
                            ("audio", mono_16k.tobytes())
                        )
                    except asyncio.QueueFull:
                        pass

                sound = pydub.AudioSegment(
                    data=arr.tobytes(),
                    sample_width=af.format.bytes,
                    frame_rate=sr,
                    channels=1,
                )
                sound_chunk += sound

            # ---- 波形顯示（完全保留你的版本）----
            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=st.session_state.sound_window_len
                    )
                sound_window_buffer += sound_chunk
                if len(sound_window_buffer) > st.session_state.sound_window_len:
                    sound_window_buffer = sound_window_buffer[
                        -st.session_state.sound_window_len :
                    ]

            if sound_window_buffer:
                sound_window_buffer = sound_window_buffer.set_channels(1)
                sample = np.array(sound_window_buffer.get_array_of_samples())

                ax_time.cla()
                times = (np.arange(-len(sample), 0)) / sound_window_buffer.frame_rate
                ax_time.plot(times, sample)
                ax_time.set_title("Time Domain")
                ax_time.grid(True, alpha=0.3)

                spec = np.fft.fft(sample)
                freq = np.fft.fftfreq(len(sample), 1.0 / sound_window_buffer.frame_rate)
                freq = freq[: len(freq) // 2]
                spec = np.abs(spec[: len(spec) // 2])

                ax_freq.cla()
                ax_freq.plot(freq, spec)
                ax_freq.set_yscale("log")
                ax_freq.set_title("Frequency Domain")
                ax_freq.grid(True, alpha=0.3)

                fig_place.pyplot(fig)

            # ---- 收 WS 回傳 ----
            while True:
                try:
                    evt = st.session_state.listen_recv_q.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if evt.get("type") == "tts_audio":
                    st.session_state.tts_last_audio = evt["audio"]
                else:
                    on_evt(evt)

            render_events(events_container)

            if st.session_state.tts_last_audio:
                tts_player.audio(st.session_state.tts_last_audio, format="audio/wav")

        else:
            st.session_state.listen_send_q.put_nowait(("end", None))
            break

    render_events(events_container)


if __name__ == "__main__":
    main()
