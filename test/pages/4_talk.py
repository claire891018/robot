import sys, json, asyncio, logging, queue, threading, time, io, base64
from datetime import datetime
from typing import Dict

import numpy as np
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import websockets
import matplotlib.pyplot as plt

try:
    BRAIN_WS_URL = st.secrets.get("BRAIN_WS_URL", "ws://140.116.158.98:9999/brain/ws/chat")
except Exception:
    BRAIN_WS_URL = "ws://140.116.158.98:9999/brain/ws/chat"

st.set_page_config(
    page_title="對話機器人 Demo",
    page_icon="https://api.dicebear.com/9.x/thumbs/svg?",
    layout="wide",
)


def _init_state():
    ss = st.session_state
    ss.setdefault("conversation", [])
    ss.setdefault("audio_buffer", [])
    ss.setdefault("tts_audio", None)
    ss.setdefault("processing", False)
    ss.setdefault("is_recording", False)


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


async def send_audio(audio_bytes: bytes):
    results = {"user_text": "", "reply_text": "", "tts_audio": None, "error": None}
    try:
        async with websockets.connect(
            BRAIN_WS_URL,
            ping_interval=None,
            max_size=2**23,
        ) as ws:
            chunk = 3200
            for i in range(0, len(audio_bytes), chunk):
                await ws.send(b"AUD0" + audio_bytes[i:i+chunk])
                await asyncio.sleep(0.05)

            received_tts = False
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=60)
                except asyncio.TimeoutError:
                    break

                if isinstance(msg, (bytes, bytearray)):
                    b = bytes(msg)
                    if len(b) >= 4 and b[:4] == b"TTS0":
                        results["tts_audio"] = b[4:]
                        received_tts = True
                        break
                else:
                    try:
                        evt = json.loads(msg)
                        if evt.get("type") == "utterance":
                            results["user_text"] = evt.get("text", "")
                        elif evt.get("type") == "reply":
                            results["reply_text"] = evt.get("text", "")
                    except:
                        pass

            return results
    except Exception as e:
        results["error"] = str(e)
        return results


def sync_send(audio_bytes: bytes):
    return asyncio.run(send_audio(audio_bytes))


def render_header():
    icon = "https://api.dicebear.com/9.x/thumbs/svg?"
    st.markdown(
        f"""
        <h2 style="display:flex;align-items:center;gap:.5rem;">
          <img src="{icon}" width="28" height="28" style="border-radius:20%; display:block;" />
          對話機器人 Demo (WebSocket)
        </h2>
        """,
        unsafe_allow_html=True,
    )
    st.caption("按 START → 說話 → STOP → AI 會回你並且說話")


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
        st.subheader("錄音狀態")
        ph = st.empty()
        if ctx.state.playing:
            st.session_state.is_recording = True
            ph.success("🔴 錄音中...")
        else:
            if st.session_state.is_recording:
                ph.info("⏹️ 停止錄音，正在處理...")
            else:
                ph.info("待機中，按 START 開始錄音")

    with col2:
        st.subheader("對話紀錄")
        with st.container(height=500):
            for role, text, ts in reversed(st.session_state.conversation):
                if role == "user":
                    st.markdown(f"**你：** {text}")
                    st.caption(f"{ts}")
                else:
                    st.markdown(f"**機器人：** {text}")
                    st.caption(f"{ts}")
                st.divider()

    if ctx.state.playing and ctx.audio_receiver:
        try:
            audio_frames = ctx.audio_receiver.get_frames(timeout=1)
            for af in audio_frames:
                arr = af.to_ndarray()
                if arr.ndim == 2:
                    mono = arr.mean(axis=0).astype(np.int16)
                else:
                    mono = arr.astype(np.int16)
                sr = af.sample_rate
                mono_16k = resample_to_16k(mono, sr)
                st.session_state.audio_buffer.append(mono_16k)
        except queue.Empty:
            pass

    elif not ctx.state.playing and st.session_state.is_recording:
        st.session_state.is_recording = False
        if not st.session_state.processing:
            st.session_state.processing = True
            with st.spinner("AI 處理中..."):
                audio = np.concatenate(st.session_state.audio_buffer)
                results = sync_send(audio.tobytes())
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if results["user_text"]:
                    st.session_state.conversation.append(("user", results["user_text"], ts))
                if results["reply_text"]:
                    st.session_state.conversation.append(("bot", results["reply_text"], ts))
                if results["tts_audio"]:
                    st.session_state.tts_audio = results["tts_audio"]

            st.session_state.audio_buffer = []
            st.session_state.processing = False
            st.rerun()

    st.subheader("機器人語音")
    if st.session_state.tts_audio:
        st.audio(st.session_state.tts_audio, format="audio/wav")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("清除對話"):
            st.session_state.conversation = []
            st.session_state.tts_audio = None
            st.session_state.audio_buffer = []
            st.rerun()
    with c2:
        if st.button("重整頁面"):
            st.rerun()


if __name__ == "__main__":
    main()
