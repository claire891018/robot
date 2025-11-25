import sys, json, asyncio, logging, queue, threading, time, io, base64
from datetime import datetime
from typing import Dict

import numpy as np
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import websockets

try:
    BRAIN_WS_URL = st.secrets.get("BRAIN_WS_URL", "ws://140.116.158.98:9999/brain/ws/chat")
except Exception:
    BRAIN_WS_URL = "ws://140.116.158.98:9999/brain/ws/chat"

st.set_page_config(page_title="對話機器人 Demo",
                   page_icon="https://api.dicebear.com/9.x/thumbs/svg?",
                   layout="wide")

def _init_state():
    ss = st.session_state
    ss.setdefault("conversation", [])
    ss.setdefault("is_recording", False)
    ss.setdefault("audio_buffer", [])
    ss.setdefault("tts_audio", None)
    ss.setdefault("processing", False)

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

async def send_ws(audio_bytes: bytes):
    result = {"user_text": "", "reply_text": "", "tts_audio": None, "error": None}

    try:
        async with websockets.connect(
            BRAIN_WS_URL,
            max_size=2**23,
            ping_interval=None
        ) as ws:
            chunk_size = 3200
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                if len(chunk) > 0:
                    await ws.send(b"AUD0" + chunk)
                    await asyncio.sleep(0.05)

            await ws.send(json.dumps({"type": "end"}))

            timeout = 0
            while timeout < 60:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1)
                    if isinstance(msg, (bytes, bytearray)):
                        b = bytes(msg)
                        if b[:4] == b"TTS0":
                            result["tts_audio"] = b[4:]
                    else:
                        evt = json.loads(msg)
                        if evt.get("type") == "utterance":
                            result["user_text"] = evt.get("text", "")
                        elif evt.get("type") == "reply":
                            result["reply_text"] = evt.get("text", "")
                    if result["tts_audio"] and result["reply_text"]:
                        break
                except asyncio.TimeoutError:
                    timeout += 1
                    continue
    except Exception as e:
        result["error"] = str(e)

    return result

def process_audio_sync(audio_bytes: bytes):
    return asyncio.run(send_ws(audio_bytes))

def header():
    icon = "https://api.dicebear.com/9.x/thumbs/svg?"
    st.markdown(
        f"""
        <h2 style="display:flex;align-items:center;gap:.5rem;">
          <img src="{icon}" width="28" height="28" />
          對話機器人 Demo (WebSocket 單段對話)
        </h2>
        """,
        unsafe_allow_html=True,
    )

def main():
    _init_state()
    header()

    ctx = webrtc_streamer(
        key="rec-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        media_stream_constraints={"audio": True},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("錄音")
        state = st.empty()
        if ctx.state.playing:
            state.success("錄音中...")
            st.session_state.is_recording = True
        else:
            if st.session_state.is_recording:
                state.info("錄音停止，正在處理...")
                st.session_state.is_recording = False
            else:
                state.info("待機中...")

    with col2:
        st.subheader("對話紀錄")
        box = st.container(height=500)
        with box:
            for role, text, ts in reversed(st.session_state.conversation):
                if role == "user":
                    st.markdown(f"**你：** {text}")
                    st.caption(ts)
                    st.divider()
                else:
                    st.markdown(f"**機器人：** {text}")
                    st.caption(ts)
                    st.divider()

    if ctx.state.playing and ctx.audio_receiver:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=1)
            for af in frames:
                arr = af.to_ndarray()
                if arr.ndim == 2:
                    mono = arr.mean(axis=0).astype(np.int16)
                else:
                    mono = arr.astype(np.int16)
                sr = af.sample_rate
                mono_16k = resample_to_16k(mono, sr)
                if mono_16k.size > 0:
                    st.session_state.audio_buffer.append(mono_16k)
        except queue.Empty:
            pass

    elif len(st.session_state.audio_buffer) > 0 and not st.session_state.processing:
        st.session_state.processing = True

        with st.spinner("AI 正在處理中..."):
            full_audio = np.concatenate(st.session_state.audio_buffer)
            audio_bytes = full_audio.tobytes()
            result = process_audio_sync(audio_bytes)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if result["user_text"]:
                st.session_state.conversation.append(("user", result["user_text"], ts))
            if result["reply_text"]:
                st.session_state.conversation.append(("bot", result["reply_text"], ts))
            if result["tts_audio"]:
                st.session_state.tts_audio = result["tts_audio"]

            st.session_state.audio_buffer = []
            st.session_state.processing = False

        st.rerun()

    st.subheader("機器人回覆語音")
    if st.session_state.tts_audio:
        st.audio(st.session_state.tts_audio, format="audio/wav")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("清除對話紀錄"):
            st.session_state.conversation = []
            st.session_state.tts_audio = None
            st.session_state.audio_buffer = []
            st.rerun()
    with c2:
        if st.button("重整"):
            st.rerun()

if __name__ == "__main__":
    main()
