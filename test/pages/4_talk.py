import streamlit as st
import websockets
import asyncio
import numpy as np
import wave
import base64
import json
from datetime import datetime

WS_URL = "ws://140.116.158.98:9999/brain/ws/chat"

st.set_page_config(
    page_title="對話機器人 Demo",
    page_icon="https://api.dicebear.com/9.x/thumbs/svg?",
    layout="wide",
)

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "tts_audio" not in st.session_state:
    st.session_state.tts_audio = None

def header():
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
    st.caption("按下錄音 → 說一句話 → AI 會回覆你並播放語音。")

header()

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("錄音")
    audio_bytes = st.audio_input("按下錄音開始", label_visibility="hidden")

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

def load_wav_16k_mono(raw):
    with wave.open(raw, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        data = wf.readframes(n)
    audio = np.frombuffer(data, dtype=np.int16)
    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    if sr != 16000:
        x = audio.astype(np.float32)
        n_in = x.shape[0]
        n_out = int(round(n_in * 16000 / sr))
        xp = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
        xn = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(xn, xp, x).astype(np.int16)
    return audio

async def ws_send_and_wait(audio_pcm: np.ndarray):
    try:
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            chunk_size = 1600
            for i in range(0, len(audio_pcm), chunk_size):
                await ws.send(b"AUD0" + audio_pcm[i:i+chunk_size].tobytes())
                await asyncio.sleep(0.05)

            user_text = ""
            reply_text = ""
            tts_audio = None

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                except:
                    break

                if isinstance(msg, bytes):
                    b = bytes(msg)
                    if b[:4] == b"TTS0":
                        tts_audio = b[4:]
                        break
                else:
                    evt = json.loads(msg)
                    if evt.get("type") == "utterance":
                        user_text = evt.get("text", "")
                    elif evt.get("type") == "reply":
                        reply_text = evt.get("text", "")

            return user_text, reply_text, tts_audio

    except Exception as e:
        return "", f"WebSocket錯誤: {e}", None

if audio_bytes is not None:
    with st.spinner("AI 思考中..."):
        audio_pcm = load_wav_16k_mono(audio_bytes)
        user_text, reply_text, tts_audio = asyncio.run(ws_send_and_wait(audio_pcm))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if user_text:
        st.session_state.conversation.append(("user", user_text, ts))
    if reply_text:
        st.session_state.conversation.append(("bot", reply_text, ts))
    if tts_audio:
        st.session_state.tts_audio = tts_audio


st.subheader("機器人回覆語音")
if st.session_state.tts_audio:
    st.audio(st.session_state.tts_audio, format="audio/wav")

c1, c2 = st.columns(2)
with c1:
    if st.button("清除對話內容"):
        st.session_state.conversation = []
        st.session_state.tts_audio = None
with c2:
    if st.button("重整頁面"):
        st.session_state.conversation = []
        st.session_state.tts_audio = None
