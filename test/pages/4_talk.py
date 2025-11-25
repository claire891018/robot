import streamlit as st
import requests
import base64
from datetime import datetime

API_URL = "http://140.116.158.98:9999/brain/voice_chat"

st.set_page_config(
    page_title="對話機器人 Demo",
    page_icon="https://api.dicebear.com/9.x/thumbs/svg?",
    layout="wide",
)

if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "tts_audio" not in st.session_state:
    st.session_state.tts_audio = None

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
    st.caption("按下錄音 → 說一句話 → AI 會回覆你並播放語音。")

render_header()

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

if audio_bytes is not None:
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    with st.spinner("AI 思考中..."):
        r = requests.post(API_URL, files=files, timeout=120)
        data = r.json()
    user_text = data.get("user_text","")
    reply_text = data.get("reply_text","")
    tts_b64 = data.get("tts","")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if user_text:
        st.session_state.conversation.append(("user", user_text, ts))
    if reply_text:
        st.session_state.conversation.append(("bot", reply_text, ts))
    if tts_b64:
        st.session_state.tts_audio = base64.b64decode(tts_b64)
    st.rerun()

st.subheader("機器人回覆語音")
if st.session_state.tts_audio:
    st.audio(st.session_state.tts_audio, format="audio/wav")

c1, c2 = st.columns(2)
with c1:
    if st.button("清除對話內容"):
        st.session_state.conversation = []
        st.session_state.tts_audio = None
        st.rerun()
with c2:
    if st.button("重整頁面"):
        st.rerun()
