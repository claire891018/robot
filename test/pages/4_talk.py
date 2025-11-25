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


async def send_audio_and_receive(audio_data: bytes):
    """發送音訊並接收回覆"""
    results = {
        "user_text": "",
        "reply_text": "",
        "tts_audio": None,
        "error": None
    }
    
    try:
        async with websockets.connect(
            BRAIN_WS_URL,
            ping_interval=20,
            ping_timeout=10,
            max_size=2**23,
        ) as ws:
            # 分段發送音訊
            chunk_size = 3200  # 16000Hz * 0.1s * 2 bytes
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                if len(chunk) > 0:
                    await ws.send(b"AUD0" + chunk)
                    await asyncio.sleep(0.05)
            
            # 音訊發送完畢，等待回覆
            print("音訊發送完畢，等待伺服器回覆...")
            
            timeout_count = 0
            max_timeout = 60  # 最多等 60 秒
            
            while timeout_count < max_timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    
                    if isinstance(msg, (bytes, bytearray)):
                        b = bytes(msg)
                        if len(b) >= 4 and b[:4] == b"TTS0":
                            results["tts_audio"] = b[4:]
                            print("收到 TTS 音訊")
                            # 收到 TTS 通常表示對話結束
                            break
                    else:
                        # JSON 訊息
                        try:
                            evt = json.loads(msg)
                            print(f"收到訊息: {evt}")
                            
                            if evt.get("type") == "utterance":
                                results["user_text"] = evt.get("text", "")
                            elif evt.get("type") == "reply":
                                results["reply_text"] = evt.get("text", "")
                                
                        except json.JSONDecodeError:
                            print(f"無法解析 JSON: {msg}")
                            
                except asyncio.TimeoutError:
                    timeout_count += 1
                    # 如果已經收到回覆，可以提早結束
                    if results["reply_text"] and results["tts_audio"]:
                        break
                    continue
                    
    except Exception as e:
        results["error"] = str(e)
        print(f"WebSocket 錯誤: {e}")
    
    return results


def process_audio_sync(audio_bytes: bytes):
    """同步包裝異步函數"""
    return asyncio.run(send_audio_and_receive(audio_bytes))


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
    st.caption("按下 START 開始錄音 → 按 STOP 停止 → AI 會辨識並回覆語音")


def main():
    _init_state()
    render_header()

    # WebRTC 錄音
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
        st.subheader("🎤 錄音狀態")
        status_placeholder = st.empty()
        
        if ctx.state.playing:
            status_placeholder.success("🔴 正在錄音中... 請說話")
            st.session_state.is_recording = True
        else:
            if st.session_state.is_recording:
                status_placeholder.info("⏹️ 錄音已停止，正在處理...")
                st.session_state.is_recording = False
            else:
                status_placeholder.info("⏸️ 待機中，點擊 START 開始錄音")

    with col2:
        st.subheader("💬 對話紀錄")
        with st.container(height=500):
            if not st.session_state.conversation:
                st.info("還沒有對話紀錄")
            else:
                for role, text, ts in reversed(st.session_state.conversation):
                    if role == "user":
                        st.markdown(f"**你：** {text}")
                        st.caption(f"🕐 {ts}")
                        st.divider()
                    else:
                        st.markdown(f"**🤖 機器人：** {text}")
                        st.caption(f"🕐 {ts}")
                        st.divider()

    # 收集音訊數據
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
                
                if mono_16k.size > 0:
                    st.session_state.audio_buffer.append(mono_16k)
                    
        except queue.Empty:
            pass
    
    # 當停止錄音時，處理音訊
    elif len(st.session_state.audio_buffer) > 0 and not st.session_state.processing:
        st.session_state.processing = True
        
        with st.spinner("🤔 AI 正在處理中..."):
            # 合併所有音訊片段
            full_audio = np.concatenate(st.session_state.audio_buffer)
            audio_bytes = full_audio.tobytes()
            
            print(f"發送音訊長度: {len(full_audio)} samples ({len(full_audio)/16000:.2f} 秒)")
            
            # 發送並接收
            results = process_audio_sync(audio_bytes)
            
            # 處理結果
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if results["error"]:
                st.error(f"❌ 錯誤: {results['error']}")
            else:
                if results["user_text"]:
                    st.session_state.conversation.append(("user", results["user_text"], ts))
                if results["reply_text"]:
                    st.session_state.conversation.append(("bot", results["reply_text"], ts))
                if results["tts_audio"]:
                    st.session_state.tts_audio = results["tts_audio"]
                
                st.success("✅ 處理完成！")
            
            # 清空緩衝區
            st.session_state.audio_buffer = []
            st.session_state.processing = False
            
        st.rerun()

    # 播放 TTS
    st.subheader("🔊 機器人回覆語音")
    if st.session_state.tts_audio:
        st.audio(st.session_state.tts_audio, format="audio/wav", autoplay=True)
    else:
        st.info("還沒有語音回覆")

    # 控制按鈕
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ 清除對話內容", type="primary"):
            st.session_state.conversation = []
            st.session_state.tts_audio = None
            st.session_state.audio_buffer = []
            st.rerun()
    with c2:
        if st.button("🔄 重新整理"):
            st.rerun()


if __name__ == "__main__":
    main()