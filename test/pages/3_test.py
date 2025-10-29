import asyncio, threading, json, io, time, queue
from datetime import datetime
import numpy as np, cv2, av, websockets
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="Simple Test", layout="wide")

def get_ws_url():
    try:
        return st.secrets.get("API_WS", "ws://140.116.158.98:9999/brain/ws")
    except Exception:
        return "ws://140.116.158.98:9999/brain/ws"

def resample_to_16k(mono_i16: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000: return mono_i16.astype(np.int16, copy=False)
    x = mono_i16.astype(np.float32); n_in = x.shape[-1]
    n_out = int(round(n_in * 16000 / sr))
    if n_out <= 0 or n_in <= 1: return np.zeros(0, dtype=np.int16)
    xp = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, xp, x).astype(np.int16)

def _init_state():
    ss = st.session_state
    ss.setdefault("results", [])
    ss.setdefault("lock", threading.Lock())
    ss.setdefault("audio_buffer", bytearray())

_init_state()

st.title("Simple Brain Test")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.divider()
    
    st.markdown("**Record Audio (optional)**")
    
    ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        media_stream_constraints={"audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    
    if ctx.state.playing and ctx.audio_receiver:
        try:
            audio_frames = ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            audio_frames = []
        
        for af in audio_frames:
            arr = af.to_ndarray()
            if arr.ndim == 2:
                mono = arr.mean(axis=0).astype(np.int16)
            else:
                mono = arr.astype(np.int16)
            sr = af.sample_rate
            pcm16 = resample_to_16k(mono, sr)
            st.session_state.audio_buffer.extend(pcm16.tobytes())
        
        audio_len = len(st.session_state.audio_buffer) / 16000 / 2
        st.info(f"Recording... {audio_len:.1f}s")
    
    if len(st.session_state.audio_buffer) > 0:
        st.success(f"Recorded {len(st.session_state.audio_buffer) / 16000 / 2:.1f}s of audio")
        if st.button("Clear Audio"):
            st.session_state.audio_buffer = bytearray()
            st.rerun()
    
    st.divider()
    
    instruction = st.text_input("Instruction", placeholder="e.g., 去告示牌")
    
    if st.button("Send to Brain", type="primary", disabled=not uploaded_image):
        if uploaded_image:
            ws_url = get_ws_url()
            img_bytes = uploaded_image.getvalue()
            
            with st.spinner("Processing..."):
                try:
                    async def send_and_receive():
                        async with websockets.connect(ws_url, max_size=2**24) as ws:
                            results = []
                            
                            if len(st.session_state.audio_buffer) > 0:
                                await ws.send(b"AUD0" + bytes(st.session_state.audio_buffer))
                                
                                try:
                                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                                    data = json.loads(response)
                                    if data.get("type") == "utterance":
                                        results.append(("ASR", data))
                                except asyncio.TimeoutError:
                                    pass
                            
                            await ws.send(img_bytes)
                            
                            response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                            data = json.loads(response)
                            if data.get("type") == "observe":
                                results.append(("Vision", data))
                            
                            return results
                    
                    results = asyncio.run(send_and_receive())
                    
                    with st.session_state.lock:
                        st.session_state.results.append({
                            "timestamp": datetime.now().strftime('%H:%M:%S'),
                            "instruction": instruction,
                            "results": results
                        })
                    
                    st.success("Done!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")

with col2:
    st.subheader("Results")
    
    with st.session_state.lock:
        results = st.session_state.results.copy()
    
    if not results:
        st.info("No results yet")
    else:
        for idx, res in enumerate(reversed(results)):
            with st.expander(f"[{res['timestamp']}] {res.get('instruction') or 'No instruction'}", expanded=(idx==0)):
                for result_type, data in res['results']:
                    st.markdown(f"**{result_type}**")
                    
                    if result_type == "ASR":
                        st.write(f"Text: {data.get('text')}")
                        st.write(f"Confidence: {data.get('confidence', 0):.2%}")
                    
                    elif result_type == "Vision":
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Intent", data.get('intent', '-'))
                            st.metric("Target", data.get('target', '-'))
                        with col_b:
                            st.metric("Direction", data.get('rel_dir', '-'))
                            st.metric("Distance", data.get('dist_label', '-'))
                        
                        if data.get('depth_m'):
                            st.metric("Depth (m)", f"{data['depth_m']:.2f}")
                        
                        if data.get('bbox'):
                            st.write(f"BBox: {data['bbox']}")
                        
                        ctrl = data.get('control', {})
                        if ctrl:
                            st.write(f"Control: v={ctrl.get('v', 0):.3f}, w={ctrl.get('w', 0):.3f}")
                        
                        perf = data.get('perf', {})
                        if perf:
                            st.caption(f"Latency: {perf.get('latency_ms', 0):.1f}ms")
                
                st.divider()

st.divider()

if st.button("Clear Results"):
    with st.session_state.lock:
        st.session_state.results = []
    st.rerun()