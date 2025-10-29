import asyncio, threading, json, io, time
from datetime import datetime
import numpy as np, cv2, websockets
import streamlit as st
from PIL import Image
from pydub import AudioSegment

st.set_page_config(page_title="Simple Test", layout="wide")

def get_ws_url():
    try:
        return st.secrets.get("API_WS", "ws://140.116.158.98:9999/brain/ws")
    except Exception:
        return "ws://140.116.158.98:9999/brain/ws"

def audio_to_pcm16(audio_bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    return audio.raw_data

def _init_state():
    ss = st.session_state
    ss.setdefault("results", [])
    ss.setdefault("lock", threading.Lock())

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
    
    audio_file = st.file_uploader("Upload Audio (optional)", type=["wav", "mp3", "m4a"])
    
    if audio_file:
        st.audio(audio_file)
    
    st.divider()
    
    instruction = st.text_input("Instruction (optional)", placeholder="e.g., 去告示牌")
    
    if st.button("Send to Brain", type="primary", disabled=not uploaded_image):
        if uploaded_image:
            ws_url = get_ws_url()
            img_bytes = uploaded_image.getvalue()
            
            with st.spinner("Processing..."):
                try:
                    async def send_and_receive():
                        async with websockets.connect(ws_url, max_size=2**24) as ws:
                            results = []
                            
                            if audio_file:
                                audio_bytes = audio_file.getvalue()
                                pcm_bytes = audio_to_pcm16(audio_bytes)
                                await ws.send(b"AUD0" + pcm_bytes)
                                
                                try:
                                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                                    data = json.loads(response)
                                    if data.get("type") == "utterance":
                                        results.append(("ASR", data))
                                except asyncio.TimeoutError:
                                    results.append(("ASR", {"error": "timeout"}))
                            
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
                        if data.get("error"):
                            st.error(f"Error: {data['error']}")
                        else:
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