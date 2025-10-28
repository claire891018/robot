import asyncio, threading, json, io, queue
from datetime import datetime
import numpy as np, cv2, av, websockets
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="Robot Demo", layout="wide")

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

def jpeg_bytes(frame_bgr: np.ndarray) -> bytes:
    im = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

async def brain_loop_async(send_q: queue.Queue, recv_q: queue.Queue, url: str):
    print(f"[BRAIN_LOOP] Connecting to {url}")
    async with websockets.connect(url, max_size=2**24) as ws:
        print(f"[BRAIN_LOOP] Connected")
        
        async def reader():
            try:
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        print(f"[BRAIN_LOOP] Got {data.get('type')}")
                    except Exception:
                        data = {"type": "raw", "value": msg}
                    recv_q.put(data)
            except Exception as e:
                print(f"[BRAIN_LOOP] Reader error: {e}")
                recv_q.put({"type": "error", "error": "ws_reader", "detail": str(e)})
        
        reader_task = asyncio.create_task(reader())
        try:
            while True:
                kind, payload = await asyncio.to_thread(send_q.get)
                if kind == "end":
                    print(f"[BRAIN_LOOP] End signal")
                    await ws.send(json.dumps({"type": "end"}))
                    break
                elif kind == "audio":
                    print(f"[BRAIN_LOOP] Sending audio {len(payload)} bytes")
                    await ws.send(b"AUD0" + payload)
                elif kind == "video":
                    print(f"[BRAIN_LOOP] Sending video {len(payload)} bytes")
                    await ws.send(payload)
        finally:
            await reader_task

def ws_worker(send_q: queue.Queue, recv_q: queue.Queue, url: str):
    asyncio.run(brain_loop_async(send_q, recv_q, url))

def _init_state():
    ss = st.session_state
    ss.setdefault("brain_send_q", queue.Queue(maxsize=64))
    ss.setdefault("brain_recv_q", queue.Queue())
    ss.setdefault("brain_ws_thread", None)
    ss.setdefault("brain_ws_running", False)
    ss.setdefault("shared", {
        "bbox":None,"depth_m":None,"intent":None,"target":None,
        "rel_dir":None,"dist_label":None,"last_ctrl":None,
        "instruction":"","pose":None,"perf":None
    })
    ss.setdefault("lock", threading.Lock())
    ss.setdefault("infer_every", 5)
    ss.setdefault("asr_history", [])

class VideoProcessor:
    def __init__(self):
        self._cnt = 0
        print(f"[VideoProcessor] init")
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._cnt += 1
        if self._cnt % 30 == 1:
            print(f"[VideoProcessor] recv called {self._cnt} times")
        
        img = frame.to_ndarray(format="bgr24")
        
        if self._cnt % int(max(1, st.session_state.get("infer_every", 5))) == 0:
            try:
                st.session_state.brain_send_q.put_nowait(("video", jpeg_bytes(img.copy())))
                print(f"[VideoProcessor] Sent video frame #{self._cnt}")
            except queue.Full:
                print(f"[VideoProcessor] Queue full, skipping frame")
        
        while True:
            try:
                data = st.session_state.brain_recv_q.get_nowait()
            except queue.Empty:
                break
            
            if data.get("type") == "observe":
                print(f"[VideoProcessor] Got observe response")
                p = data
                with st.session_state.lock:
                    st.session_state.shared["bbox"] = tuple(p["bbox"]) if p.get("bbox") else None
                    st.session_state.shared["depth_m"] = p.get("depth_m")
                    st.session_state.shared["intent"] = p.get("intent")
                    st.session_state.shared["target"] = p.get("target")
                    st.session_state.shared["rel_dir"] = p.get("rel_dir")
                    st.session_state.shared["dist_label"] = p.get("dist_label")
                    st.session_state.shared["instruction"] = p.get("instruction","")
                    c = p.get("control") or {}
                    st.session_state.shared["last_ctrl"] = (float(c.get("v",0.0)), float(c.get("w",0.0)))
            
            elif data.get("type") == "utterance":
                text = data.get("text", "").strip()
                print(f"[VideoProcessor] Got utterance: {text}")
                with st.session_state.lock:
                    st.session_state.shared["instruction"] = text
                    st.session_state.asr_history.append({
                        "text": text,
                        "confidence": data.get("confidence", 0.0),
                        "timestamp": datetime.now().strftime('%H:%M:%S')
                    })
        
        with st.session_state.lock:
            bbox = st.session_state.shared["bbox"]
            instr = st.session_state.shared["instruction"]
        
        if bbox:
            x1,y1,x2,y2 = bbox
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        
        cv2.putText(img, f"Frame {self._cnt}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if instr:
            cv2.putText(img, f"CMD: {instr}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    _init_state()
    ws_url = get_ws_url()
    
    st.title("Robot Demo")
    
    ctx = webrtc_streamer(
        key="av",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": True},
        video_processor_factory=VideoProcessor,
        async_processing=True,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
    )
    
    st.slider("Inference Interval", 1, 20, key="infer_every")
    
    if ctx.state.playing and ctx.audio_receiver:
        print(f"[MAIN] WebRTC is playing")
        
        if not st.session_state.brain_ws_running:
            print(f"[MAIN] Starting WebSocket worker")
            t = threading.Thread(
                target=ws_worker,
                args=(st.session_state.brain_send_q, st.session_state.brain_recv_q, ws_url),
                daemon=True,
            )
            t.start()
            st.session_state.brain_ws_thread = t
            st.session_state.brain_ws_running = True
            print(f"[MAIN] WebSocket worker started")
        
        try:
            audio_frames = ctx.audio_receiver.get_frames(timeout=1)
            print(f"[MAIN] Got {len(audio_frames)} audio frames")
        except queue.Empty:
            audio_frames = []
        
        for af in audio_frames:
            arr = af.to_ndarray()
            if arr.ndim == 2:
                mono = arr.mean(axis=0).astype(np.int16)
            else:
                mono = arr.astype(np.int16)
            sr = af.sample_rate
            pcm16 = resample_to_16k(mono, sr).tobytes()
            
            try:
                st.session_state.brain_send_q.put_nowait(("audio", pcm16))
                print(f"[MAIN] Sent audio {len(pcm16)} bytes")
            except queue.Full:
                print(f"[MAIN] Audio queue full")
    else:
        print(f"[MAIN] WebRTC not playing")
        # 停止 WebSocket
        if st.session_state.brain_ws_running:
            try:
                st.session_state.brain_send_q.put_nowait(("end", None))
            except Exception:
                pass
            st.session_state.brain_ws_running = False
            st.session_state.brain_ws_thread = None
    
    with st.container(border=True):
        with st.session_state.lock:
            asr_hist = st.session_state.asr_history.copy()
        
        if asr_hist:
            latest = asr_hist[-1]
            st.success(f"Last command: {latest['text']}")

if __name__ == "__main__":
    main()