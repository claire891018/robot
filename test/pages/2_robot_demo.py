import asyncio, threading, json, io, base64
import numpy as np, cv2, av, websockets
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(
    page_title="Robot Demo",
    page_icon="https://api.dicebear.com/9.x/thumbs/svg?seed=Brian",
    layout="wide",
)

def get_ws_url():
    try:
        return st.secrets.get("API_WS", "ws://127.0.0.1:9999/brain/ws")
    except Exception:
        return "ws://127.0.0.1:9999/brain/ws"

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

async def ws_loop(send_q: asyncio.Queue, recv_q: asyncio.Queue, ws_url: str):
    async with websockets.connect(ws_url, max_size=2**24) as ws:
        async def sender():
            while True:
                kind, payload = await send_q.get()
                if kind == "end":
                    await ws.send(json.dumps({"type":"end"})); break
                if kind == "bytes":
                    await ws.send(payload)
                elif kind == "text":
                    await ws.send(json.dumps(payload))
        async def receiver():
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    data = {"type":"raw","value":msg}
                await recv_q.put(data)
        task_s = asyncio.create_task(sender())
        task_r = asyncio.create_task(receiver())
        await asyncio.gather(task_s, task_r)

def _init_state():
    ss = st.session_state
    ss.setdefault("ws_started", False)
    ss.setdefault("loop", None)
    ss.setdefault("send_q", None)
    ss.setdefault("recv_q", None)
    ss.setdefault("shared", {"bbox":None,"depth_m":None,"intent":None,"target":None,"rel_dir":None,"dist_label":None,"last_ctrl":None,"instruction":"","pose":None,"perf":None})
    ss.setdefault("lock", threading.Lock())
    ss.setdefault("infer_every", 5)

def start_ws_once(ws_url: str):
    ss = st.session_state
    if not ss.get("ws_started", False):
        if ss.get("loop") is None:
            ss["loop"] = asyncio.new_event_loop()
        if ss.get("send_q") is None:
            ss["send_q"] = asyncio.Queue()
        if ss.get("recv_q") is None:
            ss["recv_q"] = asyncio.Queue()
        threading.Thread(target=lambda: ss["loop"].run_until_complete(ws_loop(ss["send_q"], ss["recv_q"], ws_url)), daemon=True).start()
        ss["ws_started"] = True

def enqueue(kind, payload):
    st.session_state.loop.call_soon_threadsafe(asyncio.create_task, st.session_state.send_q.put((kind, payload)))

class VideoProcessor:
    def __init__(self):
        self._cnt = 0
        start_ws_once(get_ws_url())
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._cnt += 1
        if self._cnt % int(max(1, st.session_state.get("infer_every", 5))) == 0:
            enqueue("bytes", jpeg_bytes(img.copy()))
        got = 0
        while True:
            try:
                data = st.session_state.recv_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            got += 1
            if data.get("type") == "observe":
                p = data
                with st.session_state.lock:
                    st.session_state.shared["bbox"] = tuple(p["bbox"]) if p.get("bbox") else None
                    st.session_state.shared["depth_m"] = p.get("depth_m")
                    st.session_state.shared["intent"] = p.get("intent")
                    st.session_state.shared["target"] = p.get("target")
                    st.session_state.shared["rel_dir"] = p.get("rel_dir")
                    st.session_state.shared["dist_label"] = p.get("dist_label")
                    st.session_state.shared["instruction"] = p.get("instruction","")
                    st.session_state.shared["pose"] = p.get("pose")
                    st.session_state.shared["perf"] = p.get("perf")
                    c = p.get("control") or {}
                    st.session_state.shared["last_ctrl"] = (float(c.get("v",0.0)), float(c.get("w",0.0)))
            elif data.get("type") == "utterance":
                with st.session_state.lock:
                    st.session_state.shared["instruction"] = data.get("text","")
        with st.session_state.lock:
            bbox = st.session_state.shared["bbox"]
            depth_m = st.session_state.shared["depth_m"]
            intent = st.session_state.shared["intent"]
            target = st.session_state.shared["target"]
            rel_dir = st.session_state.shared["rel_dir"]
            dist_label = st.session_state.shared["dist_label"]
            last_ctrl = st.session_state.shared["last_ctrl"]
            instr = st.session_state.shared["instruction"]
        if bbox:
            x1,y1,x2,y2 = bbox
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cx = (x1+x2)//2; cy = (y1+y2)//2
            h, w = img.shape[:2]
            cv2.arrowedLine(img,(w//2,h-20),(cx,cy),(0,255,255),2, tipLength=0.15)
            cv2.circle(img,(cx,cy),5,(0,255,255),-1)
        cv2.putText(img, f"intent={intent} target={target} rel={rel_dir} dist={dist_label}", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(img, f"depth_m={None if depth_m is None else round(float(depth_m),2)}", (10,54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if last_ctrl:
            cv2.putText(img, f"v={last_ctrl[0]:.2f} w={last_ctrl[1]:.2f}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        if instr:
            cv2.putText(img, f"instr={instr}", (10,106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
def render_header():
    icon = "https://api.dicebear.com/9.x/thumbs/svg?seed=Brian"
    st.markdown(
        f'''
        <h2 style="display:flex;align-items:center;gap:.5rem;">
            <img src="{icon}" width="28" height="28" style="border-radius:20%; display:block;" />
            Robot Demo
        </h2>
        ''',
        unsafe_allow_html=True,
    )

def main():
    _init_state()
    ws_url = get_ws_url()
    render_header()
    colL, colR = st.columns([2,1])
    with colL:
        st.subheader("視覺串流")
        st.slider("推理間隔(幀)", 1, 20, key="infer_every")
        ctx = webrtc_streamer(
            key="av",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": True},
            video_processor_factory=VideoProcessor,
            async_processing=True,
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        )
        if ctx.state.playing and ctx.audio_receiver:
            try:
                audio_frames = ctx.audio_receiver.get_frames(timeout=1)
            except Exception:
                audio_frames = []
            for af in audio_frames:
                arr = af.to_ndarray()
                if arr.ndim == 2:
                    mono = arr.mean(axis=0).astype(np.int16)
                else:
                    mono = arr.astype(np.int16)
                sr = af.sample_rate
                pcm16 = resample_to_16k(mono, sr).tobytes()
                enqueue("bytes", b"AUD0" + pcm16)
    with colR:
        st.subheader("大腦輸出")
        with st.container(border=True):
            with st.session_state.lock:
                s = st.session_state.shared.copy()
            st.write(f"Instruction: {s.get('instruction') or '…'}")
            st.write(f"Intent/Target: {s.get('intent')} / {s.get('target')}")
            st.write(f"RelDir/DistLabel: {s.get('rel_dir')} / {s.get('dist_label')}")
            st.write(f"Depth(m): {None if s.get('depth_m') is None else round(float(s.get('depth_m')),2)}")
            v,w = (s.get('last_ctrl') or (0.0,0.0))
            st.write(f"Control v,w: {v:.3f}, {w:.3f}")
            st.write(f"Pose: {s.get('pose')}")
            st.write(f"Perf: {s.get('perf')}")

    st.caption("此頁面同時送音訊與影像到 /brain/ws，並即時顯示大腦的觀測與控制輸出。")

if __name__ == "__main__":
    main()
