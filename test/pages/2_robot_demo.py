import asyncio, threading, json, io, base64
import numpy as np, cv2, av, websockets
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

def ws_url():
    try:
        return st.secrets.get("API_WS", "ws://140.116.158.98:9999/brain/ws")
    except Exception:
        return "ws://140.116.158.98:9999/brain/ws"

def cfg():
    st.set_page_config(page_title="Robot Demo", page_icon="🤖", layout="wide")
    st.markdown("<h2>Robot Demo（前端=感官＋監視器｜/brain/ws）</h2>", unsafe_allow_html=True)

def resample_to_16k(mono_i16: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000: return mono_i16.astype(np.int16, copy=False)
    x = mono_i16.astype(np.float32); n_in = x.shape[-1]
    n_out = int(round(n_in * 16000 / sr))
    if n_out <= 0 or n_in <= 1: return np.zeros(0, dtype=np.int16)
    xp = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, xp, x).astype(np.int16)

def jpeg_bytes(bgr: np.ndarray, q: int = 85) -> bytes:
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=q)
    return buf.getvalue()

async def ws_loop(send_q: asyncio.Queue, recv_q: asyncio.Queue, url: str):
    async with websockets.connect(url, max_size=2**24) as ws:
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
        ts = asyncio.create_task(sender())
        tr = asyncio.create_task(receiver())
        await asyncio.gather(ts, tr)

def main():
    cfg()
    url = ws_url()
    if "ws_started" not in st.session_state:
        st.session_state.ws_started = False
        st.session_state.loop = asyncio.new_event_loop()
        st.session_state.send_q = asyncio.Queue()
        st.session_state.recv_q = asyncio.Queue()
        st.session_state.shared = {"bbox":None,"depth_m":None,"v":0.0,"w":0.0,"instr":"","intent":None,"target":None,"steer":None,"turn":None,"waypoint":None,"pose":None,"latency":None,"model":None}
        st.session_state.lock = threading.Lock()
        st.session_state.last_asr = ""
        st.session_state.infer_every = 5

    colL, colR = st.columns([2,1])
    with colR:
        st.subheader("大腦輸出")
        info_instr = st.empty()
        info_state = st.empty()
        info_ctrl = st.empty()
        info_perf = st.empty()

    def start_ws_once():
        if not st.session_state.ws_started:
            threading.Thread(target=lambda: st.session_state.loop.run_until_complete(ws_loop(st.session_state.send_q, st.session_state.recv_q, url)), daemon=True).start()
            st.session_state.ws_started = True

    def send(kind, payload):
        st.session_state.loop.call_soon_threadsafe(asyncio.create_task, st.session_state.send_q.put((kind, payload)))

    class VideoProcessor:
        def __init__(self):
            self._cnt = 0
            start_ws_once()
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self._cnt += 1
            if self._cnt % int(max(1, st.session_state.infer_every)) == 0:
                send("bytes", jpeg_bytes(img.copy()))
            while True:
                try:
                    data = st.session_state.recv_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if data.get("type") == "observe":
                    with st.session_state.lock:
                        st.session_state.shared["bbox"] = tuple(data["bbox"]) if data.get("bbox") else None
                        st.session_state.shared["depth_m"] = data.get("depth_m")
                        st.session_state.shared["instr"] = data.get("instruction") or ""
                        st.session_state.shared["intent"] = data.get("intent")
                        st.session_state.shared["target"] = data.get("target")
                        g = data.get("guide") or {}
                        st.session_state.shared["steer"] = g.get("steer_angle_deg")
                        st.session_state.shared["turn"] = g.get("turn")
                        st.session_state.shared["waypoint"] = g.get("waypoint_img")
                        c = data.get("control") or {}
                        st.session_state.shared["v"] = float(c.get("v",0.0))
                        st.session_state.shared["w"] = float(c.get("w",0.0))
                        st.session_state.shared["pose"] = data.get("pose")
                        p = data.get("perf") or {}
                        st.session_state.shared["latency"] = p.get("latency_ms")
                        st.session_state.shared["model"] = p.get("model")
                elif data.get("type") == "utterance":
                    st.session_state.last_asr = data.get("text","")
            with st.session_state.lock:
                bbox = st.session_state.shared["bbox"]
                depth_m = st.session_state.shared["depth_m"]
                steer = st.session_state.shared["steer"]
                turn = st.session_state.shared["turn"]
                waypoint = st.session_state.shared["waypoint"]
                v = st.session_state.shared["v"]; w = st.session_state.shared["w"]
                instr = st.session_state.shared["instr"]
            H,W = img.shape[:2]
            if bbox:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cx,cy = int((x1+x2)/2), int((y1+y2)/2)
                cv2.circle(img,(cx,cy),6,(0,255,0),-1)
            if waypoint and isinstance(waypoint,(list,tuple)) and len(waypoint)==2:
                wx,wy = int(waypoint[0]), int(waypoint[1])
                cv2.circle(img,(wx,wy),7,(255,255,0),-1)
                cv2.arrowedLine(img,(W//2,H-30),(wx,wy),(255,255,0),2, tipLength=0.15)
            top = f"instr={instr or st.session_state.last_asr}"
            mid = f"depth_m={None if depth_m is None else round(depth_m,2)} steer={None if steer is None else round(steer,1)} turn={turn}"
            bot = f"v={v:.2f} w={w:.2f}"
            cv2.putText(img, top, (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2)
            cv2.putText(img, mid, (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
            cv2.putText(img, bot, (10,78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

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
            frames = ctx.audio_receiver.get_frames(timeout=1)
        except Exception:
            frames = []
        for af in frames:
            arr = af.to_ndarray()
            if arr.ndim == 2: mono = arr.mean(axis=0).astype(np.int16)
            else: mono = arr.astype(np.int16)
            sr = af.sample_rate
            pcm = resample_to_16k(mono, sr).tobytes()
            send("bytes", b"AUD0"+pcm)

    with st.sidebar:
        st.caption("前端只送音訊與影像；姿態由後端維護。")

    with colR:
        with st.session_state.lock:
            s = st.session_state.shared
        info_instr.markdown(f"**Instruction**：{s['instr'] or st.session_state.last_asr or '…'}")
        info_state.markdown(
            f"**Intent/Target**：{s['intent']} / {s['target']}  \n"
            f"**Depth(m)**：{None if s['depth_m'] is None else round(s['depth_m'],2)}  "
            f"**Steer(°)**：{None if s['steer'] is None else round(s['steer'],1)}  "
            f"**Turn**：{s['turn']}"
        )
        info_ctrl.markdown(
            f"**Control** v={s['v']:.2f}, w={s['w']:.2f}  \n"
            f"**Pose**：{s['pose']}"
        )
        info_perf.markdown(
            f"**Perf**：latency={s['latency']} ms  model={s['model']}"
        )

if __name__ == "__main__":
    main()
