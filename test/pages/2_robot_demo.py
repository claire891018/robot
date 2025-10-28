import asyncio, threading, json, io, base64
import numpy as np, cv2, av, websockets, requests
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

def get_ws_url():
    try:
        return st.secrets.get("API_WS", "ws://140.116.158.98:9999/brain/ws")
    except Exception:
        return "ws://140.116.158.98:9999/brain/ws"

def render_header():
    icon = "https://api.dicebear.com/9.x/thumbs/svg?seed=Brian"
    st.set_page_config(page_title="Robot Demo", page_icon=icon, layout="wide")
    st.markdown(f'''<h2 style="display:flex;align-items:center;gap:.5rem;">
    <img src="{icon}" width="28" height="28" style="border-radius:20%; display:block;" />
    Robot Demo（同頁：音訊+影像 → /brain/ws）
    </h2>''', unsafe_allow_html=True)

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
                    data = {"type":"raw", "value": msg}
                await recv_q.put(data)
        task_s = asyncio.create_task(sender())
        task_r = asyncio.create_task(receiver())
        await asyncio.gather(task_s, task_r)

def main():
    render_header()
    ws_url = get_ws_url()

    pose_col = st.columns(3)
    v_x = pose_col[0].number_input("x", value=0.0, step=0.1)
    v_y = pose_col[1].number_input("y", value=0.0, step=0.1)
    v_th = pose_col[2].number_input("theta", value=0.0, step=0.1)

    row = st.columns(3)
    infer_every = row[0].number_input("每幀間隔(推理頻率)", value=5, min_value=1, max_value=30, step=1)
    show_json = row[1].toggle("顯示 JSON", value=False)
    pose_every_s = row[2].number_input("姿態上傳秒數", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

    if "ws_started" not in st.session_state:
        st.session_state.ws_started = False
        st.session_state.loop = asyncio.new_event_loop()
        st.session_state.send_q = asyncio.Queue()
        st.session_state.recv_q = asyncio.Queue()
        st.session_state.shared = {"bbox":None,"depth_m":None,"intent":None,"target":None,"rel_dir":None,"dist_label":None,"last_ctrl":None,"instruction":""}
        st.session_state.lock = threading.Lock()
        st.session_state.last_pose_ts = 0.0

    place_json = st.empty()
    place_ctrl = st.empty()
    place_asr = st.empty()

    def start_ws_once():
        if not st.session_state.ws_started:
            threading.Thread(target=lambda: st.session_state.loop.run_until_complete(ws_loop(st.session_state.send_q, st.session_state.recv_q, ws_url)), daemon=True).start()
            st.session_state.ws_started = True

    def enqueue(kind, payload):
        st.session_state.loop.call_soon_threadsafe(asyncio.create_task, st.session_state.send_q.put((kind, payload)))

    class VideoProcessor:
        def __init__(self):
            self._cnt = 0
            start_ws_once()
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self._cnt += 1
            import time as _t
            if _t.time() - st.session_state.last_pose_ts >= pose_every_s:
                enqueue("text", {"type":"pose","pose":{"x":float(v_x),"y":float(v_y),"theta":float(v_th)}})
                st.session_state.last_pose_ts = _t.time()
            if self._cnt % int(max(1, infer_every)) == 0:
                enqueue("bytes", jpeg_bytes(img.copy()))
            got = 0
            while True:
                try:
                    data = st.session_state.recv_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                got += 1
                if show_json: place_json.json(data)
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
                        c = p.get("control") or {}
                        st.session_state.shared["last_ctrl"] = (float(c.get("v",0.0)), float(c.get("w",0.0)))
                        if c: place_ctrl.success(f"v={c.get('v',0.0):.3f}  w={c.get('w',0.0):.3f}")
                elif data.get("type") == "utterance":
                    place_asr.info(f"語音：{data.get('text','')}")
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
            cv2.putText(img, f"intent={intent} target={target} rel={rel_dir} dist={dist_label}", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(img, f"depth_m={None if depth_m is None else round(depth_m,2)}", (10,54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            if last_ctrl:
                cv2.putText(img, f"v={last_ctrl[0]:.2f} w={last_ctrl[1]:.2f}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            if instr:
                cv2.putText(img, f"instr={instr}", (10,106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="av-sendrecv",
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

    st.caption("左側麥克風與鏡頭同時啟動：音訊以 AUD0 前綴送 /brain/ws，影像每 N 幀送 JPEG，姿態定期送出，伺服器回 observe 結果即時疊在畫面。")

if __name__ == "__main__":
    main()
