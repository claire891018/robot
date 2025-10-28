# pages/2_robot_demo.py
import base64, io, threading, time
import numpy as np
import cv2
from PIL import Image
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

def get_api_base() -> str:
    try:
        return st.secrets.get("API_BASE", "http://140.116.158.98:9999")
    except Exception:
        return "http://140.116.158.98:9999"

def render_header():
    icon = "https://api.dicebear.com/9.x/thumbs/svg?seed=Brian"
    st.markdown(
        f'''
        <h2 style="display:flex;align-items:center;gap:.5rem;">
        <img src="{icon}" width="28" height="28"
            style="border-radius:20%; display:block;" />
        Robot Demo
        </h2>
        ''',
        unsafe_allow_html=True,
    )

def to_b64(frame_bgr: np.ndarray) -> str:
    im = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def infer_and_control(frame_bgr: np.ndarray, api_base: str, instr: str, pose, shared, lock, show_json: bool, send_control: bool, place_json, place_ctrl):
    try:
        b64 = to_b64(frame_bgr)
        r = requests.post(f"{api_base}/vision/infer",
                          json={"image_b64": b64, "instruction": instr},
                          timeout=30)
        r.raise_for_status()
        p = r.json()
        with lock:
            shared["bbox"] = tuple(p["bbox"]) if p.get("bbox") else None
            shared["depth_m"] = p.get("depth_m")
            shared["intent"] = p.get("intent")
            shared["target"] = p.get("target")
            shared["rel_dir"] = p.get("rel_dir")
            shared["dist_label"] = p.get("dist_label")
        if show_json:
            place_json.json(p)

        if send_control:
            payload = {"pose": pose, "perception": p}
            r2 = requests.post(f"{api_base}/brain/step", json=payload, timeout=15)
            r2.raise_for_status()
            c = r2.json()
            with lock:
                shared["last_ctrl"] = (float(c.get("v", 0.0)), float(c.get("w", 0.0)))
            place_ctrl.success(f"控制量 v={c.get('v',0.0):.3f}, w={c.get('w',0.0):.3f}")
    except Exception as e:
        place_ctrl.error(f"推理/控制錯誤：{e}")

def make_video_processor(infer_every_getter, api_base_getter, instr_getter, pose_getter, shared, lock, ui_refs):
    class VideoProcessor:
        def __init__(self):
            self._cnt = 0
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self._cnt += 1
            if self._cnt % int(max(1, infer_every_getter())) == 0:
                threading.Thread(
                    target=infer_and_control,
                    args=(
                        img.copy(),
                        api_base_getter(),
                        instr_getter(),
                        pose_getter(),
                        shared,
                        lock,
                        ui_refs["show_json"](),
                        ui_refs["send_control"](),
                        ui_refs["place_json"],
                        ui_refs["place_ctrl"],
                    ),
                    daemon=True,
                ).start()

            with lock:
                bbox = shared["bbox"]
                depth_m = shared["depth_m"]
                intent = shared["intent"]
                target = shared["target"]
                rel_dir = shared["rel_dir"]
                dist_label = shared["dist_label"]
                last_ctrl = shared["last_ctrl"]

            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            txt1 = f"intent={intent} target={target} rel={rel_dir} dist_label={dist_label}"
            txt2 = f"depth_m={None if depth_m is None else round(depth_m,2)}"
            cv2.putText(img, txt1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(img, txt2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            if last_ctrl:
                cv2.putText(img, f"v={last_ctrl[0]:.2f}, w={last_ctrl[1]:.2f}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    return VideoProcessor

def main():
    st.set_page_config(
        page_title="Robot Demo",
        page_icon="https://api.dicebear.com/9.x/thumbs/svg?seed=Brian",
        layout="wide",
    )
    render_header()

    api_base = get_api_base()
    instr = st.text_input("指令", value="請定位目標並估距離")

    pose_col = st.columns(3)
    x = pose_col[0].number_input("x", value=0.0, step=0.1)
    y = pose_col[1].number_input("y", value=0.0, step=0.1)
    theta = pose_col[2].number_input("theta", value=0.0, step=0.1)

    row = st.columns(3)
    infer_every = row[0].number_input("每幀間隔(推理頻率)", value=5, min_value=1, max_value=30, step=1)
    send_control_t = row[1].toggle("送出控制量 (/brain/step)", value=True)
    show_json_t = row[2].toggle("顯示推理 JSON", value=False)

    place_json = st.empty()
    place_ctrl = st.empty()

    if "shared" not in st.session_state:
        st.session_state.shared = {
            "bbox": None,
            "depth_m": None,
            "intent": None,
            "target": None,
            "rel_dir": None,
            "dist_label": None,
            "last_ctrl": None,
        }
    if "lock" not in st.session_state:
        st.session_state.lock = threading.Lock()

    VideoProcessor = make_video_processor(
        infer_every_getter=lambda: infer_every,
        api_base_getter=lambda: api_base,
        instr_getter=lambda: instr,
        pose_getter=lambda: {"x": x, "y": y, "theta": theta},
        shared=st.session_state.shared,
        lock=st.session_state.lock,
        ui_refs={
            "show_json": lambda: show_json_t,
            "send_control": lambda: send_control_t,
            "place_json": place_json,
            "place_ctrl": place_ctrl,
        },
    )

    webrtc_streamer(
        key="robot-webrtc",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # st.info("提示：`API_BASE` 請設為 http 後端，例如 `http://<SERVER>:9999`；此頁會持續開著相機、每 N 幀推理。")

if __name__ == "__main__":
    main()
