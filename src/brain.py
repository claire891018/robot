import threading, io, cv2, numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from src.utils.schemas import Pose, Perception, Control
from src.controller import Controller
from src.navigator import Navigator
from src.vision import Vision
from src.listener import Listener

def _jpeg_to_bgr(jpeg_bytes: bytes):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

class Brain:
    def __init__(self):
        self.ctrl = Controller()
        self.nav = Navigator()
        self.vision = Vision()
        self.pose = Pose(0.0, 0.0, 0.0)
        self.goal = None
        self.last_instruction = ""
        self._lock = threading.Lock()
        self.listener = Listener(on_utterance=self._on_asr_evt, source="external")
        self.listener.start()

    def _on_asr_evt(self, evt: dict):
        if evt.get("type") == "utterance" and evt.get("text"):
            with self._lock:
                self.last_instruction = evt["text"]

    def append_audio_pcm(self, pcm_bytes: bytes):
        self.listener.append_pcm(pcm_bytes)

    def update_pose(self, pose: Dict[str, float]):
        with self._lock:
            self.pose = Pose(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("theta", 0.0)))

    def observe_frame(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        with self._lock:
            instr = self.last_instruction or ""
            cur = Pose(self.pose.x, self.pose.y, self.pose.theta)
        frame = _jpeg_to_bgr(jpeg_bytes)
        p: Perception = self.vision.perceive(frame, instr)
        guide = self._guide_from_bbox(frame, p)
        c = self._plan(cur, p, guide)
        return {
            "instruction": instr,
            "intent": p.intent,
            "target": p.target,
            "rel_dir": p.rel_dir,
            "dist_label": p.dist_label,
            "bbox": p.bbox,
            "depth_m": p.depth_m,
            "guide": guide,
            "control": {"v": 0.0 if c is None else c.v, "w": 0.0 if c is None else c.w},
            "pose": {"x": cur.x, "y": cur.y, "theta": cur.theta}
        }

    def _guide_from_bbox(self, frame_bgr: np.ndarray, p: Perception) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        if not p.bbox:
            return {"steer_angle_deg": 0.0, "turn": "search", "distance_m": None, "waypoint_img": None, "polyline_img": []}
        x1, y1, x2, y2 = p.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dx = cx - (w // 2)
        steer = float(dx) / float(max(1, w // 2)) * 30.0
        turn = "straight" if abs(steer) < 5 else ("left" if steer < 0 else "right")
        return {"steer_angle_deg": steer, "turn": turn, "distance_m": p.depth_m, "waypoint_img": [int(cx), int(cy)], "polyline_img": []}

    def _plan(self, cur: Pose, p: Perception, guide: Dict[str, Any]) -> Optional[Control]:
        if p.intent != "navigate":
            return Control(0.0, 0.0)
        if p.depth_m is None or guide.get("turn") == "search":
            return Control(0.0, 0.0)
        ang = float(guide.get("steer_angle_deg") or 0.0)
        dist = float(p.depth_m)
        v = max(0.0, min(0.3, dist * 0.2))
        w = max(-0.6, min(0.6, -ang * 0.03))
        c = Control(v, w)
        self.ctrl.send(c)
        return c
