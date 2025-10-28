import threading, json, io, base64, cv2, numpy as np
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
        self.pose = Pose(0.0,0.0,0.0)
        self.goal = None
        self.last_instruction = ""
        self._lock = threading.Lock()
        self.listener = Listener(on_utterance=self._on_asr_evt, source="external")
        self.listener.start()

    def _on_asr_evt(self, evt: dict):
        if evt.get("type")=="utterance" and evt.get("text"):
            with self._lock:
                self.last_instruction = evt["text"]

    def append_audio_pcm(self, pcm_bytes: bytes):
        self.listener.append_pcm(pcm_bytes)

    def update_pose(self, pose: Dict[str, float]):
        with self._lock:
            self.pose = Pose(float(pose.get("x",0.0)), float(pose.get("y",0.0)), float(pose.get("theta",0.0)))

    def observe_frame(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        with self._lock:
            instr = self.last_instruction or ""
            cur = Pose(self.pose.x, self.pose.y, self.pose.theta)
        frame = _jpeg_to_bgr(jpeg_bytes)
        p: Perception = self.vision.perceive(frame, instr)
        c = self._plan(cur, p)
        return {
            "intent": p.intent, "target": p.target, "rel_dir": p.rel_dir,
            "dist_label": p.dist_label, "bbox": p.bbox, "depth_m": p.depth_m,
            "control": {"v": 0.0 if c is None else c.v, "w": 0.0 if c is None else c.w},
            "instruction": instr
        }

    def _plan(self, cur: Pose, p: Perception) -> Optional[Control]:
        if p.intent!="navigate":
            return Control(0.0,0.0)
        if p.depth_m is None:
            return Control(0.0,0.0)
        gx = cur.x + float(p.depth_m)
        gy = cur.y
        self.goal = (gx, gy)
        c = self.nav.control_to(cur, self.goal)
        if c:
            self.ctrl.send(c)
        return c
