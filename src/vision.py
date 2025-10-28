import base64, io, json, math, requests
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple
from src.utils.schemas import Perception
from src.utils.config import (
    OLLAMA_URL, OLLAMA_MODEL, FOV_DEG, TARGET_REAL_HEIGHT_M, FRAME_WIDTH,
    DEPTH_MODE, MIDAS_WEIGHTS, MIDAS_INPUT_SIZE
)

class DepthEstimator:
    def __init__(self, mode:str="heuristic"):
        self.mode = mode
        self.net = None
        if self.mode == "midas":
            self.net = cv2.dnn.readNet(MIDAS_WEIGHTS)
            self.input_size = MIDAS_INPUT_SIZE
    def predict(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.mode != "midas" or self.net is None:
            return None
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = cv2.resize(out[0,0], (w, h))
        out = (out - out.min()) / (out.max() - out.min() + 1e-9)
        return out

class Vision:
    def __init__(self, mllm_model: Optional[str]=None, depth_model: Optional[DepthEstimator]=None):
        self.model = mllm_model or OLLAMA_MODEL
        self.depth = depth_model or DepthEstimator(DEPTH_MODE)
        self.focal_px = 0.5 * FRAME_WIDTH / math.tan(math.radians(FOV_DEG/2))
    def _img_to_b64(self, frame_bgr: np.ndarray) -> str:
        img = Image.fromarray(frame_bgr[..., ::-1])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    def _ask_mllm(self, frame_bgr: np.ndarray, instruction: str) -> dict:
        sys = "你是多模態導航助理。請僅以 JSON 回答，鍵為: intent, target, rel_dir, dist_label, bbox。bbox 為 [x1,y1,x2,y2] 或 null。intent 取 navigate/chat/control 之一。dist_label 取 near/medium/far 或 null。"
        user = f"指令: {instruction}\n請輸出 JSON。"
        b64 = self._img_to_b64(frame_bgr)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": user, "images": [b64]},
            ],
            "stream": False
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "")
        try:
            s = content[content.find("{"):content.rfind("}")+1]
            return json.loads(s)
        except Exception:
            return {"intent":"navigate","target":None,"rel_dir":None,"dist_label":None,"bbox":None}
    def _depth_from_map(self, frame_bgr: np.ndarray, bbox: Optional[Tuple[int,int,int,int]]) -> Optional[float]:
        dmap = self.depth.predict(frame_bgr)
        if dmap is None:
            return None
        if bbox is None:
            return float(np.median(dmap))
        x1,y1,x2,y2 = bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(dmap.shape[1]-1, int(x2)); y2 = min(dmap.shape[0]-1, int(y2))
        crop = dmap[y1:y2, x1:x2]
        if crop.size == 0:
            return float(np.median(dmap))
        return float(np.median(crop))
    def _depth_heuristic(self, frame_bgr: np.ndarray, bbox: Optional[Tuple[int,int,int,int]], dist_label: Optional[str]) -> Optional[float]:
        if bbox is not None:
            x1,y1,x2,y2 = bbox
            h_px = max(1, int(y2 - y1))
            d = TARGET_REAL_HEIGHT_M * self.focal_px / float(h_px)
            return float(max(0.2, min(6.0, d)))
        if dist_label:
            m = {"near":0.6, "medium":1.6, "far":3.2}
            if dist_label.lower() in m:
                return float(m[dist_label.lower()])
        return None
    def perceive(self, frame_bgr: np.ndarray, instruction: str) -> Perception:
        resp = self._ask_mllm(frame_bgr, instruction or "")
        intent = resp.get("intent") or "navigate"
        target = resp.get("target")
        rel_dir = resp.get("rel_dir")
        dist_label = resp.get("dist_label")
        bbox = resp.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox = tuple(int(v) for v in bbox)
        else:
            bbox = None
        depth_m = self._depth_from_map(frame_bgr, bbox) if DEPTH_MODE == "midas" else self._depth_heuristic(frame_bgr, bbox, dist_label)
        return Perception(intent=intent, target=target, rel_dir=rel_dir, dist_label=dist_label, bbox=bbox, depth_m=depth_m)

