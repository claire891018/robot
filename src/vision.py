import base64
import os, io, json, re, math, requests
import numpy as np, cv2
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from src.utils.schemas import Perception
from src.utils.config import (
    OLLAMA_URL, OLLAMA_MODEL, FOV_DEG, TARGET_REAL_HEIGHT_M, FRAME_WIDTH,
    DEPTH_MODE, MIDAS_WEIGHTS, MIDAS_INPUT_SIZE
)

def _jpeg_b64_from_bgr(img_bgr: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _clip_bbox_to_wh(b: Tuple[int,int,int,int], w: int, h: int) -> Optional[Tuple[int,int,int,int]]:
    x1,y1,x2,y2 = map(int, b)
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1,y1,x2,y2)

def _rel_dir_from_bbox(w: int, bbox: Optional[Tuple[int,int,int,int]]) -> Optional[str]:
    if not bbox: return None
    x1,y1,x2,y2 = bbox
    cx = (x1 + x2) / 2.0
    mid = w / 2.0
    if abs(cx - mid) <= w * 0.05: return "center"
    return "left" if cx < mid else "right"

def _json_loose(s: str) -> Optional[Dict[str, Any]]:
    try:
        j = json.loads(s)
        if isinstance(j, dict): return j
    except: pass
    m = re.search(r'\{[\s\S]*\}', s)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except:
        return None

class DepthEstimator:
    def __init__(self, mode: str = "midas"):
        self.mode = (mode or "midas").lower()
        self.net = None
        self.input_size = (384, 384)
        if isinstance(MIDAS_INPUT_SIZE, (tuple, list)) and len(MIDAS_INPUT_SIZE) == 2:
            self.input_size = (int(MIDAS_INPUT_SIZE[0]), int(MIDAS_INPUT_SIZE[1]))
        elif isinstance(MIDAS_INPUT_SIZE, str):
            try:
                p = [int(x) for x in re.split(r'[\s,]+', MIDAS_INPUT_SIZE.strip()) if x]
                if len(p) == 2: self.input_size = (p[0], p[1])
            except: pass
        if self.mode == "midas" and MIDAS_WEIGHTS and os.path.exists(MIDAS_WEIGHTS):
            try:
                self.net = cv2.dnn.readNet(MIDAS_WEIGHTS)
            except:
                self.net = None

    def predict(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.mode != "midas" or self.net is None: return None
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = np.squeeze(out)
        if out.ndim == 3: out = out[0]
        dm = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        mn, mx = float(dm.min()), float(dm.max())
        if mx - mn < 1e-9: return None
        dm = (dm - mn) / (mx - mn + 1e-9)
        return dm.astype(np.float32)
    
class Vision:
    def __init__(self, mllm_model: Optional[str]=None, depth_model: Optional[DepthEstimator]=None):
        self.model = mllm_model or OLLAMA_MODEL
        self.depth = depth_model or DepthEstimator(DEPTH_MODE)
        self.focal_px = 0.5 * FRAME_WIDTH / math.tan(math.radians(FOV_DEG/2))

    def _endpoints(self):
        base = OLLAMA_URL.rstrip("/")
        if base.endswith("/api/chat") or base.endswith("/api/generate"):
            chat = base if base.endswith("/api/chat") else base.replace("/api/generate", "/api/chat")
            gen = base if base.endswith("/api/generate") else base.replace("/api/chat", "/api/generate")
        else:
            chat = base + "/api/chat"
            gen = base + "/api/generate"
        return chat, gen

    def _ask_mllm(self, frame_bgr: np.ndarray, instruction: str, H: int, W: int) -> Dict[str, Any]:
        b64 = _jpeg_b64_from_bgr(frame_bgr)
        chat_url, gen_url = self._endpoints()
        sys = "你是多模態導航助理。僅以 JSON 回答，鍵: intent,target,rel_dir,dist_label,bbox。bbox=[x1,y1,x2,y2] 或 null。intent 取 navigate/chat/control。dist_label 取 near/mid/far 或 null。"
        user = f"指令: {instruction}\n請輸出 JSON。"
        try:
            p = {"model": self.model, "messages": [{"role":"system","content":sys},{"role":"user","content":user,"images":[b64]}], "stream": False}
            r = requests.post(chat_url, json=p, timeout=60)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "") or ""
            j = _json_loose(content) or {}
        except:
            try:
                pr = {"model": self.model, "prompt": sys + "\n" + user, "images": [b64], "stream": False}
                r2 = requests.post(gen_url, json=pr, timeout=60)
                r2.raise_for_status()
                content = r2.json().get("response", "") or ""
                j = _json_loose(content) or {}
            except:
                j = {}
        intent = j.get("intent") or ("navigate" if instruction else "observe")
        target = j.get("target") or (instruction or "")
        rel_dir = j.get("rel_dir")
        dist_label = j.get("dist_label")
        if isinstance(dist_label, str) and dist_label.lower() == "medium": dist_label = "mid"
        bbox = j.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox = _clip_bbox_to_wh((int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])), W, H)
        else:
            bbox = None
        if not rel_dir: rel_dir = _rel_dir_from_bbox(W, bbox)
        return {"intent": intent, "target": target, "rel_dir": rel_dir, "dist_label": dist_label, "bbox": bbox}

    def _depth_from_map(self, frame_bgr: np.ndarray, bbox: Optional[Tuple[int,int,int,int]]) -> Optional[float]:
        dm = self.depth.predict(frame_bgr)
        if dm is None: return None
        if bbox is None: return float(np.median(dm))
        x1,y1,x2,y2 = bbox
        x1 = max(0, min(dm.shape[1]-1, int(x1)))
        x2 = max(0, min(dm.shape[1]-1, int(x2)))
        y1 = max(0, min(dm.shape[0]-1, int(y1)))
        y2 = max(0, min(dm.shape[0]-1, int(y2)))
        if x2 <= x1 or y2 <= y1: return float(np.median(dm))
        crop = dm[y1:y2, x1:x2]
        if crop.size == 0: return float(np.median(dm))
        return float(np.median(crop))

    def _depth_heuristic(self, bbox: Optional[Tuple[int,int,int,int]]) -> Optional[float]:
        if bbox is None: return None
        x1,y1,x2,y2 = bbox
        h_px = max(1, int(y2 - y1))
        d = TARGET_REAL_HEIGHT_M * self.focal_px / float(h_px)
        return float(max(0.2, min(6.0, d)))

    def _label_from_depth(self, d: Optional[float]) -> Optional[str]:
        if d is None: return None
        if d < 0.5: return "near"
        if d < 1.5: return "mid"
        return "far"

    def perceive(self, frame_bgr: np.ndarray, instruction: str) -> Perception:
        H,W = frame_bgr.shape[:2]
        j = self._ask_mllm(frame_bgr, instruction or "", H, W)
        intent = j["intent"]
        target = j["target"]
        rel_dir = j["rel_dir"]
        dist_label = j.get("dist_label")
        bbox = j["bbox"]
        depth_m = None
        if DEPTH_MODE.lower() == "midas":
            depth_m = self._depth_from_map(frame_bgr, bbox)
            if depth_m is None:
                depth_m = self._depth_heuristic(bbox)
        else:
            depth_m = self._depth_heuristic(bbox)
        dist_label = self._label_from_depth(depth_m)
        return Perception(intent=intent, target=target, rel_dir=rel_dir, dist_label=dist_label, bbox=bbox, depth_m=depth_m)
