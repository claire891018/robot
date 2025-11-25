import threading, io
from typing import Optional, Dict, Any

import cv2
import numpy as np
import httpx
from PIL import Image

from src.utils.schemas import Pose, Perception, Control
from src.controller import Controller
from src.navigator import Navigator
from src.vision import Vision
from src.listener import Listener
from src.speaker import Speaker
from src.utils.config import OLLAMA_URL, OLLAMA_MODEL, SYSPROMPT


def _jpeg_to_bgr(jpeg_bytes: bytes):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class Brain:
    def __init__(self, speaker: Optional[Speaker] = None):
        self.ctrl = Controller()
        self.nav = Navigator()
        self.vision = Vision()
        self.pose = Pose(0.0, 0.0, 0.0)
        self.goal = None

        # last ASR text (for vision)
        self.last_instruction = ""
        self._lock = threading.Lock()
        self._new_instruction = False
        self._last_perception: Optional[Perception] = None

        # speaker
        self.speaker = speaker

        # Listener = Whisper + VAD
        self.listener = Listener(on_utterance=self._on_asr_evt, source="external")
        self.listener.start()

        # dialogue memory
        self._conversations: Dict[str, list] = {}

        # async HTTP client for LLM
        self._client = httpx.AsyncClient(timeout=60)


    # ===========================================
    #     ASR event (Listener callback)
    # ===========================================
    def _on_asr_evt(self, evt: dict):
        if evt.get("type") == "utterance" and evt.get("text"):
            with self._lock:
                text = evt["text"]
                if text != self.last_instruction:
                    self.last_instruction = text
                    self._new_instruction = True
                    print(f"[BRAIN] 新指令: '{text}' (需要重新視覺推理)")


    # ===========================================
    #     Audio ingestion from /brain/ws
    # ===========================================
    def append_audio_pcm(self, pcm_bytes: bytes):
        self.listener.append_pcm(pcm_bytes)


    # ===========================================
    #     Pose
    # ===========================================
    def update_pose(self, pose: Dict[str, float]):
        with self._lock:
            self.pose = Pose(
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("theta", 0.0)),
            )


    # ===========================================
    #            VISION + NAVIGATION
    # ===========================================
    def observe_frame(self, jpeg_bytes: bytes) -> Dict[str, Any]:

        with self._lock:
            instr = self.last_instruction or ""
            cur_pose = Pose(self.pose.x, self.pose.y, self.pose.theta)
            need_vlm = self._new_instruction
            if need_vlm:
                self._new_instruction = False

        frame = _jpeg_to_bgr(jpeg_bytes)

        if need_vlm or not self._last_perception:
            print(f"[BRAIN] 重新執行視覺推理 (指令: '{instr}')")
            p = self.vision.perceive(frame, instr)
            self._last_perception = p
        else:
            print(f"[BRAIN] 重用上一次視覺結果")
            p = self._last_perception

        guide = self._guide_from_bbox(frame, p)
        c = self._plan(cur_pose, p, guide)

        return {
            "instruction": instr,
            "intent": p.intent,
            "target": p.target,
            "rel_dir": p.rel_dir,
            "dist_label": p.dist_label,
            "bbox": p.bbox,
            "depth_m": p.depth_m,
            "guide": guide,
            "control": {
                "v": 0.0 if c is None else c.v,
                "w": 0.0 if c is None else c.w,
            },
            "pose": {"x": cur_pose.x, "y": cur_pose.y, "theta": cur_pose.theta},
        }


    def _guide_from_bbox(self, frame_bgr, p: Perception) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        if not p.bbox:
            return {
                "steer_angle_deg": 0.0,
                "turn": "search",
                "distance_m": None,
                "waypoint_img": None,
                "polyline_img": [],
            }

        x1, y1, x2, y2 = p.bbox
        cx = (x1 + x2) // 2
        dx = cx - w // 2

        steer = float(dx) / float(max(1, w // 2)) * 30.0
        turn = "straight" if abs(steer) < 5 else ("left" if steer < 0 else "right")

        return {
            "steer_angle_deg": steer,
            "turn": turn,
            "distance_m": p.depth_m,
            "waypoint_img": [cx, (y1 + y2) // 2],
            "polyline_img": [],
        }


    def _plan(self, cur: Pose, p: Perception, guide: Dict[str, Any]) -> Optional[Control]:
        if p.intent != "navigate":
            return Control(0.0, 0.0)

        if p.depth_m is None or guide["turn"] == "search":
            return Control(0.0, 0.0)

        ang = guide["steer_angle_deg"]
        dist = p.depth_m

        v = max(0.0, min(0.3, dist * 0.2))
        w = max(-0.6, min(0.6, -ang * 0.03))

        c = Control(v, w)
        self.ctrl.send(c)
        return c


    # ===========================================
    #       LLM + TTS 回覆（async）
    # ===========================================
    async def handle_utterance(self, session_id: str, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {"reply_text": None, "audio": None, "need_tts": False}

        # 取得或新建對話 session
        conv = self._conversations.get(session_id)
        if conv is None:
            conv = [{"role": "system", "content": SYSPROMPT.strip()}]
            self._conversations[session_id] = conv

        conv.append({"role": "user", "content": text})

        # ===== 呼叫 LLM（用 HTTPX async）=====
        try:
            resp = await self._client.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "messages": conv, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()

        except Exception as e:
            print(f"[BRAIN/LLM_ERR] {e}")
            reply = "我這邊好像有點狀況，等一下再試試看。"
            audio = (
                await self.speaker.say(reply)
                if self.speaker is not None else None
            )
            return {"reply_text": reply, "audio": audio, "need_tts": True}

        msg = data.get("message") or {}
        reply = msg.get("content", "").strip() or "我聽不太清楚，可以再說一次嗎？"

        conv.append({"role": "assistant", "content": reply})

        # ===== 呼叫 Speaker =====
        audio = None
        if self.speaker is not None:
            try:
                audio = await self.speaker.say(reply)   # ★ 正確 await
            except Exception as e:
                print(f"[BRAIN/SPEAKER_ERR] {e}")

        return {"reply_text": reply, "audio": audio, "need_tts": True}
