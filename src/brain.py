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

        self.last_instruction = ""
        self._lock = threading.Lock()
        self._new_instruction = False
        self._last_perception: Optional[Perception] = None

        # 大腦手上有一個 Speaker，可以叫它說話
        self.speaker: Optional[Speaker] = speaker

        # ASR Listener（耳朵）
        self.listener = Listener(on_utterance=self._on_asr_evt, source="external")
        self.listener.start()

        # LLM 對話歷史（每個 session 一份）
        self._conversations: Dict[str, list] = {}

    # ======== 語音事件給「視覺 / 導航」用 ========
    def _on_asr_evt(self, evt: dict):
        """
        Listener 有新 utterance 時會呼叫這個 callback，
        這裡只負責更新 last_instruction，給 VLM / 導航用。
        """
        if evt.get("type") == "utterance" and evt.get("text"):
            with self._lock:
                new_text = evt["text"]
                if new_text != self.last_instruction:
                    self.last_instruction = new_text
                    self._new_instruction = True
                    print(f"[BRAIN] 新指令: '{new_text}' (標記需要視覺推理)")

    # ======== 將外部 PCM 丟給 Listener ========
    def append_audio_pcm(self, pcm_bytes: bytes):
        self.listener.append_pcm(pcm_bytes)

    # ======== 位姿 ========
    def update_pose(self, pose: Dict[str, float]):
        with self._lock:
            self.pose = Pose(
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("theta", 0.0)),
            )

    # ======== 視覺推理 ========
    def observe_frame(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        with self._lock:
            instr = self.last_instruction or ""
            cur = Pose(self.pose.x, self.pose.y, self.pose.theta)
            need_mllm = self._new_instruction
            if need_mllm:
                self._new_instruction = False

        frame = _jpeg_to_bgr(jpeg_bytes)

        if need_mllm or not self._last_perception:
            print(f"[BRAIN] 執行視覺推理 (指令: '{instr}')")
            p: Perception = self.vision.perceive(frame, instr)
            self._last_perception = p
        else:
            print(f"[BRAIN] 重用視覺結果 (指令: '{instr}')")
            p = self._last_perception

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
            "control": {
                "v": 0.0 if c is None else c.v,
                "w": 0.0 if c is None else c.w,
            },
            "pose": {"x": cur.x, "y": cur.y, "theta": cur.theta},
        }

    def _guide_from_bbox(self, frame_bgr: np.ndarray, p: Perception) -> Dict[str, Any]:
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
        cy = (y1 + y2) // 2
        dx = cx - (w // 2)
        steer = float(dx) / float(max(1, w // 2)) * 30.0
        turn = "straight" if abs(steer) < 5 else ("left" if steer < 0 else "right")
        return {
            "steer_angle_deg": steer,
            "turn": turn,
            "distance_m": p.depth_m,
            "waypoint_img": [int(cx), int(cy)],
            "polyline_img": [],
        }

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

    # ======== 大腦：處理一句話，要不要回、回什麼，還有聲音 ========
    async def handle_utterance(self, session_id: str, text: str) -> Dict[str, Any]:
        """
        大腦收到一句使用者的話，決定：
        - 要不要用 LLM 回覆
        - 回覆文字是什麼
        - 如果有 Speaker，順便產生 TTS 音檔

        回傳:
            {
                "reply_text": str 或 None,
                "need_tts": bool,
                "audio": Optional[bytes],
            }
        """
        text = (text or "").strip()
        if not text:
            return {"reply_text": None, "need_tts": False, "audio": None}

        # 未來這裡可以加「指令判斷」，現在先純聊天
        conv = self._conversations.get(session_id)
        if conv is None:
            conv = [{"role": "system", "content": SYSPROMPT.strip()}]
            self._conversations[session_id] = conv

        conv.append({"role": "user", "content": text})

        payload = {
            "model": OLLAMA_MODEL,
            "messages": conv,
            "stream": False,
        }

        # 用 httpx async 呼叫 Ollama
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(OLLAMA_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            print(f"[BRAIN/LLM_ERR] {e}")
            reply = "我這邊好像出了一點狀況，等等再試試看。"
            audio_bytes = None
            if self.speaker is not None:
                try:
                    audio_bytes = await self.speaker.say(reply)
                except Exception as e2:
                    print(f"[BRAIN/SPEAKER_ERR_WHEN_FAIL] {e2}")
                    audio_bytes = None
            return {"reply_text": reply, "need_tts": True, "audio": audio_bytes}

        msg = data.get("message") or {}
        reply = (msg.get("content") or "").strip()
        if not reply:
            reply = "我有點聽不清楚，可以再說一次嗎？"

        conv.append({"role": "assistant", "content": reply})

        # 在大腦裡就直接請 speaker 幫忙生音檔
        audio_bytes: Optional[bytes] = None
        if self.speaker is not None:
            try:
                audio_bytes = await self.speaker.say(reply)
            except Exception as e:
                print(f"[BRAIN/SPEAKER_ERR] {e}")
                audio_bytes = None

        return {"reply_text": reply, "need_tts": True, "audio": audio_bytes}
