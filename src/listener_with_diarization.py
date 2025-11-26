# src/listener_with_diarization.py
"""
整合 Speaker Diarization 的 Listener
繼承原本的 Listener，加上 diarization 功能
"""
import threading, time, queue
from dataclasses import dataclass, asdict
import numpy as np

from src.listener import Listener, Utterance as BaseUtterance, ErrorEvt
from src.diarizer import Diarizer

@dataclass
class UtteranceWithSpeaker:
    """帶有說話人資訊的 Utterance"""
    type: str
    text: str
    confidence: float
    lang: str
    start_ts: float
    end_ts: float
    speaker_id: int
    meta: dict

class ListenerWithDiarization(Listener):
    """
    整合 Diarization 的 Listener
    
    在原本 Listener 的基礎上，加上即時說話人辨識
    """
    
    def __init__(self, *args, **kwargs):
        """初始化"""
        super().__init__(*args, **kwargs)
        
        # 初始化 diarizer
        self.diarizer = Diarizer(
            similarity_threshold=0.75,
            ewma_alpha=0.9,
            smoothing_window=5.0,
            device=self.device
        )
        
        # 用於暫存音訊的 buffer
        self._audio_buffer = {}  # {segment_id: pcm_bytes}
        self._segment_counter = 0
        
        print(f"[ListenerWithDiarization] 已載入 (device={self.device})")
    
    def _handle_segment(self, pcm, s_ts, e_ts):
        """
        處理音訊片段
        覆寫父類別的方法，加上 diarization
        """
        try:
            audio_i16 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            if audio_i16.size == 0:
                return

            audio_f32 = audio_i16 / 32768.0
            audio_len_sec = float(e_ts - s_ts)

            # 門檻 1：太短的句子不要
            if audio_len_sec < 0.6:
                return

            # ASR 辨識
            result = self._model.transcribe(
                audio_f32,
                language=self.lang,
                fp16=(self.device == "cuda"),
                temperature=0.0,
                condition_on_previous_text=False,
                beam_size=5
            )

            text = self._cc.convert((result.get("text") or "").strip())
            segs = result.get("segments", []) or []
            no_speech = float(result.get("no_speech_prob", 0.0))

            if segs:
                avg_logprob = float(np.mean([s.get("avg_logprob", -1.0) for s in segs]))
                x = (avg_logprob + 1.2) / 1.1
                x = max(0.0, min(1.0, x))
                conf = x * (1.0 - no_speech)
            else:
                conf = 0.0

            # 門檻 2: 沒字就不要
            if not text:
                return

            # 門檻 3: 信心度太低不要
            if conf < self.min_conf:
                return

            # 門檻 4: 不是語音
            if no_speech > 0.6:
                return

            # ========== Speaker Diarization ==========
            # 使用 int16 格式的音訊進行 diarization
            audio_i16_arr = np.frombuffer(pcm, dtype=np.int16)
            
            diarization_result = self.diarizer.process_utterance(
                audio_segment=audio_i16_arr,
                text=text,
                start_ts=s_ts,
                end_ts=e_ts,
                confidence=conf
            )
            
            speaker_id = diarization_result["speaker_id"]
            
            # 建立事件
            evt = UtteranceWithSpeaker(
                type="utterance_with_speaker",
                text=text,
                confidence=conf,
                lang=self.lang,
                start_ts=float(s_ts),
                end_ts=float(e_ts),
                speaker_id=speaker_id,
                meta={
                    "audio_len_sec": round(audio_len_sec, 3),
                    "note": "ok"
                }
            )
            evt_dict = asdict(evt)

            # 回調
            if self.on_utterance:
                try:
                    self.on_utterance(evt_dict)
                except Exception as e:
                    self._emit_error("callback_error", str(e))

            # 放入 queue
            self._q.put(evt_dict)

        except Exception as e:
            self._emit_error("asr_fail", str(e))
    
    def reset_speakers(self):
        """重置說話人資料庫"""
        self.diarizer.reset()