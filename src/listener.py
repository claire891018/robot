import threading, time, queue
from dataclasses import dataclass, asdict
import numpy as np
import sounddevice as sd
import webrtcvad
import torch, whisper

SAMPLE_RATE = 16000
FRAME_MS = 30
VAD_LEVEL = 2
SILENCE_MS = 700
MAX_UTTER_SEC = 15
LANG = "zh"
MODEL_NAME = "large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_CONFIDENCE = 0.6

@dataclass
class Utterance:
    type: str
    text: str
    confidence: float
    lang: str
    start_ts: float
    end_ts: float
    meta: dict

@dataclass
class ErrorEvt:
    type: str
    error: str
    ts: float
    detail: str = ""

class Listener:
    def __init__(self,
                on_utterance=None,
                input_device=None,
                sample_rate=SAMPLE_RATE,
                frame_ms=FRAME_MS,
                vad_level=VAD_LEVEL,
                silence_ms=SILENCE_MS,
                max_utter_sec=MAX_UTTER_SEC,
                lang=LANG,
                model_name=MODEL_NAME,
                device=DEVICE,
                min_conf=MIN_CONFIDENCE
            ):
        self.on_utterance = on_utterance
        self.input_device = input_device
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad_level = vad_level
        self.silence_ms = silence_ms
        self.max_utter_sec = max_utter_sec
        self.lang = lang
        self.model_name = model_name
        self.device = device
        self.min_conf = min_conf

        self._audio_stream = None
        self._run = threading.Event()
        self._buffer = bytearray()
        self._q = queue.Queue()
        self._start_wall_ts = None

        self._vad = webrtcvad.Vad(self.vad_level)
        self._frame_bytes = int(self.sample_rate * (self.frame_ms/1000.0)) * 2
        self._silence_frames = int(self.silence_ms / self.frame_ms)
        self._max_frames = int((self.max_utter_sec*1000) / self.frame_ms)

        self._recording = False
        self._frames = []
        self._silence_count = 0
        self._utt_start_ts = None

        self._model = whisper.load_model(self.model_name, device=self.device)
        self._worker = None

    def start(self):
        if self._run.is_set():
            return
        self._run.set()
        self._start_wall_ts = time.time()
        self._open_stream()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._run.clear()
        if self._audio_stream:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

    def running(self):
        return self._run.is_set()

    def get(self, timeout=None):
        try:
            evt = self._q.get(timeout=timeout)
            return evt
        except queue.Empty:
            return None

    def _open_stream(self):
        def _cb(indata, frames, time_info, status):
            if status:
                self._emit_error("audio_status", detail=str(status))
            mono = indata[:, 0] if indata.ndim > 1 else indata
            mono_i16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
            self._buffer.extend(mono_i16.tobytes())

        self._audio_stream = sd.InputStream(
            device=self.input_device,
            channels=1,
            samplerate=self.sample_rate,
            dtype='float32',
            blocksize=int(self.sample_rate * (self.frame_ms/1000.0)),
            callback=_cb
        )
        self._audio_stream.start()

    def _loop(self):
        while self._run.is_set():
            now_ts = time.time() - self._start_wall_ts
            while len(self._buffer) >= self._frame_bytes:
                frame = self._buffer[:self._frame_bytes]
                self._buffer = self._buffer[self._frame_bytes:]
                try:
                    is_speech = self._vad.is_speech(frame, self.sample_rate)
                except Exception as e:
                    self._emit_error("vad_fail", str(e))
                    continue
                if not self._recording and is_speech:
                    self._recording = True
                    self._frames = []
                    self._silence_count = 0
                    self._utt_start_ts = now_ts
                if self._recording:
                    self._frames.append(frame)
                    if is_speech:
                        self._silence_count = 0
                    else:
                        self._silence_count += 1
                    if self._silence_count >= self._silence_frames or len(self._frames) >= self._max_frames:
                        pcm = b"".join(self._frames)
                        s_ts = self._utt_start_ts if self._utt_start_ts is not None else now_ts
                        e_ts = now_ts
                        self._recording = False
                        self._frames = []
                        self._silence_count = 0
                        self._utt_start_ts = None
                        self._handle_segment(pcm, s_ts, e_ts)
            time.sleep(0.001)

    def _handle_segment(self, pcm, s_ts, e_ts):
        try:
            audio_i16 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            if audio_i16.size == 0:
                return
            audio_f32 = audio_i16 / 32768.0
            result = self._model.transcribe(audio_f32, language=self.lang, fp16=(self.device == "cuda"))
            text = (result.get("text") or "").strip()
            segs = result.get("segments", []) or []
            if segs:
                avg_logprob = float(np.mean([s.get("avg_logprob", -1.0) for s in segs]))
                conf = max(0.0, min(1.0, avg_logprob + 1.0))
            else:
                conf = 0.0
            if not text:
                return
            evt = Utterance(
                type="utterance",
                text=text,
                confidence=conf,
                lang=self.lang,
                start_ts=float(s_ts),
                end_ts=float(e_ts),
                meta={
                    "audio_len_sec": round(e_ts - s_ts, 3),
                    "note": "low_confidence" if conf < self.min_conf else "ok"
                }
            )
            evt_dict = asdict(evt)
            if self.on_utterance:
                try:
                    self.on_utterance(evt_dict)
                except Exception as e:
                    self._emit_error("callback_error", str(e))
            self._q.put(evt_dict)
        except Exception as e:
            self._emit_error("asr_fail", str(e))

    def _emit_error(self, kind, detail=""):
        err = asdict(ErrorEvt(type="error", error=kind, ts=time.time(), detail=detail))
        if self.on_utterance:
            try:
                self.on_utterance(err)
            except Exception:
                pass
        self._q.put(err)

if __name__ == "__main__":
    def demo(evt):
        print(evt)
    lis = Listener(on_utterance=demo)
    lis.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        lis.stop()
