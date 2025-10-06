import threading, time, heapq, uuid
from dataclasses import dataclass, asdict
import pyttsx3

PRIORITY = {"high": 0, "normal": 1, "low": 2}
DEFAULT_TONE_MAP = {
    "neutral": {"rate_mul": 1.0, "volume": 1.0},
    "warm": {"rate_mul": 0.9, "volume": 1.0},
    "happy": {"rate_mul": 1.05, "volume": 1.0},
    "calm": {"rate_mul": 0.9, "volume": 1.0},
    "serious": {"rate_mul": 0.95, "volume": 1.0}
}

@dataclass
class SpeakItem:
    speak_id: str
    text: str
    tone: str
    speed: float
    priority: str
    ts: float

class Speaker:
    def __init__(self,
                on_event=None,
                voice=None,
                base_rate=200,
                base_volume=1.0,
                tone_map=None,
                max_queue=20):
        self.on_event = on_event
        self.voice = voice
        self.base_rate = base_rate
        self.base_volume = base_volume
        self.tone_map = tone_map or DEFAULT_TONE_MAP
        self.max_queue = max_queue
        self._q = []
        self._q_lock = threading.Lock()
        self._seq = 0
        self._run = threading.Event()
        self._engine = pyttsx3.init()
        if self.voice is not None:
            try:
                self._engine.setProperty("voice", self.voice)
            except Exception:
                pass
        self._current = None
        self._interrupt = threading.Event()
        self._worker = None

    def start(self):
        if self._run.is_set():
            return
        self._run.set()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._run.clear()
        self._interrupt.set()
        try:
            self._engine.stop()
        except Exception:
            pass

    def speak(self, text, tone="neutral", speed=1.0, priority="normal"):
        if not self._run.is_set():
            self.start()
        if len(text.strip()) == 0:
            return None
        pid = str(uuid.uuid4())[:8]
        item = SpeakItem(pid, text, tone, float(speed), priority, time.time())
        with self._q_lock:
            if len(self._q) >= self.max_queue:
                heapq.heappop(self._q)
            self._seq += 1
            heapq.heappush(self._q, (PRIORITY.get(priority, 1), self._seq, item))
        if self._current and PRIORITY.get(priority, 1) < PRIORITY.get(self._current.priority, 1):
            self._interrupt.set()
            try:
                self._engine.stop()
            except Exception:
                pass
        return pid

    def clear(self):
        with self._q_lock:
            self._q.clear()

    def _emit(self, evt_type, **kw):
        if self.on_event:
            try:
                self.on_event({"type": evt_type, **kw})
            except Exception:
                pass

    def _next_item(self):
        with self._q_lock:
            if not self._q:
                return None
            _, _, item = heapq.heappop(self._q)
            return item

    def _apply_voice(self, item):
        tone_cfg = self.tone_map.get(item.tone, self.tone_map["neutral"])
        rate = int(self.base_rate * tone_cfg.get("rate_mul", 1.0) * max(0.5, min(2.0, item.speed)))
        vol = float(tone_cfg.get("volume", self.base_volume))
        try:
            self._engine.setProperty("rate", rate)
        except Exception:
            pass
        try:
            self._engine.setProperty("volume", max(0.0, min(1.0, vol)))
        except Exception:
            pass

    def _loop(self):
        while self._run.is_set():
            item = self._next_item()
            if item is None:
                time.sleep(0.01)
                continue
            self._current = item
            self._interrupt.clear()
            self._apply_voice(item)
            self._emit("speak_started", speak_id=item.speak_id, text=item.text, tone=item.tone, priority=item.priority, ts=time.time())
            try:
                self._engine.say(item.text)
                self._engine.runAndWait()
            except Exception as e:
                self._emit("error", speak_id=item.speak_id, error=str(e), ts=time.time())
                self._current = None
                continue
            if self._interrupt.is_set():
                self._emit("speak_interrupted", speak_id=item.speak_id, reason="higher_priority", ts=time.time())
            else:
                self._emit("speak_finished", speak_id=item.speak_id, ts=time.time())
            self._current = None

if __name__ == "__main__":
    def demo(evt):
        print(evt)
    spk = Speaker(on_event=demo)
    spk.start()
    spk.speak("系統啟動成功。", tone="neutral", priority="normal")
    time.sleep(0.2)
    spk.speak("前方受阻，要不要繞道？", tone="calm", priority="high")
    spk.speak("這是一句低優先的閒聊。", tone="happy", priority="low")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        spk.stop()
