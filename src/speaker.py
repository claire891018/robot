import httpx
from src.utils.config import TTS_IP

class Speaker:
    def __init__(self, tts_url: str = f"http://{TTS_IP}:9998/tts", timeout: float = 60.0):
        self.tts_url = tts_url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def say(self, text: str) -> bytes:
        if not text.strip():
            return b""
        resp = await self._client.post(self.tts_url, json={"text": text})
        resp.raise_for_status()
        return resp.content
