import asyncio
import json
import websockets
from dotenv import load_dotenv
import os
load_dotenv()

async def main():
    uri = f"ws://{os.getenv('TTS_IP')}:9998/ws/tts"

    async with websockets.connect(uri) as ws:
        payload = {"text": "難怪你這趟特別開心！欸很好玩耶～！"}
        await ws.send(json.dumps(payload))

        msg = await ws.recv()

        if isinstance(msg, bytes):
            with open("temp/speak.wav", "wb") as f:
                f.write(msg)
            print("saved to temp/speak.wav")
        else:
            print("server text msg:", msg)

asyncio.run(main())
