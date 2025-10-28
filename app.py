import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.listener import Listener

VAD_LEVEL = 3
SILENCE_MS = 1000

app = FastAPI(title="Listener ASR API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    await ws.accept()

    sample_rate = 16000
    lang = "zh"
    try:
        print("Waiting for start message...")
        first = await ws.receive_text()
        try:
            data = json.loads(first)
            if isinstance(data, dict) and data.get("type") == "start":
                sample_rate = int(data.get("sr", 16000))
                lang = data.get("lang", "zh") or "zh"
            else:
                await ws.send_text(json.dumps({"type": "info", "msg": "no start message; using defaults"}))
        except json.JSONDecodeError:
            await ws.send_text(json.dumps({"type": "info", "msg": "no JSON start; defaulting 16k mono"}))
            pass
    except Exception:
        pass

    ev_q: asyncio.Queue = asyncio.Queue()

    loop = asyncio.get_running_loop()
    
    def on_evt(evt: dict):
        loop.call_soon_threadsafe(ev_q.put_nowait, evt)

    lis = Listener(
        on_utterance=on_evt,
        sample_rate=sample_rate,  # 16k
        lang=lang,
        source="external",       
        vad_level=VAD_LEVEL,
        silence_ms=SILENCE_MS,
    )
    lis.start()

    async def writer():
        try:
            while lis.running():
                evt = await ev_q.get()
                await ws.send_text(json.dumps(evt, ensure_ascii=False))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await ws.send_text(json.dumps({"type": "error", "error": "writer_fail", "detail": str(e)}))
            except Exception:
                pass

    writer_task = asyncio.create_task(writer())

    try:
        while True:
            msg = await ws.receive()
            print("DEBUG: got ws msg", msg)
            if "type" in msg and msg["type"] == "websocket.disconnect":
                break

            if "text" in msg:
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "end":
                        break
                except Exception:
                    await ws.send_text(json.dumps({"type": "error", "error": "bad_control_message"}))
            elif "bytes" in msg:
                pcm = msg["bytes"] 
                if pcm:
                    lis.append_pcm(pcm)
            else:
                await ws.send_text(json.dumps({"type": "error", "error": "unknown_message"}))
    except WebSocketDisconnect:
        pass
    finally:
        try:
            lis.stop()
        except Exception:
            pass
        try:
            await asyncio.wait_for(writer_task, timeout=1.0)
        except Exception:
            writer_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999, factory=False)  
