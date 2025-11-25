import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.brain import Brain
from src.speaker import Speaker

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

brain = Brain(speaker=Speaker())
stats = {"asr_ws": 0, "brain_ws": 0, "chat_ws": 0, "audio_pkts": 0, "video_pkts": 0, "utterances": 0, "observes": 0}

def _pyify(obj):
    import numpy as np
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_pyify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _pyify(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return _pyify(obj.tolist())
    return str(obj)


@app.get("/health")
def health():
    return {"ok": True, "stats": stats}


@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    await ws.accept()
    stats["asr_ws"] += 1
    ws_id = stats["asr_ws"]
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                b = msg["bytes"]
                if len(b) >= 4 and b[:4] == b"AUD0":
                    brain.append_audio_pcm(b[4:])
                    evt = await asyncio.to_thread(brain.listener.get, 0.3)
                    if evt:
                        await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
    except WebSocketDisconnect:
        pass


@app.websocket("/brain/ws")
async def brain_ws(ws: WebSocket):
    await ws.accept()
    stats["brain_ws"] += 1
    ws_id = stats["brain_ws"]
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                b = msg["bytes"]
                if len(b) >= 4 and b[:4] == b"AUD0":
                    brain.append_audio_pcm(b[4:])
                    evt = await asyncio.to_thread(brain.listener.get, 0.3)
                    if evt:
                        await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
                else:
                    jpeg = b
                    out = await asyncio.to_thread(brain.observe_frame, jpeg)
                    await ws.send_text(json.dumps(_pyify(out), ensure_ascii=False))
    except WebSocketDisconnect:
        pass


@app.websocket("/brain/ws/chat")
async def brain_chat_ws(ws: WebSocket):
    await ws.accept()
    stats["chat_ws"] += 1
    ws_id = stats["chat_ws"]
    session_id = f"chat#{ws_id}"
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                b = msg["bytes"]
                if len(b) >= 4 and b[:4] == b"AUD0":
                    brain.append_audio_pcm(b[4:])
                    evt = await asyncio.to_thread(brain.listener.get, 0.4)
                    if evt and evt.get("type") == "utterance":
                        text = evt["text"]
                        await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
                        result = await brain.handle_utterance(session_id, text)
                        await ws.send_text(json.dumps({
                            "type": "reply",
                            "text": result["reply_text"]
                        }, ensure_ascii=False))
                        if result["audio"]:
                            await ws.send_bytes(b"TTS0" + result["audio"])
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999)
