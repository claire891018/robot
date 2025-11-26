import json, asyncio, time
from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File

import base64
import numpy as np
import time
# import torch

from src.brain import Brain
from src.speaker import Speaker

# from src.listener_with_diarization import ListenerWithDiarization

app = FastAPI(title="Robot API", version="0.3.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

speaker = Speaker()
brain = Brain(speaker=speaker)

# listener_with_diar = ListenerWithDiarization(
#     sample_rate=16000,
#     device="cuda",
#     source="diarization"
# )
# listener_with_diar.start()

stats = {"asr_ws": 0, "chat_ws": 0, "brain_ws": 0, "audio_pkts": 0, "video_pkts": 0, "utterances": 0, "observes": 0}

def _pyify(obj):
    import numpy as _np
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_pyify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _pyify(v) for k, v in obj.items()}
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.ndarray,)):
        return _pyify(obj.tolist())
    return str(obj)

@app.get("/health")
def health():
    return {"ok": True, "stats": stats}

@app.post("/pose/update")
def pose_update(payload: dict = Body(...)):
    brain.update_pose(payload or {})
    p = brain.pose
    return {"ok": True, "pose": {"x": p.x, "y": p.y, "theta": p.theta}}

@app.get("/pose")
def pose_get():
    p = brain.pose
    return {"x": p.x, "y": p.y, "theta": p.theta}

async def _asr_writer(ws: WebSocket, running_flag, tag: str):
    try:
        while running_flag["on"]:
            evt = await asyncio.to_thread(brain.listener.get, 0.2)
            if not evt:
                continue
            try:
                await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
            except:
                break
    except:
        pass

@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    await ws.accept()
    stats["asr_ws"] += 1
    ws_id = stats["asr_ws"]
    running = {"on": True}
    writer_task = asyncio.create_task(_asr_writer(ws, running, f"asr#{ws_id}"))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                b = msg["bytes"]
                stats["audio_pkts"] += 1
                brain.append_audio_pcm(b[4:] if len(b) >= 4 and b[:4] == b"AUD0" else b)
                await ws.send_text(json.dumps({"type": "asr_ack"}))
            elif msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "end":
                        break
                except:
                    await ws.send_text(json.dumps({"type": "error", "error": "bad_text_json"}))
    except WebSocketDisconnect:
        pass
    finally:
        running["on"] = False
        writer_task.cancel()
        await ws.close()

@app.websocket("/brain/ws")
async def brain_ws(ws: WebSocket):
    await ws.accept()
    stats["brain_ws"] += 1
    ws_id = stats["brain_ws"]
    running = {"on": True}
    writer_task = asyncio.create_task(_asr_writer(ws, running, f"brain#{ws_id}"))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                b = msg["bytes"]
                if len(b) >= 4 and b[:4] == b"AUD0":
                    stats["audio_pkts"] += 1
                    brain.append_audio_pcm(b[4:])
                    timeout = 15.0
                    start = time.time()
                    while time.time() - start < timeout:
                        evt = await asyncio.to_thread(brain.listener.get, 0.5)
                        if evt:
                            await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
                            break
            elif msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "end":
                        break
                except:
                    await ws.send_text(json.dumps({"type": "error", "error": "bad_json"}))
    except WebSocketDisconnect:
        pass
    finally:
        running["on"] = False
        writer_task.cancel()
        await ws.close()

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
    
@app.post("/brain/voice_chat")
async def voice_chat(file: UploadFile = File(...)):
    raw = await file.read()
    pcm = np.frombuffer(raw, dtype=np.int16)

    result = {}

    def _cb(evt):
        if evt.get("type") == "utterance":
            result["evt"] = evt

    old = brain.listener.on_utterance
    brain.listener.on_utterance = _cb
    brain.listener.append_pcm(pcm.tobytes())

    for _ in range(200):
        if "evt" in result:
            break
        time.sleep(0.05)

    brain.listener.on_utterance = old

    if "evt" not in result:
        return {"user_text": "", "reply_text": "我聽不清楚", "tts": ""}

    user_text = result["evt"]["text"]
    out = await brain.handle_utterance("rest", user_text)

    audio = out.get("audio")
    tts_b64 = base64.b64encode(audio).decode() if audio else ""

    return {
        "user_text": user_text,
        "reply_text": out.get("reply_text",""),
        "tts": tts_b64
    }

# @app.websocket("/asr/diarization")
# async def asr_diarization_ws(ws: WebSocket):
#     """
#     ASR + Speaker Diarization WebSocket
#     完整的即時語音辨識 + 說話人辨識
#     """
#     await ws.accept()
#     stats["asr_ws"] += 1
#     ws_id = stats["asr_ws"]
    
#     running = {"on": True}
    
#     # Writer: 將辨識結果傳給前端
#     async def writer():
#         try:
#             while running["on"]:
#                 evt = await asyncio.to_thread(listener_with_diar.get, 0.2)
#                 if not evt:
#                     continue
#                 try:
#                     await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
#                 except:
#                     break
#         except:
#             pass
    
#     writer_task = asyncio.create_task(writer())
    
#     try:
#         while True:
#             msg = await ws.receive()
            
#             if msg.get("type") == "websocket.disconnect":
#                 break
            
#             if msg.get("bytes") is not None:
#                 # 音訊資料
#                 b = msg["bytes"]
#                 stats["audio_pkts"] += 1
                
#                 # 去除 header 後送給 listener
#                 pcm = b[4:] if len(b) >= 4 and b[:4] == b"AUD0" else b
#                 listener_with_diar.append_pcm(pcm)
                
#                 # 確認訊息
#                 await ws.send_text(json.dumps({"type": "ack"}))
                
#             elif msg.get("text") is not None:
#                 # 控制訊息
#                 try:
#                     data = json.loads(msg["text"])
                    
#                     if data.get("type") == "end":
#                         break
#                     elif data.get("type") == "reset_speakers":
#                         # 重置說話人資料庫
#                         listener_with_diar.reset_speakers()
#                         await ws.send_text(json.dumps({
#                             "type": "speakers_reset",
#                             "message": "說話人資料已重置"
#                         }))
                        
#                 except Exception as e:
#                     await ws.send_text(json.dumps({
#                         "type": "error",
#                         "error": "bad_json",
#                         "detail": str(e)
#                     }))
    
#     except WebSocketDisconnect:
#         pass
#     finally:
#         running["on"] = False
#         writer_task.cancel()
#         await ws.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999)
