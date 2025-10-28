from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
from src.brain import Brain

app = FastAPI(title="Robot API", version="0.2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

brain = Brain()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/pose/update")
def pose_update(payload: dict = Body(...)):
    brain.update_pose(payload or {})
    p = brain.pose
    return {"ok": True, "pose": {"x": p.x, "y": p.y, "theta": p.theta}}

@app.get("/pose")
def pose_get():
    p = brain.pose
    return {"x": p.x, "y": p.y, "theta": p.theta}

@app.websocket("/brain/ws")
async def brain_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if "type" in msg and msg["type"] == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                b = msg["bytes"]
                if len(b) >= 4 and b[:4] == b"AUD0":
                    brain.append_audio_pcm(b[4:])
                    await ws.send_text(json.dumps({"type": "asr_ack"}))
                else:
                    out = brain.observe_frame(b)
                    await ws.send_text(json.dumps({"type": "observe", **out}))
            elif "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                    t = data.get("type")
                    if t == "end":
                        break
                    elif t == "pose":
                        brain.update_pose(data.get("pose", {}))
                        await ws.send_text(json.dumps({"type": "pose_ack"}))
                    else:
                        await ws.send_text(json.dumps({"type": "error", "error": "unknown_text"}))
                except Exception:
                    await ws.send_text(json.dumps({"type": "error", "error": "bad_text_json"}))
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if "type" in msg and msg["type"] == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                b = msg["bytes"]
                if len(b) >= 4 and b[:4] == b"AUD0":
                    brain.append_audio_pcm(b[4:])
                    await ws.send_text(json.dumps({"type": "asr_ack"}))
                else:
                    brain.append_audio_pcm(b)
                    await ws.send_text(json.dumps({"type": "asr_ack"}))
            elif "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "end":
                        break
                except Exception:
                    await ws.send_text(json.dumps({"type": "error", "error": "bad_text_json"}))
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999)