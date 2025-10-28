import json, asyncio, time
from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.brain import Brain

app = FastAPI(title="Robot API", version="0.3.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

brain = Brain()
stats = {"asr_ws": 0, "brain_ws": 0, "audio_pkts": 0, "video_pkts": 0, "utterances": 0, "observes": 0}

def _pyify(obj):
    import numpy as _np
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [ _pyify(x) for x in obj ]
    if isinstance(obj, dict):
        return { str(k): _pyify(v) for k,v in obj.items() }
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
    print(f"[POSE] update x={p.x:.3f} y={p.y:.3f} th={p.theta:.3f}")
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
            t = evt.get("type")
            if t == "utterance":
                stats["utterances"] += 1
                txt = (evt.get("text") or "").strip()
                conf = evt.get("confidence")
                print(f"[ASR->{tag}] #{stats['utterances']} text='{txt}' conf={None if conf is None else round(conf,3)} "
                    f"len={evt.get('meta',{}).get('audio_len_sec')}")
            elif t == "error":
                print(f"[ASR_ERR->{tag}] {evt.get('error')} detail={evt.get('detail')}")
            try:
                await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
            except Exception as e:
                print(f"[ASR_SEND_ERR->{tag}] {e}")
    except Exception as e:
        print(f"[ASR_WRITER_FAIL->{tag}] {e}")

@app.websocket("/brain/ws")
async def brain_ws(ws: WebSocket):
    await ws.accept()
    stats["brain_ws"] += 1
    ws_id = stats["brain_ws"]
    print(f"[WS/brain] open #{ws_id}")
    running = {"on": True}
    writer_task = asyncio.create_task(_asr_writer(ws, running, f"brain#{ws_id}"))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                print(f"[WS/brain] disconnect #{ws_id}")
                break

            if msg.get("bytes") is not None:
                b = msg["bytes"]
                if len(b) >= 4 and b[:4] == b"AUD0":
                    stats["audio_pkts"] += 1
                    print(f"[WS/brain] audio pkt #{stats['audio_pkts']} bytes={len(b)-4}")
                    brain.append_audio_pcm(b[4:])
                    await ws.send_text(json.dumps({"type": "asr_ack"}))
                else:
                    stats["video_pkts"] += 1
                    t0 = time.perf_counter()
                    out = await asyncio.to_thread(brain.observe_frame, b)
                    dt = (time.perf_counter() - t0) * 1000.0
                    stats["observes"] += 1
                    payload = {"type": "observe", **out, "perf": {"latency_ms": round(dt,2)}}
                    print(f"[OBS] #{stats['observes']} dt={dt:.2f}ms "
                        f"bbox={out.get('bbox')} depth={out.get('depth_m')} "
                        f"v={out.get('control',{}).get('v')} w={out.get('control',{}).get('w')}")
                    await ws.send_text(json.dumps(_pyify(payload), ensure_ascii=False))

            elif msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                    t = data.get("type")
                    if t == "end":
                        print(f"[WS/brain] end by client #{ws_id}")
                        break
                    elif t == "pose":
                        brain.update_pose(data.get("pose", {}))
                        await ws.send_text(json.dumps({"type": "pose_ack"}))
                    else:
                        print(f"[WS/brain] unknown text type={t}")
                        await ws.send_text(json.dumps({"type": "error", "error": "unknown_text"}))
                except Exception as e:
                    print(f"[WS/brain] bad_text_json err={e}")
                    await ws.send_text(json.dumps({"type": "error", "error": "bad_text_json"}))
    except WebSocketDisconnect:
        print(f"[WS/brain] closed #{ws_id}")
    finally:
        running["on"] = False
        try:
            writer_task.cancel()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        print(f"[WS/brain] finalize #{ws_id}")

@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    await ws.accept()
    stats["asr_ws"] += 1
    ws_id = stats["asr_ws"]
    print(f"[WS/asr] open #{ws_id}")
    running = {"on": True}
    writer_task = asyncio.create_task(_asr_writer(ws, running, f"asr#{ws_id}"))
    try:
        while True:
            msg = await ws.receive()
            print("DEBUG: got ws msg", msg)
            if msg.get("type") == "websocket.disconnect":
                print(f"[WS/asr] disconnect #{ws_id}")
                break

            if msg.get("bytes") is not None:
                b = msg["bytes"]
                if len(b) >= 4 and b[:4] == b"AUD0":
                    stats["audio_pkts"] += 1
                    print(f"[WS/asr] audio pkt #{stats['audio_pkts']} bytes={len(b)-4}")
                    brain.append_audio_pcm(b[4:])
                else:
                    stats["audio_pkts"] += 1
                    print(f"[WS/asr] raw audio pkt #{stats['audio_pkts']} bytes={len(b)}")
                    brain.append_audio_pcm(b)
                await ws.send_text(json.dumps({"type": "asr_ack"}))

            elif msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "end":
                        print(f"[WS/asr] end by client #{ws_id}")
                        break
                except Exception as e:
                    print(f"[WS/asr] bad_text_json err={e}")
                    await ws.send_text(json.dumps({"type": "error", "error": "bad_text_json"}))
    except WebSocketDisconnect:
        print(f"[WS/asr] closed #{ws_id}")
    finally:
        running["on"] = False
        try:
            writer_task.cancel()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        print(f"[WS/asr] finalize #{ws_id}")

if __name__ == "__main__":
    import uvicorn
    print("[BOOT] Robot API 0.3.1 starting on 0.0.0.0:9999")
    uvicorn.run(app, host="0.0.0.0", port=9999)