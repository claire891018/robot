import json, asyncio, time
from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import websockets  # ← 新增

import base64
import numpy as np
import time

from src.brain import Brain
from src.speaker import Speaker

app = FastAPI(title="Robot API", version="0.3.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

speaker = Speaker()
brain = Brain(speaker=speaker)

# 4090 的 diarization server 位址
DIARIZATION_SERVER = "ws://140.116.158.98:9997/diarization"  # ← 填入 4090 IP

stats = {"asr_ws": 0, "chat_ws": 0, "brain_ws": 0, "audio_pkts": 0, "video_pkts": 0, "utterances": 0, "observes": 0, "diar_ws": 0}

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


# ============================================================
# 新增：ASR + Diarization 整合 endpoint
# ============================================================
@app.websocket("/asr/diarization")
async def asr_diarization_ws(ws: WebSocket):
    """
    ASR + Diarization 整合 WebSocket
    
    流程：
    1. 接收前端音訊
    2. Whisper 轉錄（5090）
    3. 送給 4090 做 diarization
    4. 返回 (文字 + speaker_id) 給前端
    """
    await ws.accept()
    stats["diar_ws"] += 1
    ws_id = stats["diar_ws"]
    
    print(f"[Diar WS #{ws_id}] 連線建立")
    
    running = {"on": True}
    diar_ws = None
    
    # 連接到 4090 的 diarization server
    try:
        diar_ws = await websockets.connect(DIARIZATION_SERVER, max_size=2**25)
        print(f"[Diar WS #{ws_id}] 已連接到 4090 diarization server")
    except Exception as e:
        print(f"[Diar WS #{ws_id}] 無法連接到 diarization server: {e}")
        await ws.send_text(json.dumps({
            "type": "error",
            "error": "diarization_unavailable",
            "detail": str(e)
        }))
    
    # 暫存音訊的字典 {utterance_id: audio_bytes}
    audio_cache = {}
    utterance_counter = 0
    
    # Writer: 監聽 ASR 結果並送給 4090
    async def asr_writer():
        nonlocal utterance_counter
        
        try:
            while running["on"]:
                evt = await asyncio.to_thread(brain.listener.get, 0.2)
                if not evt:
                    continue
                
                if evt.get("type") == "utterance":
                    utterance_counter += 1
                    utt_id = utterance_counter
                    
                    text = evt.get("text", "")
                    start_ts = evt.get("start_ts", 0.0)
                    end_ts = evt.get("end_ts", 0.0)
                    confidence = evt.get("confidence", 0.0)
                    
                    print(f"[Diar WS #{ws_id}] ASR: {text[:50]}...")
                    
                    # 取得對應的音訊
                    audio_pcm = audio_cache.pop(utt_id, None)
                    
                    if diar_ws and audio_pcm:
                        try:
                            # 送給 4090: 控制訊息
                            await diar_ws.send(json.dumps({
                                "type": "diarize",
                                "text": text,
                                "start_ts": start_ts,
                                "end_ts": end_ts,
                                "confidence": confidence,
                                "audio_len": len(audio_pcm)
                            }))
                            
                            # 送給 4090: 音訊 (binary)
                            await diar_ws.send(audio_pcm)
                            
                            # 接收 4090 的回應
                            response = await asyncio.wait_for(
                                diar_ws.recv(),
                                timeout=10.0
                            )
                            
                            result = json.loads(response)
                            speaker_id = result.get("speaker_id", 0)
                            
                            print(f"[Diar WS #{ws_id}] Speaker {speaker_id}: {text[:50]}...")
                            
                            # 組合結果
                            evt["speaker_id"] = speaker_id
                            evt["type"] = "utterance_with_speaker"
                            
                        except asyncio.TimeoutError:
                            print(f"[Diar WS #{ws_id}] Diarization timeout")
                            evt["speaker_id"] = 0
                            evt["type"] = "utterance_with_speaker"
                        except Exception as e:
                            print(f"[Diar WS #{ws_id}] Diarization error: {e}")
                            evt["speaker_id"] = 0
                            evt["type"] = "utterance_with_speaker"
                    else:
                        # 沒有 diarization，直接返回
                        evt["speaker_id"] = 0
                        evt["type"] = "utterance_with_speaker"
                    
                    # 傳給前端
                    await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
                    
                else:
                    # 其他事件直接轉發
                    await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
                    
        except Exception as e:
            print(f"[Diar WS #{ws_id}] ASR Writer error: {e}")
            import traceback
            traceback.print_exc()
    
    writer_task = asyncio.create_task(asr_writer())
    
    try:
        while True:
            msg = await ws.receive()
            
            if msg.get("type") == "websocket.disconnect":
                break
            
            if msg.get("bytes") is not None:
                # 音訊資料
                b = msg["bytes"]
                stats["audio_pkts"] += 1
                
                # 去除 header
                pcm = b[4:] if len(b) >= 4 and b[:4] == b"AUD0" else b
                
                # 暫存音訊（用於後續 diarization）
                utterance_counter += 1
                audio_cache[utterance_counter] = pcm
                
                # 送給 Whisper
                brain.append_audio_pcm(pcm)
                
            elif msg.get("text") is not None:
                # 控制訊息
                try:
                    data = json.loads(msg["text"])
                    
                    if data.get("type") == "end":
                        print(f"[Diar WS #{ws_id}] 收到結束訊號")
                        break
                        
                    elif data.get("type") == "reset_speakers" and diar_ws:
                        # 轉發給 4090
                        await diar_ws.send(json.dumps({"type": "reset_speakers"}))
                        response = await diar_ws.recv()
                        await ws.send_text(response)
                        print(f"[Diar WS #{ws_id}] 已重置說話人")
                        
                except Exception as e:
                    print(f"[Diar WS #{ws_id}] 控制訊息錯誤: {e}")
    
    except WebSocketDisconnect:
        print(f"[Diar WS #{ws_id}] 連線中斷")
    except Exception as e:
        print(f"[Diar WS #{ws_id}] 錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        running["on"] = False
        writer_task.cancel()
        if diar_ws:
            await diar_ws.close()
        await ws.close()
        print(f"[Diar WS #{ws_id}] 連線關閉")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999)