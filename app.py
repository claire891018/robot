import json
import asyncio
import time

from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.brain import Brain
from src.speaker import Speaker

app = FastAPI(title="Robot API", version="0.3.1")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

speaker = Speaker()
brain = Brain(speaker=speaker)

stats = {
    "asr_ws": 0,
    "brain_ws": 0,
    "audio_pkts": 0,
    "video_pkts": 0,
    "utterances": 0,
    "observes": 0,
}

MIN_CONFIDENCE = 0.6  # ASR 基本門檻


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
    print(f"[POSE] update x={p.x:.3f} y={p.y:.3f} th={p.theta:.3f}")
    return {"ok": True, "pose": {"x": p.x, "y": p.y, "theta": p.theta}}


@app.get("/pose")
def pose_get():
    p = brain.pose
    return {"x": p.x, "y": p.y, "theta": p.theta}


async def _asr_writer(ws: WebSocket, running_flag, tag: str):
    """
    純 ASR streaming 用在 /asr
    """
    try:
        print("==" * 15)
        print(f"[ASR_WRITER-{tag}] started")
        while running_flag["on"]:
            evt = await asyncio.to_thread(brain.listener.get, 0.2)
            if not evt:
                continue

            t = evt.get("type")

            if t == "utterance":
                txt = (evt.get("text") or "").strip()
                conf = evt.get("confidence")

                if not txt:
                    continue
                if conf is not None and conf < MIN_CONFIDENCE:
                    continue

                stats["utterances"] += 1
                print(
                    f"[ASR->{tag}] #{stats['utterances']} text='{txt}' "
                    f"conf={None if conf is None else round(conf, 3)} "
                    f"len={evt.get('meta', {}).get('audio_len_sec')}"
                )

            elif t == "error":
                print(
                    f"[ASR_ERR->{tag}] {evt.get('error')} detail={evt.get('detail')}"
                )

            try:
                await ws.send_text(json.dumps(_pyify(evt), ensure_ascii=False))
            except Exception as e:
                print(f"[ASR_SEND_ERR->{tag}] {e}")
    except Exception as e:
        print(f"[ASR_WRITER_FAIL-{tag}] {e}")


@app.websocket("/brain/ws")
async def brain_ws(ws: WebSocket):
    await ws.accept()
    stats["brain_ws"] += 1
    ws_id = stats["brain_ws"]
    print(f"[WS/brain] open #{ws_id}")
    t_start = time.perf_counter()
    try:
        while True:
            msg = await ws.receive()

            if msg.get("bytes"):
                print(f"[DEBUG-Brain] got bytes len={len(msg['bytes'])}")
            else:
                print(f"[DEBUG-Brain] got ws msg {str(msg)[:100]}")

            if msg.get("type") == "websocket.disconnect":
                print(f"[WS/brain] disconnect #{ws_id}")
                break

            if msg.get("bytes") is not None:
                b = msg["bytes"]

                # 音訊封包：AUD0 + PCM16
                if len(b) >= 4 and b[:4] == b"AUD0":
                    stats["audio_pkts"] += 1
                    payload = b[4:]
                    print(
                        f"[WS/brain] audio pkt #{stats['audio_pkts']} bytes={len(payload)}"
                    )

                    # 丟給 Listener
                    t_append = time.perf_counter()
                    brain.append_audio_pcm(payload)
                    print(
                        f"[TIME] append_audio_pcm: {(time.perf_counter()-t_append)*1000:.2f}ms"
                    )

                    # 等待一個 ASR 結果
                    print(f"[WS/brain] waiting for ASR result...")
                    timeout = 15.0
                    start = time.time()
                    asr_wait_start = time.perf_counter()
                    while time.time() - start < timeout:
                        evt = await asyncio.to_thread(brain.listener.get, 0.5)
                        if not evt:
                            continue

                        asr_wait_time = (time.perf_counter() - asr_wait_start) * 1000
                        print(f"[TIME] ASR wait: {asr_wait_time:.2f}ms")
                        t = evt.get("type")

                        if t == "utterance":
                            txt = (evt.get("text") or "").strip()
                            conf = evt.get("confidence")

                            if not txt:
                                print("[WS/brain-ASR] empty text, skip")
                                break
                            if conf is not None and conf < MIN_CONFIDENCE:
                                print(
                                    f"[WS/brain-ASR] low confidence {conf}, skip text='{txt}'"
                                )
                                break

                            stats["utterances"] += 1
                            print(f"[WS/brain-ASR] text='{txt}' conf={conf}")

                            # 1. 先把 ASR 結果丟回前端（即時字幕）
                            await ws.send_text(
                                json.dumps(_pyify(evt), ensure_ascii=False)
                            )

                            # 2. 交給大腦：想要怎麼回、順便請 Speaker 說出來
                            session_id = f"brain#{ws_id}"
                            brain_result = await brain.handle_utterance(
                                session_id, txt
                            )

                            reply_text = (brain_result.get("reply_text") or "").strip()
                            need_tts = bool(brain_result.get("need_tts", False))
                            audio_bytes = brain_result.get("audio")

                            # 3. 如果有聲音，就送給前端播
                            if need_tts and audio_bytes:
                                print(
                                    f"[WS/brain-TTS] send {len(audio_bytes)} bytes (from Brain)"
                                )
                                await ws.send_bytes(b"TTS0" + audio_bytes)

                            break

                        elif t == "error":
                            print(f"[WS/brain-ASR] error={evt.get('error')}")
                            await ws.send_text(
                                json.dumps(_pyify(evt), ensure_ascii=False)
                            )
                            break
                    else:
                        print(f"[WS/brain] ASR timeout after {timeout}s")

                # 影片畫面（觀察）
                else:
                    stats["video_pkts"] += 1
                    print(f"[WS/brain] start observe_frame...")
                    t0 = time.perf_counter()
                    out = await asyncio.to_thread(brain.observe_frame, b)
                    dt = (time.perf_counter() - t0) * 1000.0
                    print(f"[TIME] observe_frame total: {dt:.2f}ms")
                    stats["observes"] += 1
                    payload = {
                        "type": "observe",
                        **out,
                        "perf": {"latency_ms": round(dt, 2)},
                    }
                    print(
                        f"[OBS] #{stats['observes']} dt={dt:.2f}ms "
                        f"bbox={out.get('bbox')} depth={out.get('depth_m')} "
                        f"v={out.get('control',{}).get('v')} w={out.get('control',{}).get('w')}"
                    )

                    t_send = time.perf_counter()
                    await ws.send_text(
                        json.dumps(_pyify(payload), ensure_ascii=False)
                    )
                    print(
                        f"[TIME] send_text: {(time.perf_counter()-t_send)*1000:.2f}ms"
                    )

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
                        await ws.send_text(
                            json.dumps(
                                {"type": "error", "error": "unknown_text"},
                                ensure_ascii=False,
                            )
                        )
                except Exception as e:
                    print(f"[WS/brain] bad_text_json err={e}")
                    await ws.send_text(
                        json.dumps(
                            {"type": "error", "error": "bad_text_json"},
                            ensure_ascii=False,
                        )
                    )
    except WebSocketDisconnect:
        print(f"[WS/brain] closed #{ws_id}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        t_total = (time.perf_counter() - t_start) * 1000
        print(f"[WS/brain] total time: {t_total:.2f}ms")
        print(f"[WS/brain] finalize #{ws_id}")


@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    """
    純 ASR WebSocket：只負責把音訊送進 Listener，
    然後用 _asr_writer 把結果 streaming 回來，不經過大腦 / TTS。
    """
    await ws.accept()
    stats["asr_ws"] += 1
    ws_id = stats["asr_ws"]
    print(f"[WS/asr] open #{ws_id}")
    running = {"on": True}
    writer_task = asyncio.create_task(_asr_writer(ws, running, f"asr#{ws_id}"))
    try:
        while True:
            msg = await ws.receive()
            print(f"[DEBUG-ASR] got ws msg {str(msg)[:100]}")
            if msg.get("type") == "websocket.disconnect":
                print(f"[WS/asr] disconnect #{ws_id}")
                break

            if msg.get("bytes") is not None:
                b = msg["bytes"]
                if len(b) >= 4 and b[:4] == b"AUD0":
                    stats["audio_pkts"] += 1
                    payload = b[4:]
                    print(
                        f"[WS/asr] audio pkt #{stats['audio_pkts']} bytes={len(payload)}"
                    )
                    t_append = time.perf_counter()
                    brain.append_audio_pcm(payload)
                    print(
                        f"[TIME] append_audio_pcm: {(time.perf_counter()-t_append)*1000:.2f}ms"
                    )
                else:
                    stats["audio_pkts"] += 1
                    print(
                        f"[WS/asr] raw audio pkt #{stats['audio_pkts']} bytes={len(b)}"
                    )
                    t_append = time.perf_counter()
                    brain.append_audio_pcm(b)
                    print(
                        f"[TIME] append_audio_pcm: {(time.perf_counter()-t_append)*1000:.2f}ms"
                    )

                t_ack = time.perf_counter()
                await ws.send_text(json.dumps({"type": "asr_ack"}))
                print(
                    f"[TIME] send ack: {(time.perf_counter()-t_ack)*1000:.2f}ms"
                )

            elif msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "end":
                        print(f"[WS/asr] end by client #{ws_id}")
                        break
                except Exception as e:
                    print(f"[WS/asr] bad_text_json err={e}")
                    await ws.send_text(
                        json.dumps(
                            {"type": "error", "error": "bad_text_json"},
                            ensure_ascii=False,
                        )
                    )
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
