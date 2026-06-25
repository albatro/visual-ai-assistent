"""Live2D / Rive rendering branch.

Server-side: LLM → TTS → viseme stream. Client-side: PIXI + pixi-live2d-display
(or @rive-app/canvas) renders the rig and drives mouth/eyes/head from the
parameter updates pushed over the WebSocket.

Bandwidth: ~30 Hz of ~200-byte JSON messages plus one audio blob per LLM
sentence. Two orders of magnitude lighter than the MuseTalk branch (which
ships JPEGs at ~50 KB × 24 fps).

WebSocket protocol (server → client)
────────────────────────────────────
  {"type": "stack", "llm": ..., "tts": ..., "rig": "live2d-cubism"}
  {"type": "thinking"}                                     // ask mode only

  // per LLM sentence (or per direct-say request):
  {"type": "segment", "idx": int, "text": str,
   "audio_size": int, "audio_format": "mp3"|"wav",
   "duration_ms": int, "tts": "edge"|"silero",
   "llm_ms": int, "tts_ms": int}
  <audio bytes>                                            // browser plays this

  // streamed at WINDOW_HZ during the segment:
  {"type": "rig", "idx": int, "t": float,                  // seconds into segment
   "params": {"ParamMouthOpenY": 0.6, "ParamA": 0.7, ...}}

  {"type": "segment_end", "idx": int}
  {"type": "answer_complete", "text": str, "elapsed": float,
   "segments": int, "ttft_ms": int}                        // ask mode only

Client → server
───────────────
  {"text": "...", "mode": "ask"|"say", "tts": "edge"|"silero"?}
"""
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from core import llm, tts
from core.visemes import audio_to_visemes

load_dotenv()

HOST = "0.0.0.0"
PORT = 8082
TTS_ENGINE = os.environ.get("TTS_ENGINE", "edge").lower()
WINDOW_HZ = 30                              # rig-param updates per second
WINDOW_SAMPLES = tts.TARGET_SAMPLE_RATE // WINDOW_HZ  # 16000 / 30 ≈ 533

RIG_KIND = os.environ.get("LIVE2D_RIG", "live2d-cubism")


# ─── RPM-viseme → Live2D Cubism parameter mapping ────────────────────────────
# Cubism param names below are the *standard* ones from the Cubism SDK
# template (ParamA, ParamI, ..., ParamMouthOpenY, ParamMouthForm). If your
# .model3.json uses non-standard names, change them in *one place* here and
# nowhere else needs to know.
#
# ParamMouthForm: -1 = round (O, U), 0 = neutral, +1 = wide (I, E, SS)
# ParamMouthOpenY: 0 = closed, 1 = wide open
RPM_TO_CUBISM: dict[str, dict[str, float]] = {
    "viseme_sil": {"ParamMouthOpenY": 0.0},
    "viseme_aa":  {"ParamA": 1.0, "ParamMouthOpenY": 0.85},
    "viseme_O":   {"ParamO": 1.0, "ParamMouthOpenY": 0.70, "ParamMouthForm": -0.5},
    "viseme_E":   {"ParamE": 1.0, "ParamMouthOpenY": 0.40, "ParamMouthForm":  0.4},
    "viseme_I":   {"ParamI": 1.0, "ParamMouthOpenY": 0.25, "ParamMouthForm":  0.6},
    "viseme_RR":  {"ParamE": 0.5, "ParamMouthOpenY": 0.30},
    "viseme_PP":  {"ParamMouthOpenY": 0.0, "ParamMouthForm":  0.0},  # closed (p/b/m)
    "viseme_nn":  {"ParamI": 0.3, "ParamMouthOpenY": 0.10},
    "viseme_DD":  {"ParamE": 0.3, "ParamMouthOpenY": 0.15},
    "viseme_SS":  {"ParamI": 0.2, "ParamMouthOpenY": 0.10, "ParamMouthForm":  0.4},
    "viseme_FF":  {"ParamMouthOpenY": 0.10, "ParamMouthForm":  0.2},
    "viseme_CH":  {"ParamI": 0.4, "ParamMouthOpenY": 0.15, "ParamMouthForm":  0.3},
}


def map_to_rig(weights: dict[str, float]) -> dict[str, float]:
    """Blend RPM viseme weights into Cubism rig parameters.
    Strongest viseme wins for mouth shape; amplitude scales jaw openness."""
    amplitude = weights.pop("amplitude", 0.0)
    if not weights:
        return {"ParamMouthOpenY": 0.0}

    # Weighted blend across all active visemes. Strongest determines shape;
    # the others taper their contribution proportionally to their weight.
    out: dict[str, float] = {}
    total_w = sum(weights.values()) or 1.0
    for vis, w in weights.items():
        mapping = RPM_TO_CUBISM.get(vis)
        if not mapping:
            continue
        share = w / total_w
        for param, val in mapping.items():
            out[param] = out.get(param, 0.0) + val * share

    # Scale mouth openness by amplitude so quiet syllables don't gape.
    if "ParamMouthOpenY" in out:
        out["ParamMouthOpenY"] *= max(0.2, amplitude)

    return out


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Live2D Avatar Branch")
llm_client = None

static_dir = Path(__file__).parent.parent / "web_live2d"
static_dir.mkdir(exist_ok=True)


@app.on_event("startup")
async def _startup():
    global llm_client
    llm_client = llm.make_client()
    if llm_client is None:
        print("[Live2D] WARNING: OPENAI_API_KEY not set — 'ask' mode disabled")
    if TTS_ENGINE == "silero":
        tts.init_silero()
    print(f"[Live2D] Ready on http://localhost:{PORT} (rig={RIG_KIND}, tts={TTS_ENGINE})")


@app.get("/")
async def index():
    idx = static_dir / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"status": "ok", "hint": "drop a Live2D index.html into web_live2d/"}


@app.get("/health")
async def health():
    return {"status": "ok", "llm": llm_client is not None, "rig": RIG_KIND}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({
        "type": "stack",
        "llm": llm.DEFAULT_MODEL if llm_client else "disabled (no OPENAI_API_KEY)",
        "tts": tts.stack_desc(TTS_ENGINE),
        "rig": RIG_KIND,
        "window_hz": WINDOW_HZ,
    })

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            text = msg.get("text", "").strip()
            mode = msg.get("mode", "ask")
            tts_engine = (msg.get("tts") or TTS_ENGINE).lower()

            if not text:
                await ws.send_json({"type": "error", "message": "Empty text"})
                continue
            if tts_engine not in ("edge", "silero"):
                await ws.send_json({"type": "error",
                                    "message": f"Unknown tts={tts_engine!r}"})
                continue

            if mode == "ask":
                if llm_client is None:
                    await ws.send_json({"type": "error",
                                        "message": "OPENAI_API_KEY not configured"})
                    continue
                await _handle_ask(ws, text, tts_engine)
            else:
                await _handle_say(ws, text, tts_engine)
    except WebSocketDisconnect:
        print("[Live2D-WS] disconnected")
    except Exception as e:
        print(f"[Live2D-WS] Error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _handle_say(ws: WebSocket, text: str, tts_engine: str) -> None:
    await _render_segment(ws, text, idx=0, tts_engine=tts_engine, llm_ms=0)
    await ws.send_json({"type": "answer_complete",
                        "text": text, "segments": 1, "elapsed": 0.0,
                        "ttft_ms": 0})


async def _handle_ask(ws: WebSocket, question: str, tts_engine: str) -> None:
    await ws.send_json({"type": "thinking"})
    t_start = time.time()

    send_lock = asyncio.Lock()
    pending: list[asyncio.Task] = []
    idx = 0
    ttft_ms: float | None = None
    answer_text = ""

    async for sentence, elapsed_ms, _is_tail in llm.stream_sentences(
        llm_client, question,
    ):
        if ttft_ms is None:
            ttft_ms = elapsed_ms
        answer_text += (" " if answer_text else "") + sentence
        pending.append(asyncio.create_task(
            _render_segment(ws, sentence, idx, tts_engine, elapsed_ms, send_lock)
        ))
        idx += 1

    if pending:
        await asyncio.gather(*pending)

    await ws.send_json({
        "type": "answer_complete",
        "text": answer_text,
        "elapsed": round(time.time() - t_start, 2),
        "ttft_ms": round(ttft_ms or 0),
        "segments": idx,
    })


async def _render_segment(
    ws: WebSocket, text: str, idx: int, tts_engine: str,
    llm_ms: float, send_lock: asyncio.Lock | None = None,
) -> None:
    """TTS + viseme stream for one sentence. send_lock serialises WS writes
    across segments so binary blobs from different sentences don't interleave."""
    t0 = time.time()
    wav_path, audio_bytes = await tts.synth(text, tts_engine)
    tts_ms = (time.time() - t0) * 1000

    # Load full PCM, scan in WINDOW_SAMPLES strides, compute viseme weights
    # then map → rig parameters. All work happens before we acquire the lock
    # so other segments can synthesise in parallel.
    pcm, sr = sf.read(wav_path, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    duration_ms = int(1000 * len(pcm) / sr)
    rig_frames: list[tuple[float, dict[str, float]]] = []
    pos = 0
    while pos < len(pcm):
        window = pcm[pos:pos + WINDOW_SAMPLES]
        t_sec = pos / sr
        weights = audio_to_visemes(window, sr, t_sec)
        rig_frames.append((t_sec, map_to_rig(weights)))
        pos += WINDOW_SAMPLES

    audio_format = "mp3" if tts_engine == "edge" else "wav"

    lock = send_lock or asyncio.Lock()
    async with lock:
        await ws.send_json({
            "type": "segment",
            "idx": idx,
            "text": text,
            "audio_size": len(audio_bytes),
            "audio_format": audio_format,
            "duration_ms": duration_ms,
            "tts": tts_engine,
            "llm_ms": round(llm_ms),
            "tts_ms": round(tts_ms),
        })
        await ws.send_bytes(audio_bytes)

        # Pace rig frames to the audio clock so they arrive ~in real time.
        # Client should still interpolate between updates for smoothness.
        t_seg_start = time.time()
        for t_sec, params in rig_frames:
            wait = t_sec - (time.time() - t_seg_start)
            if wait > 0:
                await asyncio.sleep(wait)
            await ws.send_json({
                "type": "rig", "idx": idx, "t": round(t_sec, 3),
                "params": {k: round(v, 3) for k, v in params.items()},
            })

        await ws.send_json({"type": "segment_end", "idx": idx})

    try:
        os.unlink(wav_path)
    except OSError:
        pass

    print(f"[Live2D Seg {idx}] llm={llm_ms:.0f}ms tts={tts_ms:.0f}ms "
          f"audio={duration_ms}ms frames={len(rig_frames)} text={text[:60]!r}")


if __name__ == "__main__":
    uvicorn.run(
        "branches.live2d_branch:app",
        host=HOST, port=PORT, log_level="warning",
        ws="wsproto",  # default "websockets" backend duplicates Date header
                       # and breaks strict clients like websockets>=16.
    )
