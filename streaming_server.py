"""
Streaming server for MuseTalk avatar with chunked generation.
Text → TTS → Split audio into ~2s chunks → Generate & stream frames per chunk.

Run: python streaming_server.py
Open: http://localhost:8080
"""

import asyncio
import io
import json
import os
import random
import re
import sys
import tempfile
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import edge_tts
import torch
import torchaudio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from openai import AsyncOpenAI

load_dotenv()  # read .env in CWD so OPENAI_API_KEY etc. are picked up

# Add MuseTalk to path
sys.path.insert(0, str(Path(__file__).parent / "MuseTalk"))

from musetalk_wrapper import MuseTalkEngine

# ─── Configuration ───────────────────────────────────────────────────────────
AVATAR_VIDEO = "kling_peter_cropped.mp4"      # 800x800 head/shoulders for MuseTalk
FULL_BODY_VIDEO = "kling_peter_ref.mp4"       # 1176x1764 original for background
CROP_X, CROP_Y = 190, 0                       # offset of crop inside full body
AVATAR_ID = "peter_stream"

# TTS engine selection: "edge" (cloud Microsoft) or "silero" (local PyTorch).
# Default via env: TTS_ENGINE=silero python streaming_server.py
# Per-request override via WS message field: {"text": "...", "tts": "silero"}
TTS_ENGINE = os.environ.get("TTS_ENGINE", "edge").lower()

# edge-tts
TTS_VOICE = "ru-RU-DmitryNeural"
TTS_RATE = "-5%"

# Silero (v4_ru speakers: aidar, baya, kseniya, xenia, eugene, random)
SILERO_SPEAKER = os.environ.get("SILERO_SPEAKER", "aidar")
SILERO_SAMPLE_RATE = 24000   # native; resampled to 16k for MuseTalk
SILERO_PUT_ACCENT = os.environ.get("SILERO_PUT_ACCENT", "1") != "0"
SILERO_PUT_YO = os.environ.get("SILERO_PUT_YO", "1") != "0"
MUSETALK_SAMPLE_RATE = 16000

FPS = 24
JPEG_QUALITY = 85
HOST = "0.0.0.0"
PORT = 8080
CHUNK_SECONDS = 4.0  # Split audio into chunks of this duration

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_SYSTEM = (
    "Ты — Пётр Первый, основатель Санкт-Петербурга, российский император. "
    "Отвечай кратко (2-4 предложения), от первого лица, как Пётр I в речи, "
    "но современным русским языком. Без эмодзи и markdown."
)
LLM_MAX_TOKENS = 250

# ─── Global engine (loaded once) ─────────────────────────────────────────────
engine: MuseTalkEngine = None
avatar = None
full_body_cycle: list = []   # forward + reverse, matches avatar.coord_list_cycle length
engine_lock = threading.Lock()
llm_client: AsyncOpenAI = None

# Silero TTS (lazy-loaded on first use unless TTS_ENGINE=silero at startup)
silero_model = None
silero_lock = threading.Lock()

# Filler clips (pre-generated short phrases played while LLM/TTS/render is in flight)
FILLERS_DIR = Path(__file__).parent / "fillers"
fillers: list[dict] = []                   # [{id, text, audio_bytes, duration_ms}]
recent_fillers: deque = deque(maxlen=5)   # anti-repeat ring


def init_engine():
    """Load MuseTalk models, prepare avatar, cache full-body background, init LLM."""
    global engine, avatar, full_body_cycle, llm_client
    if os.environ.get("OPENAI_API_KEY"):
        llm_client = AsyncOpenAI()
    else:
        print("[Server] WARNING: OPENAI_API_KEY not set — 'ask' mode disabled, 'say' mode still works")
        llm_client = None
    print("[Server] Initializing MuseTalk engine...")
    engine = MuseTalkEngine(fps=FPS)
    engine.load_models()
    avatar = engine.prepare_avatar(AVATAR_ID, AVATAR_VIDEO)

    print(f"[Server] Loading full-body background from {FULL_BODY_VIDEO}...")
    cap = cv2.VideoCapture(FULL_BODY_VIDEO)
    bg_frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        bg_frames.append(f)
    cap.release()
    full_body_cycle = bg_frames + bg_frames[::-1]
    print(f"[Server] Cached {len(full_body_cycle)} background frames "
          f"({bg_frames[0].shape[1]}x{bg_frames[0].shape[0]})")

    load_fillers()

    if TTS_ENGINE == "silero":
        init_silero()

    print("[Server] Engine ready!")


def init_silero() -> None:
    """Load Silero v4_ru TTS model onto GPU (or CPU fallback) and warm it up.
    Idempotent: safe to call repeatedly; subsequent calls are no-ops.
    Downloads ~60 MB on first run via torch.hub."""
    global silero_model
    with silero_lock:
        if silero_model is not None:
            return
        print(f"[Silero] Loading v4_ru (speaker={SILERO_SPEAKER})...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
            trust_repo=True,
        )
        model.to(device)
        # Warmup x3 with varied lengths to JIT all GPU kernels and accent paths.
        # Skipping this leaves the first ~2 real requests at 1-3s each.
        warmup_texts = [
            "Готов к работе.",
            "Я построил Петербург на болоте, чтобы открыть России окно в Европу.",
            "Расскажу тебе кратко, в двух предложениях, о моих делах и победах, "
            "о городе на Неве и о морском флоте.",
        ]
        t0 = time.time()
        with torch.inference_mode():
            for wt in warmup_texts:
                _ = model.apply_tts(
                    text=wt,
                    speaker=SILERO_SPEAKER,
                    sample_rate=SILERO_SAMPLE_RATE,
                    put_accent=SILERO_PUT_ACCENT,
                    put_yo=SILERO_PUT_YO,
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        silero_model = model
        print(f"[Silero] Ready on {device} (warmup x{len(warmup_texts)} "
              f"in {(time.time()-t0)*1000:.0f}ms, accent={SILERO_PUT_ACCENT}, "
              f"yo={SILERO_PUT_YO})")


def load_fillers() -> None:
    """Load pre-generated filler audio clips into memory. Safe to call when
    fillers/ is missing — just leaves the list empty."""
    global fillers
    manifest_path = FILLERS_DIR / "manifest.json"
    if not manifest_path.exists():
        print("[Fillers] No manifest.json — run generate_fillers.py first")
        return
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    loaded = []
    for entry in manifest:
        mp3_path = FILLERS_DIR / entry["mp3"]
        if not mp3_path.exists():
            continue
        loaded.append({
            "id": entry["id"],
            "text": entry["text"],
            "audio_bytes": mp3_path.read_bytes(),
            "duration_ms": entry["duration_ms"],
        })
    fillers = loaded
    total_kb = sum(len(x["audio_bytes"]) for x in fillers) / 1024
    print(f"[Fillers] Loaded {len(fillers)} clips ({total_kb:.0f} KB in RAM)")


def pick_filler() -> dict | None:
    """Random filler, avoiding the last 5 used."""
    if not fillers:
        return None
    pool = [f for f in fillers if f["id"] not in recent_fillers] or fillers
    pick = random.choice(pool)
    recent_fillers.append(pick["id"])
    return pick


def composite_one(frame, i: int):
    """Paste a single cropped lip-synced frame onto the i-th full-body background."""
    if not full_body_cycle:
        return frame
    bg = full_body_cycle[i % len(full_body_cycle)].copy()
    ch, cw = frame.shape[:2]
    bg[CROP_Y:CROP_Y + ch, CROP_X:CROP_X + cw] = frame
    return bg


# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="MuseTalk Avatar Streaming")

static_dir = Path(__file__).parent / "web_stream"
static_dir.mkdir(exist_ok=True)


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": engine is not None}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected")

    await ws.send_json({
        "type": "stack",
        "llm": LLM_MODEL if llm_client else f"{LLM_MODEL} (disabled — no OPENAI_API_KEY)",
        "tts": _tts_stack_desc(),
        "lipsync": "MuseTalk v1.5",
        "fps": FPS,
    })

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            text = msg.get("text", "").strip()
            mode = msg.get("mode", "ask")  # "ask" = LLM, "say" = direct TTS
            tts_override = msg.get("tts")  # "edge" | "silero" | None=server default

            if not text:
                await ws.send_json({"type": "error", "message": "Empty text"})
                continue

            if tts_override and tts_override.lower() not in ("edge", "silero"):
                await ws.send_json({
                    "type": "error",
                    "message": f"Unknown tts={tts_override!r}; expected 'edge' or 'silero'",
                })
                continue

            tts_engine = (tts_override or TTS_ENGINE).lower()
            print(f"[WS] Received ({mode}, tts={tts_engine}): {text[:80]}...")

            if mode == "ask":
                if llm_client is None:
                    await ws.send_json({
                        "type": "error",
                        "message": "OPENAI_API_KEY not configured; use mode='say' or set the env var",
                    })
                    continue
                try:
                    await process_question_streaming(ws, text, tts_engine=tts_engine)
                except Exception as e:
                    print(f"[LLM/Stream] Error: {e}")
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue
            else:
                await process_say(ws, text, tts_engine=tts_engine)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass


def _tts_stack_desc() -> str:
    """Human-readable summary of TTS engine state for the stack handshake."""
    silero_state = "loaded" if silero_model is not None else "lazy"
    return (
        f"default={TTS_ENGINE} · "
        f"edge={TTS_VOICE}@{TTS_RATE} · "
        f"silero={SILERO_SPEAKER} ({silero_state})"
    )


async def ask_llm(question: str) -> str:
    """Ask the LLM, return plain-text answer (non-streaming)."""
    resp = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.7,
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip()


SENTENCE_END = re.compile(r"[.!?…](?=\s|$)")


async def process_question_streaming(
    ws: WebSocket, question: str, tts_engine: str | None = None
):
    """Stream LLM token-by-token; on each completed sentence, kick off
    TTS+MuseTalk in parallel and ship audio+frames to the client.
    Drops time-to-first-voice from ~7s to ~2s."""
    await ws.send_json({"type": "thinking"})
    t_start = time.time()

    # Kick off a filler clip immediately to mask LLM/TTS/render latency.
    # Client interrupts the filler audio when the first real segment lands.
    filler = pick_filler()
    if filler:
        await ws.send_json({
            "type": "filler",
            "id": filler["id"],
            "text": filler["text"],
            "duration_ms": filler["duration_ms"],
            "audio_size": len(filler["audio_bytes"]),
        })
        await ws.send_bytes(filler["audio_bytes"])

    stream = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.7,
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": question},
        ],
        stream=True,
    )

    buffer = ""
    answer_text = ""
    segment_idx = 0
    pending: list[asyncio.Task] = []
    send_lock = asyncio.Lock()  # only one segment streams bytes at a time
    ttft_ms: float | None = None  # time-to-first-token

    async for ev in stream:
        delta = ev.choices[0].delta.content or ""
        if not delta:
            continue
        if ttft_ms is None:
            ttft_ms = (time.time() - t_start) * 1000
        buffer += delta
        answer_text += delta

        # Flush every completed sentence
        while True:
            m = SENTENCE_END.search(buffer)
            if not m:
                break
            cut = m.end()
            sentence = buffer[:cut].strip()
            buffer = buffer[cut:].lstrip()
            if sentence:
                llm_ms = (time.time() - t_start) * 1000
                if segment_idx == 0:
                    print(f"[LLM] first sentence ready at +{llm_ms:.0f}ms (ttft {ttft_ms:.0f}ms)")
                pending.append(asyncio.create_task(
                    render_segment(ws, sentence, segment_idx, send_lock, llm_ms,
                                   tts_engine=tts_engine)
                ))
                segment_idx += 1

    # Flush remainder (text without terminating punctuation)
    tail = buffer.strip()
    if tail:
        llm_ms = (time.time() - t_start) * 1000
        pending.append(asyncio.create_task(
            render_segment(ws, tail, segment_idx, send_lock, llm_ms,
                           tts_engine=tts_engine)
        ))
        segment_idx += 1

    # Wait for all segments to finish streaming
    if pending:
        await asyncio.gather(*pending)

    await ws.send_json({
        "type": "answer_complete",
        "text": answer_text,
        "elapsed": round(time.time() - t_start, 2),
        "ttft_ms": round(ttft_ms or 0),
        "segments": segment_idx,
    })
    print(f"[Stream] Total {segment_idx} segments in {time.time()-t_start:.1f}s")


async def render_segment(
    ws: WebSocket, text: str, idx: int, send_lock: asyncio.Lock, llm_ms: float,
    tts_engine: str | None = None,
):
    """TTS + MuseTalk for one sentence. Serialised via send_lock so segments
    arrive at the client in order. llm_ms = time from request start to when
    this sentence was ready from the LLM stream."""
    loop = asyncio.get_event_loop()
    t0 = time.time()

    wav_path, audio_bytes = await generate_tts(text, engine=tts_engine)
    tts_ms = (time.time() - t0) * 1000

    t_gen = time.time()
    frames = await loop.run_in_executor(None, _generate_frames_sync, wav_path)
    gen_ms = (time.time() - t_gen) * 1000

    # Serialise the WS send so binary blobs of different segments don't interleave
    async with send_lock:
        await ws.send_json({
            "type": "segment",
            "idx": idx,
            "text": text,
            "audio_size": len(audio_bytes),
            "frame_count": len(frames),
            "fps": FPS,
            "tts": (tts_engine or TTS_ENGINE).lower(),
            "llm_ms": round(llm_ms),
            "tts_ms": round(tts_ms),
            "gen_ms": round(gen_ms),
            "gen_fps": round(len(frames) / (gen_ms / 1000), 1) if gen_ms else 0,
        })
        await ws.send_bytes(audio_bytes)
        for i, frame in enumerate(frames):
            composed = composite_one(frame, i)
            _, jpeg = cv2.imencode(
                ".jpg", composed, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            await ws.send_bytes(jpeg.tobytes())
            del composed
        await ws.send_json({"type": "segment_end", "idx": idx})

    print(f"[Seg {idx}] llm={llm_ms:.0f}ms tts={tts_ms:.0f}ms gen={gen_ms:.0f}ms "
          f"frames={len(frames)} text={text[:60]!r}")

    try:
        os.unlink(wav_path)
    except OSError:
        pass


async def process_say(ws: WebSocket, text: str, tts_engine: str | None = None):
    """say mode: text → TTS → MuseTalk, using the same segment protocol as ask mode."""
    await ws.send_json({"type": "thinking"})
    t0 = time.time()
    lock = asyncio.Lock()
    await render_segment(ws, text, 0, lock, 0.0, tts_engine)
    await ws.send_json({
        "type": "answer_complete",
        "text": text,
        "elapsed": round(time.time() - t0, 2),
        "segments": 1,
        "ttft_ms": 0,
    })


async def process_text_chunked(
    ws: WebSocket, text: str, tts_engine: str | None = None
):
    """Process text with chunked audio generation for real-time feel."""
    loop = asyncio.get_event_loop()

    # Step 1: Generate full TTS audio
    await ws.send_json({"type": "status", "message": "Generating speech..."})
    wav_path, audio_bytes = await generate_tts(text, engine=tts_engine)

    # Step 2: Send full MP3 to client (browser will play it)
    await ws.send_json({"type": "audio", "size": len(audio_bytes)})
    await ws.send_bytes(audio_bytes)

    # Step 3: Split WAV into chunks and generate frames per chunk
    await ws.send_json({"type": "stream_start", "fps": FPS})

    chunk_paths = split_audio(wav_path, CHUNK_SECONDS)
    total_frames_sent = 0
    t0 = time.time()

    for i, chunk_path in enumerate(chunk_paths):
        await ws.send_json({
            "type": "status",
            "message": f"Generating chunk {i+1}/{len(chunk_paths)}..."
        })

        # Generate frames for this chunk in thread pool
        frames = await loop.run_in_executor(
            None, _generate_frames_sync, chunk_path
        )

        # Stream frames immediately as JPEG (composite onto fullbody per-frame
        # to avoid OOM on long answers)
        for j, frame in enumerate(frames):
            composed = composite_one(frame, total_frames_sent)
            _, jpeg_buf = cv2.imencode(
                ".jpg", composed, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            await ws.send_bytes(jpeg_buf.tobytes())
            total_frames_sent += 1
            del composed

        # Cleanup chunk file
        try:
            os.unlink(chunk_path)
        except OSError:
            pass

    elapsed = time.time() - t0
    await ws.send_json({
        "type": "stream_end",
        "frames_sent": total_frames_sent,
        "elapsed": round(elapsed, 2),
    })
    print(f"[WS] Streamed {total_frames_sent} frames in {elapsed:.1f}s "
          f"({total_frames_sent / elapsed:.1f} FPS)")

    # Cleanup
    try:
        os.unlink(wav_path)
    except OSError:
        pass


def split_audio(wav_path: str, chunk_seconds: float) -> list[str]:
    """Split a WAV file into chunks of chunk_seconds duration."""
    data, sr = sf.read(wav_path, dtype="float32")
    chunk_samples = int(chunk_seconds * sr)
    total_samples = len(data)

    chunk_paths = []
    offset = 0
    idx = 0

    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        chunk_data = data[offset:end]

        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_chunk{idx}.wav", delete=False, dir="."
        )
        tmp.close()
        sf.write(tmp.name, chunk_data, sr)
        chunk_paths.append(tmp.name)

        offset = end
        idx += 1

    print(f"[Audio] Split into {len(chunk_paths)} chunks of ~{chunk_seconds}s")
    return chunk_paths


async def generate_tts(text: str, engine: str | None = None) -> tuple[str, bytes]:
    """Dispatch TTS to the selected engine. Override engine per call by passing
    "edge" or "silero"; otherwise falls back to the server-default TTS_ENGINE.
    Returns (wav_path_for_musetalk, audio_bytes_for_client)."""
    eng = (engine or TTS_ENGINE).lower()
    if eng == "silero":
        return await _generate_tts_silero(text)
    if eng == "edge":
        return await _generate_tts_edge(text)
    raise ValueError(f"Unknown TTS engine: {eng!r} (expected 'edge' or 'silero')")


async def _generate_tts_edge(text: str) -> tuple[str, bytes]:
    """edge-tts streaming: mp3 chunks → ffmpeg (mp3→wav 16k mono) in parallel.
    By the time the last chunk arrives, ffmpeg is nearly done.
    Returns (wav_path, mp3_bytes)."""
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=".")
    wav_tmp.close()

    ff = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "mp3", "-i", "pipe:0",
        "-ar", str(MUSETALK_SAMPLE_RATE), "-ac", "1",
        wav_tmp.name,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    mp3_chunks: list[bytes] = []
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data = chunk["data"]
            mp3_chunks.append(data)
            ff.stdin.write(data)
    ff.stdin.close()
    await ff.wait()

    return wav_tmp.name, b"".join(mp3_chunks)


async def _generate_tts_silero(text: str) -> tuple[str, bytes]:
    """Silero v4_ru local TTS. Synthesises at 24k, resamples to 16k for MuseTalk.
    One WAV-encode pass: same bytes go to client and to disk (MuseTalk needs a path).
    Returns (wav_path, wav_bytes)."""
    if silero_model is None:
        init_silero()

    loop = asyncio.get_event_loop()
    audio_16k = await loop.run_in_executor(None, _silero_synth_sync, text)

    t_enc = time.time()
    buf = io.BytesIO()
    sf.write(buf, audio_16k, MUSETALK_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=".")
    wav_tmp.write(wav_bytes)
    wav_tmp.close()
    print(f"[Silero] encode+write {len(wav_bytes)//1024}KB in {(time.time()-t_enc)*1000:.0f}ms")
    return wav_tmp.name, wav_bytes


def _silero_synth_sync(text: str) -> np.ndarray:
    """Blocking Silero synthesis. Returns float32 mono numpy at MUSETALK_SAMPLE_RATE."""
    t0 = time.time()
    with silero_lock, torch.inference_mode():
        tensor = silero_model.apply_tts(
            text=text,
            speaker=SILERO_SPEAKER,
            sample_rate=SILERO_SAMPLE_RATE,
            put_accent=SILERO_PUT_ACCENT,
            put_yo=SILERO_PUT_YO,
        )
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if SILERO_SAMPLE_RATE != MUSETALK_SAMPLE_RATE:
            tensor = torchaudio.functional.resample(
                tensor, SILERO_SAMPLE_RATE, MUSETALK_SAMPLE_RATE
            )
        t_synth = (time.time() - t0) * 1000
        arr = tensor.squeeze(0).cpu().numpy().astype(np.float32)
    print(f"[Silero] synth={t_synth:.0f}ms total={((time.time()-t0)*1000):.0f}ms "
          f"({len(arr)/MUSETALK_SAMPLE_RATE:.2f}s audio, {len(text)}ch)")
    return arr


def _generate_frames_sync(audio_path: str) -> list:
    """Synchronous MuseTalk frame generation. Returns cropped 800x800 frames
    (~1.9 MiB each); fullbody compositing is done lazily per frame at send time
    to keep peak RAM low."""
    with engine_lock:
        return engine.generate_frames(avatar, audio_path)


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_engine()
    print(f"[Server] Starting on http://localhost:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", ws="wsproto")
