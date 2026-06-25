"""TTS adapters: edge-tts (cloud Microsoft) and Silero v4_ru (local PyTorch).

Both engines produce a 16k mono WAV on disk (suitable for MuseTalk and viseme
analysis) plus the raw audio bytes the browser plays directly:
  * edge   — MP3 bytes (browser-friendly, smaller)
  * silero — WAV bytes (no extra encode pass)

Silero is lazy-loaded on first use; call init_silero() at startup to pay the
~3-5 s load+warmup cost up front instead of in the first user request.
"""
import asyncio
import io
import os
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000  # MuseTalk + viseme analysis both consume 16k mono

# Edge defaults (override via env)
EDGE_VOICE = os.environ.get("EDGE_VOICE", "ru-RU-DmitryNeural")
EDGE_RATE = os.environ.get("EDGE_RATE", "-5%")

# Silero defaults (override via env)
SILERO_SPEAKER = os.environ.get("SILERO_SPEAKER", "aidar")
SILERO_NATIVE_SR = 24000
SILERO_PUT_ACCENT = os.environ.get("SILERO_PUT_ACCENT", "1") != "0"
SILERO_PUT_YO = os.environ.get("SILERO_PUT_YO", "1") != "0"

_silero_model = None
_silero_lock = threading.Lock()


def init_silero() -> None:
    """Idempotent. Load Silero v4_ru on CUDA if available, otherwise CPU,
    and warm it up 3x to JIT GPU kernels and accent paths. Skipping the warmup
    leaves the first ~2 real requests at 1-3 s each."""
    global _silero_model
    import torch

    with _silero_lock:
        if _silero_model is not None:
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

        warmup_texts = [
            "Готов к работе.",
            "Я построил Петербург на болоте, чтобы открыть России окно в Европу.",
            "Расскажу тебе кратко, в двух предложениях, о моих делах.",
        ]
        t0 = time.time()
        with torch.inference_mode():
            for wt in warmup_texts:
                _ = model.apply_tts(
                    text=wt,
                    speaker=SILERO_SPEAKER,
                    sample_rate=SILERO_NATIVE_SR,
                    put_accent=SILERO_PUT_ACCENT,
                    put_yo=SILERO_PUT_YO,
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        _silero_model = model
        print(
            f"[Silero] Ready on {device} "
            f"(warmup x{len(warmup_texts)} in {(time.time()-t0)*1000:.0f}ms, "
            f"accent={SILERO_PUT_ACCENT}, yo={SILERO_PUT_YO})"
        )


async def synth(text: str, engine: str) -> tuple[str, bytes]:
    """Synthesise speech. Returns (wav_path_16k_mono, client_audio_bytes).
    client_audio_bytes is MP3 for edge, WAV for silero — both browser-playable."""
    eng = engine.lower()
    if eng == "silero":
        return await _synth_silero(text)
    if eng == "edge":
        return await _synth_edge(text)
    raise ValueError(f"Unknown TTS engine: {engine!r} (expected 'edge' or 'silero')")


async def _synth_edge(text: str) -> tuple[str, bytes]:
    """Stream MP3 chunks from edge-tts into ffmpeg in parallel; by the time the
    last chunk arrives, the 16k WAV is nearly done."""
    import edge_tts

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=".")
    wav_tmp.close()

    ff = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "mp3", "-i", "pipe:0",
        "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
        wav_tmp.name,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    mp3_chunks: list[bytes] = []
    communicate = edge_tts.Communicate(text, voice=EDGE_VOICE, rate=EDGE_RATE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data = chunk["data"]
            mp3_chunks.append(data)
            ff.stdin.write(data)
    ff.stdin.close()
    await ff.wait()

    return wav_tmp.name, b"".join(mp3_chunks)


async def _synth_silero(text: str) -> tuple[str, bytes]:
    """Silero v4_ru local TTS. Synthesises at 24k, resamples to 16k.
    One WAV-encode pass: same bytes go to client and to disk."""
    if _silero_model is None:
        init_silero()

    loop = asyncio.get_event_loop()
    audio_16k = await loop.run_in_executor(None, _silero_synth_sync, text)

    t_enc = time.time()
    buf = io.BytesIO()
    sf.write(buf, audio_16k, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=".")
    wav_tmp.write(wav_bytes)
    wav_tmp.close()
    print(
        f"[Silero] encode+write {len(wav_bytes)//1024}KB "
        f"in {(time.time()-t_enc)*1000:.0f}ms"
    )
    return wav_tmp.name, wav_bytes


def _silero_synth_sync(text: str) -> np.ndarray:
    """Blocking Silero call. Returns float32 mono numpy at TARGET_SAMPLE_RATE."""
    import torch
    import torchaudio

    t0 = time.time()
    with _silero_lock, torch.inference_mode():
        tensor = _silero_model.apply_tts(
            text=text,
            speaker=SILERO_SPEAKER,
            sample_rate=SILERO_NATIVE_SR,
            put_accent=SILERO_PUT_ACCENT,
            put_yo=SILERO_PUT_YO,
        )
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if SILERO_NATIVE_SR != TARGET_SAMPLE_RATE:
            tensor = torchaudio.functional.resample(
                tensor, SILERO_NATIVE_SR, TARGET_SAMPLE_RATE
            )
        arr = tensor.squeeze(0).cpu().numpy().astype(np.float32)
    print(
        f"[Silero] synth {len(arr)/TARGET_SAMPLE_RATE:.2f}s audio "
        f"in {(time.time()-t0)*1000:.0f}ms ({len(text)}ch)"
    )
    return arr


def stack_desc(default_engine: str) -> str:
    """One-line human-readable summary of the TTS stack — used in WS handshakes."""
    state = "loaded" if _silero_model is not None else "lazy"
    return (
        f"default={default_engine} · "
        f"edge={EDGE_VOICE}@{EDGE_RATE} · "
        f"silero={SILERO_SPEAKER} ({state})"
    )
