"""
Pre-generate filler audio (TTS) for all phrases in fillers/texts.json.

For each entry:
  - edge-tts stream -> mp3 saved to fillers/audio/<id>.mp3
  - ffmpeg converts to wav 16kHz mono -> fillers/audio/<id>.wav
  - measure duration via soundfile
Writes fillers/manifest.json with metadata.

Idempotent: skips entries whose .mp3 + .wav already exist.

Run: python generate_fillers.py
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import edge_tts
import soundfile as sf

# Must match streaming_server.py voice/rate so live voice ≈ filler voice.
TTS_VOICE = "ru-RU-DmitryNeural"
TTS_RATE = "-5%"

FILLERS_DIR = Path(__file__).parent / "fillers"
AUDIO_DIR = FILLERS_DIR / "audio"
TEXTS_PATH = FILLERS_DIR / "texts.json"
MANIFEST_PATH = FILLERS_DIR / "manifest.json"


async def synth_one(entry: dict, force: bool) -> dict:
    fid = entry["id"]
    text = entry["text"]
    mp3_path = AUDIO_DIR / f"{fid}.mp3"
    wav_path = AUDIO_DIR / f"{fid}.wav"

    if not force and mp3_path.exists() and wav_path.exists():
        # Already generated — just read duration
        data, sr = sf.read(str(wav_path))
        duration_ms = round(len(data) / sr * 1000)
        print(f"  [skip] {fid}: {text!r} ({duration_ms} ms)")
    else:
        # Edge-TTS streaming -> mp3 bytes
        chunks: list[bytes] = []
        comm = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE)
        async for ch in comm.stream():
            if ch["type"] == "audio":
                chunks.append(ch["data"])
        mp3_bytes = b"".join(chunks)
        mp3_path.write_bytes(mp3_bytes)

        # ffmpeg -> wav 16k mono
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(mp3_path),
             "-ar", "16000", "-ac", "1",
             str(wav_path)],
            check=True,
        )

        data, sr = sf.read(str(wav_path))
        duration_ms = round(len(data) / sr * 1000)
        print(f"  [done] {fid}: {text!r} -> {len(mp3_bytes)}b mp3, {duration_ms} ms")

    return {
        "id": fid,
        "text": text,
        "category": entry.get("category"),
        "mp3": f"audio/{fid}.mp3",
        "wav": f"audio/{fid}.wav",
        "duration_ms": duration_ms,
    }


async def main():
    force = "--force" in sys.argv
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEXTS_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    print(f"[Fillers] Synthesising {len(entries)} phrases (voice={TTS_VOICE}, rate={TTS_RATE})")

    manifest = []
    # Synthesise sequentially to avoid edge-tts rate limiting
    for entry in entries:
        try:
            manifest.append(await synth_one(entry, force))
        except Exception as e:
            print(f"  [FAIL] {entry['id']}: {e}", file=sys.stderr)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_ms = sum(m["duration_ms"] for m in manifest)
    print(f"[Fillers] Wrote {len(manifest)} entries to manifest.json "
          f"(total audio: {total_ms / 1000:.1f}s, avg {total_ms / len(manifest):.0f} ms/clip)")


if __name__ == "__main__":
    asyncio.run(main())
