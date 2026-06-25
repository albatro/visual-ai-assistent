"""WAV chunking — split a long WAV into ~N-second files for incremental rendering.

Used by branches that need to start streaming frames/visemes before the whole
TTS output is ready (MuseTalk per-chunk, Live2D per-segment).
"""
import tempfile

import soundfile as sf


def split_wav(
    wav_path: str, chunk_seconds: float, *, work_dir: str = "."
) -> list[str]:
    """Split wav_path into N temp WAV files of ~chunk_seconds each.
    Returns the list of paths; caller is responsible for unlinking them."""
    data, sr = sf.read(wav_path, dtype="float32")
    chunk_samples = int(chunk_seconds * sr)
    chunk_paths: list[str] = []
    offset = 0
    idx = 0

    while offset < len(data):
        end = min(offset + chunk_samples, len(data))
        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_chunk{idx}.wav", delete=False, dir=work_dir,
        )
        tmp.close()
        sf.write(tmp.name, data[offset:end], sr)
        chunk_paths.append(tmp.name)
        offset = end
        idx += 1

    return chunk_paths
