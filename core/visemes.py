"""PCM audio window → viseme weights.

Uses RMS amplitude + 3-band FFT energy ratios + time-based phoneme cycling.
Output keys follow ARKit / Ready Player Me naming (viseme_aa, viseme_PP, ...),
which is rig-agnostic — each rendering branch maps these to its own parameter
space (RPM blendshapes, Live2D Cubism Param*, Rive state-machine inputs).

Replace with a phoneme-level recogniser (wav2vec2-phoneme, Allosaurus, Rhubarb)
when the dancing-mouth artefact becomes noticeable.
"""
import random

import numpy as np

VISEME_OPEN   = ['viseme_aa', 'viseme_O']
VISEME_MID    = ['viseme_E',  'viseme_I',  'viseme_RR']
VISEME_CLOSED = ['viseme_PP', 'viseme_nn', 'viseme_DD']
VISEME_FRIC   = ['viseme_SS', 'viseme_FF', 'viseme_CH']

ALL_VISEMES = (
    ['viseme_sil'] + VISEME_OPEN + VISEME_MID + VISEME_CLOSED + VISEME_FRIC
)


def _band_energy(
    fft_mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float
) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    return float(fft_mag[mask].mean()) if mask.any() else 0.0


def audio_to_visemes(
    chunk: np.ndarray, sr: int, t: float
) -> dict[str, float]:
    """Returns viseme_name → weight (0..1) plus 'amplitude' (also 0..1).
    `t` is absolute time in seconds; drives the 3-5 Hz phoneme cycling so
    consecutive frames don't all pick the same viseme."""
    if len(chunk) == 0:
        return {'viseme_sil': 1.0, 'amplitude': 0.0}

    rms = float(np.sqrt(np.mean(chunk ** 2)))
    amplitude = min(1.0, rms * 12.0)

    if amplitude < 0.04:
        return {'viseme_sil': 1.0, 'amplitude': amplitude}

    fft_mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs   = np.fft.rfftfreq(len(chunk), 1.0 / sr)

    low   = _band_energy(fft_mag, freqs,    0,  500)
    mid   = _band_energy(fft_mag, freqs,  500, 2500)
    high  = _band_energy(fft_mag, freqs, 2500, 8000)
    total = low + mid + high + 1e-8
    low_r, high_r = low / total, high / total

    cycle = (t * 4.0) % 1.0
    if high_r > 0.5:
        v = VISEME_FRIC[int(cycle * len(VISEME_FRIC)) % len(VISEME_FRIC)]
    elif cycle < 0.35:
        v = random.choice(VISEME_OPEN)
    elif cycle < 0.60:
        v = random.choice(VISEME_MID)
    elif cycle < 0.78:
        v = random.choice(VISEME_CLOSED)
    else:
        v = random.choice(VISEME_MID)

    openness = amplitude * (0.6 + low_r * 0.4)
    weights: dict[str, float] = {v: openness}

    if amplitude > 0.6 and v not in VISEME_OPEN:
        weights['viseme_aa'] = (amplitude - 0.6) * 2.0 * openness

    weights['viseme_sil'] = max(0.0, 0.15 - openness * 0.15)
    weights['amplitude'] = amplitude
    return weights
