"""
Generates a simple test WAV file with varying amplitude
so you can test the avatar without real speech.
"""
import numpy as np
import soundfile as sf

SR = 22050
duration = 8.0
t = np.linspace(0, duration, int(SR * duration))

# Simulate speech-like bursts
envelope = np.zeros_like(t)
bursts = [(0.5, 0.8), (1.2, 1.6), (2.0, 2.5), (3.0, 3.2),
          (3.8, 4.4), (5.0, 5.8), (6.2, 6.6), (7.0, 7.5)]
for start, end in bursts:
    mask = (t >= start) & (t <= end)
    rise = np.linspace(0, 1, mask.sum() // 3 + 1)[:mask.sum() // 3]
    hold_len = mask.sum() - 2 * (mask.sum() // 3)
    fall = np.linspace(1, 0, mask.sum() // 3 + 1)[:mask.sum() // 3]
    env_seg = np.concatenate([rise, np.ones(max(0, hold_len)), fall])
    env_seg = env_seg[:mask.sum()]
    envelope[mask] = env_seg

# Carrier: mix of harmonics to mimic voice
voice = (
    0.5 * np.sin(2 * np.pi * 180 * t) +
    0.3 * np.sin(2 * np.pi * 360 * t) +
    0.15 * np.sin(2 * np.pi * 720 * t) +
    0.05 * np.random.randn(len(t))
)
audio = (voice * envelope * 0.7).astype(np.float32)

sf.write("test_speech.wav", audio, SR)
print("Generated test_speech.wav")
