"""
WebSocket server + HTTP server for the 3D avatar.
- Serves web/ directory over HTTP (port 8766)
- WebSocket on port 8765 sends viseme animation data
- Audio playback with real-time amplitude + spectral analysis → visemes
"""
import asyncio
import threading
import json
import time
import math
import random
import numpy as np
import soundfile as sf
import sounddevice as sd
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import websockets
from websockets.server import serve as ws_serve


# ─────────────────────────────────────────────────────────────────────────────
# Viseme table (ARKit / Ready Player Me names)
# ─────────────────────────────────────────────────────────────────────────────
# Weighted viseme pools used during speech animation cycling
VISEME_OPEN   = ['viseme_aa', 'viseme_O']        # open vowels
VISEME_MID    = ['viseme_E',  'viseme_I',  'viseme_RR']  # mid/front vowels
VISEME_CLOSED = ['viseme_PP', 'viseme_nn', 'viseme_DD']  # closed/consonants
VISEME_FRIC   = ['viseme_SS', 'viseme_FF', 'viseme_CH']  # fricatives

ALL_VISEMES = (
    ['viseme_sil'] + VISEME_OPEN + VISEME_MID + VISEME_CLOSED + VISEME_FRIC
)

CHUNK = 1024
SR_DEFAULT = 22050


# ─────────────────────────────────────────────────────────────────────────────
# Audio analysis → viseme weights
# ─────────────────────────────────────────────────────────────────────────────

def _band_energy(fft_mag: np.ndarray, freqs: np.ndarray,
                 lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    return float(fft_mag[mask].mean()) if mask.any() else 0.0


def audio_to_visemes(chunk: np.ndarray, sr: int,
                     t: float) -> dict[str, float]:
    """
    Returns a dict of viseme_name → weight (0-1).
    Uses amplitude + spectral content + time-based cycling.
    """
    if len(chunk) == 0:
        return {'viseme_sil': 1.0, 'amplitude': 0.0}

    rms = float(np.sqrt(np.mean(chunk ** 2)))
    amplitude = min(1.0, rms * 12.0)   # scale to 0-1

    if amplitude < 0.04:
        return {'viseme_sil': 1.0, 'amplitude': amplitude}

    # Spectral analysis
    fft_mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs   = np.fft.rfftfreq(len(chunk), 1.0 / sr)

    low  = _band_energy(fft_mag, freqs,    0,  500)
    mid  = _band_energy(fft_mag, freqs,  500, 2500)
    high = _band_energy(fft_mag, freqs, 2500, 8000)
    total = low + mid + high + 1e-8

    low_r  = low  / total
    mid_r  = mid  / total
    high_r = high / total

    # Speech cycling at ~3-5 Hz to simulate phoneme changes
    cycle = (t * 4.0) % 1.0

    weights: dict[str, float] = {}

    # Pick dominant viseme based on spectrum + cycle
    if high_r > 0.5:
        # Fricative (s, f, sh)
        v = VISEME_FRIC[int(cycle * len(VISEME_FRIC)) % len(VISEME_FRIC)]
    elif cycle < 0.35:
        v = random.choice(VISEME_OPEN)    # open vowel phase
    elif cycle < 0.60:
        v = random.choice(VISEME_MID)     # mid vowel phase
    elif cycle < 0.78:
        v = random.choice(VISEME_CLOSED)  # consonant phase
    else:
        v = random.choice(VISEME_MID)

    # How open: driven by amplitude + low-frequency energy
    openness = amplitude * (0.6 + low_r * 0.4)

    # Primary viseme
    weights[v] = openness

    # Blend toward open vowel when very loud
    if amplitude > 0.6 and v not in VISEME_OPEN:
        weights['viseme_aa'] = (amplitude - 0.6) * 2.0 * openness

    # Always keep a touch of silence (avoids 100% open look)
    weights['viseme_sil'] = max(0.0, 0.15 - openness * 0.15)

    weights['amplitude'] = amplitude
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Shared state between audio thread and WebSocket
# ─────────────────────────────────────────────────────────────────────────────
class AnimState:
    def __init__(self):
        self._lock = threading.Lock()
        self._weights: dict[str, float] = {'viseme_sil': 1.0, 'amplitude': 0.0}
        self._playing = False

    def set(self, weights: dict[str, float]) -> None:
        with self._lock:
            self._weights = weights

    def get(self) -> dict[str, float]:
        with self._lock:
            return dict(self._weights)

    @property
    def playing(self) -> bool:
        return self._playing

    @playing.setter
    def playing(self, v: bool):
        self._playing = v


anim_state = AnimState()
_play_stop = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Audio playback thread
# ─────────────────────────────────────────────────────────────────────────────

def play_audio_file(path: str) -> None:
    """
    Plays an audio file and continuously updates anim_state with viseme
    weights derived from the audio content.
    """
    try:
        data, sr = sf.read(path, dtype='float32', always_2d=True)
    except Exception as e:
        print(f'[audio] cannot read {path}: {e}')
        return

    mono = data.mean(axis=1)
    n_chunks = len(mono) // CHUNK
    anim_state.playing = True
    _play_stop.clear()

    done_event = threading.Event()

    def callback(outdata, frames, time_info, status):
        start = callback.pos
        end   = start + frames
        seg   = data[start:end]
        if len(seg) < frames:
            outdata[:len(seg)] = seg
            outdata[len(seg):] = 0
            raise sd.CallbackStop
        outdata[:] = seg
        callback.pos = end

        # Analyse the mono chunk for visemes
        mono_chunk = mono[start:end]
        t = start / sr
        weights = audio_to_visemes(mono_chunk, sr, t)
        anim_state.set(weights)

    callback.pos = 0

    with sd.OutputStream(
        samplerate=sr, channels=data.shape[1],
        callback=callback,
        finished_callback=done_event.set,
        dtype='float32',
    ):
        while not done_event.is_set() and not _play_stop.is_set():
            time.sleep(0.01)

    anim_state.set({'viseme_sil': 1.0, 'amplitude': 0.0})
    anim_state.playing = False


def stop_audio() -> None:
    _play_stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket server
# ─────────────────────────────────────────────────────────────────────────────
class _WSBroadcaster:
    def __init__(self):
        self.clients: set = set()

    async def handler(self, websocket) -> None:
        self.clients.add(websocket)
        print(f'[ws] client connected  ({len(self.clients)} total)')
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)
            print(f'[ws] client disconnected ({len(self.clients)} total)')

    async def push_loop(self) -> None:
        while True:
            await asyncio.sleep(1 / 30)
            if not self.clients:
                continue
            weights = anim_state.get()
            amplitude = weights.pop('amplitude', 0.0)
            msg = json.dumps({
                'type':      'visemes',
                'weights':   weights,
                'amplitude': amplitude,
            })
            dead = set()
            for ws in self.clients:
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            self.clients -= dead

    async def run(self) -> None:
        print('[ws] listening on ws://localhost:8765')
        async with ws_serve(self.handler, 'localhost', 8765):
            await self.push_loop()


_broadcaster = _WSBroadcaster()


async def start_ws_server() -> None:
    await _broadcaster.run()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP server (serves web/ directory)
# ─────────────────────────────────────────────────────────────────────────────

def start_http_server(web_dir: str, port: int = 8766) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=web_dir, **kw)

        def log_message(self, *_):
            pass   # suppress access logs

    server = HTTPServer(('localhost', port), Handler)
    print(f'[http] serving {web_dir} at http://localhost:{port}')
    server.serve_forever()


# ─────────────────────────────────────────────────────────────────────────────
# Public API (used by demo_3d.py)
# ─────────────────────────────────────────────────────────────────────────────

def start_servers(web_dir: str) -> None:
    """Start HTTP + WebSocket servers in background threads."""
    t = threading.Thread(
        target=start_http_server, args=(web_dir,), daemon=True
    )
    t.start()
    asyncio.run(start_ws_server())   # blocks (run in its own thread)


def start_servers_background(web_dir: str) -> threading.Thread:
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # HTTP in its own thread
        ht = threading.Thread(
            target=start_http_server, args=(web_dir,), daemon=True
        )
        ht.start()

        loop.run_until_complete(start_ws_server())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
