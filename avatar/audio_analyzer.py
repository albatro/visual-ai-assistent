"""
Real-time audio amplitude tracker.
Works in two modes:
  - file mode: reads a WAV/FLAC file and streams amplitude over time
  - live mode: reads from sounddevice input stream
"""
import threading
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, lfilter


def _butter_lowpass(cutoff: float, fs: float, order: int = 4):
    nyq = fs / 2
    Wn = min(cutoff / nyq, 0.99)
    b, a = butter(order, Wn, btype="low")
    return b, a


class AudioAnalyzer:
    """
    Thread-safe amplitude provider.
    Call `start_file(path)` to begin playback analysis.
    Read `amplitude` property (0.0–1.0) from any thread.
    """

    SMOOTHING = 0.15          # low-pass time constant (seconds)
    CHUNK = 1024              # samples per analysis frame

    def __init__(self):
        self._amplitude = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._playing = False
        self._duration = 0.0
        self._position = 0.0

    @property
    def amplitude(self) -> float:
        with self._lock:
            return self._amplitude

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        with self._lock:
            return self._position

    @property
    def duration(self) -> float:
        return self._duration

    # ------------------------------------------------------------------
    def start_file(self, path: str) -> None:
        """Load and play an audio file, tracking amplitude in real time."""
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._play_file_thread, args=(path,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._playing = False
        with self._lock:
            self._amplitude = 0.0

    # ------------------------------------------------------------------
    def _play_file_thread(self, path: str) -> None:
        try:
            data, samplerate = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            print(f"[AudioAnalyzer] cannot read {path}: {e}")
            return

        # Mix to mono for amplitude analysis
        mono = data.mean(axis=1)
        self._duration = len(mono) / samplerate

        # Low-pass filter to smooth amplitude (~20 Hz envelope)
        b, a = _butter_lowpass(20.0, samplerate)

        # Pre-compute RMS envelope
        chunk = self.CHUNK
        n_chunks = len(mono) // chunk
        rms_envelope = np.zeros(n_chunks, dtype=np.float32)
        for i in range(n_chunks):
            seg = mono[i * chunk:(i + 1) * chunk]
            rms_envelope[i] = float(np.sqrt(np.mean(seg ** 2)))

        # Normalise
        peak = rms_envelope.max()
        if peak > 1e-6:
            rms_envelope = rms_envelope / peak

        self._playing = True
        event = threading.Event()

        def callback(outdata, frames, time_info, status):
            start = callback.pos
            end = start + frames
            chunk_data = data[start:end]
            if len(chunk_data) < frames:
                outdata[:len(chunk_data)] = chunk_data
                outdata[len(chunk_data):] = 0
                raise sd.CallbackStop
            else:
                outdata[:] = chunk_data
            callback.pos = end

            # Update amplitude
            idx = start // chunk
            if idx < len(rms_envelope):
                # Smooth toward current RMS
                current_rms = rms_envelope[idx]
                with self._lock:
                    prev = self._amplitude
                    self._amplitude = prev * 0.6 + current_rms * 0.4
                    self._position = start / samplerate

        callback.pos = 0

        with sd.OutputStream(
            samplerate=samplerate,
            channels=data.shape[1],
            callback=callback,
            finished_callback=event.set,
            dtype="float32",
        ):
            while not event.is_set() and not self._stop_event.is_set():
                time.sleep(0.01)

        with self._lock:
            self._amplitude = 0.0
        self._playing = False
