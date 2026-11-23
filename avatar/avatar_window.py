"""
Pygame-based display window that composes all animation layers:
  - mouth deformation (driven by audio amplitude)
  - blink animation
  - idle head sway
Runs at ~30 fps.
"""
import time
import threading
import numpy as np
import pygame
import cv2

from .face_processor import FaceData
from .mouth_animator import MouthAnimator
from .idle_animator import IdleAnimator
from .audio_analyzer import AudioAnalyzer


FPS = 30
BG_COLOR = (15, 15, 20)


class AvatarWindow:
    def __init__(
        self,
        face_data: FaceData,
        window_title: str = "Visual AI Assistant",
    ):
        self._face = face_data
        self._mouth = MouthAnimator(face_data)
        self._idle = IdleAnimator(face_data)
        self._audio = AudioAnalyzer()

        h, w = face_data.image.shape[:2]
        self._img_size = (w, h)
        self._win_size = (max(w, 480), max(h, 480))

        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the window in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._audio.stop()
        if self._thread:
            self._thread.join(timeout=3.0)

    def play_audio(self, path: str) -> None:
        """Play an audio file and animate the avatar to it."""
        self._audio.start_file(path)

    def wait_for_audio(self) -> None:
        """Block until current audio finishes."""
        while self._audio.is_playing:
            time.sleep(0.05)

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode(self._win_size)
        pygame.display.set_caption("Visual AI Assistant")
        clock = pygame.time.Clock()

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False

            t = time.time()
            frame = self._compose_frame(t)
            surface = self._bgr_to_surface(frame)

            # Centre the avatar on a dark background
            screen.fill(BG_COLOR)
            iw, ih = self._img_size
            ox = (self._win_size[0] - iw) // 2
            oy = (self._win_size[1] - ih) // 2
            screen.blit(surface, (ox, oy))

            # Status bar
            self._draw_status(screen, t)

            pygame.display.flip()
            clock.tick(FPS)

        pygame.quit()

    def _compose_frame(self, t: float) -> np.ndarray:
        amplitude = self._audio.amplitude

        # 1. Mouth deformation
        frame = self._mouth.render(amplitude)

        # 2. Idle: blink + sway
        blink, M = self._idle.get_transform(t)
        frame = self._idle.apply_blink(frame, blink)
        frame = self._idle.apply_sway(frame, M)

        return frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bgr_to_surface(self, bgr: np.ndarray) -> pygame.Surface:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def _draw_status(self, screen: pygame.Surface, t: float) -> None:
        if not hasattr(self, "_font"):
            self._font = pygame.font.SysFont("monospace", 14)

        if self._audio.is_playing:
            pos = self._audio.position
            dur = self._audio.duration
            pct = int(pos / dur * 100) if dur > 0 else 0
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            text = f" ▶  {bar}  {pos:.1f}s / {dur:.1f}s"
            color = (100, 220, 100)
        else:
            text = "  idle — call play_audio(path) to animate"
            color = (100, 100, 120)

        surf = self._font.render(text, True, color)
        screen.blit(surf, (8, self._win_size[1] - 22))
