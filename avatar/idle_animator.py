"""
Generates idle animations: blinking and subtle head sway.
All animations are time-based so they look natural without audio.
"""
import math
import random
import time
import numpy as np
import cv2
from .face_processor import FaceData


class IdleAnimator:
    def __init__(self, face_data: FaceData):
        self.face = face_data
        self._h, self._w = face_data.image.shape[:2]

        # Blink state
        self._next_blink = time.time() + random.uniform(2.0, 5.0)
        self._blink_phase = 0.0      # 0 = open, 1 = closed
        self._blinking = False
        self._blink_start = 0.0
        self._BLINK_DURATION = 0.12  # seconds

        # Sway state — very subtle
        self._sway_freq = random.uniform(0.08, 0.14)   # Hz
        self._sway_amp = random.uniform(1.5, 3.0)       # pixels
        self._sway_phase = random.uniform(0, 2 * math.pi)

    # ------------------------------------------------------------------
    def get_transform(self, t: float) -> tuple[float, np.ndarray]:
        """
        Returns (blink_amount, affine_matrix) where:
          blink_amount: 0.0 (eyes open) → 1.0 (eyes closed)
          affine_matrix: 2x3 float32, identity + sway offset
        """
        blink = self._update_blink(t)
        M = self._sway_matrix(t)
        return blink, M

    # ------------------------------------------------------------------
    def _update_blink(self, t: float) -> float:
        if not self._blinking and t >= self._next_blink:
            self._blinking = True
            self._blink_start = t
            self._next_blink = t + random.uniform(2.5, 6.0)

        if self._blinking:
            elapsed = t - self._blink_start
            half = self._BLINK_DURATION / 2
            if elapsed < half:
                phase = elapsed / half           # 0 → 1
            elif elapsed < self._BLINK_DURATION:
                phase = (self._BLINK_DURATION - elapsed) / half  # 1 → 0
            else:
                self._blinking = False
                phase = 0.0
            return float(np.clip(phase, 0.0, 1.0))

        return 0.0

    def _sway_matrix(self, t: float) -> np.ndarray:
        dx = self._sway_amp * math.sin(2 * math.pi * self._sway_freq * t + self._sway_phase)
        dy = self._sway_amp * 0.4 * math.sin(2 * math.pi * self._sway_freq * t * 0.7 + 1.2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return M

    # ------------------------------------------------------------------
    def apply_blink(self, img: np.ndarray, blink: float) -> np.ndarray:
        """Draws closed eyelids over the eyes proportional to blink [0..1]."""
        if blink < 0.05:
            return img
        result = img.copy()
        self._close_eye(result, self.face.left_eye_center, self.face.left_eye_height, blink)
        self._close_eye(result, self.face.right_eye_center, self.face.right_eye_height, blink)
        return result

    def _close_eye(
        self,
        img: np.ndarray,
        center: np.ndarray,
        eye_h: float,
        blink: float,
    ) -> None:
        cx, cy = int(center[0]), int(center[1])
        ew = int(eye_h * 3.5)
        eh = int(eye_h * blink * 1.2)
        if ew <= 0 or eh <= 0:
            return
        # Sample skin color from just above the eye
        sy = max(0, cy - int(eye_h * 1.5))
        skin_region = img[max(0, sy - 4):sy + 4, max(0, cx - 8):cx + 8]
        if skin_region.size > 0:
            skin_color = tuple(int(v) for v in skin_region.mean(axis=(0, 1)))
        else:
            skin_color = (200, 180, 160)

        cv2.ellipse(
            img,
            (cx, cy),
            (max(1, ew // 2), max(1, eh)),
            0, 0, 360,
            skin_color,
            -1,
        )

    def apply_sway(self, img: np.ndarray, M: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
