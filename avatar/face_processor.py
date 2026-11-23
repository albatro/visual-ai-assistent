"""
Detects face landmarks from a photo and prepares avatar layers
for real-time mouth animation.
Uses MediaPipe Tasks API (mediapipe >= 0.10).
"""
import os
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# Path to the bundled model file — placed next to this package
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")


# MediaPipe face mesh landmark indices for mouth region
UPPER_LIP_TOP = [13, 312, 311, 310, 415, 308]      # inner upper lip top
LOWER_LIP_BOTTOM = [14, 317, 402, 318, 324, 308]    # inner lower lip bottom
MOUTH_CORNERS = [61, 291]                             # left/right corners
MOUTH_OUTLINE = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146, 61
]
INNER_MOUTH = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 78
]

# Indices for blinking
LEFT_EYE_TOP = [159, 160, 161]
LEFT_EYE_BOTTOM = [145, 144, 153]
RIGHT_EYE_TOP = [386, 387, 388]
RIGHT_EYE_BOTTOM = [374, 373, 380]
LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


@dataclass
class FaceData:
    image: np.ndarray                  # original BGR image
    landmarks: np.ndarray              # (468, 2) pixel coords
    mouth_center: np.ndarray           # (x, y)
    mouth_top: np.ndarray              # (x, y) — top of inner lip
    mouth_bottom: np.ndarray           # (x, y) — bottom of inner lip
    mouth_left: np.ndarray             # (x, y)
    mouth_right: np.ndarray            # (x, y)
    mouth_height_neutral: float        # pixels between top/bottom at rest
    left_eye_center: np.ndarray
    right_eye_center: np.ndarray
    left_eye_height: float
    right_eye_height: float
    image_rgb: np.ndarray              # RGB for pygame


class FaceProcessor:
    def __init__(self, model_path: str | None = None):
        path = os.path.abspath(model_path or _MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Face landmarker model not found at: {path}\n"
                "Download it with:\n"
                "  curl -L -o face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )
        options = FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=path),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._detector = FaceLandmarker.create_from_options(options)

    def process(self, image_path: str) -> Optional[FaceData]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        # Resize to a comfortable display size, keep aspect ratio
        h, w = img.shape[:2]
        max_dim = 720
        scale = min(max_dim / w, max_dim / h)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        h, w = img.shape[:2]
        # Tasks API needs an mp.Image in RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            raise RuntimeError("No face detected in image.")

        lm = result.face_landmarks[0]   # list of NormalizedLandmark
        # Convert normalised coords → pixel coords (Tasks API returns 478 pts: 468 + 10 iris)
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        mouth_top = pts[[13, 312, 311]].mean(axis=0)
        mouth_bottom = pts[[14, 317, 402]].mean(axis=0)
        mouth_left = pts[61]
        mouth_right = pts[291]
        mouth_center = (mouth_top + mouth_bottom) / 2
        mouth_height_neutral = float(np.linalg.norm(mouth_bottom - mouth_top))

        left_eye_top = pts[LEFT_EYE_TOP].mean(axis=0)
        left_eye_bottom = pts[LEFT_EYE_BOTTOM].mean(axis=0)
        right_eye_top = pts[RIGHT_EYE_TOP].mean(axis=0)
        right_eye_bottom = pts[RIGHT_EYE_BOTTOM].mean(axis=0)
        left_eye_center = (left_eye_top + left_eye_bottom) / 2
        right_eye_center = (right_eye_top + right_eye_bottom) / 2

        return FaceData(
            image=img,
            landmarks=pts,
            mouth_center=mouth_center,
            mouth_top=mouth_top,
            mouth_bottom=mouth_bottom,
            mouth_left=mouth_left,
            mouth_right=mouth_right,
            mouth_height_neutral=mouth_height_neutral,
            left_eye_center=left_eye_center,
            right_eye_center=right_eye_center,
            left_eye_height=float(np.linalg.norm(left_eye_bottom - left_eye_top)),
            right_eye_height=float(np.linalg.norm(right_eye_bottom - right_eye_top)),
            image_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        )
