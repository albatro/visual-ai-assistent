"""
Alternative preprocessing using MediaPipe FaceLandmarker (Tasks API) instead of mmpose/DWPose.
This avoids the need for mmcv compiled C++ extensions on Windows.
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

# Initialize MediaPipe FaceLandmarker (new Tasks API)
_BASE_OPTIONS = mp.tasks.BaseOptions
_FACE_LANDMARKER = mp.tasks.vision.FaceLandmarker
_FACE_LANDMARKER_OPTIONS = mp.tasks.vision.FaceLandmarkerOptions
_RUNNING_MODE = mp.tasks.vision.RunningMode

# Find the face_landmarker model
_MODEL_PATH = None
_POSSIBLE_PATHS = [
    Path(__file__).parent.parent.parent.parent / "face_landmarker.task",  # project root
    Path(__file__).parent.parent.parent / "models" / "face_landmarker.task",
]
for p in _POSSIBLE_PATHS:
    if p.exists():
        _MODEL_PATH = str(p)
        break

if _MODEL_PATH is None:
    # Try to download it
    import requests
    _MODEL_PATH = str(_POSSIBLE_PATHS[0])
    print(f"[preprocessing_mp] Downloading face_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    r = requests.get(url)
    _POSSIBLE_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    with open(_MODEL_PATH, "wb") as f:
        f.write(r.content)
    print(f"[preprocessing_mp] Downloaded to {_MODEL_PATH}")

_options = _FACE_LANDMARKER_OPTIONS(
    base_options=_BASE_OPTIONS(model_asset_path=_MODEL_PATH),
    running_mode=_RUNNING_MODE.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
_landmarker = _FACE_LANDMARKER.create_from_options(_options)

# Placeholder for no-face frames
coord_placeholder = (0.0, 0.0, 0.0, 0.0)

# MediaPipe face oval indices
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Nose bridge indices for vertical split
NOSE_BRIDGE_INDICES = [6, 197, 195, 5, 4, 1, 19]


def read_imgs(img_list):
    """Read images from file paths."""
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _get_face_landmarks_mp(image_bgr):
    """
    Extract face landmarks using MediaPipe FaceLandmarker (Tasks API).
    Returns landmarks as (N, 2) numpy array in pixel coordinates, or None.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = _landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    h, w = image_bgr.shape[:2]
    face_lm = result.face_landmarks[0]

    landmarks = np.array(
        [(lm.x * w, lm.y * h) for lm in face_lm],
        dtype=np.float32,
    )
    return landmarks


def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    Extract face bounding boxes from images using MediaPipe.
    Drop-in replacement for the mmpose-based version.
    Returns (coords_list, frames).
    """
    frames = read_imgs(img_list)
    coords_list = []

    if upperbondrange != 0:
        print(f'get face bounding boxes with bbox_shift: {upperbondrange}')
    else:
        print('get face bounding boxes with default value')

    for frame in tqdm(frames, desc="Detecting faces"):
        landmarks = _get_face_landmarks_mp(frame)

        if landmarks is None:
            coords_list.append(coord_placeholder)
            continue

        h, w = frame.shape[:2]

        # Get face oval landmarks for bounding box
        face_oval = landmarks[FACE_OVAL_INDICES]

        # Get nose bridge for vertical split
        nose_bridge = landmarks[NOSE_BRIDGE_INDICES]

        # half_face_y: nose tip area (similar to mmpose keypoint 29)
        half_face_y = nose_bridge[3][1]

        if upperbondrange != 0:
            half_face_y += upperbondrange

        x_min = int(np.min(face_oval[:, 0]))
        x_max = int(np.max(face_oval[:, 0]))
        y_max = int(np.max(face_oval[:, 1]))

        half_face_dist = y_max - half_face_y
        upper_bond = max(0, int(half_face_y - half_face_dist))

        f_landmark = (x_min, upper_bond, x_max, y_max)
        x1, y1, x2, y2 = f_landmark

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            x1 = max(0, int(np.min(face_oval[:, 0])))
            y1 = max(0, int(np.min(face_oval[:, 1])))
            x2 = min(w, int(np.max(face_oval[:, 0])))
            y2 = min(h, int(np.max(face_oval[:, 1])))
            coords_list.append((x1, y1, x2, y2))
        else:
            coords_list.append(f_landmark)

    return coords_list, frames
