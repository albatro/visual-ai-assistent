"""
Mouth animation via smooth pixel displacement:
  - Lip-shaped mask (polygon from actual landmarks, not rectangle)
  - cv2.remap: pixels spread outward from center line with cosine falloff
  - Cavity: dark ellipse + teeth strip, shaped to actual mouth corners
  - All blending is soft (no hard edges)
"""
import cv2
import numpy as np
from .face_processor import FaceData

# Outer lip contour indices (MediaPipe 478-point model)
_OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
               375, 321, 405, 314, 17, 84, 181, 91, 146]


class MouthAnimator:
    MAX_SHIFT_RATIO = 0.30   # fraction of mouth width → max open distance

    def __init__(self, face_data: FaceData):
        lm = face_data.landmarks
        h, w = face_data.image.shape[:2]
        self._h, self._w = h, w
        self._base = face_data.image.copy()

        mw = float(np.linalg.norm(face_data.mouth_right - face_data.mouth_left))
        self._max_shift = max(8, int(mw * self.MAX_SHIFT_RATIO))
        self._mouth_width = mw
        self._mid_y = float((face_data.mouth_top[1] + face_data.mouth_bottom[1]) / 2)
        self._cx = float((face_data.mouth_left[0] + face_data.mouth_right[0]) / 2)
        self._half_w = max(1.0, mw / 2.0)

        # Bounding box (tight) for cavity drawing
        xs = lm[_OUTER_LIP, 0]
        ys = lm[_OUTER_LIP, 1]
        self._bx0 = max(0, int(xs.min()) - 2)
        self._bx1 = min(w, int(xs.max()) + 2)
        self._by0 = max(0, int(ys.min()) - 2)
        self._by1 = min(h, int(ys.max()) + 2)

        # Pre-compute pixel coordinate grids (float32, full image)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self._xx = xx
        self._yy = yy

        # Lip-shaped soft mask  (float32 0-1, full image size)
        self._lip_mask = self._build_lip_mask(lm, h, w)

        # Displacement weight: strongest at center of mouth, cosine falloff to edges
        self._weight = self._build_weight_map(xx, yy)

        # Pre-build cavity image (updated per frame in _draw_cavity)
        self._cavity_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    def render(self, openness: float) -> np.ndarray:
        openness = float(np.clip(openness, 0.0, 1.0))
        if openness < 0.02:
            return self._base

        shift = self._max_shift * openness

        # 1. Cavity layer (dark interior + teeth)
        cavity_img = self._get_cavity(int(round(shift)))

        # 2. Displacement map: pixels above mid_y go up, below go down
        sign = np.where(self._yy < self._mid_y, -1.0, 1.0).astype(np.float32)
        src_y = np.clip(self._yy - sign * shift * self._weight, 0, self._h - 1).astype(np.float32)

        # 3. Remap base image
        warped = cv2.remap(self._base, self._xx, src_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)

        # 4. Composite: cavity → warped lips → original (soft blend at edges)
        m = self._lip_mask[:, :, np.newaxis]           # (H,W,1)

        # Weight the warp by openness so it's subtle when barely open
        warp_alpha = np.clip(m * (openness * 3.0), 0.0, 1.0)
        lip_layer = warped * warp_alpha + self._base * (1.0 - warp_alpha)

        # Cavity shows only in the inner mouth (stronger mask, shrunk)
        cav_alpha = np.clip(m * (openness * 5.0), 0.0, 1.0)
        result = cavity_img * cav_alpha + lip_layer * (1.0 - cav_alpha)

        return result.astype(np.uint8)

    # ------------------------------------------------------------------
    def _build_lip_mask(self, lm: np.ndarray, h: int, w: int) -> np.ndarray:
        outer = lm[_OUTER_LIP].astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [outer], 255)
        # Dilate slightly so displacement covers the whole lip body
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, k)
        # Feather edges
        blur_r = max(3, int(self._mouth_width * 0.12))
        if blur_r % 2 == 0:
            blur_r += 1
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (blur_r, blur_r), blur_r / 3)
        return (mask_f / 255.0).astype(np.float32)

    def _build_weight_map(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        # Horizontal: cosine from mouth center (0 at corners, 1 at center)
        h_norm = np.clip((xx - self._cx) / self._half_w, -1.0, 1.0)
        h_w = np.cos(h_norm * (np.pi / 2)).astype(np.float32)

        # Vertical: linear from 1 at mid_y to 0 at lip boundary
        lip_h = max(1.0, (self._by1 - self._by0) / 2.0)
        v_norm = np.abs(yy - self._mid_y) / lip_h
        v_w = np.clip(1.0 - v_norm, 0.0, 1.0).astype(np.float32)

        weight = h_w * v_w * self._lip_mask
        return weight

    def _get_cavity(self, shift_px: int) -> np.ndarray:
        """Returns the base image with a mouth cavity drawn on it (cached by shift)."""
        if shift_px in self._cavity_cache:
            return self._cavity_cache[shift_px]

        img = self._base.copy().astype(np.float32)
        cx = int(self._cx)
        mid = int(self._mid_y)

        cav_w = max(4, int(self._mouth_width * 0.82) // 2)
        cav_h = max(2, shift_px)

        # Dark mouth interior
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, mid), (cav_w, cav_h), 0, 0, 360, (28, 18, 18), -1)

        # Slight shadow gradient — blend 80% to soften hard edge
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, overlay)
        img = overlay

        # Teeth (upper arc, narrow)
        if shift_px >= 5:
            teeth_h = max(1, int(cav_h * 0.38))
            teeth_w = max(2, int(cav_w * 0.78))
            t_center_y = mid - max(1, int(cav_h * 0.25))
            cv2.ellipse(img, (cx, t_center_y), (teeth_w, teeth_h),
                        0, 180, 360, (232, 228, 222), -1)

            # Lower teeth (smaller, darker)
            bt_h = max(1, int(cav_h * 0.22))
            bt_w = max(2, int(cav_w * 0.60))
            bt_center_y = mid + max(1, int(cav_h * 0.25))
            cv2.ellipse(img, (cx, bt_center_y), (bt_w, bt_h),
                        0, 0, 180, (200, 196, 190), -1)

        # Cache max 20 different sizes
        if len(self._cavity_cache) < 20:
            self._cavity_cache[shift_px] = img.astype(np.float32)

        return img.astype(np.float32)
