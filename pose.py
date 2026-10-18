"""Lightweight physical pose estimation (pitch / yaw / roll) for the held object.

NumPy only (no OpenCV needed), ~1-2 ms per frame on 72x72 pixels:
  * roll  - principal-axis angle of the foreground mask (image moments), 0 = upright, + = clockwise.
  * yaw   - view matching: the live descriptor is compared with the enrolled cardinal centroids and the matches are
            blended on the yaw circle (front 0, right 90, back 180, left 270 = -90), i.e.
            yaw = atan2(w_right - w_left, w_front - w_back) with softmax view weights - unambiguous over 360 degrees.
            Legacy banks with a single "side" view fall back to +/-90 with the last known side. The descriptor is DINOv2 ViT-S/14 (NPU pose worker in receiver.py) when the class
            was enrolled with DINOv2 view centroids, else the MobileNet embedding. For phones the apparent width/height
            ratio (oriented-box foreshortening) refines the magnitude.
  * pitch - top-view weight plus the perspective taper of the mask along its major axis; > 0 = top toward the camera.
The result is an approximation intended to drive the dashboard's 3D model, not a metrology-grade pose.
"""

from __future__ import annotations

import math
import time

import numpy as np
from PIL import Image

ANALYSIS_SIZE = 72
BORDER = 4
MIN_FILL, MAX_FILL = 0.03, 0.92
MIN_CONTRAST = 18.0
VIEW_TEMPERATURE = 0.06
YAW_ANGLES = {"front": 0.0, "right": 90.0, "back": 180.0, "left": -90.0, "side": 90.0}
TOP_VIEW_PITCH = 40.0  # the enrolled "top" view is captured at a 35-45 deg tilt toward the camera


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _otsu(values: np.ndarray, bins: int = 32) -> float:
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(edges[-1])
    centers = (edges[:-1] + edges[1:]) * 0.5
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centers)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (w0 * w1) * ((m0 / w0 - (mt - m0) / w1) ** 2)
    between = np.nan_to_num(between)
    return float(centers[int(np.argmax(between))])


def _morph(mask: np.ndarray, erode: bool) -> np.ndarray:
    p = np.pad(mask, 1, constant_values=erode)
    stack = [p[dy:dy + mask.shape[0], dx:dx + mask.shape[1]] for dy in range(3) for dx in range(3)]
    return np.logical_and.reduce(stack) if erode else np.logical_or.reduce(stack)


def segment(small: np.ndarray) -> np.ndarray | None:
    """Foreground mask of the object against the median colour of the crop border; None if no clear object."""
    ring = np.concatenate([
        small[:BORDER].reshape(-1, 3), small[-BORDER:].reshape(-1, 3),
        small[BORDER:-BORDER, :BORDER].reshape(-1, 3), small[BORDER:-BORDER, -BORDER:].reshape(-1, 3),
    ])
    background = np.median(ring, axis=0)
    dist = np.sqrt(((small - background) ** 2).sum(axis=2))
    if float(np.percentile(dist, 95)) < MIN_CONTRAST:
        return None
    mask = dist > max(_otsu(dist.ravel()), MIN_CONTRAST * 0.6)
    mask = _morph(_morph(mask, erode=True), erode=False)
    fill = float(mask.mean())
    return mask if MIN_FILL <= fill <= MAX_FILL else None


def measure(mask: np.ndarray) -> dict:
    """Principal-axis geometry of a binary mask."""
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy
    mu20, mu02, mu11 = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)  # major-axis angle from +x, image coords (y down)
    spread = math.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0.0))
    lam_major, lam_minor = (mu20 + mu02 + spread) / 2, max((mu20 + mu02 - spread) / 2, 1e-9)
    elongation = 1.0 - math.sqrt(lam_minor / max(lam_major, 1e-9))

    a = math.degrees(theta)
    roll = a - 90.0 if a > 0 else a + 90.0
    ux, uy = math.cos(theta), math.sin(theta)
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux
    lo_a, hi_a = np.percentile(along, [2, 98])
    lo_c, hi_c = np.percentile(across, [2, 98])
    length, width = max(hi_a - lo_a, 1e-6), max(hi_c - lo_c, 1e-6)

    # Perspective taper: the end of the object nearer the camera looks wider. "Top" is the image-up end.
    top_is_low = (uy * -1.0) < 0  # sign of the major-axis direction that points up the image
    sign = -1.0 if top_is_low else 1.0
    pos = along * sign  # larger = nearer the image top
    third = length / 3.0
    top_sel, bot_sel = pos > (np.percentile(pos, 98) - third), pos < (np.percentile(pos, 2) + third)
    top_w = np.ptp(across[top_sel]) if top_sel.sum() > 3 else width
    bot_w = np.ptp(across[bot_sel]) if bot_sel.sum() > 3 else width
    taper = (top_w - bot_w) / max(top_w + bot_w, 1e-6)
    return {
        "roll": roll, "length": float(length), "width": float(width), "aspect": float(width / length),
        "elongation": float(elongation), "taper": float(taper), "fill": float(mask.mean()),
        "cx": float(cx / mask.shape[1]), "cy": float(cy / mask.shape[0]),
    }


class PoseEstimator:
    """Stateful estimator: smooths the pose and learns the object's frontal aspect ratio."""

    def __init__(self, smoothing: float = 0.45) -> None:
        self.alpha = smoothing
        self.reset()

    def reset(self) -> None:
        self._pose: dict | None = None
        self._ref_aspect: float | None = None
        self._label: str | None = None

    @staticmethod
    def _view_yaw(view_sims: dict[str, float], prev_yaw: float) -> tuple[float | None, float, str | None]:
        """Blend view-centroid similarities on the yaw circle. Returns (yaw, top_weight, best_view)."""
        if not view_sims:
            return None, 0.0, None
        top = max(view_sims.values())
        weights = {v: math.exp((s - top) / VIEW_TEMPERATURE) for v, s in view_sims.items()}
        norm = sum(weights.values())
        weights = {v: w / norm for v, w in weights.items()}
        best = max(weights, key=weights.get)
        # a legacy single "side" view cannot tell left from right: keep the last side (right/left banks never need this)
        side_sign = -1.0 if prev_yaw < -10.0 and not ({"right", "left"} & weights.keys()) else 1.0
        x = y = 0.0
        for view, w in weights.items():
            if view in YAW_ANGLES:
                ang = math.radians(YAW_ANGLES[view] * (side_sign if view == "side" else 1.0))
                x += w * math.cos(ang)
                y += w * math.sin(ang)
        yaw = math.degrees(math.atan2(y, x)) if (x or y) else None
        return yaw, weights.get("top", 0.0), best

    def update(self, image: Image.Image, box: tuple[int, int, int], view_sims: dict[str, float] | None,
               label: str | None = None, category: str | None = None) -> dict:
        t0 = time.perf_counter()
        if label != self._label:
            self.reset()
            self._label = label
        size, left, top = box
        small = np.asarray(
            image.crop((left, top, left + size, top + size)).resize((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.BILINEAR),
            dtype=np.float32,
        )
        mask = segment(small)
        prev = self._pose or {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        if mask is None:
            out = {**prev, "valid": False, "view": None, "aspect": None, "fill": 0.0, "yaw360": round(prev["yaw"] % 360.0, 1)}
            out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return out
        geo = measure(mask)

        roll = max(-45.0, min(45.0, geo["roll"])) if geo["elongation"] > 0.12 else 0.0
        yaw_view, top_w, best = self._view_yaw(view_sims or {}, prev["yaw"])
        yaw = yaw_view if yaw_view is not None else prev["yaw"]

        if category == "smartphone" and best in ("front", "back") and geo["elongation"] > 0.12:
            aspect = geo["aspect"]
            self._ref_aspect = aspect if self._ref_aspect is None else max(0.9 * self._ref_aspect + 0.1 * aspect, aspect * 0.9)
        if category == "smartphone" and self._ref_aspect and best != "top":
            asp_yaw = math.degrees(math.acos(max(0.0, min(1.0, geo["aspect"] / self._ref_aspect))))
            fold = min(abs(yaw), 180.0 - abs(yaw)) if yaw_view is not None else 0.0
            fold = 0.5 * fold + 0.5 * asp_yaw
            sign = -1.0 if yaw < 0 else 1.0
            yaw = sign * (fold if abs(yaw) <= 90.0 else 180.0 - fold)

        # Pitch > 0 = the object's top tilted toward the camera. The "top" wizard view is a 35-45 deg tilt, and a
        # top edge that images wider than the bottom edge (perspective taper > 0) is nearer the camera, i.e. tilted
        # toward it - both terms share that sign.
        pitch = TOP_VIEW_PITCH * top_w + max(-45.0, min(45.0, geo["taper"] * 120.0))

        if self._pose is None:
            sm = {"pitch": pitch, "yaw": yaw, "roll": roll}
        else:
            a = self.alpha
            sm = {
                "pitch": (1 - a) * prev["pitch"] + a * pitch,
                "roll": (1 - a) * prev["roll"] + a * roll,
                "yaw": _wrap180(prev["yaw"] + a * _wrap180(yaw - prev["yaw"])),
            }
        self._pose = sm
        return {
            "pitch": round(sm["pitch"], 1), "yaw": round(sm["yaw"], 1), "roll": round(sm["roll"], 1),
            "yaw360": round(sm["yaw"] % 360.0, 1),
            "valid": True, "view": best, "aspect": round(geo["aspect"], 3), "fill": round(geo["fill"], 3),
            "ms": round((time.perf_counter() - t0) * 1000, 2),
        }
