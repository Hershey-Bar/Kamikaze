"""Live 360-degree can label capture: stitch the physical can's real packaging from camera crops, RAM only.

While the operator turns the can in front of the camera (amber LED guidance), every tight can crop from the detector is
fed to LabelStitcher.feed(). The central band of the crop (|theta| <= BAND_DEG, where foreshortening and glare are
lowest) is resampled from image x = r sin(theta) to arc length s = r theta, so a rotation of the can becomes a pure
horizontal translation. Consecutive bands are registered with phase correlation (high-passed, Hanning-windowed) and
accumulated into a periodic buffer one circumference (2 pi r) wide; the capture is complete when every column of the
label has been seen. Nothing is written to disk: texture() returns the label as an array and the caller decides.
"""

from __future__ import annotations

import math
import threading
import time

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

BAND_DEG = 30.0
MIN_RESPONSE = 0.08
MAX_DY_PX = 4.0
COMPLETE_AT = 0.985
HEIGHT_PX = 384
TEX_W = 2048
SESSION_TTL_S = 180.0


class StitchError(RuntimeError):
    pass


class LabelStitcher:
    def __init__(self, label: str = "", height: int = HEIGHT_PX, band_deg: float = BAND_DEG) -> None:
        self.label = label
        self.h = int(height)
        self.band = math.radians(band_deg)
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.updated = self.started
        self.w_ref = 0
        self.r = 0.0
        self.circ = 0
        self.acc: np.ndarray | None = None
        self.wsum: np.ndarray | None = None
        self.prev: np.ndarray | None = None
        self.prev_hp: np.ndarray | None = None
        self.pos = 0.0  # label coordinate (px of arc) of the current band's centre
        self.travel = 0.0
        self.frames = 0
        self.accepted = 0
        self.rejected = 0
        self.window: np.ndarray | None = None
        self.last_reason = ""

    # -- geometry -------------------------------------------------------------------------------------------------
    def _setup(self, w: int) -> None:
        self.w_ref = w
        self.r = w / 2.0
        self.circ = int(round(2 * math.pi * self.r))
        self.acc = np.zeros((self.h, self.circ, 3), np.float64)
        self.wsum = np.zeros((self.h, self.circ), np.float64)
        n = int(round(2 * self.band * self.r))
        self.s = (np.arange(n) - (n - 1) / 2.0)  # arc-length sample positions, px
        self.map_x = (self.r + self.r * np.sin(self.s / self.r) - 0.5).astype(np.float32)
        self.taper = np.cos(np.linspace(-math.pi / 2, math.pi / 2, n)) ** 2 + 1e-3

    def _unwrap(self, crop: np.ndarray) -> np.ndarray:
        mx = np.tile(self.map_x, (self.h, 1))
        my = np.tile(np.arange(self.h, dtype=np.float32)[:, None], (1, len(self.map_x)))
        return cv2.remap(crop, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _highpass(band: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return g - cv2.GaussianBlur(g, (0, 0), 6)

    def _deposit(self, band: np.ndarray, centre: float) -> None:
        cols = (np.round(centre + self.s).astype(np.int64)) % self.circ
        np.add.at(self.acc, (slice(None), cols), band.astype(np.float64) * self.taper[None, :, None])
        np.add.at(self.wsum, (slice(None), cols), np.broadcast_to(self.taper, (self.h, len(cols))))

    # -- public ---------------------------------------------------------------------------------------------------
    def feed(self, crop_bgr: np.ndarray) -> dict:
        """Add one tight can crop (BGR). Returns status()."""
        if crop_bgr is None or crop_bgr.ndim != 3 or min(crop_bgr.shape[:2]) < 16:
            raise StitchError("empty crop")
        with self.lock:
            self.frames += 1
            self.updated = time.monotonic()
            ch, cw = crop_bgr.shape[:2]
            w = max(16, int(round(cw * self.h / ch)))
            if not self.w_ref:
                self._setup(w)
            elif abs(w - self.w_ref) > 0.15 * self.w_ref:
                self.rejected += 1
                self.last_reason = f"crop width {w}px vs {self.w_ref}px: can moved closer / farther or box jitter"
                return self._status()
            crop = cv2.resize(crop_bgr, (self.w_ref, self.h), interpolation=cv2.INTER_AREA)
            band = self._unwrap(crop)
            hp = self._highpass(band)
            if self.window is None or self.window.shape != hp.shape:
                self.window = cv2.createHanningWindow((hp.shape[1], hp.shape[0]), cv2.CV_32F)
            if self.prev_hp is None:
                self._deposit(band, self.pos)
                self.prev_hp = hp
                self.accepted += 1
                return self._status()
            (dx, dy), resp = cv2.phaseCorrelate(self.prev_hp, hp, self.window)
            if resp < MIN_RESPONSE or abs(dy) > MAX_DY_PX or abs(dx) > 0.45 * len(self.s):
                self.rejected += 1
                self.last_reason = f"registration failed (response {resp:.2f}, dx {dx:.1f}, dy {dy:.1f})"
                return self._status()
            # content moved by dx in the image -> the band centre now shows label coordinate pos - dx
            self.pos -= dx
            self.travel += abs(dx)
            self._deposit(band, self.pos)
            self.prev_hp = hp
            self.accepted += 1
            self.last_reason = ""
            return self._status()

    def coverage(self) -> float:
        if self.wsum is None:
            return 0.0
        return float((self.wsum.max(axis=0) > 0.05).mean())

    def _status(self) -> dict:
        cov = self.coverage()
        return {"label": self.label, "frames": self.frames, "accepted": self.accepted, "rejected": self.rejected,
                "coverage": round(cov, 3), "rotation_deg": round(math.degrees(self.travel / self.r), 1) if self.r else 0.0,
                "complete": cov >= COMPLETE_AT, "circumference_px": self.circ, "reason": self.last_reason,
                "age_s": round(time.monotonic() - self.started, 1)}

    def status(self) -> dict:
        with self.lock:
            return self._status()

    def texture(self, width: int = TEX_W, allow_partial: bool = False) -> np.ndarray:
        """Label (BGR uint8, `width` wide, height from the crop aspect) with the session's first view at u = 0.5."""
        with self.lock:
            if self.acc is None:
                raise StitchError("no frames yet")
            cov = self.coverage()
            if cov < COMPLETE_AT and not allow_partial:
                raise StitchError(f"label {100 * cov:.0f}% captured - keep turning the can")
            seen = self.wsum.max(axis=0) > 0.05
            tex = self.acc / np.maximum(self.wsum, 1e-9)[:, :, None]
            if not seen.all():
                idx = np.nonzero(seen)[0]
                for c in np.nonzero(~seen)[0]:
                    tex[:, c] = tex[:, idx[np.argmin(np.minimum(abs(idx - c), self.circ - abs(idx - c)))]]
            tex = np.roll(tex, self.circ // 2, axis=1)  # first view (label coordinate 0) to the centre
            height = int(round(width * self.h / self.circ))
            return np.clip(cv2.resize(tex.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA),
                           0, 255).astype(np.uint8)

    def texture_jpeg(self, quality: int = 90, allow_partial: bool = False) -> bytes:
        ok, enc = cv2.imencode(".jpg", self.texture(allow_partial=allow_partial), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise StitchError("JPEG encode failed")
        return enc.tobytes()

    def expired(self) -> bool:
        return time.monotonic() - self.updated > SESSION_TTL_S


def render_cylinder(label: np.ndarray, phi: float, width: int, height: int) -> np.ndarray:
    """Orthographic view of a cylinder wrapped with `label` (u = 0.5 facing the camera at phi = 0) rotated by phi
    radians. Used by the tests and the self-check to validate the stitcher against a known texture."""
    lh, lw = label.shape[:2]
    r = width / 2.0
    x = (np.arange(width) + 0.5 - r) / r
    theta = np.arcsin(np.clip(x, -1, 1))
    u = ((theta + phi) / (2 * math.pi) + 0.5) % 1.0
    mx = np.tile((u * lw - 0.5).astype(np.float32), (height, 1))
    my = np.tile(((np.arange(height) + 0.5) / height * lh - 0.5).astype(np.float32)[:, None], (1, width))
    return cv2.remap(label, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
