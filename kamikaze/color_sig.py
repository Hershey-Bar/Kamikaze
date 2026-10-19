"""Colour-tone signature of the reticle crop, the third modality of the identity fusion (vision + OCR text + colour).

A packaging colour scheme separates brands that a grey-ish embedding can confuse (Coca-Cola red vs Pepsi blue vs
Sprite green vs a pink/silver Kirkland Grapefruit can). The signature is an 18-bin hue histogram, every pixel weighted
by its saturation (near-black pixels ramped out), plus 3 achromatic bins (dark / grey / bright) for the remainder,
soft-binned and normalised to sum 1. Similarity is histogram intersection, mapped to a calibrated score

    C_color = clip((I - I_LO) / (I_HI - I_LO), 0, 1)

so the same object under the same light (I ~0.65-0.9) scores ~1 and a different colour scheme (I ~0.1-0.3) scores ~0.
It is computed on a 64 px downsample of the crop (~0.3 ms) and only the 21 floats are ever stored, never pixels.
"""

from __future__ import annotations

import os

import numpy as np

HUE_BINS = 18
DIM = HUE_BINS + 3
SIDE = 64
I_LO = float(os.environ.get("KAMIKAZE_COLOR_I_LO", "0.30"))
I_HI = float(os.environ.get("KAMIKAZE_COLOR_I_HI", "0.70"))


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rgb float32 in [0, 1], shape (..., 3) -> hue in [0, 1), saturation, value."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    delta = mx - mn
    safe = np.where(delta > 1e-6, delta, 1.0)
    h = np.where(mx == r, ((g - b) / safe) % 6.0, np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) / 6.0
    h = np.where(delta > 1e-6, h, 0.0) % 1.0
    s = np.where(mx > 1e-6, delta / np.where(mx > 1e-6, mx, 1.0), 0.0)
    return h, s, mx


def signature(rgb: np.ndarray) -> np.ndarray:
    """21-d normalised colour signature of an RGB uint8 (or float 0-1) image array (H, W, 3).

    Soft binning everywhere, so a small white-balance / exposure shift moves mass smoothly instead of across a hard
    boundary: each pixel's chromatic weight sat x (dark-ramp of value) is split linearly between the two nearest hue
    bin centres; the remaining (achromatic) weight is split between dark / grey / bright by triangular memberships."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        raise ValueError("expected an (H, W, 3) image")
    h, w = arr.shape[:2]
    if max(h, w) > SIDE:
        ys = np.linspace(0, h - 1, min(h, SIDE)).astype(int)
        xs = np.linspace(0, w - 1, min(w, SIDE)).astype(int)
        arr = arr[ys][:, xs]
    x = arr[..., :3].astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    hue, sat, val = (a.ravel() for a in _rgb_to_hsv(x))
    chroma = sat * np.clip((val - 0.12) / 0.23, 0.0, 1.0)  # hue of near-black pixels is noise
    pos = hue * HUE_BINS - 0.5
    lo = np.floor(pos)
    frac = pos - lo
    lo = lo.astype(int) % HUE_BINS
    hist = np.bincount(lo, weights=chroma * (1.0 - frac), minlength=HUE_BINS) + \
        np.bincount((lo + 1) % HUE_BINS, weights=chroma * frac, minlength=HUE_BINS)
    # circular [1/4, 1/2, 1/4] smoothing: tolerance for a white-balance shift of about one bin (20 deg)
    hist = 0.5 * hist + 0.25 * np.roll(hist, 1) + 0.25 * np.roll(hist, -1)
    rest = 1.0 - chroma
    dark = np.clip((0.5 - val) / 0.35, 0.0, 1.0)
    bright = np.clip((val - 0.5) / 0.35, 0.0, 1.0)
    achro = np.array([(rest * dark).sum(), (rest * (1.0 - dark - bright)).sum(), (rest * bright).sum()])
    out = np.concatenate([hist, achro]).astype(np.float32)
    return out / max(float(out.sum()), 1e-9)


def intersection(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)).sum())


def score(live: np.ndarray, refs: dict | list | None) -> float | None:
    """Calibrated colour agreement of a live signature with a class's per-view signatures (best view); None if the
    class has no colour reference."""
    if live is None or not refs:
        return None
    values = refs.values() if isinstance(refs, dict) else refs
    best = max(intersection(live, r) for r in values)
    return float(np.clip((best - I_LO) / max(1e-6, I_HI - I_LO), 0.0, 1.0))


def mean_signature(sigs: list) -> np.ndarray:
    m = np.mean(np.stack([np.asarray(s, dtype=np.float32) for s in sigs]), axis=0)
    return m / max(float(m.sum()), 1e-9)


def dominant(sig: np.ndarray) -> str:
    """Human-readable dominant tone, e.g. "red", "blue", "silver/grey" (HUD diagnostics)."""
    names = ["red", "orange", "orange", "yellow", "yellow", "lime", "green", "green", "teal", "cyan", "cyan", "blue",
             "blue", "violet", "purple", "magenta", "pink", "red"]
    sig = np.asarray(sig)
    i = int(np.argmax(sig))
    if i < HUE_BINS:
        return names[i]
    return ("black", "silver/grey", "white")[i - HUE_BINS]

# Lime vs lemon hue tiebreaker (OpenCV hue scale 0-180; saturation / value on 0-255). Kirkland lime / lemon cans share
# one silver-white design and differ only in print, so the median hue of the strongly coloured pixels decides:
# yellow lemon print sits at ~22-34, green lime print at ~40-75. Reds / pinks (grapefruit) fall outside both bands.
CITRUS_S_MIN = float(os.environ.get("KAMIKAZE_CITRUS_S_MIN", "60"))
CITRUS_V_MIN = float(os.environ.get("KAMIKAZE_CITRUS_V_MIN", "60"))
CITRUS_SPLIT = float(os.environ.get("KAMIKAZE_CITRUS_SPLIT", "38"))
CITRUS_BAND = (15.0, 90.0)  # outside this the colour says neither lemon nor lime
CITRUS_MIN_FRAC = float(os.environ.get("KAMIKAZE_CITRUS_MIN_FRAC", "0.03"))


def hue_stats(rgb: np.ndarray) -> dict:
    """{"median_hue": OpenCV-scale median hue of pixels with S > CITRUS_S_MIN and V > CITRUS_V_MIN (None if too few),
    "fraction": share of such pixels} of an RGB image (downsampled to SIDE px first)."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        raise ValueError("expected an (H, W, 3) image")
    h, w = arr.shape[:2]
    if max(h, w) > SIDE:
        ys = np.linspace(0, h - 1, min(h, SIDE)).astype(int)
        xs = np.linspace(0, w - 1, min(w, SIDE)).astype(int)
        arr = arr[ys][:, xs]
    x = arr[..., :3].astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    hue, sat, val = (a.ravel() for a in _rgb_to_hsv(x))
    mask = (sat * 255.0 > CITRUS_S_MIN) & (val * 255.0 > CITRUS_V_MIN)
    frac = float(mask.mean()) if mask.size else 0.0
    if frac < CITRUS_MIN_FRAC:
        return {"median_hue": None, "fraction": round(frac, 4)}
    return {"median_hue": round(float(np.median(hue[mask])) * 180.0, 2), "fraction": round(frac, 4)}


def citrus_verdict(stats: dict | None) -> str | None:
    """"lime" (median hue > CITRUS_SPLIT), "lemon" (inside the citrus band, <= CITRUS_SPLIT) or None (no evidence)."""
    hue = (stats or {}).get("median_hue")
    if hue is None or not CITRUS_BAND[0] <= hue <= CITRUS_BAND[1]:
        return None
    return "lime" if hue > CITRUS_SPLIT else "lemon"
