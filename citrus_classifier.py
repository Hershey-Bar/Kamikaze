"""Multimodal citrus flavour disambiguation (colour + OCR) for otherwise identical sparkling-water cans.

Kirkland lime / lemon / grapefruit cans share one silver design and differ only in the flavour print, and RapidOCR
confuses "lime" / "lemon" often enough that text alone cannot decide. This module grounds the decision in colour:

1. Masked surface extraction: the detector box of the can -> inner 60 % ellipse (no background, table reflections or
   edge shadows) -> drop specular glare (V > 240 and S < 30) and dark ink (V < 40). Only chromatic pixels (S >= 60)
   carry flavour evidence; silver / white / grey body pixels are neutral.
2. Dual colour space analysis: k-means (k = 3, deterministic) on the CIELAB chromaticity (a*, b*) of the chromatic
   pixels; every cluster is scored against each flavour's OpenCV-scale HSV hue band (0-180) and CIELAB constraints
   (a*, b* signed, 8-bit OpenCV LAB minus 128):
       lime        hue 36-82                 a* < -12
       lemon       hue 22-35, S > 100        b* > 30
       grapefruit  hue 160-180 or 0-11       a* 10-45 (coral pink / ruby / pastel orange)
       orange      hue 12-21, high S         a* > 10, b* > 40
   S_color(f) = sum over clusters of share x (0.7 band x saturation gate + 0.3 LAB), in [0, 1].
3. Fusion: P(f) = 0.55 S_color(f) + 0.45 S_OCR(f), S_OCR = 1.0 when the flavour word was read on the latest OCR pass,
   0.6 when only in the rolling buffer. When the colour verdict is decisive and the OCR flavour's S_color <= 0.15
   (OCR read "lemon" on a strictly green can) the OCR flavour is a conflict: it is suppressed and colour decides.

Frames stay in RAM; nothing is written to disk.
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np

FLAVORS = ("lime", "lemon", "grapefruit", "orange")
WORDS = {
    "lime": {"lime", "limes"},
    "lemon": {"lemon", "lemons"},
    "grapefruit": {"grapefruit", "grapefruits", "pamplemousse"},
    "orange": {"orange", "oranges", "mandarin", "mandarine", "tangerine", "clementine"},
}
W_COLOR, W_OCR = 0.55, 0.45
S_OCR_RECENT, S_OCR_ROLLING = 1.0, 0.6

SIDE = int(os.environ.get("KAMIKAZE_CITRUS_SIDE", "96"))  # crop is downsampled to this longer side
ELLIPSE = 0.60  # inner ellipse axes as a fraction of the box
GLARE_V, GLARE_S, INK_V = 240, 30, 40
CHROMA_S = int(os.environ.get("KAMIKAZE_CITRUS_CHROMA_S", "60"))
MIN_FRACTION = float(os.environ.get("KAMIKAZE_CITRUS_MIN_FRAC", "0.03"))  # chromatic share of the surface
K = 3
VERDICT_MIN, VERDICT_MARGIN, STRONG = 0.35, 0.15, 0.60
CONFLICT_MAX = 0.15  # an OCR flavour whose colour score is at most this is contradicted by the can
SOFT = 3.0  # hue band edge falloff (OpenCV hue units)

BANDS = {
    "lime": ((36.0, 82.0),),
    "lemon": ((22.0, 35.0),),
    "grapefruit": ((160.0, 180.0), (0.0, 11.0)),
    "orange": ((12.0, 21.0),),
}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _hue_dist(h: float, lo: float, hi: float) -> float:
    """Circular distance (OpenCV hue, period 180) from h to the interval [lo, hi]; 0 inside."""
    if lo <= h <= hi:
        return 0.0
    d = min(abs(h - lo), abs(h - hi))
    return min(d, 180.0 - d)


def band_membership(flavor: str, hue: float) -> float:
    d = min(_hue_dist(hue, lo, hi) for lo, hi in BANDS[flavor])
    return _clip01(1.0 - d / SOFT)


def sat_gate(flavor: str, sat: float) -> float:
    if flavor == "lemon":
        return _clip01((sat - 85.0) / 30.0)  # S > 100 -> ~0.5..1
    if flavor == "orange":
        return _clip01((sat - 120.0) / 40.0)
    return 1.0


def lab_membership(flavor: str, a: float, b: float) -> float:
    if flavor == "lime":
        return _clip01((-a - 4.0) / 8.0)  # a* <= -12 -> 1
    if flavor == "lemon":
        # yellow: b* > 30, neither reddish nor green-shifted (lime prints also have a high b*)
        return _clip01((b - 22.0) / 8.0) * _clip01((15.0 - a) / 10.0) * _clip01((a + 16.0) / 8.0)
    if flavor == "grapefruit":
        # coral / pastel a* 10-45; saturated ruby prints reach a* ~65, so the upper edge is soft
        return _clip01((a - 4.0) / 6.0) * _clip01((75.0 - a) / 15.0) * _clip01((45.0 - b) / 10.0)
    if flavor == "orange":
        return _clip01((a - 4.0) / 6.0) * _clip01((b - 32.0) / 8.0)
    return 0.0


def cluster_scores(hue: float, sat: float, a: float, b: float) -> dict[str, float]:
    """Membership of one colour cluster (circular-mean hue, median S, mean a*, b*) in every flavour.
    The hue band (with its saturation gate) is necessary; CIELAB only confirms it. A purely additive LAB term let a
    pure-yellow lemon print (a* < -12) earn ~0.2 "lime", enough for an OCR "lime" misread to outvote the colour."""
    return {f: band_membership(f, hue) * sat_gate(f, sat) * (0.7 + 0.3 * lab_membership(f, a, b)) for f in FLAVORS}


def to_pixel_box(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) in pixels; a box whose values are all <= 1 is treated as normalised."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if max(x0, y0, x1, y1) <= 1.0:
        x0, x1, y0, y1 = x0 * width, x1 * width, y0 * height, y1 * height
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def surface_pixels(rgb: np.ndarray, bbox=None) -> tuple[np.ndarray, np.ndarray, int]:
    """(HSV, LAB) uint8 pixels of the masked can surface (inner ellipse, no glare / ink), plus the masked area."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        raise ValueError("expected an (H, W, 3) RGB image")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if bbox is not None:
        x0, y0, x1, y1 = to_pixel_box(bbox, arr.shape[1], arr.shape[0])
        if x1 - x0 < 4 or y1 - y0 < 4:
            raise ValueError("bounding box too small")
        arr = arr[y0:y1, x0:x1]
    h, w = arr.shape[:2]
    if max(h, w) > SIDE:
        s = SIDE / max(h, w)
        arr = cv2.resize(arr, (max(4, int(round(w * s))), max(4, int(round(h * s)))), interpolation=cv2.INTER_AREA)
        h, w = arr.shape[:2]
    arr = np.ascontiguousarray(arr)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ry, rx = max(1.0, ELLIPSE * h / 2.0), max(1.0, ELLIPSE * w / 2.0)
    ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    s, v = hsv[..., 1], hsv[..., 2]
    keep = ellipse & ~((v > GLARE_V) & (s < GLARE_S)) & ~(v < INK_V)
    return hsv[keep], lab[keep], int(keep.sum())


def _circular_mean_hue(h: np.ndarray) -> float:
    ang = h.astype(np.float64) * (2.0 * np.pi / 180.0)
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float((m * 180.0 / (2.0 * np.pi)) % 180.0)


def color_evidence(rgb: np.ndarray, bbox=None) -> dict:
    """Colour evidence of a can crop: {"scores": {flavor: S_color}, "verdict", "strong", "fraction", "clusters"}.
    verdict is the best flavour when S >= 0.35 and it leads the runner-up by >= 0.15, else None."""
    hsv, lab, area = surface_pixels(rgb, bbox)
    zero = {f: 0.0 for f in FLAVORS}
    chroma = hsv[:, 1] >= CHROMA_S
    n = int(chroma.sum())
    frac = n / area if area else 0.0
    out = {"scores": dict(zero), "verdict": None, "strong": False, "fraction": round(frac, 4), "clusters": []}
    if area == 0 or frac < MIN_FRACTION or n < 12:
        return out
    hsv_c, lab_c = hsv[chroma], lab[chroma].astype(np.float32)
    ab = np.ascontiguousarray(lab_c[:, 1:3] - 128.0)
    k = min(K, n)
    cv2.setRNGSeed(0)
    _, labels, _ = cv2.kmeans(ab, k, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3,
                              cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    scores = dict(zero)
    for c in range(k):
        m = labels == c
        share = float(m.mean())
        if share <= 0:
            continue
        hue = _circular_mean_hue(hsv_c[m, 0])
        sat = float(np.median(hsv_c[m, 1]))
        a, b = (float(x) for x in ab[m].mean(axis=0))
        cs = cluster_scores(hue, sat, a, b)
        for f in FLAVORS:
            scores[f] += share * cs[f]
        best = max(cs, key=cs.get)
        out["clusters"].append({"share": round(share, 3), "hue": round(hue, 1), "sat": round(sat, 1),
                                "a": round(a, 1), "b": round(b, 1), "best": best if cs[best] > 0 else None})
    out["clusters"].sort(key=lambda c: -c["share"])
    out["scores"] = {f: round(_clip01(s), 4) for f, s in scores.items()}
    ranked = sorted(out["scores"].items(), key=lambda kv: -kv[1])
    (f1, s1), (_, s2) = ranked[0], ranked[1]
    if s1 >= VERDICT_MIN and s1 - s2 >= VERDICT_MARGIN:
        out["verdict"], out["strong"] = f1, s1 >= STRONG
    return out


def _token_set(tokens) -> set[str]:
    if not tokens:
        return set()
    return {str(t).lower() for t in tokens}


def ocr_scores(recent=None, rolling=None) -> dict[str, float]:
    """S_OCR per flavour: 1.0 read on the latest OCR pass, 0.6 only in the rolling multi-pass buffer, else 0."""
    rec, roll = _token_set(recent), _token_set(rolling)
    out = {}
    for f, ws in WORDS.items():
        out[f] = S_OCR_RECENT if ws & rec else S_OCR_ROLLING if ws & roll else 0.0
    return out


def conflicts(evidence: dict | None) -> set[str]:
    """Flavours the can's colour contradicts: with a decisive colour verdict, every other flavour whose S_color is at
    most CONFLICT_MAX (a strictly green can is never lemon, grapefruit or orange)."""
    if not evidence or not evidence.get("verdict"):
        return set()
    sc = evidence["scores"]
    return {f for f in FLAVORS if f != evidence["verdict"] and sc.get(f, 0.0) <= CONFLICT_MAX}


def fuse(evidence: dict | None, ocr: dict[str, float] | None) -> dict:
    """P(f) = 0.55 S_color + 0.45 S_OCR with conflicting OCR flavours suppressed.
    Returns {"flavor", "p", "color", "ocr", "suppressed", "conflict"}; flavor is None without enough evidence."""
    color = dict((evidence or {}).get("scores") or {f: 0.0 for f in FLAVORS})
    ocr = dict(ocr or {f: 0.0 for f in FLAVORS})
    bad = conflicts(evidence)
    suppressed = sorted(f for f in bad if ocr.get(f, 0.0) > 0.0)
    for f in suppressed:
        ocr[f] = 0.0
    p = {f: round(W_COLOR * color.get(f, 0.0) + W_OCR * ocr.get(f, 0.0), 4) for f in FLAVORS}
    ranked = sorted(p.items(), key=lambda kv: -kv[1])
    (f1, p1), (_, p2) = ranked[0], ranked[1]
    flavor = f1 if p1 >= W_COLOR * VERDICT_MIN and p1 - p2 >= 0.05 else None
    return {"flavor": flavor, "p": p, "color": color, "ocr": ocr, "suppressed": suppressed,
            "conflict": bool(suppressed)}


def gate_tokens(tokens, evidence: dict | None):
    """Drop the OCR words of every flavour the can's colour contradicts (fuzzy "lemon" on a green can).
    Accepts a set / list (returns a set) or a {token: count} dict (returns a dict)."""
    bad = conflicts(evidence)
    if not bad or not tokens:
        return tokens
    banned = set().union(*(WORDS[f] for f in bad))
    if isinstance(tokens, dict):
        return {t: n for t, n in tokens.items() if str(t).lower() not in banned}
    return {t for t in tokens if str(t).lower() not in banned}


def flavor_of(meta: dict | None, label: str = "") -> str | None:
    """The single citrus flavour a class names (flavour / variant / name / label words), else None ("lemon-lime")."""
    meta = meta or {}
    text = " ".join(str(meta.get(f, "")) for f in ("flavor", "variant", "raw_name", "name")) + " " + \
        (label or "").replace("_", " ")
    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = [f for f, ws in WORDS.items() if ws & words]
    return hits[0] if len(hits) == 1 else None


def arbitrate(ranked: list, fused: dict, flavor_of_label, evidence: dict | None, recent=None, rolling=None,
              lexicon_label: str | None = None, calibrate=None, max_gap: float = 0.25, margin: float = 0.15,
              demote_to: float = 0.45) -> dict | None:
    """Correct the visual ranking of citrus-flavoured classes with the fused colour + OCR flavour decision.

    Applies when the top-1 class names a citrus flavour and the decision names a different one:
      * a class of the decided flavour within `max_gap` fused confidence of the top is promoted to top-1 (it takes
        the top's confidence; the old top drops to `margin` + 0.05 below it);
      * otherwise, when the colour verdict is strong and contradicts the top's flavour (S_color <= 0.15), the top is
        demoted to at most `demote_to` (a lemon can must never be identified as the only enrolled lime class).
    A class the OCR keyword lexicon confirmed is left alone (its tokens were already colour-gated).
    Mutates ranked / fused in place; returns diagnostics or None when no citrus class leads."""
    if not ranked:
        return None
    cal = calibrate or (lambda s: s)
    top = ranked[0][0]
    ftop = flavor_of_label(top)
    if ftop is None:
        return None
    dec = fuse(evidence, ocr_scores(recent, rolling))
    info = {"top": top, "top_flavor": ftop, "flavor": dec["flavor"], "p": dec["p"],
            "verdict": (evidence or {}).get("verdict"), "strong": bool((evidence or {}).get("strong")),
            "suppressed": dec["suppressed"], "action": None}
    if dec["flavor"] is None or dec["flavor"] == ftop or lexicon_label == top:
        return info
    f_top = fused.get(top, cal(ranked[0][1]))
    for i in range(1, len(ranked)):
        lab = ranked[i][0]
        if flavor_of_label(lab) != dec["flavor"]:
            continue
        f_lab = fused.get(lab, cal(ranked[i][1]))
        if f_top - f_lab <= max_gap:
            hi = max(f_top, f_lab)
            fused[lab], fused[top] = hi, min(f_top, f_lab, max(0.0, hi - margin - 0.05))
            ranked.insert(0, ranked.pop(i))
            info.update(action="promoted", winner=lab)
            return info
        break
    if info["strong"] and dec["flavor"] == info["verdict"] and ftop in conflicts(evidence):
        fused[top] = min(f_top, demote_to)
        ranked.sort(key=lambda r: -fused.get(r[0], cal(r[1])))
        info.update(action="demoted", winner=ranked[0][0])
    return info
