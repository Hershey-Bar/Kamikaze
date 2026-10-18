"""Fine-grained phone model distinguisher (offline): structural cues + DINOv2 camera-island embeddings.

Reference index models/phone_embeddings.npz (scripts/populate_device_cache.py): every cached device GLB is rendered
orthographically from the back (render_view), the top 45 % (camera island) is cropped and embedded with the board's
DINOv2 ViT-S/14 (npu.DinoPoseEmbedder, CLS + mean patch, 768-d unit vector).

Live, for a detected phone crop the classifier fuses:
  * DINOv2 cosine of the live camera-island crop vs every reference (only when the index + embedder exist),
  * lens layout from Hough circles on the island (triangle = Apple Pro triple square, diagonal pair = iPhone 13/14/15
    base, vertical pair = iPhone 16 pill, vertical column = Samsung),
  * camera housing presence (raised plate between lenses: S20/S21/S22 / Fold 3-4) vs floating rings (S22 Ultra, S23+),
  * front cut-out when the screen is lit (notch = iPhone 13 / 14 base, Dynamic Island = 14 Pro / 15 / 16, punch hole =
    Galaxy),
  * outline aspect H/W from the detector box (Z Fold closed 2.25-2.36 vs slabs 2.05-2.20).
Every cue is optional; scores are normalised over the cues that were measured. The result is advisory (it never
overrides an enrolled-class identification; it reports agreement / conflict).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import device_cache

ROOT = Path(__file__).resolve().parent
EMB_PATH = ROOT / "models" / "phone_embeddings.npz"
ISLAND_FRAC = 0.45
EMB_SIZE = 224
PAD_GREY = 128

# island_type -> expected lens layout seen on the back
LAYOUT_OF = {
    "triple_square": "triangle", "diagonal_dual": "diagonal", "vertical_pill": "vertical_pair",
    "rect_housing": "column", "contour_housing": "column", "contour_island": "column", "vertical_housing": "column",
    "floating_rings": "column",
}
HOUSED = {"rect_housing", "contour_housing", "contour_island", "vertical_housing"}
# layouts that are easily confused by Hough (partial credit)
LAYOUT_NEAR = {("column", "vertical_pair"), ("vertical_pair", "column"), ("triangle", "column"),
               ("column", "triangle")}
WEIGHTS = {"dino": 0.45, "layout": 0.2, "housing": 0.1, "front": 0.15, "aspect": 0.1}
ASPECT_SIGMA = 0.07


# --------------------------------------------------------------------------------------
# Offline rendering of reference GLBs
# --------------------------------------------------------------------------------------


def _material_colours(doc: dict) -> list[np.ndarray]:
    out = []
    for mat in doc.get("materials", []):
        f = (mat.get("pbrMetallicRoughness") or {}).get("baseColorFactor") or [0.8, 0.8, 0.8, 1.0]
        out.append(np.clip(np.asarray(f[:3], dtype=np.float64), 0, 1))
    return out


def render_view(glb: bytes, view: str = "back", size: int = 448, pad: float = 0.04) -> np.ndarray:
    """Orthographic Lambert render (RGB uint8, size x size, phone centred on PAD_GREY) of a device-cache GLB
    (screen +Z, height +Y). view "back" looks at -Z (the viewer's right is the model's -X), "front" looks at +Z."""
    doc, binary = device_cache.parse_glb(glb)
    cols = _material_colours(doc)
    tris, tcol = [], []
    for t, mat in device_cache.triangles(doc, binary):
        tris.append(t)
        c = cols[mat] if mat is not None and mat < len(cols) else np.array([0.8, 0.8, 0.8])
        tcol.append(np.repeat(c[None], len(t), axis=0))
    if not tris:
        raise ValueError("GLB has no triangles")
    tris, tcol = np.concatenate(tris), np.concatenate(tcol)
    if view == "back":
        u, v, depth = -tris[..., 0], tris[..., 1], tris[..., 2]  # camera at -Z: smaller z is nearer
        light = np.array([0.35, 0.45, -1.0])
    elif view == "front":
        u, v, depth = tris[..., 0], tris[..., 1], -tris[..., 2]
        light = np.array([-0.35, 0.45, 1.0])
    else:
        raise ValueError("view must be back or front")
    light = light / np.linalg.norm(light)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    shade = 0.28 + 0.72 * np.abs(n @ light)
    rgb = np.clip(tcol * shade[:, None] * 255.0, 0, 255)
    umin, umax, vmin, vmax = u.min(), u.max(), v.min(), v.max()
    scale = (1 - 2 * pad) * size / max(umax - umin, vmax - vmin, 1e-9)
    px = (u - (umin + umax) / 2) * scale + size / 2
    py = size / 2 - (v - (vmin + vmax) / 2) * scale
    order = np.argsort(-depth.mean(axis=1))  # far -> near (painter's algorithm)
    img = np.full((size, size, 3), PAD_GREY, np.uint8)
    pts = np.round(np.stack([px, py], -1) * 16).astype(np.int32)  # 4 fractional bits
    e1, e2 = (pts[:, 1] - pts[:, 0]).astype(np.int64), (pts[:, 2] - pts[:, 0]).astype(np.int64)
    area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 512.0
    for i in order:
        if area[i] < 0.02:
            continue
        cv2.fillConvexPoly(img, pts[i], tuple(int(c) for c in rgb[i]), lineType=cv2.LINE_AA, shift=4)
    return img


def phone_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the non-background pixels of a render."""
    mask = np.any(np.abs(img.astype(np.int16) - PAD_GREY) > 3, axis=2)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty render")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def island_crop(phone_rgb: np.ndarray) -> np.ndarray:
    """Top ISLAND_FRAC of an upright phone-back crop."""
    h = phone_rgb.shape[0]
    return phone_rgb[: max(1, int(round(h * ISLAND_FRAC)))]


def square_pad(rgb: np.ndarray, size: int = EMB_SIZE) -> np.ndarray:
    """Aspect-preserving resize onto a PAD_GREY square (the same framing for renders and live crops)."""
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    out = np.full((size, size, 3), PAD_GREY, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def to_nhwc01(rgb: np.ndarray) -> np.ndarray:
    return (square_pad(rgb).astype(np.float32) / 255.0)[None]


# --------------------------------------------------------------------------------------
# Structural cues on live crops
# --------------------------------------------------------------------------------------


def find_lenses(island_rgb: np.ndarray, phone_width_px: int | None = None) -> list[tuple[float, float, float]]:
    """Circles (x, y, r) in island pixels that look like camera lens rings."""
    gray = cv2.cvtColor(island_rgb, cv2.COLOR_RGB2GRAY)
    w = phone_width_px or island_rgb.shape[1]
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(4, int(0.13 * w)), param1=110,
                               param2=24, minRadius=max(3, int(0.045 * w)), maxRadius=max(5, int(0.16 * w)))
    if circles is None:
        return []
    out = []
    for x, y, r in circles[0][:8]:
        # a lens is darker inside than its ring / surroundings
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, (int(x), int(y)), max(1, int(r * 0.6)), 255, -1)
        ring = np.zeros(gray.shape, np.uint8)
        cv2.circle(ring, (int(x), int(y)), int(r * 1.35), 255, -1)
        cv2.circle(ring, (int(x), int(y)), int(r * 1.05), 0, -1)
        inner = cv2.mean(gray, mask)[0]
        outer = cv2.mean(gray, ring)[0] if ring.any() else 255.0
        if inner < outer - 8:
            out.append((float(x), float(y), float(r)))
    return out


def lens_layout(lenses: list[tuple[float, float, float]], width_px: float) -> str:
    """"triangle" | "diagonal" | "vertical_pair" | "column" | "unknown" from lens centres (normalised by width)."""
    if len(lenses) < 2:
        return "unknown"
    pts = np.array([(x, y) for x, y, _ in lenses]) / max(width_px, 1.0)
    dx, dy = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
    if len(lenses) == 2:
        if dx > 0.1 and dy > 0.1:
            return "diagonal"
        if dx <= 0.08 and dy > 0.1:
            return "vertical_pair"
        return "unknown"
    if dx <= 0.12 and dy >= 0.3:
        return "column"
    if dx > 0.12 and dy > 0.12 and dy < 0.5:
        return "triangle"
    return "column" if dy > 2.0 * dx else "unknown"


def housing_present(island_rgb: np.ndarray, lenses: list[tuple[float, float, float]]) -> float | None:
    """Contrast (0..1) of the strip between neighbouring lenses against the open back panel: a raised housing /
    contour plate differs from the panel, floating rings sit directly on it. None when < 2 lenses."""
    if len(lenses) < 2:
        return None
    lab = cv2.cvtColor(island_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    pts = sorted(lenses, key=lambda c: c[1])
    between = []
    for (x0, y0, r0), (x1, y1, r1) in zip(pts, pts[1:]):
        gap = math.hypot(x1 - x0, y1 - y0) - r0 - r1
        if gap < 3:
            continue
        mx, my = int((x0 + x1) / 2), int((y0 + y1) / 2)
        k = max(1, int(gap / 4))
        between.append(lab[max(0, my - k):my + k + 1, max(0, mx - k):mx + k + 1].reshape(-1, 3))
    if not between:
        return None
    xs = [c[0] for c in lenses]
    far_x = int(min(w - 1, max(0, (w * 0.15 if np.mean(xs) > w / 2 else w * 0.85))))
    panel = lab[int(h * 0.2):int(h * 0.9), max(0, far_x - 4):far_x + 5].reshape(-1, 3)
    if not len(panel):
        return None
    d = np.linalg.norm(np.median(np.concatenate(between), 0) - np.median(panel, 0))
    return float(np.clip((d - 8.0) / 22.0, 0.0, 1.0))


def front_cutout(screen_rgb: np.ndarray) -> str:
    """"notch" | "dynamic_island" | "punch_hole" | "unknown" from a lit screen crop (upright)."""
    hsv = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    if np.median(hsv[..., 2]) < 90:
        return "unknown"  # screen off: the cut-out is invisible
    band = hsv[: max(4, int(0.12 * h)), int(0.15 * w):int(0.85 * w), 2]
    dark = (band < 50).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cx = (x + bw / 2) / band.shape[1]
        if area < 6 or not 0.3 <= cx <= 0.7:
            continue
        if best is None or area > best[4]:
            best = (x, y, bw, bh, area)
    if best is None:
        return "unknown"
    x, y, bw, bh, _ = best
    wf = bw / w
    if y <= 1 and wf >= 0.25:
        return "notch"
    if 0.14 <= wf <= 0.4 and bw / max(bh, 1) >= 2.0:
        return "dynamic_island"
    if wf <= 0.09 and 0.6 <= bw / max(bh, 1) <= 1.6:
        return "punch_hole"
    return "unknown"


def aspect_score(measured: float | None, entry: dict) -> float | None:
    if not measured or measured <= 0:
        return None
    return float(math.exp(-0.5 * ((measured - entry["height_mm"] / entry["width_mm"]) / ASPECT_SIGMA) ** 2))


# --------------------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------------------


@dataclass
class PhoneIndex:
    ids: list[str]
    emb: np.ndarray  # (n, d) unit vectors of the camera-island renders
    embedder: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> "PhoneIndex | None":
        path = Path(path or EMB_PATH)
        if not path.is_file():
            return None
        z = np.load(path, allow_pickle=False)
        return cls([str(s) for s in z["ids"]], z["island"].astype(np.float32), str(z["embedder"]) if "embedder" in z else "")


@dataclass
class PhoneClassifier:
    index: PhoneIndex | None = None
    embed: object = None  # callable (1, 224, 224, 3) float [0,1] -> unit vector, e.g. DinoPoseEmbedder.embed_rgb01
    entries: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.entries:
            self.entries = device_cache.devices()

    @classmethod
    def create(cls, embed=None, path: Path | None = None) -> "PhoneClassifier":
        return cls(PhoneIndex.load(path), embed)

    def cues(self, phone_rgb: np.ndarray, view: str | None = None, box_aspect: float | None = None) -> dict:
        """Structural evidence from an upright phone crop (RGB). view "front" -> cut-out, otherwise rear cues."""
        out: dict = {"aspect": round(box_aspect, 3) if box_aspect else None}
        h, w = phone_rgb.shape[:2]
        if view == "front":
            out["front"] = front_cutout(phone_rgb)
            return out
        isl = island_crop(phone_rgb)
        lenses = find_lenses(isl, w)
        out["lenses"] = len(lenses)
        out["layout"] = lens_layout(lenses, w)
        hp = housing_present(isl, lenses)
        out["housing"] = None if hp is None else round(hp, 3)
        return out

    def dino_scores(self, phone_rgb: np.ndarray) -> dict[str, float]:
        if self.index is None or self.embed is None or not len(self.index.ids):
            return {}
        v = np.asarray(self.embed(to_nhwc01(island_crop(phone_rgb))), dtype=np.float32).reshape(-1)
        if v.shape[0] != self.index.emb.shape[1]:
            return {}
        sims = self.index.emb @ (v / max(float(np.linalg.norm(v)), 1e-9))
        return {d: float(s) for d, s in zip(self.index.ids, sims)}

    def score(self, cues: dict, dino: dict[str, float] | None = None) -> list[dict]:
        """Ranked [{"id", "name", "score", "parts"}] over every devices_db entry."""
        dino = dino or {}
        if dino:
            vals = np.array(list(dino.values()))
            lo, hi = float(vals.min()), float(vals.max())
        ranked = []
        for dev, e in self.entries.items():
            parts: dict[str, float] = {}
            if dev in dino:
                parts["dino"] = (dino[dev] - lo) / (hi - lo) if hi > lo else 0.5
            layout = cues.get("layout")
            if layout and layout != "unknown":
                want = LAYOUT_OF.get(e.get("island_type"), "")
                parts["layout"] = 1.0 if layout == want else 0.5 if (layout, want) in LAYOUT_NEAR else 0.0
            if cues.get("housing") is not None and LAYOUT_OF.get(e.get("island_type")) == "column":
                housed = e.get("island_type") in HOUSED
                parts["housing"] = cues["housing"] if housed else 1.0 - cues["housing"]
            front = cues.get("front")
            if front and front != "unknown":
                parts["front"] = 1.0 if front == e.get("front_type") else 0.0
            a = aspect_score(cues.get("aspect"), e)
            if a is not None:
                parts["aspect"] = a
            wsum = sum(WEIGHTS[k] for k in parts)
            s = sum(WEIGHTS[k] * v for k, v in parts.items()) / wsum if wsum else 0.0
            ranked.append({"id": dev, "name": f"{e['brand']} {e['name']}", "score": round(s, 4),
                           "parts": {k: round(v, 3) for k, v in parts.items()}})
        ranked.sort(key=lambda r: -r["score"])
        return ranked

    def classify(self, phone_rgb: np.ndarray, view: str | None = None, box_aspect: float | None = None,
                 top: int = 5) -> dict:
        cues = self.cues(phone_rgb, view, box_aspect)
        dino = self.dino_scores(phone_rgb) if view != "front" else {}
        ranked = self.score(cues, dino)
        best = ranked[0] if ranked else None
        return {"cues": cues, "dino": bool(dino), "top": ranked[:top],
                "family": family_of(best["id"], self.entries) if best else None}


def family_of(dev: str, entries: dict | None = None) -> dict:
    e = (entries or device_cache.devices())[dev]
    return {"form": e.get("form"), "front_type": e.get("front_type"), "island_type": e.get("island_type"),
            "layout": LAYOUT_OF.get(e.get("island_type"))}


def family_conflicts(label_dev: str | None, cues: dict, entries: dict | None = None) -> list[str]:
    """Human-readable contradictions between an identified device and measured structural cues."""
    if not label_dev:
        return []
    entries = entries or device_cache.devices()
    if label_dev not in entries:
        return []
    e = entries[label_dev]
    out = []
    want = LAYOUT_OF.get(e.get("island_type"))
    lay = cues.get("layout")
    if lay and lay != "unknown" and want and lay != want and (lay, want) not in LAYOUT_NEAR:
        out.append(f"lens layout looks {lay.replace('_', ' ')}, {e['name']} has a {want.replace('_', ' ')} layout")
    front = cues.get("front")
    if front and front != "unknown" and front != e.get("front_type"):
        out.append(f"front shows a {front.replace('_', ' ')}, {e['name']} has a {e['front_type'].replace('_', ' ')}")
    a = aspect_score(cues.get("aspect"), e)
    if a is not None and a < 0.05:
        out.append(f"outline aspect {cues['aspect']:.2f} does not match {e['name']} "
                   f"({e['height_mm'] / e['width_mm']:.2f})")
    return out


class ScanAccumulator:
    """Per-SMART-scan vote over phone classifications (sum of per-frame scores)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.totals: dict[str, float] = {}
        self.frames = 0
        self.cues: dict[str, dict[str, int]] = {}
        self.dino = False

    def add(self, result: dict) -> None:
        self.frames += 1
        self.dino |= bool(result.get("dino"))
        for r in result.get("top", []):
            self.totals[r["id"]] = self.totals.get(r["id"], 0.0) + r["score"]
        for k in ("layout", "front"):
            v = (result.get("cues") or {}).get(k)
            if v and v != "unknown":
                self.cues.setdefault(k, {})[v] = self.cues.setdefault(k, {}).get(v, 0) + 1

    def summary(self, label_dev: str | None = None) -> dict | None:
        if not self.frames or not self.totals:
            return None
        ranked = sorted(self.totals.items(), key=lambda kv: -kv[1])[:3]
        entries = device_cache.devices()
        majority = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in self.cues.items()}
        best = ranked[0][0]
        return {
            "id": best, "name": f"{entries[best]['brand']} {entries[best]['name']}" if best in entries else best,
            "score": round(ranked[0][1] / self.frames, 3), "frames": self.frames, "dino": self.dino,
            "runner_up": [{"id": d, "score": round(s / self.frames, 3)} for d, s in ranked[1:]],
            "cues": majority, "family": family_of(best, entries) if best in entries else None,
            "label_device": label_dev, "agrees": (label_dev == best) if label_dev else None,
            "conflicts": family_conflicts(label_dev, majority, entries),
        }


def upright_crop(rgb: np.ndarray, max_side: int = 448) -> tuple[np.ndarray, float]:
    """A phone crop rotated to portrait (landscape -> 90 degrees) and reduced to <= max_side; returns (crop, H/W)."""
    h, w = rgb.shape[:2]
    if w > h:
        rgb = np.ascontiguousarray(np.rot90(rgb))
        h, w = w, h
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        rgb = cv2.resize(rgb, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return rgb, h / max(1, w)


class DeviceMatcher:
    """Background fine-grained phone matcher for the SMART kiosk scan: the inference loop submits the detector crop of
    the phone (latest wins, rate-capped at `hz`), a worker thread classifies it and accumulates the 5 s scan's votes.
    Crops are only held in RAM until processed."""

    def __init__(self, classifier: PhoneClassifier, hz: float = 2.0):
        self.classifier = classifier
        self.period = 1.0 / max(0.1, hz)
        self.acc = ScanAccumulator()
        self._cv = threading.Condition()
        self._job: tuple | None = None
        self._gen = 0
        self._last = 0.0
        self._thread: threading.Thread | None = None
        self.last: dict | None = None
        self.runs = self.errors = 0
        self.ms = 0.0

    def submit(self, rgb: np.ndarray, view: str | None = None) -> bool:
        now = time.monotonic()
        with self._cv:
            if now - self._last < self.period:
                return False
            self._last = now
            self._job = (self._gen, rgb, view)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="phone-classifier", daemon=True)
                self._thread.start()
            self._cv.notify()
        return True

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._job is not None)
                gen, rgb, view = self._job
                self._job = None
            try:
                t = time.perf_counter()
                crop, aspect = upright_crop(rgb)
                res = self.classifier.classify(crop, "front" if view == "front" else "back", aspect)
                self.ms = (time.perf_counter() - t) * 1000.0
                self.runs += 1
                with self._cv:
                    if gen == self._gen:  # a scan reset while classifying discards the stale result
                        self.acc.add(res)
                        self.last = res
            except Exception:
                self.errors += 1

    def reset(self) -> None:
        with self._cv:
            self._gen += 1
            self._job, self.last = None, None
            self.acc.reset()

    def summary(self, label_dev: str | None = None) -> dict | None:
        with self._cv:
            return self.acc.summary(label_dev)

    def info(self) -> dict:
        idx = self.classifier.index
        return {"index": len(idx.ids) if idx else 0, "embedder": idx.embedder if idx else None,
                "dino": bool(idx is not None and self.classifier.embed is not None), "runs": self.runs,
                "errors": self.errors, "ms": round(self.ms, 1), "rate_hz": round(1.0 / self.period, 1)}

