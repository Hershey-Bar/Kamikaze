"""Beverage-can packaging from Open Food Facts: real product photos wrapped onto a dimension-calibrated can.

Game-art repositories rarely model a specific flavour; the packaging itself is what identifies a can. Open Food Facts
(https://world.openfoodfacts.org, images CC BY-SA 3.0, no API key) hosts user photos of the front of most products.

    find_products(product)      search-a-licious hits that pass the strict gate (brand + flavour + tier + can volume)
    label_from_photo(jpeg)      can box -> upright / proportion checks -> 360 deg label texture (front 140 deg real,
                                back filled with the per-row colour of the label edge and flagged as an approximation)
    build(product)              -> (attribution dict, GLB bytes) on a Crown-dimension can (spec_search.can_specs)

Nothing is synthesised from logos or brand guesses: if no photo of the exact product passes the gate the result is
None. The live alternative is label_stitch.LabelStitcher, which captures the full 360 deg label from the camera.
"""

from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

import cv2
import numpy as np

import netguard
import procedural_glb
import spec_search

log = logging.getLogger("kamikaze.packaging")

OFF_SEARCH = "https://search.openfoodfacts.org/search"
OFF_PRODUCT = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
OFF_PAGE = "https://world.openfoodfacts.org/product/{code}"
OFF_LICENSE = "CC BY-SA 3.0"
OFF_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
USER_AGENT = "Kamikaze/1.0 (+https://github.com/Hershey-Bar/Kamikaze; edge CAD retrieval)"
TIMEOUT_S = 15.0
MAX_IMAGE_BYTES = 12 * 1024 * 1024
TEX_W = 2048
VISIBLE_DEG = 70.0  # beyond this the cylinder is too foreshortened / glare-prone to sample
BLEND_DEG = 8.0
ASPECT_TOL = 0.15
MAX_TILT_DEG = 4.0
MIN_FILL = 0.85
MAX_WIDTH_CV = 0.06
PHOTO_TRIES = 4
FLAVOURS = {"lemon", "lime", "grapefruit", "cherry", "vanilla", "orange", "mango", "raspberry", "berry", "peach",
            "cucumber", "ginger", "lemonade", "strawberry", "watermelon", "pineapple", "coconut", "apple", "grape",
            "passion", "blackberry", "blueberry", "pomegranate", "citrus", "tropical", "cranberry", "mandarin"}
TIER_WORDS = {"zero": {"zero", "max"}, "diet": {"diet", "light", "lite"}}
NOT_A_CAN = {"bottle", "pet", "glass", "multipack", "pack", "box", "carton", "syrup", "powder", "concentrate"}
# (box, category) pairs from an injected object detector; the receiver supplies its YOLO-World wrapper
Detector = Callable[[np.ndarray], list]
_detector: Detector | None = None


class PackagingError(RuntimeError):
    pass


def set_detector(fn: Detector | None) -> None:
    """fn(bgr) -> [(xmin, ymin, xmax, ymax normalised, category), ...]; used before the GrabCut fallback."""
    global _detector
    _detector = fn


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower().replace("-", " ")))


def _get(url: str, limit: int = 4 * 1024 * 1024, accept: str = "application/json") -> bytes:
    netguard.check(url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        body = resp.read(limit + 1)
    if len(body) > limit:
        raise PackagingError("response too large")
    return body


# --------------------------------------------------------------------------------------
# Search + strict product gate
# --------------------------------------------------------------------------------------


def lucene_query(product) -> str:
    brand = (getattr(product, "brand", "") or "").lower()
    words = [w for w in (getattr(product, "flavor", ""), getattr(product, "drink_type", "")) if w]
    tier = getattr(product, "sub_tier", "")
    if tier in TIER_WORDS:
        words.append(tier)
    brand_q = f'brands:"{brand}"' if brand else ""
    return " ".join([brand_q] + words).strip()


def product_gate(product, hit: dict) -> tuple[bool, str]:
    """Exact brand / flavour / tier / can-volume match of an OFF hit for a CanonicalProduct."""
    brands = _words(" ".join(hit.get("brands") or []) if isinstance(hit.get("brands"), list) else hit.get("brands", ""))
    name = _words(hit.get("product_name", ""))
    want_brand = _words(getattr(product, "brand", "")) - {"the", "company"}
    if not want_brand or not want_brand <= (brands | name):
        return False, "brand"
    flavour = (getattr(product, "flavor", "") or "").lower()
    if flavour and flavour not in name:
        return False, f"flavour {flavour!r} missing"
    if (name & FLAVOURS) - {flavour} - _words(getattr(product, "drink_type", "")):
        return False, f"other flavour {sorted((name & FLAVOURS) - {flavour})}"
    tier = getattr(product, "sub_tier", "")
    for key, syn in TIER_WORDS.items():
        if (tier == key) != bool(name & syn):
            return False, f"tier {key}"
    drink = _words(getattr(product, "drink_type", ""))
    if drink and not drink <= name:
        return False, "drink type"
    if name & NOT_A_CAN:
        return False, "not a single can"
    ml = spec_search.volume_ml(str(hit.get("quantity") or ""))
    if ml is None:
        return False, f"quantity {hit.get('quantity')!r} is not a standard can size"
    want_ml = spec_search.volume_ml(getattr(product, "container_volume", "") or "")
    if want_ml and want_ml != ml:
        return False, f"{ml} ml != {want_ml} ml"
    if not hit.get("image_front_url"):
        return False, "no front photo"
    return True, f"{ml} ml"


def find_products(product, limit: int = 40) -> list[dict]:
    q = lucene_query(product)
    if not q:
        return []
    params = urllib.parse.urlencode({"q": q, "page_size": limit, "fields": "code,product_name,brands,quantity,"
                                     "unique_scans_n,image_front_url,lang"})
    data = json.loads(_get(f"{OFF_SEARCH}?{params}"))
    out = []
    for hit in data.get("hits") or []:
        ok, why = product_gate(product, hit)
        if ok:
            out.append({**hit, "volume_ml": spec_search.volume_ml(str(hit.get("quantity"))), "gate": why})
    out.sort(key=lambda h: -int(h.get("unique_scans_n") or 0))
    return out


def full_image_url(url: str) -> str:
    return re.sub(r"\.(\d+)\.jpg$", ".full.jpg", url or "")


# --------------------------------------------------------------------------------------
# Can segmentation + label unwrap
# --------------------------------------------------------------------------------------


def segment(bgr: np.ndarray, rect: tuple[int, int, int, int] | None = None) -> tuple[tuple[int, int, int, int], dict,
                                                                                     np.ndarray]:
    """GrabCut the object (inside `rect` when a detector box is known): bounding box of the can-shaped part of the
    largest foreground blob + silhouette metrics (fill ratio, centre-line tilt, row-width variation) + its mask
    (uint8, box-sized)."""
    h, w = bgr.shape[:2]
    s = min(1.0, 600.0 / max(h, w))
    small = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    sh, sw = small.shape[:2]
    if rect is None:
        r = (int(0.04 * sw), int(0.02 * sh), int(0.92 * sw), int(0.96 * sh))
    else:
        x0, y0, x1, y1 = (int(v * s) for v in rect)
        px, py = int(0.04 * (x1 - x0)) + 2, int(0.03 * (y1 - y0)) + 2
        x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(sw - 1, x1 + px), min(sh - 1, y1 + py)
        r = (x0, y0, max(2, x1 - x0), max(2, y1 - y0))
    mask = np.zeros((sh, sw), np.uint8)
    cv2.grabCut(small, mask, r, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        raise PackagingError("no foreground object")
    fg = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    widths = fg.sum(1)
    rows = np.nonzero(widths)[0]
    med = float(np.median(widths[rows]))
    rows = rows[widths[rows] <= 1.3 * med]  # drop table / hand rows wider than the can
    if len(rows) < 10:
        raise PackagingError("foreground is not can-shaped")
    y0, y1 = int(rows.min()), int(rows.max())
    cols = np.nonzero(fg[y0:y1 + 1].sum(0) > 0.5 * (y1 - y0))[0]
    if not len(cols):
        raise PackagingError("foreground is not can-shaped")
    x0, x1 = int(cols.min()), int(cols.max())
    sub = fg[y0:y1 + 1, x0:x1 + 1]
    mid = sub[int(0.1 * len(sub)):int(0.9 * len(sub)) + 1]
    have = mid.any(1)
    left = np.argmax(mid, 1)[have]
    right = (mid.shape[1] - 1 - np.argmax(mid[:, ::-1], 1))[have]
    ys = np.nonzero(have)[0].astype(np.float64)
    width = (right - left + 1).astype(np.float64)
    slope = np.polyfit(ys, (left + right) / 2.0, 1)[0] if len(ys) > 5 else 0.0
    metrics = {"fill": round(float(sub.mean()), 3), "tilt_deg": round(math.degrees(math.atan(slope)), 2),
               "width_cv": round(float(width.std() / max(width.mean(), 1.0)), 3) if len(width) else 1.0}
    box = (int(x0 / s), int(y0 / s), int((x1 + 1) / s), int((y1 + 1) / s))
    crop_mask = cv2.resize(sub.astype(np.uint8), (max(1, box[2] - box[0]), max(1, box[3] - box[1])),
                           interpolation=cv2.INTER_NEAREST)
    return box, metrics, crop_mask


def can_box(bgr: np.ndarray) -> tuple[tuple[int, int, int, int], dict, np.ndarray]:
    h, w = bgr.shape[:2]
    rect, how = None, "grabcut"
    if _detector is not None:
        try:
            dets = [d for d in _detector(bgr) if d[4] == "beverage_can"]
        except Exception as exc:  # detector is optional; fall back to GrabCut alone
            log.warning("packaging detector failed: %s", exc)
            dets = []
        if dets:
            x0, y0, x1, y1, _ = max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            rect, how = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)), "detector+grabcut"
    box, metrics, mask = segment(bgr, rect)
    return box, {"segmentation": how, **metrics}, mask


def check_photo(bgr: np.ndarray, spec: dict) -> tuple[tuple[int, int, int, int], dict, np.ndarray]:
    """Can box + upright / cylinder-silhouette / proportion / framing checks against the Crown dimensions."""
    h, w = bgr.shape[:2]
    box, report, mask = can_box(bgr)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bh < 0.4 * h or bw < 0.15 * w:
        raise PackagingError(f"can too small in the photo ({bw}x{bh} of {w}x{h})")
    if x0 <= 1 or x1 >= w - 1 or y0 <= 1 or y1 >= h - 1:
        raise PackagingError("can is cut off by the photo border")
    want = spec["height_mm"] / spec["diameter_mm"]
    aspect = bh / max(bw, 1)
    report.update(box=list(box), aspect=round(aspect, 3), expected_aspect=round(want, 3))
    if abs(aspect - want) / want > ASPECT_TOL:
        raise PackagingError(f"box aspect {aspect:.2f} vs can {want:.2f}: not an upright side view")
    if abs(report["tilt_deg"]) > MAX_TILT_DEG:
        raise PackagingError(f"can leans {report['tilt_deg']:.1f} deg")
    if report["fill"] < MIN_FILL or report["width_cv"] > MAX_WIDTH_CV:
        raise PackagingError(f"silhouette is not an upright cylinder (fill {report['fill']}, width cv "
                             f"{report['width_cv']}) - occluded, tilted or perspective-distorted")
    return box, report, mask


def row_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row silhouette left / right edge (float px) of a box-sized can mask, median-smoothed; empty rows take the
    nearest filled row."""
    h, w = mask.shape
    have = mask.any(1)
    if not have.any():
        return np.zeros(h), np.full(h, w - 1.0)
    left = np.where(have, np.argmax(mask, 1), -1).astype(np.float64)
    right = np.where(have, w - 1 - np.argmax(mask[:, ::-1], 1), -1).astype(np.float64)
    idx = np.nonzero(have)[0]
    k = max(3, (h // 60) | 1)
    out = []
    for arr in (left, right):
        arr = np.interp(np.arange(h), idx, arr[idx])
        padded = np.pad(arr, k // 2, mode="edge")
        out.append(np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1))
    return out[0], out[1]


def unwrap_front(bgr: np.ndarray, box: tuple[int, int, int, int], spec: dict,
                 mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """360 deg label texture (BGR, TEX_W wide, v = full can height). The photographed front (|theta| <= VISIBLE_DEG)
    is resampled through the cylinder projection x = sin(theta); the unseen back takes each row's median edge colour."""
    x0, y0, x1, y1 = box
    crop = bgr[y0:y1, x0:x1].astype(np.float32)
    ch, cw = crop.shape[:2]
    tex_h = int(round(TEX_W * spec["height_mm"] / (math.pi * spec["diameter_mm"])))
    rows = cv2.resize(crop, (cw, tex_h), interpolation=cv2.INTER_AREA if ch > tex_h else cv2.INTER_LINEAR)
    if mask is not None and mask.shape == (ch, cw):
        left, right = row_edges(cv2.resize(mask, (cw, tex_h), interpolation=cv2.INTER_NEAREST))
        inset = 0.01 * (right - left)  # stay off the anti-aliased silhouette
        left, right = left + inset, right - inset
    else:
        left, right = np.zeros(tex_h), np.full(tex_h, cw - 1.0)
    theta = (np.arange(TEX_W) + 0.5) / TEX_W * 2 * math.pi - math.pi  # u = 0.5 -> front
    frac = 0.5 + 0.5 * np.sin(np.clip(theta, -math.pi / 2, math.pi / 2))
    map_x = (left[:, None] + frac[None, :] * (right - left)[:, None]).astype(np.float32)
    map_y = np.tile(np.arange(tex_h, dtype=np.float32)[:, None], (1, TEX_W))
    front = cv2.remap(rows, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lim = math.radians(VISIBLE_DEG)
    edge = (np.abs(theta) > math.radians(VISIBLE_DEG - 15)) & (np.abs(theta) <= lim)
    fill = np.median(front[:, edge], axis=1)  # per-row colour of the label edge (bands stay continuous)
    fill = cv2.GaussianBlur(fill.reshape(-1, 1, 3), (1, 2 * (tex_h // 40) + 1), 0).reshape(-1, 3)
    blend = np.clip((np.abs(theta) - (lim - math.radians(BLEND_DEG))) / math.radians(BLEND_DEG), 0, 1)
    tex = front * (1 - blend[None, :, None]) + fill[:, None, :] * blend[None, :, None]
    covered = float((np.abs(theta) <= lim).mean())
    return np.clip(tex, 0, 255).astype(np.uint8), {"photographed_fraction": round(covered, 3),
                                                    "approximation": f"back {100 - 100 * covered:.0f}% of the label "
                                                                     "not photographed: filled with the label's per-row "
                                                                     "edge colour"}


def label_from_photo(jpeg: bytes, spec: dict) -> tuple[bytes, dict]:
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise PackagingError("photo is not a decodable image")
    box, report, mask = check_photo(bgr, spec)
    tex, info = unwrap_front(bgr, box, spec, mask)
    ok, enc = cv2.imencode(".jpg", tex, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise PackagingError("texture encode failed")
    return enc.tobytes(), {**report, **info}


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def build(product, extras: dict | None = None, hits: list[dict] | None = None) -> tuple[dict, bytes] | None:
    """(attribution, GLB) for the first OFF product whose front photo passes check_photo, else None."""
    hits = find_products(product) if hits is None else hits
    for hit in hits[:PHOTO_TRIES]:
        spec = spec_search.can_specs(getattr(product, "raw_query", ""), f"{hit['volume_ml']}ml")
        if not spec.get("verified"):
            continue
        url = full_image_url(hit["image_front_url"])
        try:
            jpeg = _get(url, MAX_IMAGE_BYTES, "image/jpeg")
            label, report = label_from_photo(jpeg, spec)
        except (urllib.error.URLError, OSError, ValueError, PackagingError, cv2.error) as exc:
            log.info("OFF %s photo rejected: %s", hit.get("code"), exc)
            continue
        page = OFF_PAGE.format(code=hit["code"])
        meta = {"off_code": hit["code"], "off_photo": url, "label": report, "spec_sources": spec["sources"],
                **(extras or {})}
        glb = procedural_glb.build_from_spec(spec, getattr(product, "raw_query", ""), meta, label=label)
        attr = {
            "title": f"{hit.get('product_name') or product.canonical_name} ({hit['volume_ml']} ml, packaging photo)",
            "author": "Open Food Facts contributors", "author_url": page, "license": OFF_LICENSE,
            "license_url": OFF_LICENSE_URL, "source_url": page, "thumbnail_url": hit["image_front_url"],
            "attribution_text": (f"Label photo: Open Food Facts contributors, {page} ({OFF_LICENSE}); "
                                 f"geometry: Kamikaze spec generator from {spec['sources'][0]['publisher']} "
                                 f"{spec['diameter_mm']} x {spec['height_mm']} mm ({spec['sources'][0]['url']})"),
            "modified": True, "off_code": hit["code"], "label_report": report, "spec": spec,
            "approximations": spec.get("approximations", []) + [report["approximation"]],
        }
        return attr, glb
    return None
