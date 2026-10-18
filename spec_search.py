"""Verified physical dimensions for dimension-calibrated GLB generation (see procedural_glb.build_*_spec).

Nothing here guesses. A spec is returned with verified=True only when every number comes from a cited source:

  smartphones   GSMArena product page (height x width x depth in mm, rear camera module count). The phone is located
                through GSMArena's own quicksearch index (the search page sits behind a Cloudflare challenge); the maker
                must match the brand, so "S25 Ultra" never resolves to another vendor's phone of the same name.
  beverage cans Crown Holdings beverage-can specifications (overall diameter x height in mm per volume and profile).
                The fill volume must be known (from the class name / metadata, e.g. "coke zero 330ml"); without it the
                result is verified=False and nothing is generated.

Approximations are flagged explicitly in the returned dict ("approximations") and carried into the GLB sidecar.
"""

from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request

log = logging.getLogger("kamikaze.specs")

GSMARENA = "https://www.gsmarena.com"
BROWSER_UA = ("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")
TIMEOUT_S = 12.0
INDEX_TTL_S = 6 * 3600.0
CROWN_URL = "https://www.crowncork.com/beverage-packaging/products/beverage-cans/{slug}"

# (fill ml, profile) -> (overall diameter mm, overall height mm, Crown spec page slug)
CROWN_CANS: dict[tuple[int, str], tuple[float, float, str]] = {
    (200, "slim"): (53.0, 111.2, "67oz-200ml-slim"),
    (237, "sleek"): (58.0, 88.05, "8oz-237-ml-crownsleek"),
    (240, "slim"): (53.0, 129.5, "81oz-240ml-slim"),
    (250, "slim"): (53.0, 133.0, "83oz-250ml-slim"),
    (250, "standard"): (66.0, 91.69, "83oz-250ml-standard"),
    (300, "standard"): (66.0, 106.3, "101oz-300ml-standard"),
    (310, "sleek"): (58.0, 138.20, "104oz-310ml-crownsleek"),
    (330, "standard"): (66.0, 115.2, "113oz-330ml-standard"),
    (330, "sleek"): (58.0, 145.4, "113oz-330ml-crownsleek"),
    (355, "standard"): (66.0, 122.22, "12oz-355ml-standard"),
    (355, "sleek"): (58.0, 156.59, "12oz-355ml-crownsleek"),
    (440, "standard"): (66.0, 149.9, "149oz-440ml-standard"),
    (473, "standard"): (66.0, 157.23, "16oz-473ml-standard"),
    (500, "standard"): (66.0, 168.0, "169oz-500ml-standard"),
    (550, "standard"): (66.0, 182.5, "186oz-550ml-standard"),
}
DEFAULT_PROFILE = {250: "slim", 310: "sleek"}
# lid ("end") diameters by body diameter: 202 end ~ 54 mm on 66 mm bodies, 200 end ~ 50.8 mm on 53/58 mm bodies
NECK_DIAMETER = {66.0: 54.0, 58.0: 50.8, 53.0: 50.8}
NECK_HEIGHT_MM = 10.0
ML_PER_OZ = 29.5735
_VOLUME_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ml|cl|l|fl\s*oz|oz)\b", re.I)
_PROFILE_RE = re.compile(r"\b(slim|sleek|standard)\b", re.I)

_index_lock = threading.Lock()
_index: tuple[float, dict, list] | None = None


class SpecError(RuntimeError):
    pass


def _get(url: str, timeout: float = TIMEOUT_S) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA, "Accept": "text/html,application/json,*/*",
                                               "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --------------------------------------------------------------------------------------
# Smartphones (GSMArena)
# --------------------------------------------------------------------------------------


def _quicksearch_index() -> tuple[dict, list]:
    """GSMArena's quicksearch JSON: ({maker_id: maker name}, [[maker_id, phone_id, name, keywords, ...], ...])."""
    global _index
    with _index_lock:
        if _index and time.monotonic() - _index[0] < INDEX_TTL_S:
            return _index[1], _index[2]
        home = _get(f"{GSMARENA}/").decode("utf-8", "replace")
        m = re.search(r"quicksearch-(\d+)\.jpg", home)
        if not m:
            raise SpecError("GSMArena quicksearch index not referenced on the home page")
        data = json.loads(_get(f"{GSMARENA}/quicksearch-{m.group(1)}.jpg", timeout=TIMEOUT_S * 2).decode("utf-8", "replace"))
        if not (isinstance(data, list) and len(data) >= 2 and isinstance(data[0], dict) and isinstance(data[1], list)):
            raise SpecError("unexpected GSMArena quicksearch format")
        makers = {int(k): str(v) for k, v in data[0].items()}
        _index = (time.monotonic(), makers, data[1])
        return makers, data[1]


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9+]+", " ", text.lower().replace("plus", "+"))
    return re.sub(r"\s+", " ", re.sub(r"\s*\+", "+", text)).strip()


def find_phone(brand: str, model: str, index: tuple[dict, list] | None = None) -> tuple[str, int, str] | None:
    """(maker, gsmarena phone id, exact name) for brand + model ("Samsung", "Galaxy S25 Ultra"); None if not listed."""
    makers, phones = index or _quicksearch_index()
    want_maker = brand.strip().lower()
    want = _norm(re.sub(rf"^{re.escape(brand)}\s+", "", model.strip(), flags=re.I))
    for entry in phones:
        if not isinstance(entry, list) or len(entry) < 3:
            continue
        maker = makers.get(int(entry[0]), "")
        if maker.lower() != want_maker:
            continue
        if _norm(str(entry[2])) == want:
            return maker, int(entry[1]), str(entry[2])
    return None


def phone_page_url(maker: str, phone_id: int, name: str) -> str:
    slug = re.sub(r"[^a-z0-9+]+", "_", f"{maker} {name}".lower()).strip("_")
    return f"{GSMARENA}/{slug}-{phone_id}.php"


def parse_phone_page(page: str) -> dict:
    """Dimensions (mm) and rear camera module count from a GSMArena spec page."""
    m = re.search(r'data-spec="dimensions">\s*([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)\s*mm', page)
    if not m:
        raise SpecError("dimensions not found on the GSMArena page")
    height, width, depth = (float(v) for v in m.groups())
    lenses = 0
    cam = re.search(r'data-spec="cam1modules">(.*?)</td>', page, re.S)
    if cam:
        lines = [html.unescape(re.sub(r"<[^>]+>", "", ln)).strip() for ln in re.split(r"<br\s*/?>", cam.group(1))]
        lenses = sum(1 for ln in lines if re.search(r"\bMP\b", ln))
    weight = re.search(r'data-spec="weight">\s*([\d.]+)\s*g', page)
    return {"height_mm": height, "width_mm": width, "depth_mm": depth, "rear_cameras": lenses,
            "weight_g": float(weight.group(1)) if weight else None}


def phone_specs(brand: str, model: str) -> dict:
    if brand not in ("Apple", "Samsung") or not model:
        return {"verified": False, "category": "smartphone", "reason": f"no verified spec source for {brand} {model}".strip()}
    try:
        hit = find_phone(brand, model)
    except (urllib.error.URLError, OSError, ValueError, SpecError) as exc:
        return {"verified": False, "category": "smartphone", "reason": f"GSMArena index unavailable: {exc}"}
    if not hit:
        return {"verified": False, "category": "smartphone", "reason": f"{brand} {model} is not listed on GSMArena"}
    maker, pid, name = hit
    url = phone_page_url(maker, pid, name)
    try:
        spec = parse_phone_page(_get(url).decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, SpecError) as exc:
        return {"verified": False, "category": "smartphone", "reason": f"GSMArena page {url} unreadable: {exc}"}
    if not spec["rear_cameras"]:
        return {"verified": False, "category": "smartphone", "reason": f"rear camera count missing on {url}"}
    return {
        "verified": True, "category": "smartphone", "maker": maker, "name": f"{maker} {name}", **spec,
        "sources": [{"what": "dimensions, rear camera modules", "url": url, "publisher": "GSMArena"}],
        "approximations": ["corner radius = 11% of width (not published)",
                           "camera island layout follows the maker's generation design language; lens count is exact"],
    }


# --------------------------------------------------------------------------------------
# Beverage cans (Crown Holdings)
# --------------------------------------------------------------------------------------


def volume_ml(text: str) -> int | None:
    """Fill volume snapped to a Crown table size (within 3 %): "330ml" -> 330, "12 oz" -> 355, "0.5l" -> 500."""
    m = _VOLUME_RE.search(text or "")
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    unit = re.sub(r"\s+", "", m.group(2).lower())
    ml = value * {"ml": 1.0, "cl": 10.0, "l": 1000.0, "oz": ML_PER_OZ, "floz": ML_PER_OZ}[unit]
    sizes = sorted({v for v, _ in CROWN_CANS})
    best = min(sizes, key=lambda s: abs(s - ml))
    return best if abs(best - ml) <= 0.03 * best else None


def can_specs(text: str, volume: str = "", profile: str | None = None) -> dict:
    ml = volume_ml(volume) or volume_ml(text)
    if ml is None:
        return {"verified": False, "category": "beverage_can",
                "reason": "can volume unknown - add it to the class name (e.g. '330ml' or '12oz') so the Crown "
                          "manufacturer dimensions can be used"}
    wanted = (profile or (_PROFILE_RE.search(text or "") or [None, None])[1] or "").lower() or None
    assumed = wanted is None
    prof = wanted or DEFAULT_PROFILE.get(ml, "standard")
    if (ml, prof) not in CROWN_CANS:
        options = sorted(p for v, p in CROWN_CANS if v == ml)
        return {"verified": False, "category": "beverage_can",
                "reason": f"no {prof} {ml} ml can in the Crown range (available: {', '.join(options)})"}
    diameter, height, slug = CROWN_CANS[(ml, prof)]
    return {
        "verified": True, "category": "beverage_can", "volume_ml": ml, "profile": prof, "profile_assumed": assumed,
        "diameter_mm": diameter, "height_mm": height, "neck_diameter_mm": NECK_DIAMETER[diameter],
        "neck_height_mm": NECK_HEIGHT_MM,
        "sources": [{"what": "overall diameter and height", "url": CROWN_URL.format(slug=slug), "publisher": "Crown Holdings"}],
        "approximations": [f"lid/neck diameter {NECK_DIAMETER[diameter]} mm from the standard end size",
                           f"neck taper height {NECK_HEIGHT_MM} mm"]
                          + ([f"{prof} profile assumed (most common for {ml} ml); add 'slim'/'sleek'/'standard' to override"]
                             if assumed else []),
    }


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def product_specs(product, profile: str | None = None) -> dict:
    """Verified dimensions for an entity_resolver.CanonicalProduct (or anything with the same attributes)."""
    category = getattr(product, "category", "")
    if category == "smartphone":
        return phone_specs(getattr(product, "brand", ""), getattr(product, "model_or_variant", ""))
    if category == "beverage_can":
        return can_specs(getattr(product, "raw_query", ""), getattr(product, "container_volume", ""), profile)
    return {"verified": False, "category": category, "reason": f"no verified spec source for category {category!r}"}


def search(query: str, category: str = "") -> dict:
    """Free-text lookup ("galaxy s25 ultra", "coke zero 330ml can")."""
    import entity_resolver
    import taxonomy

    try:
        product = entity_resolver.resolve(query, category=category)
    except taxonomy.MetadataError as exc:
        return {"verified": False, "category": category, "reason": str(exc)}
    return {"query": query, "product": product.canonical_name, **product_specs(product)}
