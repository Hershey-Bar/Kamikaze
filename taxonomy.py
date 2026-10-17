"""Product taxonomy for Kamikaze: metadata parsing, enrollment views, phone structure rules, CAD queries.

Pure Python (no torch / network), so it is importable and testable anywhere.
"""

from __future__ import annotations

import re

CATEGORIES = ("beverage_can", "beverage_bottle", "smartphone")
VIEWS = ("front", "back", "side", "detail")
VIEW_YAW = {"front": 0.0, "side": 90.0, "back": 180.0}

VIEW_STEPS = {
    "beverage_can": {
        "front": ("Front (0°)", "Face the main label / branding toward the camera."),
        "back": ("Back (180°)", "Rotate the can to show the barcode and nutrition facts."),
        "side": ("Side (90°)", "Turn the can a quarter-turn to show its profile and seam."),
        "detail": ("Top / detail", "Tilt the can toward the camera to show the pull-tab."),
    },
    "beverage_bottle": {
        "front": ("Front (0°)", "Face the main label toward the camera."),
        "back": ("Back (180°)", "Rotate the bottle to show the barcode and nutrition facts."),
        "side": ("Side (90°)", "Turn the bottle a quarter-turn to show its profile."),
        "detail": ("Cap / detail", "Hold the bottle so the cap and neck fill the box."),
    },
    "smartphone": {
        "front": ("Front (0°)", "Show the screen side facing the camera."),
        "back": ("Back (180°)", "Flip the phone to show the rear shell and camera island."),
        "side": ("Side (90°)", "Show the phone edge-on to expose the button rails and frame."),
        "detail": ("Camera island", "Bring the rear camera island close, filling the box."),
    },
}

MSG_PHONE_FLIP = "Phone detected. Flip to the rear side to reveal the camera island and branding."
MSG_PHONE_TILT = "Tilt phone 30° toward light to expose lens count and LiDAR sensor."

_BRANDS = (
    (re.compile(r"coca[\s\-]?cola|\bcoke\b"), "Coca-Cola", "beverage_can"),
    (re.compile(r"pepsi"), "Pepsi", "beverage_can"),
    (re.compile(r"sprite"), "Sprite", "beverage_can"),
    (re.compile(r"red[\s\-]?bull"), "Red Bull", "beverage_can"),
    (re.compile(r"iphone|\bapple\b"), "Apple", "smartphone"),
    (re.compile(r"galaxy|samsung|\bs\s?2[0-5]\b"), "Samsung", "smartphone"),
)
_VOLUME_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ml|cl|l|oz|fl\s*oz)\b")
_IPHONE_RE = re.compile(r"iphone\s*(\d{1,2})\s*(pro\s*max|pro|plus|mini)?")
_GALAXY_RE = re.compile(r"(?:galaxy\s*)?\bs\s?(2[0-5])\s*(ultra|plus|\+|fe)?")


class MetadataError(ValueError):
    pass


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm_volume(text: str) -> str:
    m = _VOLUME_RE.search(text.lower())
    if not m:
        return ""
    number = m.group(1).replace(",", ".")
    if number.endswith(".0"):
        number = number[:-2]
    unit = "".join(m.group(2).split())
    return number + unit


def _phone_model(brand: str, text: str) -> tuple[str, str]:
    """Return (model name, tier-variant) such as ("iPhone 15 Pro Max", "pro_max")."""
    low = text.lower()
    if brand == "Apple" or "iphone" in low:
        m = _IPHONE_RE.search(low)
        if m:
            suffix = re.sub(r"\s+", "_", (m.group(2) or "").strip())
            name = f"iPhone {m.group(1)}" + (f" {m.group(2).title()}" if m.group(2) else "")
            return re.sub(r"\s+", " ", name), suffix or "standard"
    m = _GALAXY_RE.search(low)
    if m:
        suffix = {"+": "plus"}.get(m.group(2) or "", m.group(2) or "")
        name = f"Galaxy S{m.group(1)}" + (f" {suffix.title() if suffix != 'fe' else 'FE'}" if suffix else "")
        return name, suffix or "standard"
    return "", ""


def parse_metadata(req: dict) -> dict:
    """Validate and complete the enrollment metadata dict (explicit fields win over inference)."""
    if not isinstance(req, dict):
        raise MetadataError("metadata must be a JSON object")
    raw = _clean(req.get("raw_name"))
    brand = _clean(req.get("brand"))
    variant = _clean(req.get("variant")).lower()
    if not raw and not brand:
        raise MetadataError("raw_name (or brand) is required")
    haystack = f"{raw} {brand} {variant}".lower()

    inferred_cat = None
    for pattern, canon, default_cat in _BRANDS:
        if pattern.search(haystack):
            brand = brand or canon
            inferred_cat = default_cat
            break
    category = _clean(req.get("category")).lower()
    if not category:
        if re.search(r"\bbottle\b|\bpet\b", haystack):
            category = "beverage_bottle"
        elif re.search(r"\bcan\b|\btin\b", haystack):
            category = "beverage_can"
        elif re.search(r"phone|iphone|galaxy", haystack):
            category = "smartphone"
        else:
            category = inferred_cat or ""
    if category not in CATEGORIES:
        raise MetadataError(f"category must be one of {', '.join(CATEGORIES)}")

    package = _clean(req.get("package_type")).lower()
    volume = _norm_volume(_clean(req.get("container_volume")) or raw)
    if category == "smartphone":
        model, tier = _phone_model(brand, haystack)
        package = package or "phone"
        variant = variant or tier or "standard"
        volume = ""
        if model and not re.search(r"\d", raw):
            raw = f"{brand} {model}"
    else:
        package = package or ("bottle" if category == "beverage_bottle" else "can")
        if not variant:
            if re.search(r"zero|no sugar|sugar[\s\-]?free", haystack):
                variant = "zero"
            elif re.search(r"\bdiet\b|\blight\b", haystack):
                variant = "diet"
            else:
                variant = "original"
    if not brand:
        raise MetadataError("could not infer brand - pass it explicitly")
    return {
        "raw_name": raw or f"{brand} {variant}".strip(),
        "category": category,
        "brand": brand,
        "variant": variant,
        "package_type": package,
        "container_volume": volume,
    }


def phone_model_name(meta: dict) -> str:
    """Human model name for a phone entry, e.g. "Samsung Galaxy S24 Ultra"."""
    model, _ = _phone_model(meta.get("brand", ""), f"{meta.get('raw_name', '')} {meta.get('variant', '')}")
    if not model:
        return meta.get("raw_name", "smartphone")
    return f"{meta.get('brand', '')} {model}".strip()


def label_from_metadata(meta: dict) -> str:
    """Stable bank label (a-z0-9_, max 40 chars) derived from metadata."""
    if meta.get("category") == "smartphone":
        model, _ = _phone_model(meta.get("brand", ""), f"{meta.get('raw_name', '')} {meta.get('variant', '')}")
        parts = [model or meta.get("raw_name", "phone")]
    else:
        variant = meta.get("variant", "")
        parts = [meta.get("brand", ""), "" if variant == "original" else variant, meta.get("container_volume", ""),
                 meta.get("package_type", "")]
    label = re.sub(r"[^a-z0-9]+", "_", " ".join(p for p in parts if p).lower()).strip("_")
    return (label or "object")[:40].rstrip("_")


def display_name(meta: dict) -> str:
    if meta.get("category") == "smartphone":
        return phone_model_name(meta)
    bits = [meta.get("brand", "")]
    if meta.get("variant") and meta["variant"] != "original":
        bits.append(meta["variant"].title())
    bits += [meta.get("container_volume", ""), meta.get("package_type", "")]
    return " ".join(b for b in bits if b)


def phone_profile(meta: dict) -> dict:
    """Structural cues that separate standard / Pro (Apple) and standard / Ultra (Samsung) phones."""
    brand = meta.get("brand", "")
    variant = (meta.get("variant") or "standard").lower()
    if brand == "Apple":
        pro = "pro" in variant
        return {
            "maker": "Apple", "tier": "pro" if pro else "standard", "lenses": 3 if pro else 2, "lidar": pro,
            "cues": (["3 large lenses in a triangle", "black LiDAR dot", "matte textured back", "titanium/steel rails"]
                     if pro else ["2 diagonal or vertical lenses", "glossy rear glass", "aluminium frame"]),
        }
    ultra = "ultra" in variant
    return {
        "maker": brand or "Samsung", "tier": "ultra" if ultra else "standard", "lenses": 4 if ultra else 3, "lidar": False,
        "cues": (["boxy sharp corners", "4+ lenses with periscope telephoto", "laser autofocus sensor", "S-Pen silhouette"]
                 if ultra else ["rounded corners", "3 vertical circular lenses", "compact width"]),
    }


def search_queries(meta: dict) -> list[str]:
    """Ordered CAD search queries, most specific first."""
    cat = meta.get("category")
    brand = meta.get("brand", "")
    queries: list[str] = []
    if cat == "smartphone":
        name = phone_model_name(meta)
        profile = phone_profile(meta)
        queries += [f"{name} 3D model glb", f"{name}", f"{brand} smartphone {profile['lenses']} cameras glb", "smartphone 3D model glb"]
    elif cat == "beverage_bottle":
        queries += [f"{brand} {meta.get('variant', '')} bottle glb".replace("  ", " "), f"{brand} bottle 3D model", "plastic bottle glb"]
    else:
        variant = meta.get("variant", "")
        v = "" if variant in ("", "original") else f" {variant}"
        queries += [f"{brand}{v} aluminium can glb", f"{brand} can 3D model", "soda can glb"]
    seen, out = set(), []
    for q in queries:
        q = _clean(q)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def view_step(category: str, view: str) -> tuple[str, str]:
    return VIEW_STEPS.get(category, VIEW_STEPS["beverage_can"])[view]
