"""Product taxonomy for Kamikaze: metadata parsing, enrollment views, phone structure rules, CAD queries.

Pure Python (no torch / network), so it is importable and testable anywhere.
"""

from __future__ import annotations

import re

CATEGORIES = ("beverage_can", "beverage_bottle", "smartphone")
# Every view key a class can hold. Four cardinal yaw views (front 0, right 90, back 180, left 270) make the yaw circle
# unambiguous and "top" (tilted 35-45 deg) anchors pitch. Which of them the guided wizard captures, in which order and
# with which prompt is category-specific (WIZARD_VIEWS / view_step): a can is rotationally near-symmetric, so its
# "side seam" step is stored as the +90 "right" view and it has no separate left view.
VIEWS = ("front", "right", "back", "left", "top")
VIEW_YAW = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}
LEGACY_VIEW_ALIASES = {"detail": "top", "side": "right"}

WIZARD_VIEWS = {
    "smartphone": ("front", "back", "right", "left", "top"),
    "beverage_can": ("front", "back", "right", "top"),
    "beverage_bottle": ("front", "back", "right", "top"),
}

# (title, prompt, what the step must capture)
VIEW_STEPS = {
    "beverage_can": {
        "front": ("Front branding (0°)", "Position the main brand logo directly facing the camera, centered and upright.",
                  "Primary brand graphics and flavour name."),
        "back": ("Rear panel - nutrition & barcode (180°)", "Rotate the can 180°. Show the nutrition facts grid, barcode and manufacturer text.",
                 "Ingredients table, barcode, regulatory markings."),
        "right": ("Side seam (90°)", "Turn the can 90°. Show the lateral seam, volume markings and side artwork.",
                  "Cylindrical curvature, side packaging text."),
        "left": ("Other side (270°)", "Turn the can 90° the other way to show the remaining side artwork.",
                 "Opposite side artwork."),
        "top": ("Top rim & pull-tab (tilt 35-45°)",
                "Tilt the top of the can 35-45° toward the camera to reveal the pull-tab and lid opening.",
                "Pull-tab colour / style, stay-on tab geometry, aluminium rim."),
    },
    "beverage_bottle": {
        "front": ("Front label (0°)", "Face the main label directly toward the camera, centered and upright.",
                  "Brand label and flavour name."),
        "back": ("Rear label (180°)", "Rotate the bottle 180°. Show the nutrition facts and barcode.",
                 "Ingredients table, barcode."),
        "right": ("Side (90°)", "Turn the bottle 90° to show its side profile and label edge.", "Bottle profile."),
        "left": ("Other side (270°)", "Turn the bottle 90° the other way.", "Opposite side."),
        "top": ("Cap & neck (tilt 35-45°)", "Tilt the bottle 35-45° so the cap and neck face the camera.",
                "Cap colour and shape."),
    },
    "smartphone": {
        "front": ("Front face (display & selfie camera)", "Face the screen directly toward the camera, centered and upright.",
                  "Screen surface, bezels, Dynamic Island / punch-hole camera."),
        "back": ("Rear back shell (180°)", "Rotate the phone 180°. Show the rear glass, camera bump, lenses and flash.",
                 "Lens cluster, LiDAR / sensors, rear logo."),
        "right": ("Right side profile (90°)", "Turn the phone 90° so its right edge faces the camera.",
                  "Button placement, rail curvature, frame material."),
        "left": ("Left side profile (270°)", "Turn the phone 90° the other way so its left edge faces the camera.",
                 "Left-rail buttons, antenna bands."),
        "top": ("Top / camera cluster (tilt 35-45°)",
                "Tilt the phone's top edge 35-45° toward the camera to expose lens elevation and depth.",
                "Periscope cut-out, lens ring height, 3D contours."),
    },
}
# Where the physical buttons really are, so the rail prompts name a feature the user can see instead of a rotation
# direction ("clockwise" is ambiguous depending on whether you look from above or from the camera).
PHONE_RAILS = {
    "Apple": {"right": ("with the side (power) button", "Side button, rail curvature, frame material."),
              "left": ("with the volume buttons and Action button / mute switch",
                       "Volume buttons, Action button / mute switch, antenna bands.")},
    "Samsung": {"right": ("with the power and volume keys", "Power + volume keys, rail curvature, frame material."),
                "left": ("(no buttons - antenna bands only)", "Plain rail, antenna bands, frame material.")},
}

MSG_PHONE_FLIP = "Screen detected. Flip phone 180° to expose camera island and sensors."
MSG_PHONE_TILT = "Tilt phone 30° toward light to expose lens count and LiDAR sensor."

_BRANDS = (
    (re.compile(r"coca[\s\-]?cola|\bcoke\b"), "Coca-Cola", "beverage_can"),
    (re.compile(r"pepsi"), "Pepsi", "beverage_can"),
    (re.compile(r"sprite"), "Sprite", "beverage_can"),
    (re.compile(r"red[\s\-]?bull"), "Red Bull", "beverage_can"),
    (re.compile(r"iphone|\bapple\b"), "Apple", "smartphone"),
    (re.compile(r"galaxy|samsung|\bs\s?2[0-5]\b"), "Samsung", "smartphone"),
)
KNOWN_BRANDS = tuple(canon for _, canon, _ in _BRANDS)
# Flavours / drink types let unknown store brands ("Kirkland Grapefruit", "La Croix lime") resolve to a can product
# without ever substituting a known brand. "apple" is deliberately absent (it is the phone brand).
FLAVORS = (
    "pink grapefruit", "grapefruit", "blood orange", "key lime", "lemon lime", "lemon-lime", "passion fruit",
    "passionfruit", "black cherry", "wild berry", "mixed berry", "lime", "lemon", "orange", "tangerine", "mandarin",
    "citrus", "mango", "cherry", "peach", "raspberry", "strawberry", "blackberry", "blueberry", "cranberry", "berry",
    "pineapple", "watermelon", "melon", "coconut", "grape", "pomegranate", "pear", "kiwi", "guava", "lychee", "vanilla",
    "ginger", "cucumber", "hibiscus",
)
DRINK_TYPES = {
    "sparkling water": "sparkling water", "sparkling": "sparkling water", "seltzer": "seltzer",
    "mineral water": "mineral water", "tonic water": "tonic water", "tonic": "tonic water",
    "energy drink": "energy drink", "iced tea": "iced tea", "ice tea": "iced tea", "tea": "iced tea",
    "lemonade": "lemonade", "kombucha": "kombucha", "cold brew": "cold brew coffee", "coffee": "cold brew coffee",
    "juice": "juice", "soda": "soda", "pop": "soda", "cola": "cola", "water": "water",
}
SIGNATURE_FLAVOR = {"Sprite": "lemon lime"}
_FLAVOR_RE = re.compile(r"\b(" + "|".join(re.escape(f) for f in sorted(FLAVORS, key=len, reverse=True)) + r")\b")
_DRINK_RE = re.compile(r"\b(" + "|".join(re.escape(d) for d in sorted(DRINK_TYPES, key=len, reverse=True)) + r")\b")
_PACKAGE_WORDS = {"can", "cans", "tin", "bottle", "pet", "aluminum", "aluminium", "flavor", "flavored", "flavour",
                  "flavoured", "drink", "beverage", "zero", "diet", "light", "sugar", "free", "no", "original", "taste"}
_DESCRIPTOR_WORDS = _PACKAGE_WORDS | {w for f in FLAVORS for w in re.split(r"[\s\-]", f)} | {
    w for d in DRINK_TYPES for w in d.split()}
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


def find_flavor(text: str) -> str:
    """Canonical flavour named in `text` ("" when none), e.g. "Kirkland Grapefruit" -> "grapefruit"."""
    m = _FLAVOR_RE.search((text or "").lower())
    return m.group(1).replace("-", " ") if m else ""


def find_drink_type(text: str) -> str:
    """Canonical drink type named in `text` ("" when none), e.g. "LaCroix seltzer" -> "seltzer"."""
    m = _DRINK_RE.search((text or "").lower())
    return DRINK_TYPES[m.group(1)] if m else ""


def product_flavor(meta: dict) -> str:
    """Flavour of a beverage entry that is not already implied by its brand (Sprite is lemon-lime by definition)."""
    if meta.get("category") == "smartphone":
        return ""
    flavor = find_flavor(f"{meta.get('raw_name', '')} {meta.get('variant', '')}")
    return "" if SIGNATURE_FLAVOR.get(meta.get("brand", "")) == flavor else flavor


def _leading_brand(raw: str) -> str:
    """Words before the first flavour / drink-type / package / volume word: "Kirkland Grapefruit" -> "Kirkland"."""
    words = []
    for word in raw.split():
        key = re.sub(r"[^a-z0-9\-]", "", word.lower())
        if not key or key in _DESCRIPTOR_WORDS or re.match(r"\d", key) or any(p in _DESCRIPTOR_WORDS for p in key.split("-")):
            break
        words.append(word)
    brand = " ".join(words)
    return brand.title() if brand == brand.lower() else brand


def _descriptor(raw: str, brand: str) -> str:
    """The product line after the brand, minus package / volume words: "Kirkland Sparkling Water Grapefruit Can"
    -> "sparkling water grapefruit"."""
    rest = raw[len(brand):] if raw.lower().startswith(brand.lower()) else raw
    rest = _VOLUME_RE.sub(" ", rest.lower())
    words = [w for w in re.split(r"[^a-z0-9\-]+", rest) if w and w not in _PACKAGE_WORDS]
    return " ".join(words)


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
            category = inferred_cat or ("beverage_can" if find_flavor(haystack) or find_drink_type(haystack) else "")
    if category not in CATEGORIES:
        raise MetadataError(f"category must be one of {', '.join(CATEGORIES)}")
    if not brand and category != "smartphone":
        brand = _leading_brand(raw)

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
            if re.search(r"zero|no sugar|sugar[\s\-]?free|pepsi\s*max", haystack):
                variant = "zero"
            elif re.search(r"\bdiet\b|\blight\b", haystack):
                variant = "diet"
            elif brand and brand not in KNOWN_BRANDS:
                variant = _descriptor(raw, brand) or "original"
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
        flavor = product_flavor(meta)
        parts = [meta.get("brand", ""), "" if variant == "original" else variant,
                 flavor if flavor and flavor not in variant else "", meta.get("container_volume", ""),
                 meta.get("package_type", "")]
    label = re.sub(r"[^a-z0-9]+", "_", " ".join(p for p in parts if p).lower()).strip("_")
    return (label or "object")[:40].rstrip("_")


def display_name(meta: dict) -> str:
    if meta.get("category") == "smartphone":
        return phone_model_name(meta)
    bits = [meta.get("brand", "")]
    variant = meta.get("variant") or ""
    if variant and variant != "original":
        bits.append(variant.title())
    flavor = product_flavor(meta)
    if flavor and flavor not in variant:
        bits.append(flavor.title())
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


def canonical_view(name: str) -> str:
    """Map a legacy view key (``detail``) onto the current one (``top``)."""
    return LEGACY_VIEW_ALIASES.get(name, name)


def wizard_views(category: str | None) -> tuple[str, ...]:
    """Ordered view keys the guided enrollment captures for a category."""
    return WIZARD_VIEWS.get(category or "", WIZARD_VIEWS["beverage_can"])


def view_step(category: str, view: str, brand: str = "") -> tuple[str, str]:
    """(title, prompt) for one wizard step; phone rail prompts name the brand's real buttons."""
    title, prompt, _ = view_detail(category, view, brand)
    return title, prompt


def view_detail(category: str, view: str, brand: str = "") -> tuple[str, str, str]:
    """(title, prompt, capture target) for one wizard step."""
    title, prompt, target = VIEW_STEPS.get(category, VIEW_STEPS["beverage_can"])[view]
    rails = PHONE_RAILS.get(brand) if category == "smartphone" else None
    if rails and view in rails:
        desc, target = rails[view]
        prompt = f"Turn the phone 90° so its {view} edge {desc} faces the camera."
    return title, prompt, target
