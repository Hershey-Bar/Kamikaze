"""CAD harvester: find, quality-check, license-check, download and cache a 3D GLB for a canonical product, with author
attribution. It never fabricates geometry: when no verified model exists the result is an explicit NOT_FOUND.

    harvest(resolve("samsung galaxy s24 ultra"))  ->  static/models/galaxy_s24_ultra.glb + galaxy_s24_ultra.json

Search queries come from build_search_queries()/search_plan(): colloquial text is resolved to a canonical product and
expanded into exact -> similar -> category queries ("s25 ultra" -> "Samsung Galaxy S25 Ultra" ... "Galaxy S24 Ultra" ...
"Smartphone").

Automatic harvest (harvest / retrieve) - exact-product level only:
  Sketchfab Data API (SKETCHFAB_TOKEN) + Poly Pizza API (POLY_PIZZA_API_KEY, header X-Auth-Token) are pooled;
  candidates are relevance-filtered (exact model + tier + flavour, no competing brand), licence-filtered (CC0 / CC-BY /
  CC-BY-SA / CC-BY-ND / Sketchfab Standard; NonCommercial rejected), single-object and face-count capped, then ranked by
      relevance + licence + popularity - mesh weight,   popularity = 2.5*ln(1+likes) + 0.75*ln(1+views) + 4*staff_pick
  (capped at 25). Each download must pass validate_glb(): binary glTF 2.0, MIN_VERTICES..MAX_VERTICES unique vertices,
  <= MAX_GLB_MB and at least one material - primitive placeholders and untextured meshes are deleted and the next
  candidate is tried. If nothing passes, the result is status "not_found" (no substitute model is shown).

User-driven paths (never automatic):
  candidates()          - top ranked models across exact / similar / category levels for a picker UI (no download);
  select()              - download one picked candidate;
  upload_glb()          - install an operator-supplied GLB (same quality gate);
  generate_from_specs() - dimension-calibrated GLB built only from verified published specs (spec_search.py), badged
                          "Estimated Model: Dimensionally Calibrated via Spec Search".

Every cached model gets a sidecar static/models/<class_id>.json holding the attribution (title, author, author URL,
licence, licence URL, source URL, match level, query). credits() / write_credits_md() aggregate them (served at
GET /api/credits and written to static/models/CREDITS.md).
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import entity_resolver
import glb_orient
import mcp_server as mcp
import procedural_glb
import spec_search
import taxonomy

log = logging.getLogger("kamikaze.cad")

PROJECT_URL = "https://github.com/Hershey-Bar/Kamikaze"
BUDGET_S = float(os.environ.get("KAMIKAZE_CAD_BUDGET", "40"))
SEARCH_TIMEOUT_S = float(os.environ.get("KAMIKAZE_CAD_SEARCH_TIMEOUT", "10"))
MAX_FACES = int(os.environ.get("KAMIKAZE_CAD_MAX_FACES", "150000"))
MAX_GLB_MB = float(os.environ.get("KAMIKAZE_CAD_MAX_MB", "35"))
MIN_VERTICES = int(os.environ.get("KAMIKAZE_CAD_MIN_VERTICES", "200"))
MAX_VERTICES = int(os.environ.get("KAMIKAZE_CAD_MAX_VERTICES", "350000"))
ALLOW_NONCOMMERCIAL = os.environ.get("KAMIKAZE_CAD_ALLOW_NC", "0") == "1"
DOWNLOAD_TRIES = 4
POPULARITY_CAP = 25.0
FLAVOR_CONFLICT = 12.0
TIER_CONFLICT = 6.0
POLY_PIZZA_SEARCH = "https://api.poly.pizza/v1.1/search/{query}"
LEVEL_SCORE = {"exact": 0.0, "similar": 30.0, "category": 10.0}
PROVIDERS = ("sketchfab", "polypizza")
SOURCES = ("auto", "sketchfab", "polypizza", "local", "upload")
OFFLINE_SOURCES = ("local", "upload")  # the only sources that make no network request (dashboard Offline mode)
SPEC_BADGE = "Estimated Model: Dimensionally Calibrated via Spec Search"

PHONE_TIERS = {"pro", "max", "plus", "ultra", "mini", "fe", "edge", "air", "lite"}
CAN_TIER_SYNONYMS = {"zero": "zero", "max": "zero", "diet": "diet", "light": "diet", "lite": "diet"}
REJECT_COMMON = {"pack", "crate", "fridge", "vending", "machine", "dispenser", "scene", "pile", "collection", "set",
                 "cooler", "display", "shelf", "store", "room", "stack", "carton", "rack", "bundle", "truck", "series",
                 "cans", "bottles", "phones", "smartphones", "sixpack", "case", "cases", "logo", "sign", "poster",
                 "billboard", "crushed", "broken", "destroyed", "grenade", "bomb", "weapon", "explosive"}
# shape-changing words that disqualify a model even when they only appear in its tags / description
META_REJECT = {"crushed", "broken", "destroyed", "dented", "smashed", "grenade", "bomb", "weapon", "explosive"}
REJECT_PHONE = {"cover", "holder", "stand", "charger", "box", "packaging", "tablet", "ipad", "fold", "flip", "watch",
                "buds", "protector", "wallpaper", "desk", "hand", "holding", "dock", "tripod", "gimbal", "accessory",
                "pen", "spen", "stylus", "pencil", "airpods", "earbuds", "headphones", "airtag", "strap", "lens"}
REJECT_CAN = {"bottle", "glass", "cup", "keg", "tin", "fanart", "sonic", "character", "sprite3d"}
REJECT_BOTTLE = {"can", "glass", "cup", "keg", "tin", "fanart", "sonic", "character", "sprite3d", "wine", "beer"}
CAN_WORDS = {"can", "soda", "drink", "beverage", "lemon", "lime", "cola", "aluminum", "aluminium", "energy", "seltzer",
             "sparkling", "fizzy"}
BOTTLE_WORDS = {"bottle", "soda", "drink", "beverage", "pet", "plastic"}
PHONE_WORDS = {"phone", "smartphone", "mobile", "cellphone", "handset"}
PHONE_BRANDS = {"iphone", "apple", "samsung", "galaxy", "pixel", "google", "xiaomi", "redmi", "huawei", "oneplus",
                "nokia", "motorola", "moto", "sony", "xperia", "oppo", "vivo", "realme", "honor", "nothing", "asus"}
PHONE_FAMILY = {"Apple": {"iphone", "apple"}, "Samsung": {"galaxy", "samsung"}}
# brands looked for in free-text descriptions; words that are also common English ("nothing", "honor") are left out
META_PHONE_BRANDS = {"iphone", "apple", "samsung", "galaxy", "pixel", "xiaomi", "redmi", "huawei", "oneplus", "motorola",
                     "xperia", "oppo", "realme"}
BRAND_DRINK = {"Coca-Cola": "cola", "Pepsi": "cola", "Red Bull": "energy drink"}
DRINK_GROUP = {"seltzer": "water", "sparkling water": "water", "water": "water", "mineral water": "water"}
BEVERAGE_BRANDS = {"coke", "coca", "cocacola", "pepsi", "sprite", "fanta", "redbull", "monster", "dew", "schweppes",
                   "rockstar", "drpepper", "mirinda", "lipton", "nestea", "heineken", "budweiser", "beer", "corona",
                   "carlsberg", "guinness", "arizona", "celsius"}
LICENSE_RANK = (("cc0", 8.0), ("public domain", 8.0), ("by-sa", 4.0), ("sharealike", 4.0), ("by-nd", 3.0),
                ("noderiv", 3.0), ("attribution", 6.0), ("cc-by", 6.0), ("standard", 1.0))
LICENSE_URLS = (
    ("cc0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    ("public domain", "https://creativecommons.org/publicdomain/zero/1.0/"),
    ("sharealike", "https://creativecommons.org/licenses/by-sa/4.0/"),
    ("by-sa", "https://creativecommons.org/licenses/by-sa/4.0/"),
    ("noderiv", "https://creativecommons.org/licenses/by-nd/4.0/"),
    ("by-nd", "https://creativecommons.org/licenses/by-nd/4.0/"),
    ("attribution", "https://creativecommons.org/licenses/by/4.0/"),
    ("cc-by", "https://creativecommons.org/licenses/by/4.0/"),
    ("standard", "https://sketchfab.com/licenses"),
)

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_blocked_until: dict[str, float] = {}
_cand_guard = threading.Lock()
_cand_cache: dict[str, dict[str, "Candidate"]] = {}


@dataclass
class Candidate:
    provider: str
    title: str
    score: float
    author: str = ""
    author_url: str = ""
    license: str = ""
    license_url: str = ""
    source_url: str = ""
    thumbnail_url: str = ""
    uid: str = ""
    download_url: str = ""
    faces: int = 0
    attribution_text: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Paths, sidecars, credits
# --------------------------------------------------------------------------------------


def models_dir() -> Path:
    return Path(mcp.MODELS_DIR)


def _slug(class_id: str) -> str:
    return mcp.slugify(class_id)


def sidecar_path(class_id: str) -> Path:
    return models_dir() / f"{_slug(class_id)}.json"


def glb_path(class_id: str) -> Path:
    return mcp.local_glb_path(_slug(class_id))


def glb_url(class_id: str) -> str:
    return f"/static/models/{_slug(class_id)}.glb"


def read_attribution(class_id: str) -> dict | None:
    try:
        data = json.loads(sidecar_path(class_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


write_json = _write_json


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def attribution_line(attr: dict) -> str:
    """One-line human credit: '"Coke Zero" by <author> (<url>) - licensed under CC Attribution (<url>) - via Sketchfab (<url>)'."""
    changed = " - modified: re-oriented by Kamikaze" if (
        attr.get("modified") and (attr.get("orientation_fix") or {}).get("identity") is False) else ""
    if attr.get("attribution_text"):
        return str(attr["attribution_text"]) + changed
    title = attr.get("title") or "Untitled"
    author = attr.get("author") or "unknown author"
    parts = [f"\u201c{title}\u201d by {author}" + (f" ({attr['author_url']})" if attr.get("author_url") else "")]
    if attr.get("license"):
        parts.append(f"licensed under {attr['license']}" + (f" ({attr['license_url']})" if attr.get("license_url") else ""))
    if attr.get("source"):
        parts.append(f"via {attr['source']}" + (f" ({attr['source_url']})" if attr.get("source_url") else ""))
    return " - ".join(parts) + changed


def credits() -> list[dict]:
    """Attribution records for every cached GLB that has a sidecar, sorted by class_id."""
    out = []
    root = models_dir()
    if not root.is_dir():
        return out
    for js in sorted(root.glob("*.json")):
        if not mcp.is_valid_glb(js.with_suffix(".glb")):
            continue
        try:
            rec = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict):
            out.append({**rec, "class_id": js.stem, "glb_url": f"/static/models/{js.stem}.glb", "credit": attribution_line(rec)})
    return out


def write_credits_md() -> Path | None:
    records = credits()
    lines = ["# 3D model credits", "",
             "Models cached by Kamikaze's CAD harvester. Keep this attribution when showing or redistributing them.", ""]
    for rec in records:
        lines.append(f"- **{rec['class_id']}** - {rec['credit']}")
    path = models_dir() / "CREDITS.md"
    try:
        models_dir().mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    except OSError:
        return None


def delete_cached(class_id: str) -> bool:
    """Remove the cached GLB + attribution sidecar of one class. Returns True when a GLB was removed."""
    removed = False
    for path in (glb_path(class_id), sidecar_path(class_id)):
        try:
            path.unlink()
            removed = removed or path.suffix == ".glb"
        except OSError:
            pass
    if removed:
        write_credits_md()
    return removed


# --------------------------------------------------------------------------------------
# Relevance + licence filtering (pure functions)
# --------------------------------------------------------------------------------------


def search_plan(product) -> list[tuple[str, str]]:
    """[(query, level)] for a product, most specific first (levels: exact -> similar -> category)."""
    product = _coerce(product)
    return entity_resolver.query_plan(product.category, product.brand, product.model_or_variant, product.sub_tier,
                                      product.container_volume, product.flavor, product.drink_type)


def plan_queries(product, level: str) -> list[str]:
    return [query for query, lv in search_plan(product) if lv == level]


def build_search_queries(raw_text: str) -> list[str]:
    """Normalise colloquial text into ordered CAD search queries (most specific -> category fallback).

        "coke zero can 330ml" -> ["Coca Cola Zero Can", "Coke Zero 330ml Can", "Coke Zero Can", "Coca Cola Can", "Soda Can"]
        "s25 ultra"           -> ["Samsung Galaxy S25 Ultra", "Samsung S25 Ultra", "Galaxy S25 Ultra", "Galaxy S24 Ultra",
                                  "Smartphone"]
        "iphone 15 pro max"   -> ["iPhone 15 Pro Max", "Apple iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 14 Pro Max",
                                  "Smartphone"]

    Text the resolver cannot place (unknown brand) degrades to the title-cased text itself.
    """
    try:
        product = entity_resolver.resolve(raw_text)
    except taxonomy.MetadataError:
        text = re.sub(r"\s+", " ", str(raw_text or "")).strip()
        return [text.title()] if text else []
    return [query for query, _ in search_plan(product)]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", (text or "").lower().replace("+", " plus "))


def _has_seq(tokens: list[str], seq: list[str]) -> bool:
    n = len(seq)
    return n > 0 and any(tokens[i:i + n] == seq for i in range(len(tokens) - n + 1))


def _brand_present(brand: str, tokens: list[str]) -> bool:
    b = brand.lower()
    if "coca" in b:
        return "coke" in tokens or "cocacola" in tokens or ("coca" in tokens and "cola" in tokens)
    if "red bull" in b:
        return "redbull" in tokens or ("red" in tokens and "bull" in tokens)
    wanted = _tokens(brand)
    return bool(wanted) and all(w in tokens for w in wanted)


def _reject_set(product) -> set[str]:
    if product.category == "smartphone":
        return REJECT_COMMON | REJECT_PHONE
    return REJECT_COMMON | (REJECT_BOTTLE if product.category == "beverage_bottle" else REJECT_CAN)


def _container_words(product) -> set[str]:
    return BOTTLE_WORDS if product.category == "beverage_bottle" else CAN_WORDS


def name_score(product: entity_resolver.CanonicalProduct, name: str) -> float | None:
    """How well a model title matches the exact product (None = reject). Variant tiers must match exactly."""
    tokens = _tokens(name)
    if not tokens:
        return None
    if _reject_set(product) & set(tokens):
        return None
    if product.category == "smartphone":
        model = _tokens(product.model_or_variant)
        want_tier = {t for t in model if t in PHONE_TIERS}
        base = [t for t in model if t not in PHONE_TIERS and t != "galaxy"]
        if not _has_seq(tokens, base):
            return None
        if {t for t in tokens if t in PHONE_TIERS} != want_tier:
            return None
        score = 60.0
    else:
        if not _brand_present(product.brand, tokens):
            return None
        brand_key = product.brand.lower()
        if not ("coca" in brand_key or "pepsi" in brand_key) and not _container_words(product) & set(tokens):
            return None  # ambiguous brand words (e.g. "sprite") need a can/soda cue in the title
        want_tier = {"zero"} if product.sub_tier == "zero" else {"diet"} if product.sub_tier == "diet" else set()
        have_tier = {CAN_TIER_SYNONYMS[t] for t in tokens if t in CAN_TIER_SYNONYMS}
        if have_tier != want_tier:
            return None
        if product.flavor and not _has_seq(tokens, _tokens(product.flavor)):
            return None  # "Kirkland Grapefruit" needs grapefruit in the title to count as the exact product
        if not product.flavor and _title_flavor(product, name):
            return None  # a Cherry Coke model is not the plain Coca-Cola can
        pkg = "bottle" if product.category == "beverage_bottle" else "can"
        score = 50.0 + (10.0 if pkg in tokens else 0.0)
    return score - min(10.0, 0.5 * max(0, len(tokens) - 8))


def _title_flavor(product, name: str) -> str:
    """Flavour named in a model title, ignoring the brand's own signature flavour (Sprite = lemon-lime)."""
    flavor = taxonomy.find_flavor(name)
    signature = taxonomy.SIGNATURE_FLAVOR.get(product.brand, "")
    return "" if flavor and set(flavor.split()) <= set(signature.split()) else flavor


def _flavor_conflict(product, name: str, tokens: list[str]) -> bool:
    """The title names a different flavour (e.g. "Orange Soda" for a grapefruit can)."""
    other = _title_flavor(product, name)
    return bool(other) and not (product.flavor and _has_seq(tokens, _tokens(product.flavor)))


def _drink_conflict(product, name: str) -> bool:
    """The title names a different drink type (e.g. "Cola Can" for sparkling water). Plain "soda" is generic."""
    theirs = taxonomy.find_drink_type(name)
    if not theirs or theirs == "soda":
        return False
    mine = product.drink_type or BRAND_DRINK.get(product.brand, "soda")
    group = lambda d: DRINK_GROUP.get(d, d)  # noqa: E731
    if mine == "soda" and group(theirs) == "water":
        return False
    return group(theirs) != group(mine)


def _tag_names(tags) -> list[str]:
    """Sketchfab tags are [{"name": ...}], Poly Pizza tags are plain strings."""
    out = []
    for tag in tags or []:
        name = tag.get("name") or tag.get("slug") if isinstance(tag, dict) else tag
        if name:
            out.append(str(name))
    return out


def metadata_score(product, title: str, tags, description: str = "") -> float | None:
    """Brand / flavour evidence that a generic title hides, from the model's tags and description (None = reject).

    A Sketchfab "Soda Can" whose description reads "A simple old Pepsi Soda can" (or that is tagged pepsi / dew / cola)
    carries a Pepsi / Coke livery: it must never stand in for a Kirkland can. When the title itself names the product's
    brand the title is trusted (descriptions often compare with competitors). A different flavour in the tags only
    lowers the score; crushed / grenade variants are rejected.
    """
    tag_text = " ".join(_tag_names(tags))
    tokens = set(_tokens(tag_text + " " + (description or "")))
    if not tokens:
        return 0.0
    if META_REJECT & set(_tokens(tag_text)):
        return None
    title_tokens = _tokens(title)
    if product.category == "smartphone":
        family = PHONE_FAMILY.get(product.brand, set(_tokens(product.brand)))
        if family & set(title_tokens):
            return 0.0
        return None if (META_PHONE_BRANDS - family) & tokens else 0.0
    if _brand_present(product.brand, title_tokens):
        return 0.0
    if _foreign_beverage(product, sorted(tokens)):
        return None
    if not _drink_conflict(product, title) and any(_drink_conflict(product, t) for t in _tag_names(tags)):
        return None  # a generic "Soda Can" tagged cola is textured as a cola can
    if not _flavor_conflict(product, title, title_tokens) and any(
            _flavor_conflict(product, t, _tokens(t)) for t in _tag_names(tags)):
        return -FLAVOR_CONFLICT / 2
    return 0.0


def popularity(likes: int = 0, views: int = 0, staff_picked: bool = False) -> float:
    """Log-scaled community signal: 2.5·ln(1+likes) + 0.75·ln(1+views) + 4 if staff-picked, capped at 25.

    Log scaling keeps a 5,000-like model from out-voting relevance; neither Sketchfab search nor Poly Pizza exposes
    download counts, so likes / views / staff picks are the available popularity signals.
    """
    score = 2.5 * math.log1p(max(0, likes)) + 0.75 * math.log1p(max(0, views)) + (4.0 if staff_picked else 0.0)
    return min(POPULARITY_CAP, score)


def _foreign_beverage(product, tokens: list[str]) -> bool:
    own = set(_tokens(product.brand)) | ({"coke", "coca", "cocacola"} if "coca" in product.brand.lower() else set())
    if "red bull" in product.brand.lower():
        own.add("redbull")
    elif "red" in tokens and "bull" in tokens:
        return True
    return bool((BEVERAGE_BRANDS - own) & set(tokens))


def match_score(product: entity_resolver.CanonicalProduct, name: str, level: str = "exact") -> float | None:
    """Relevance of a model title at a search level (None = reject).

    exact    - the exact product and tier (name_score).
    similar  - same brand family / product line (another generation, tier or flavour), e.g. Galaxy S24 Ultra for S25 Ultra,
               or an unbranded can of the same flavour ("Grapefruit Seltzer Can" for "Kirkland Grapefruit").
    category - a generic object of the same category with no competing brand, e.g. "Soda Can", "Smartphone"; a title
               naming another flavour or drink type ("Orange Soda", "Cola Can" for a grapefruit seltzer) is penalised.
    """
    exact = name_score(product, name)
    if exact is not None or level == "exact":
        return exact
    tokens = _tokens(name)
    if not tokens or _reject_set(product) & set(tokens):
        return None
    penalty = min(10.0, 0.5 * max(0, len(tokens) - 8))
    if product.category == "smartphone":
        family = PHONE_FAMILY.get(product.brand, set(_tokens(product.brand)))
        if family & set(tokens) and not (PHONE_BRANDS - family) & set(tokens):
            return LEVEL_SCORE["similar"] - penalty
        if level == "category" and PHONE_WORDS & set(tokens) and not PHONE_BRANDS & set(tokens):
            return LEVEL_SCORE["category"] - penalty
        return None
    words = _container_words(product)
    foreign = _foreign_beverage(product, tokens)
    if _brand_present(product.brand, tokens) and words & set(tokens) and not foreign:
        return LEVEL_SCORE["similar"] - penalty
    drink_conflict = _drink_conflict(product, name)
    if (product.flavor and _has_seq(tokens, _tokens(product.flavor)) and words & set(tokens) and not foreign
            and not drink_conflict):
        return LEVEL_SCORE["similar"] - 4.0 - penalty
    pkg = "bottle" if product.category == "beverage_bottle" else "can"
    if level == "category" and (pkg in tokens or "soda" in tokens) and not foreign and not drink_conflict:
        want_tier = {"zero"} if product.sub_tier == "zero" else {"diet"} if product.sub_tier == "diet" else set()
        have_tier = {CAN_TIER_SYNONYMS[t] for t in tokens if t in CAN_TIER_SYNONYMS}
        return (LEVEL_SCORE["category"] - penalty - (FLAVOR_CONFLICT if _flavor_conflict(product, name, tokens) else 0.0)
                - (TIER_CONFLICT if have_tier - want_tier else 0.0))
    return None


def license_allowed(label: str) -> bool:
    low = (label or "").lower()
    if not low:
        return False
    if re.search(r"non[\s\-]?commercial|(?:^|[\s\-])nc(?:$|[\s\-])", low) and not ALLOW_NONCOMMERCIAL:
        return False
    return any(key in low for key, _ in LICENSE_RANK)


def license_rank(label: str) -> float:
    low = (label or "").lower()
    for key, rank in LICENSE_RANK:
        if key in low:
            return rank
    return 0.0


def license_url(label: str) -> str:
    low = (label or "").lower()
    for key, url in LICENSE_URLS:
        if key in low:
            return url
    return ""


def rank_sketchfab(product: entity_resolver.CanonicalProduct, results: list[dict], level: str = "exact") -> list[Candidate]:
    """Filter and rank Sketchfab search results for the product at a search level, best first."""
    ranked: list[Candidate] = []
    for model in results or []:
        uid, name = model.get("uid"), str(model.get("name") or "")
        lic = (model.get("license") or {}).get("label", "")
        faces = int(model.get("faceCount") or 0)
        if not uid or not license_allowed(lic) or not 0 < faces <= MAX_FACES:
            continue
        if model.get("isDownloadable") is False:
            continue
        base = match_score(product, name, level)
        if base is None:
            continue
        meta = metadata_score(product, name, model.get("tags"), str(model.get("description") or ""))
        if meta is None:
            continue
        likes, views = int(model.get("likeCount") or 0), int(model.get("viewCount") or 0)
        pop = popularity(likes, views, bool(model.get("staffpickedAt")))
        score = base + meta + license_rank(lic) + pop - min(8.0, faces / 25000.0)
        user = model.get("user") or {}
        ranked.append(Candidate(
            provider="sketchfab", title=name, score=score,
            author=str(user.get("displayName") or user.get("username") or ""),
            author_url=str(user.get("profileUrl") or ""), license=lic, license_url=license_url(lic),
            source_url=f"https://sketchfab.com/3d-models/{uid}", uid=uid, faces=faces,
            thumbnail_url=mcp._pick_thumbnail(model.get("thumbnails")),
            extra={"likes": likes, "views": views, "popularity": round(pop, 2)},
        ))
    ranked.sort(key=lambda c: -c.score)
    return ranked


def _poly_field(item: dict, *names: str):
    for n in names:
        if item.get(n) not in (None, ""):
            return item[n]
    return ""


def _attribution_license_url(text: str) -> str:
    found = re.search(r"https?://creativecommons\.org/[^\s\"')]+", text or "")
    return found.group(0).rstrip(".,;") if found else ""


def rank_polypizza(product: entity_resolver.CanonicalProduct, results: list[dict], level: str = "exact") -> list[Candidate]:
    """Filter and rank Poly Pizza search results (tolerates both capitalised and lower-case field names).

    Poly Pizza returns no likes / views, so its own relevance order stands in for popularity (6 points, -0.5 per rank).
    """
    ranked: list[Candidate] = []
    for position, item in enumerate(results or []):
        title = str(_poly_field(item, "Title", "title"))
        lic = str(_poly_field(item, "Licence", "License", "licence", "license"))
        url = str(_poly_field(item, "Download", "download"))
        tris = int(_poly_field(item, "Tri Count", "triCount", "TriCount") or 0)
        base = match_score(product, title, level)
        if not url or base is None or not license_allowed(lic) or tris > MAX_FACES:
            continue
        meta = metadata_score(product, title, _poly_field(item, "Tags", "tags") or [],
                              str(_poly_field(item, "Description", "description")))
        if meta is None:
            continue
        creator = _poly_field(item, "Creator", "creator")
        creator = creator if isinstance(creator, dict) else {}
        pid = str(_poly_field(item, "ID", "id"))
        username = str(_poly_field(creator, "Username", "username", "name"))
        profile = str(_poly_field(creator, "profileLink", "ProfileLink"))
        if profile.startswith("/"):
            profile = "https://poly.pizza" + profile
        if not profile and username:
            profile = f"https://poly.pizza/u/{urllib.parse.quote(username)}"
        attribution = str(_poly_field(item, "Attribution", "attribution"))
        pop = max(0.0, 6.0 - 0.5 * position)
        ranked.append(Candidate(
            provider="polypizza", title=title, score=base + meta + license_rank(lic) + pop,
            author=username, author_url=profile,
            license=lic, license_url=_attribution_license_url(attribution) or license_url(lic),
            source_url=f"https://poly.pizza/m/{pid}" if pid else "https://poly.pizza", download_url=url,
            thumbnail_url=str(_poly_field(item, "Thumbnail", "thumbnail", "previewUrl")), uid=pid, faces=tris,
            attribution_text=attribution, extra={"popularity": pop},
        ))
    ranked.sort(key=lambda c: -c.score)
    return ranked


# --------------------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------------------


def _blocked(provider: str) -> bool:
    return time.monotonic() < _blocked_until.get(provider, 0.0)


def _block(provider: str, seconds: float, why: str) -> None:
    _blocked_until[provider] = time.monotonic() + seconds
    log.warning("CAD source %s disabled for %.0fs: %s", provider, seconds, why)


def _http_failure(provider: str, exc: Exception) -> None:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            _block(provider, 3600.0, f"HTTP {exc.code} - check the API key")
            return
        if exc.code == 429:
            _block(provider, 120.0, "rate limited")
            return
    log.warning("CAD source %s failed: %s", provider, exc)


def _record(product, cand: Candidate, source_name: str) -> dict:
    return {
        "title": cand.title, "author": cand.author, "author_url": cand.author_url, "license": cand.license,
        "license_url": cand.license_url, "source": source_name, "source_key": cand.provider, "source_url": cand.source_url,
        "thumbnail_url": cand.thumbnail_url, "attribution_text": cand.attribution_text, "modified": False,
        "faces": cand.faces, "class_id": product.class_id, "canonical_name": product.canonical_name,
        "match": cand.extra.get("match", "exact"),
        "query": cand.extra.get("query", ""),
        "rank_score": round(cand.score, 2), "popularity": cand.extra.get("popularity"),
        "likes": cand.extra.get("likes"), "views": cand.extra.get("views"),
        "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def _sketchfab_download_url(uid: str, token: str) -> str | None:
    try:
        info = mcp._http_json(f"https://api.sketchfab.com/v3/models/{uid}/download", {"Authorization": f"Token {token}"},
                              timeout=SEARCH_TIMEOUT_S)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        _http_failure("sketchfab", exc)
        return None
    glb = (info or {}).get("glb") or {}
    if not glb.get("url"):
        return None
    if glb.get("size") and int(glb["size"]) > MAX_GLB_MB * 1024 * 1024:
        log.info("Skipping Sketchfab model %s: GLB is %.1f MB (> %.0f MB)", uid, int(glb["size"]) / 1e6, MAX_GLB_MB)
        return None
    return str(glb["url"])


def _keep_best(best: dict[str, Candidate], cands: list[Candidate], query: str, level: str) -> None:
    for cand in cands:
        cand.extra.update(query=query, match=level)
        if cand.uid not in best or cand.score > best[cand.uid].score:
            best[cand.uid] = cand


def _search_sketchfab(product, queries: list[str], level: str, deadline: float) -> list[Candidate]:
    """Every relevant, licence-clean Sketchfab model across all queries of one level (deduplicated by uid)."""
    token = os.environ.get("SKETCHFAB_TOKEN")
    best: dict[str, Candidate] = {}
    if not token:
        return []
    for query in queries:
        if time.monotonic() > deadline or _blocked("sketchfab"):
            break
        params = urllib.parse.urlencode({"type": "models", "q": query, "downloadable": "true", "count": 24})
        try:
            data = mcp._http_json(f"https://api.sketchfab.com/v3/search?{params}", {"Authorization": f"Token {token}"},
                                  timeout=SEARCH_TIMEOUT_S)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            _http_failure("sketchfab", exc)
            continue
        _keep_best(best, rank_sketchfab(product, (data or {}).get("results", []), level), query, level)
    return list(best.values())


def _search_polypizza(product, queries: list[str], level: str, deadline: float) -> list[Candidate]:
    """Every relevant CC0 / CC-BY Poly Pizza model across all queries of one level (deduplicated by id)."""
    key = os.environ.get("POLY_PIZZA_API_KEY")
    best: dict[str, Candidate] = {}
    if not key:
        return []
    for query in queries:
        if time.monotonic() > deadline or _blocked("polypizza"):
            break
        try:
            data = mcp._http_json(POLY_PIZZA_SEARCH.format(query=urllib.parse.quote(query)), {"X-Auth-Token": key},
                                  timeout=SEARCH_TIMEOUT_S)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            _http_failure("polypizza", exc)
            continue
        _keep_best(best, rank_polypizza(product, (data or {}).get("results", []), level), query, level)
    return list(best.values())


SOURCE_NAMES = {"sketchfab": "Sketchfab", "polypizza": "Poly Pizza", "local": "Local Cache", "upload": "Custom Scan Upload",
                "library": "Local Library"}


def _fetch(cand: Candidate, slug: str) -> bool:
    if cand.provider == "sketchfab":
        token = os.environ.get("SKETCHFAB_TOKEN", "")
        url = _sketchfab_download_url(cand.uid, token) if token else None
        return bool(url and mcp.download_glb(url, slug))
    return bool(cand.download_url and mcp.download_glb(cand.download_url, slug))


def _download_and_validate(cand: Candidate, slug: str) -> bool:
    """Download one candidate and run the quality gate; a rejected file is deleted so the next candidate can be tried."""
    if not _fetch(cand, slug):
        return False
    path = mcp.local_glb_path(slug)
    ok, why, info = validate_glb(path)
    cand.extra["quality"] = info
    if ok:
        return True
    log.info("CAD candidate %r (%s) rejected by the quality gate: %s", cand.title, cand.provider, why)
    try:
        path.unlink()
    except OSError:
        pass
    return False


def _pool(product, queries: list[str], level: str, deadline: float, providers: tuple[str, ...]) -> list[Candidate]:
    pool: list[Candidate] = []
    if "sketchfab" in providers:
        pool += _search_sketchfab(product, queries, level, deadline)
    if "polypizza" in providers:
        pool += _search_polypizza(product, queries, level, deadline)
    pool.sort(key=lambda c: -c.score)
    return pool


def _try_online(product, slug: str, deadline: float, queries: list[str], level: str,
                providers: tuple[str, ...] = PROVIDERS) -> tuple[Candidate, str] | None:
    """Pool Sketchfab + Poly Pizza candidates for one search level, rank them by relevance + licence + popularity and
    download the best one that passes the GLB quality gate (up to DOWNLOAD_TRIES attempts)."""
    pool = _pool(product, queries, level, deadline, providers)
    if pool:
        log.info("CAD %s [%s] top candidates: %s", product.class_id, level,
                 "; ".join(f"{c.title!r} ({SOURCE_NAMES.get(c.provider, c.provider)}, {c.score:.1f})" for c in pool[:3]))
    for cand in pool[:DOWNLOAD_TRIES]:
        if time.monotonic() > deadline:
            break
        if _download_and_validate(cand, slug):
            return cand, SOURCE_NAMES.get(cand.provider, cand.provider)
    return None


# --------------------------------------------------------------------------------------
# GLB quality gate
# --------------------------------------------------------------------------------------


def validate_glb(path: Path) -> tuple[bool, str, dict]:
    """(ok, reason, info) - rejects non-GLB files, primitive placeholder geometry (< MIN_VERTICES unique vertices),
    meshes too heavy for the dashboard (> MAX_VERTICES or > MAX_GLB_MB) and material-less (untextured) models."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            header = fh.read(20)
            if len(header) < 20 or header[:4] != b"glTF":
                return False, "not a binary glTF (GLB)", {}
            version = struct.unpack("<I", header[4:8])[0]
            json_len, json_type = struct.unpack("<II", header[12:20])
            if version != 2 or json_type != 0x4E4F534A or json_len > size:
                return False, "malformed GLB header", {}
            doc = json.loads(fh.read(json_len).decode("utf-8"))
    except (OSError, ValueError, struct.error) as exc:
        return False, f"unreadable GLB: {exc}", {}
    accessors = doc.get("accessors") or []
    seen: set[int] = set()
    vertices = 0
    for mesh in doc.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            ref = (prim.get("attributes") or {}).get("POSITION")
            if isinstance(ref, int) and 0 <= ref < len(accessors) and ref not in seen:
                seen.add(ref)
                vertices += int(accessors[ref].get("count") or 0)
    info = {"vertices": vertices, "size_mb": round(size / 1e6, 2), "materials": len(doc.get("materials") or []),
            "meshes": len(doc.get("meshes") or []), "textures": len(doc.get("textures") or [])}
    if size > MAX_GLB_MB * 1024 * 1024:
        return False, f"{size / 1e6:.1f} MB exceeds the {MAX_GLB_MB:.0f} MB budget", info
    if vertices < MIN_VERTICES:
        return False, f"only {vertices} vertices (< {MIN_VERTICES}): primitive / placeholder geometry", info
    if vertices > MAX_VERTICES:
        return False, f"{vertices} vertices exceeds the {MAX_VERTICES} budget", info
    if not info["materials"]:
        return False, "no materials (would render as an untextured mesh)", info
    return True, "ok", info


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _lock_for(slug: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(slug, threading.Lock())


def _coerce(product) -> entity_resolver.CanonicalProduct:
    if isinstance(product, entity_resolver.CanonicalProduct):
        return product
    if isinstance(product, dict):
        return entity_resolver.resolve_metadata(product)
    return entity_resolver.resolve(str(product))


def _with_class_id(product: entity_resolver.CanonicalProduct, class_id: str) -> entity_resolver.CanonicalProduct:
    data = product.to_dict()
    data["class_id"] = class_id
    data["cad_search_terms"] = tuple(data["cad_search_terms"])
    return entity_resolver.CanonicalProduct(**data)


def _providers(source: str) -> tuple[str, ...]:
    source = (source or "auto").lower()
    if source in ("auto", "online", "all"):
        return PROVIDERS
    if source in PROVIDERS:
        return (source,)
    if source in OFFLINE_SOURCES:
        return ()
    raise ValueError(f"unknown CAD source {source!r} ({', '.join(SOURCES)})")


def _result(product, attr: dict, status: str, source: str, cached: bool, query: str = "") -> dict:
    return {
        "class_id": product.class_id, "title": attr.get("title") or product.canonical_name, "glb_url": glb_url(product.class_id),
        "status": status, "source": source, "query": query, "cached": cached, "attribution": attr,
        "credit": attribution_line(attr), "thumbnail_url": attr.get("thumbnail_url", ""), "match": attr.get("match", ""),
        "badge": attr.get("badge", ""),
    }


def not_found(class_id: str, title: str, reason: str, status: str = "not_found") -> dict:
    """The explicit NOT_FOUND outcome: no model is shown rather than a substitute. status "offline_missing" = nothing
    is cached locally and Offline mode forbids searching online."""
    return {"class_id": class_id, "title": title, "glb_url": "", "status": status, "source": "none", "query": "",
            "cached": False, "attribution": {}, "credit": "", "thumbnail_url": "", "match": "", "badge": "",
            "reason": reason}


def _auto_orient(class_id: str, category: str | None) -> dict | None:
    """Stand a freshly landed GLB up for its category (glb_orient). Never fails the landing."""
    try:
        info = glb_orient.auto_orient(glb_path(class_id), category)
    except (OSError, ValueError) as exc:
        log.warning("auto-orient skipped for %s: %s", class_id, exc)
        return None
    if info.get("changed"):
        log.info("CAD %s auto-oriented: %s (%s)", class_id, info["description"], info["note"])
    return info


def _orientation_fields(info: dict | None) -> dict:
    if not info:
        return {}
    out = {"orientation_fix": {k: info[k] for k in ("rotation", "description", "note", "identity")}}
    if not info.get("identity"):
        out["modified"] = True  # CC-BY: indicate that the model was changed (re-oriented)
    return out


def reorient(class_id: str, op: str, axis: str = "y", deg: float = 90.0, category: str | None = None) -> dict:
    """Dashboard orientation fix for a cached GLB: op = "auto" | "rotate" | "reset" | "status". Updates the sidecar."""
    path = glb_path(class_id)
    if not mcp.is_valid_glb(path):
        raise FileNotFoundError(f"no cached GLB for {class_id!r}")
    with _lock_for(_slug(class_id)):
        if op == "auto":
            info = glb_orient.auto_orient(path, category)
        elif op == "rotate":
            info = glb_orient.rotate(path, axis, deg)
        elif op == "reset":
            info = glb_orient.reset(path)
        elif op == "status":
            return glb_orient.status(path)
        else:
            raise ValueError("op must be auto, rotate, reset or status")
        attr = read_attribution(class_id)
        if attr is not None:
            attr["orientation_fix"] = _orientation_fields(info)["orientation_fix"]
            if attr.get("source_key") not in ("upload", "spec_generated"):
                attr["modified"] = not info.get("identity")  # CC-BY: indicate changes
            _write_json(sidecar_path(class_id), attr)
            write_credits_md()
    return info


def _finish(product, cand: Candidate, source_name: str, status: str, extra: dict | None = None) -> dict:
    attr = {**_record(product, cand, source_name), **(extra or {})}
    attr.update(_orientation_fields(_auto_orient(product.class_id, product.category)))
    _write_json(sidecar_path(product.class_id), attr)
    write_credits_md()
    log.info("CAD for %s: %s | %s", product.class_id, source_name, attribution_line(attr))
    return _result(product, attr, status, cand.provider, False, cand.extra.get("query", ""))


def _library_match(product) -> dict | None:
    try:
        return model_library.find(product)
    except (OSError, ValueError) as exc:
        log.warning("local library lookup failed for %s: %s", product.class_id, exc)
        return None


def harvest(product, force: bool = False, offline: bool = False, budget_s: float | None = None,
            source: str = "auto") -> dict:
    """Resolve + cache a GLB for `product` (CanonicalProduct | metadata dict | free text) and write its attribution.

    Only exact-product matches are accepted automatically; similar / category models are offered through candidates()
    for an explicit user choice. offline=True (or source "local" / "upload") only consults the local cache and makes
    no network request. Returns {"class_id", "title", "glb_url", "status": "online"|"cached"|"not_found"|
    "offline_missing", "source", "query", "cached", "attribution", "credit", "reason"?}.
    """
    product = _coerce(product)
    providers = _providers(source)
    slug = _slug(product.class_id)
    dest = glb_path(product.class_id)
    with _lock_for(slug):
        if force:
            delete_cached(product.class_id)
        attr = read_attribution(product.class_id)
        if mcp.is_valid_glb(dest):
            if attr:
                return _result(product, attr, "cached", "local-cache", True)
            try:
                dest.unlink()  # a GLB without attribution cannot be shown - re-harvest it
            except OSError:
                pass
        if (source or "auto").lower() not in PROVIDERS:  # an explicit Sketchfab / Poly Pizza pick skips the library
            lib = _library_match(product)
            if lib is not None:
                attr = model_library.install(lib["id"], product.class_id)
                return _result(product, attr, "cached", "library", True)
        if offline:
            return not_found(product.class_id, product.canonical_name,
                             "offline mode: no local 3D model is cached for this class", "offline_missing")
        if not providers:
            return not_found(product.class_id, product.canonical_name,
                             f"source {SOURCE_NAMES.get(source, source)}: no 3D model is cached for this class - "
                             "upload a GLB or pick an online source")
        deadline = time.monotonic() + (BUDGET_S if budget_s is None else budget_s)
        queries = [q for q, lv in search_plan(product) if lv == "exact"]
        found = None
        if queries:
            try:
                found = _try_online(product, slug, deadline, queries, "exact", providers)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError, KeyError) as exc:
                log.warning("CAD search failed for %s: %s", product.class_id, exc)
        if not found:
            names = " + ".join(SOURCE_NAMES[p] for p in providers)
            return not_found(product.class_id, product.canonical_name,
                             f"no exact-match model on {names} passed the quality gate - browse similar options, "
                             "upload a GLB or generate one from verified specs")
        cand, source_name = found
        return _finish(product, cand, source_name, "online")


def candidates(product, limit: int = 4, source: str = "auto", budget_s: float | None = None) -> list[dict]:
    """Top `limit` ranked online models across exact -> similar -> category levels (nothing is downloaded)."""
    product = _coerce(product)
    providers = _providers(source)
    if not providers:
        return []
    deadline = time.monotonic() + (BUDGET_S if budget_s is None else budget_s)
    plan = search_plan(product)
    picked: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for level in entity_resolver.QUERY_LEVELS:
        queries = [q for q, lv in plan if lv == level]
        if not queries or len(picked) >= limit or time.monotonic() > deadline:
            continue
        for cand in _pool(product, queries, level, deadline, providers):
            if (cand.provider, cand.uid) not in seen:
                seen.add((cand.provider, cand.uid))
                picked.append(cand)
    picked = picked[:max(1, int(limit))]
    with _cand_guard:
        _cand_cache[_slug(product.class_id)] = {c.uid: c for c in picked}
    return [{
        "uid": c.uid, "provider": c.provider, "source": SOURCE_NAMES.get(c.provider, c.provider), "title": c.title,
        "thumbnail_url": c.thumbnail_url, "author": c.author, "author_url": c.author_url, "license": c.license,
        "license_url": c.license_url, "source_url": c.source_url, "faces": c.faces, "likes": c.extra.get("likes"),
        "views": c.extra.get("views"), "level": c.extra.get("match", ""), "query": c.extra.get("query", ""),
        "score": round(c.score, 1),
    } for c in picked]


def select(product, uid: str) -> dict:
    """Download the candidate `uid` (from the last candidates() call for this class) as the class model."""
    product = _coerce(product)
    slug = _slug(product.class_id)
    with _cand_guard:
        cand = (_cand_cache.get(slug) or {}).get(uid)
    if cand is None:
        raise KeyError(f"candidate {uid!r} is not in the last search for {product.class_id}; browse options again")
    with _lock_for(slug):
        delete_cached(product.class_id)
        if not _download_and_validate(cand, slug):
            why = (cand.extra.get("quality") or {})
            raise ValueError(f"{cand.title!r} could not be downloaded or failed the quality gate {why or ''}".strip())
        return _finish(product, cand, SOURCE_NAMES.get(cand.provider, cand.provider), "online", {"selected_by_user": True})


def upload_glb(class_id: str, data: bytes, filename: str = "", title: str = "", category: str | None = None) -> dict:
    """Install a user-supplied GLB for `class_id` after the quality gate. Raises ValueError when it is rejected."""
    slug = _slug(class_id)
    dest = glb_path(class_id)
    tmp = dest.with_suffix(".upload.part")
    with _lock_for(slug):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        ok, why, info = validate_glb(tmp)
        if not ok:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"upload rejected: {why}")
        delete_cached(class_id)
        os.replace(tmp, dest)
        name = Path(filename or "").name
        attr = {
            "title": title or name or class_id, "author": "user upload", "author_url": "", "license": "provided by the user",
            "license_url": "", "source": "Custom upload", "source_key": "upload", "source_url": "", "thumbnail_url": "",
            "attribution_text": f"Custom upload{f' ({name})' if name else ''} - supplied by the operator", "modified": False,
            "class_id": class_id, "match": "user", "quality": info, "filename": name,
            "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        info = _auto_orient(class_id, category)
        if info:
            attr["orientation_fix"] = _orientation_fields(info)["orientation_fix"]
        _write_json(sidecar_path(class_id), attr)
        write_credits_md()
    product = SimpleNamespace(class_id=class_id, canonical_name=title or class_id)
    return _result(product, attr, "cached", "upload", True)


def use_library(class_id: str, entry_id: str) -> dict:
    """Explicit operator choice: make library model `entry_id` the CAD of `class_id` (replaces any cached model)."""
    with _lock_for(_slug(class_id)):
        delete_cached(class_id)
        attr = model_library.install(entry_id, class_id)
    product = SimpleNamespace(class_id=class_id, canonical_name=attr.get("title") or class_id)
    return _result(product, attr, "cached", "library", True)


def generate_from_specs(product, profile: str | None = None) -> dict:
    """Build a dimension-calibrated GLB from verified published specs (explicit user action only)."""
    product = _coerce(product)
    spec = spec_search.product_specs(product, profile)
    if not spec.get("verified"):
        return {**not_found(product.class_id, product.canonical_name, spec.get("reason", "specs not verified")),
                "spec": spec}
    data = procedural_glb.build_from_spec(spec, product.raw_query, {"class_id": product.class_id})
    slug = _slug(product.class_id)
    dest = glb_path(product.class_id)
    with _lock_for(slug):
        delete_cached(product.class_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        ok, why, info = validate_glb(tmp)
        if not ok:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"generated GLB failed the quality gate: {why}")
        os.replace(tmp, dest)
        src = spec["sources"][0]
        dims = (f"{spec['height_mm']} x {spec['width_mm']} x {spec['depth_mm']} mm, {spec['rear_cameras']} rear cameras"
                if spec["category"] == "smartphone" else
                f"{spec['diameter_mm']} x {spec['height_mm']} mm ({spec['volume_ml']} ml {spec['profile']})")
        attr = {
            "title": f"{product.canonical_name} (dimensionally calibrated)", "author": "Kamikaze spec generator",
            "author_url": PROJECT_URL, "license": "MIT (project licence)", "license_url": f"{PROJECT_URL}/blob/main/LICENSE",
            "source": "Spec search", "source_key": "spec_generated", "source_url": src["url"], "thumbnail_url": "",
            "attribution_text": f"Generated by Kamikaze from published dimensions ({dims}) - {src['publisher']} ({src['url']})",
            "modified": False, "class_id": product.class_id, "canonical_name": product.canonical_name, "match": "spec",
            "badge": SPEC_BADGE, "spec": spec, "quality": info,
            "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        _write_json(sidecar_path(product.class_id), attr)
        write_credits_md()
    log.info("Spec-generated CAD for %s: %s", product.class_id, attr["attribution_text"])
    return {**_result(product, attr, "generated", "spec_generated", False), "spec": spec}


def retrieve(label: str, meta: dict | None = None, force: bool = False, offline: bool = False,
             source: str = "auto") -> dict:
    """Harvest the GLB for bank label `label` (+ optional enrollment metadata). Unparseable labels are NOT_FOUND."""
    try:
        product = entity_resolver.resolve_metadata(meta) if meta else entity_resolver.resolve(label.replace("_", " "))
    except taxonomy.MetadataError as exc:
        return not_found(label, label.replace("_", " "), f"cannot build a CAD query for {label!r}: {exc}")
    return harvest(_with_class_id(product, label), force=force, offline=offline, source=source)


def product_for(label: str, meta: dict | None = None) -> entity_resolver.CanonicalProduct:
    """CanonicalProduct for a bank label (raises taxonomy.MetadataError when it cannot be parsed)."""
    product = entity_resolver.resolve_metadata(meta) if meta else entity_resolver.resolve(label.replace("_", " "))
    return _with_class_id(product, label)


import model_library  # noqa: E402  (circular: model_library needs every definition above)
