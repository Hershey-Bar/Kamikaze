"""CAD harvester: find, license-check, download and cache a 3D GLB for a canonical product, with author attribution.

    harvest(resolve("samsung galaxy s24 ultra"))  ->  static/models/galaxy_s24_ultra.glb + galaxy_s24_ultra.json

Source order (first acceptable GLB wins):
  1. Sketchfab Data API  (SKETCHFAB_TOKEN)   - results are relevance-filtered (exact model + tier), licence-filtered
                                               (CC0 / CC-BY / CC-BY-SA / CC-BY-ND / Sketchfab Standard; NonCommercial rejected),
                                               single-object, face-count and download-size capped.
  2. Poly Pizza API      (POLY_PIZZA_API_KEY) - CC0 / CC-BY models only.
  3. Open repos          - KhronosGroup/glTF-Sample-Assets (strict name match; credits read from each model README).
  4. Local parametric model (original work, brand/tier aware) built by procedural_glb.py; generic templates live in
     static/models/base/{soda_can,bottle,smartphone}.glb.

Every cached model gets a sidecar static/models/<class_id>.json holding the attribution (title, author, author URL,
licence, licence URL, source URL). credits() / write_credits_md() aggregate them (served at GET /api/credits and
written to static/models/CREDITS.md).
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import entity_resolver
import mcp_server as mcp
import procedural_glb
import taxonomy

log = logging.getLogger("kamikaze.cad")

KHRONOS_INDEX = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/model-index.json"
KHRONOS_RAW = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
KHRONOS_TREE = "https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models"
PROJECT_URL = "https://github.com/Hershey-Bar/Kamikaze"
BUDGET_S = float(os.environ.get("KAMIKAZE_CAD_BUDGET", "40"))
SEARCH_TIMEOUT_S = float(os.environ.get("KAMIKAZE_CAD_SEARCH_TIMEOUT", "10"))
MAX_FACES = int(os.environ.get("KAMIKAZE_CAD_MAX_FACES", "150000"))
MAX_GLB_MB = float(os.environ.get("KAMIKAZE_CAD_MAX_MB", "24"))
ALLOW_NONCOMMERCIAL = os.environ.get("KAMIKAZE_CAD_ALLOW_NC", "0") == "1"
CANDIDATES_PER_QUERY = 3
QUERIES_PER_SOURCE = 4

PHONE_TIERS = {"pro", "max", "plus", "ultra", "mini", "fe", "edge", "air", "lite"}
CAN_TIER_SYNONYMS = {"zero": "zero", "max": "zero", "diet": "diet", "light": "diet", "lite": "diet"}
REJECT_COMMON = {"pack", "crate", "fridge", "vending", "machine", "dispenser", "scene", "pile", "collection", "set",
                 "cooler", "display", "shelf", "store", "room", "stack", "carton", "rack", "bundle", "truck", "series",
                 "cans", "bottles", "phones", "smartphones", "sixpack", "case", "cases"}
REJECT_PHONE = {"cover", "holder", "stand", "charger", "box", "packaging", "tablet", "ipad", "fold", "flip", "watch",
                "buds", "protector", "wallpaper", "desk", "hand", "holding", "dock", "tripod", "gimbal", "accessory"}
REJECT_CAN = {"bottle", "glass", "cup", "keg", "tin", "fanart", "sonic", "character", "sprite3d"}
CAN_WORDS = {"can", "soda", "drink", "beverage", "lemon", "lime", "cola", "aluminum", "aluminium", "energy"}
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
_index_lock = threading.Lock()
_index_cache: list | None = None


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


def attribution_line(attr: dict) -> str:
    """One-line human credit: '"Coke Zero" by <author> (<url>) - licensed under CC Attribution (<url>) - via Sketchfab (<url>)'."""
    if attr.get("attribution_text"):
        return str(attr["attribution_text"])
    title = attr.get("title") or "Untitled"
    author = attr.get("author") or "unknown author"
    parts = [f"\u201c{title}\u201d by {author}" + (f" ({attr['author_url']})" if attr.get("author_url") else "")]
    if attr.get("license"):
        parts.append(f"licensed under {attr['license']}" + (f" ({attr['license_url']})" if attr.get("license_url") else ""))
    if attr.get("source"):
        parts.append(f"via {attr['source']}" + (f" ({attr['source_url']})" if attr.get("source_url") else ""))
    return " - ".join(parts)


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


def name_score(product: entity_resolver.CanonicalProduct, name: str) -> float | None:
    """How well a model title matches the exact product (None = reject). Variant tiers must match exactly."""
    tokens = _tokens(name)
    if not tokens:
        return None
    reject = REJECT_COMMON | (REJECT_PHONE if product.category == "smartphone" else REJECT_CAN)
    if reject & set(tokens):
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
        if not ("coca" in brand_key or "pepsi" in brand_key) and not CAN_WORDS & set(tokens):
            return None  # ambiguous brand words (e.g. "sprite") need a can/soda cue in the title
        want_tier = {"zero"} if product.sub_tier == "zero" else {"diet"} if product.sub_tier == "diet" else set()
        have_tier = {CAN_TIER_SYNONYMS[t] for t in tokens if t in CAN_TIER_SYNONYMS}
        if have_tier != want_tier:
            return None
        score = 50.0 + (10.0 if "can" in tokens else 0.0)
    return score - min(10.0, 0.5 * max(0, len(tokens) - 8))


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


def rank_sketchfab(product: entity_resolver.CanonicalProduct, results: list[dict]) -> list[Candidate]:
    """Filter and rank Sketchfab search results for the exact product, best first."""
    ranked: list[Candidate] = []
    for model in results or []:
        uid, name = model.get("uid"), str(model.get("name") or "")
        lic = (model.get("license") or {}).get("label", "")
        faces = int(model.get("faceCount") or 0)
        if not uid or not license_allowed(lic) or not 0 < faces <= MAX_FACES:
            continue
        if model.get("isDownloadable") is False:
            continue
        base = name_score(product, name)
        if base is None:
            continue
        likes = int(model.get("likeCount") or 0)
        score = base + license_rank(lic) + min(6.0, 2.0 * math.log10(likes + 1)) - min(8.0, faces / 25000.0)
        user = model.get("user") or {}
        ranked.append(Candidate(
            provider="sketchfab", title=name, score=score,
            author=str(user.get("displayName") or user.get("username") or ""),
            author_url=str(user.get("profileUrl") or ""), license=lic, license_url=license_url(lic),
            source_url=f"https://sketchfab.com/3d-models/{uid}", uid=uid, faces=faces,
            thumbnail_url=mcp._pick_thumbnail(model.get("thumbnails")),
        ))
    ranked.sort(key=lambda c: -c.score)
    return ranked


def _poly_field(item: dict, *names: str):
    for n in names:
        if item.get(n) not in (None, ""):
            return item[n]
    return ""


def rank_polypizza(product: entity_resolver.CanonicalProduct, results: list[dict]) -> list[Candidate]:
    """Filter and rank Poly Pizza search results (tolerates both capitalised and lower-case field names)."""
    ranked: list[Candidate] = []
    for item in results or []:
        title = str(_poly_field(item, "Title", "title"))
        lic = str(_poly_field(item, "Licence", "License", "licence", "license")) or "CC0"
        url = str(_poly_field(item, "Download", "download"))
        base = name_score(product, title)
        if not url or base is None or not license_allowed(lic):
            continue
        creator = _poly_field(item, "Creator", "creator")
        creator = creator if isinstance(creator, dict) else {}
        pid = str(_poly_field(item, "ID", "id"))
        profile = str(_poly_field(creator, "profileLink", "ProfileLink", "DPURL"))
        if profile.startswith("/"):
            profile = "https://poly.pizza" + profile
        ranked.append(Candidate(
            provider="polypizza", title=title, score=base + license_rank(lic),
            author=str(_poly_field(creator, "Username", "username", "name")), author_url=profile,
            license=lic, license_url=license_url(lic),
            source_url=f"https://poly.pizza/m/{pid}" if pid else "https://poly.pizza", download_url=url,
            thumbnail_url=str(_poly_field(item, "Thumbnail", "thumbnail", "previewUrl")), uid=pid,
            attribution_text=str(_poly_field(item, "Attribution", "attribution")),
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


def _try_sketchfab(product, slug: str, deadline: float) -> tuple[Candidate, str] | None:
    token = os.environ.get("SKETCHFAB_TOKEN")
    if not token or _blocked("sketchfab"):
        return None
    seen: set[str] = set()
    for query in product.cad_search_terms[:QUERIES_PER_SOURCE]:
        if time.monotonic() > deadline or _blocked("sketchfab"):
            break
        params = urllib.parse.urlencode({"type": "models", "q": query, "downloadable": "true", "count": 24})
        try:
            data = mcp._http_json(f"https://api.sketchfab.com/v3/search?{params}", {"Authorization": f"Token {token}"},
                                  timeout=SEARCH_TIMEOUT_S)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            _http_failure("sketchfab", exc)
            continue
        for cand in rank_sketchfab(product, data.get("results", []))[:CANDIDATES_PER_QUERY]:
            if cand.uid in seen or time.monotonic() > deadline:
                continue
            seen.add(cand.uid)
            url = _sketchfab_download_url(cand.uid, token)
            if url and mcp.download_glb(url, slug):
                cand.extra["query"] = query
                return cand, "Sketchfab"
    return None


def _try_polypizza(product, slug: str, deadline: float) -> tuple[Candidate, str] | None:
    key = os.environ.get("POLY_PIZZA_API_KEY")
    if not key or _blocked("polypizza"):
        return None
    for query in product.cad_search_terms[:QUERIES_PER_SOURCE]:
        if time.monotonic() > deadline or _blocked("polypizza"):
            break
        try:
            data = mcp._http_json(f"https://api.poly.pizza/v1.1/search/{urllib.parse.quote(query)}", {"x-auth-token": key},
                                  timeout=SEARCH_TIMEOUT_S)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            _http_failure("polypizza", exc)
            continue
        for cand in rank_polypizza(product, (data or {}).get("results", []))[:CANDIDATES_PER_QUERY]:
            if mcp.download_glb(cand.download_url, slug):
                cand.extra["query"] = query
                return cand, "Poly Pizza"
    return None


def _khronos_index() -> list:
    global _index_cache
    with _index_lock:
        if _index_cache is None:
            data = mcp._http_json(KHRONOS_INDEX, timeout=SEARCH_TIMEOUT_S)
            _index_cache = data if isinstance(data, list) else []
        return _index_cache


def _khronos_credit(name: str) -> tuple[str, str, str]:
    """(author, license label, license url) from a sample model's README 'Legal' section."""
    try:
        req = urllib.request.Request(f"{KHRONOS_RAW}/{name}/README.md", headers={"User-Agent": mcp.USER_AGENT})
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT_S) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return "", "", ""
    legal = text.split("## Legal", 1)[-1]
    lic = re.search(r"\[([^\]]+)\]\((https?://[^)]+)\)", legal)
    authors = re.findall(r"^\s*-\s+(.+?)\s+for\s+", legal, flags=re.M)
    return ", ".join(authors), (lic.group(1) if lic else ""), (lic.group(2) if lic else "")


def _try_khronos(product, slug: str, deadline: float) -> tuple[Candidate, str] | None:
    try:
        index = _khronos_index()
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        log.warning("CAD source github failed: %s", exc)
        return None
    for entry in index:
        name = str(entry.get("name", ""))
        if time.monotonic() > deadline or "glTF-Binary" not in (entry.get("variants") or {}):
            continue
        if name_score(product, f"{name} {entry.get('label', '')}") is None:
            continue
        url = f"{KHRONOS_RAW}/{name}/glTF-Binary/{entry['variants']['glTF-Binary']}"
        if not mcp.download_glb(url, slug):
            continue
        author, lic, lic_url = _khronos_credit(name)
        return Candidate(
            provider="github", title=str(entry.get("label") or name), score=0.0, author=author or "see model README",
            license=lic or "see model README", license_url=lic_url, source_url=f"{KHRONOS_TREE}/{name}",
            thumbnail_url=f"{KHRONOS_RAW}/{name}/{entry.get('screenshot', 'screenshot/screenshot.jpg')}",
        ), "KhronosGroup/glTF-Sample-Assets"
    return None


# --------------------------------------------------------------------------------------
# Local parametric fallback
# --------------------------------------------------------------------------------------

BASE_MODELS = {
    "soda_can": lambda: procedural_glb.build_can("", "original"),
    "bottle": lambda: procedural_glb.build_can("", "original", bottle=True),
    "smartphone": lambda: procedural_glb.build_phone("Generic", "standard", 3),
}


def _atomic_write(dest: Path, data: bytes) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return mcp.is_valid_glb(dest)


def ensure_base_models() -> dict[str, Path]:
    """Generate the generic parametric templates in static/models/base/ (idempotent)."""
    out = {}
    for name, build in BASE_MODELS.items():
        path = models_dir() / "base" / f"{name}.glb"
        if not mcp.is_valid_glb(path):
            try:
                _atomic_write(path, build())
            except Exception:
                log.exception("Could not build base model %s", name)
                continue
        out[name] = path
    return out


def _base_for(category: str) -> str:
    return {"smartphone": "smartphone", "beverage_bottle": "bottle"}.get(category, "soda_can")


def _procedural(product, dest: Path) -> Candidate | None:
    try:
        ok = _atomic_write(dest, procedural_glb.build_for(product.metadata()))
    except Exception:
        log.exception("Procedural model failed for %s - using the base template", product.class_id)
        ok = False
    if not ok:
        base = ensure_base_models().get(_base_for(product.category))
        if not base:
            return None
        shutil.copyfile(base, dest)
        ok = mcp.is_valid_glb(dest)
    if not ok:
        return None
    return Candidate(
        provider="procedural", title=f"{product.canonical_name} (parametric model)", score=0.0,
        author="Kamikaze project (procedurally generated, original work)", author_url=PROJECT_URL,
        license="MIT (project licence)", license_url=f"{PROJECT_URL}/blob/main/LICENSE", source_url=PROJECT_URL,
    )


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


def _result(product, attr: dict, status: str, source: str, cached: bool, query: str = "") -> dict:
    return {
        "class_id": product.class_id, "title": attr.get("title") or product.canonical_name, "glb_url": glb_url(product.class_id),
        "status": status, "source": source, "query": query, "cached": cached, "attribution": attr,
        "credit": attribution_line(attr), "thumbnail_url": attr.get("thumbnail_url", ""),
    }


def harvest(product, force: bool = False, offline: bool = False, budget_s: float | None = None) -> dict:
    """Resolve + cache a GLB for `product` (CanonicalProduct | metadata dict | free text) and write its attribution.

    Returns {"class_id", "title", "glb_url", "status": "online"|"cached", "source", "query", "cached", "attribution", "credit"}.
    """
    product = _coerce(product)
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
        deadline = time.monotonic() + (BUDGET_S if budget_s is None else budget_s)
        providers = () if offline else (("sketchfab", _try_sketchfab), ("polypizza", _try_polypizza), ("github", _try_khronos))
        found: tuple[Candidate, str] | None = None
        for name, provider in providers:
            try:
                found = provider(product, slug, deadline)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError, KeyError) as exc:
                log.warning("CAD source %s failed for %s: %s", name, product.class_id, exc)
                found = None
            if found:
                break
        status = "online"
        if not found:
            status = "cached"
            cand = _procedural(product, dest)
            found = (cand, "Kamikaze procedural generator") if cand else None
        if not found:
            return {"class_id": product.class_id, "title": product.canonical_name, "glb_url": "", "status": "cached",
                    "source": "none", "query": "", "cached": False, "attribution": {}, "credit": ""}
        cand, source_name = found
        attr = _record(product, cand, source_name)
        _write_json(sidecar_path(product.class_id), attr)
        write_credits_md()
        log.info("CAD for %s: %s | %s", product.class_id, source_name, attribution_line(attr))
        return _result(product, attr, status, cand.provider, False, cand.extra.get("query", ""))


def retrieve(label: str, meta: dict | None = None, force: bool = False, offline: bool = False) -> dict:
    """Backwards-compatible entry point: harvest the GLB for bank label `label` (+ optional enrollment metadata)."""
    try:
        product = entity_resolver.resolve_metadata(meta) if meta else entity_resolver.resolve(label.replace("_", " "))
    except taxonomy.MetadataError:
        found = mcp.search_3d_asset(label)
        if not found.get("glb_url"):
            return {**found, "source": found.get("status", "cached"), "attribution": {}, "credit": ""}
        host = urllib.parse.urlparse(found.get("model_url", "")).netloc or "public source"
        attr = {"title": found.get("title") or label, "author": "see source", "license": "see source",
                "source": host, "source_key": "fallback", "source_url": found.get("model_url", ""), "modified": False,
                "attribution_text": found.get("attribution") or "",
                "thumbnail_url": found.get("thumbnail_url", ""), "class_id": label,
                "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        _write_json(sidecar_path(label), attr)
        write_credits_md()
        return {**found, "class_id": label, "source": found.get("status", "cached"), "attribution": attr,
                "credit": attribution_line(attr)}
    return harvest(_with_class_id(product, label), force=force, offline=offline)
