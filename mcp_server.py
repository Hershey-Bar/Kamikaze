#!/usr/bin/env python3
"""Kamikaze MCP server: 3D asset retrieval/caching, Modulino Pixels actuation, enrollment and guidance.

Run over stdio:   python mcp_server.py
Import directly:  from mcp_server import set_leds, search_3d_asset, actuate_modulino, trigger_guidance

Tools: search_3d_asset, actuate_modulino, trigger_guidance, enroll_object, get_guidance.
enroll_object / get_guidance talk to the running receiver.py (the single owner of the prototype bank)
over its local HTTP API, so an agent can drive the same live state the dashboard shows.

Environment (all optional):
  SKETCHFAB_TOKEN        Sketchfab API token (required for GLB download via the Data API)
  POLY_PIZZA_API_KEY     Poly Pizza API key
  KAMIKAZE_ROUTER_SOCK   Router socket path (default /var/run/arduino-router.sock)
  KAMIKAZE_RECEIVER_URL  Receiver base URL (default http://127.0.0.1:5000)
  KAMIKAZE_MODELS_DIR    GLB cache directory (default static/models next to this file)
  KAMIKAZE_FALLBACKS     JSON file overriding/extending the cached model table
                         (default: fallback_models.json next to this file)
  KAMIKAZE_HTTP_TIMEOUT  Per-request network timeout in seconds (default 3.0)

stdout is reserved for the MCP stdio protocol; all logging goes to stderr.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import msgpack
from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=os.environ.get("KAMIKAZE_LOG", "INFO"),
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("kamikaze.mcp")

mcp = FastMCP("kamikaze")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = Path(os.environ.get("KAMIKAZE_MODELS_DIR", STATIC_DIR / "models"))
ROUTER_SOCK_PATH = os.environ.get("KAMIKAZE_ROUTER_SOCK", "/var/run/arduino-router.sock")
RECEIVER_URL = os.environ.get("KAMIKAZE_RECEIVER_URL", "http://127.0.0.1:5000").rstrip("/")
HTTP_TIMEOUT = float(os.environ.get("KAMIKAZE_HTTP_TIMEOUT", "3.0"))
DOWNLOAD_TIMEOUT = float(os.environ.get("KAMIKAZE_DOWNLOAD_TIMEOUT", "30"))
MAX_GLB_BYTES = 64 * 1024 * 1024
RPC_TIMEOUT = 1.0
ONLINE_CACHE_TTL = 600.0
USER_AGENT = "kamikaze-edge/1.0"

# --------------------------------------------------------------------------------------
# Target domain catalog
# --------------------------------------------------------------------------------------

IPHONE_GENERATIONS = (12, 13, 14, 15, 16)
IPHONE_VARIANTS = ("", "Pro", "Pro Max")
GALAXY_GENERATIONS = (20, 21, 22, 23, 24)
GALAXY_VARIANTS = ("", "Plus", "Ultra")
CAN_LABELS = ("Coca-Cola Classic", "Diet Coke", "Coke Zero", "Pepsi", "Sprite")


def _phone_labels() -> list[str]:
    labels = []
    for gen in IPHONE_GENERATIONS:
        for var in IPHONE_VARIANTS:
            labels.append(f"Apple iPhone {gen} {var}".strip())
    for gen in GALAXY_GENERATIONS:
        for var in GALAXY_VARIANTS:
            labels.append(f"Samsung Galaxy S{gen} {var}".strip())
    return labels


PHONE_LABELS: tuple[str, ...] = tuple(_phone_labels())
TARGET_LABELS: tuple[str, ...] = PHONE_LABELS + CAN_LABELS


def label_category(label: str) -> str | None:
    if label in PHONE_LABELS:
        return "phone"
    if label in CAN_LABELS:
        return "can"
    return None


# --------------------------------------------------------------------------------------
# Cached fallback model table (direct public .glb URLs, always reachable offline of APIs)
# --------------------------------------------------------------------------------------
# Khronos glTF-Sample-Assets are stable, public, CORS-enabled GLBs hosted on GitHub raw.
# Each target category has its own entry so any product can be swapped for a real model
# by editing fallback_models.json without touching code.

_KHRONOS = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"


def _khronos(name: str) -> str:
    return f"{_KHRONOS}/{name}/glTF-Binary/{name}.glb"


_PHONE_GLB = _khronos("Box")
_CAN_GLB = _khronos("WaterBottle")


def _thumb(name: str) -> str:
    ext = "png" if name == "Box" else "jpg"
    return f"{_KHRONOS}/{name}/screenshot/screenshot.{ext}"


def _build_default_fallbacks() -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    for gen in IPHONE_GENERATIONS:
        table[f"iphone {gen}"] = {
            "title": f"Apple iPhone {gen}",
            "model_url": _PHONE_GLB,
            "thumbnail_url": _thumb("Box"),
        }
    for gen in GALAXY_GENERATIONS:
        table[f"galaxy s{gen}"] = {
            "title": f"Samsung Galaxy S{gen}",
            "model_url": _PHONE_GLB,
            "thumbnail_url": _thumb("Box"),
        }
    for key, title in (
        ("coca-cola", "Coca-Cola Classic Can"),
        ("diet coke", "Diet Coke Can"),
        ("coke zero", "Coke Zero Can"),
        ("pepsi", "Pepsi Can"),
        ("sprite", "Sprite Can"),
    ):
        table[key] = {
            "title": title,
            "model_url": _CAN_GLB,
            "thumbnail_url": _thumb("WaterBottle"),
        }
    table["_generic_phone"] = {
        "title": "Smartphone",
        "model_url": _PHONE_GLB,
        "thumbnail_url": _thumb("Box"),
    }
    table["_generic_can"] = {
        "title": "Beverage Can",
        "model_url": _CAN_GLB,
        "thumbnail_url": _thumb("WaterBottle"),
    }
    return table


def _load_fallbacks() -> dict[str, dict[str, str]]:
    table = _build_default_fallbacks()
    path = Path(os.environ.get("KAMIKAZE_FALLBACKS", Path(__file__).with_name("fallback_models.json")))
    if path.is_file():
        try:
            overrides = json.loads(path.read_text(encoding="utf-8"))
            for key, entry in overrides.items():
                if isinstance(entry, dict) and entry.get("model_url"):
                    base = table.get(key.lower(), {})
                    table[key.lower()] = {
                        "title": entry.get("title", base.get("title", key)),
                        "model_url": entry["model_url"],
                        "thumbnail_url": entry.get("thumbnail_url", base.get("thumbnail_url", "")),
                    }
            log.info("Loaded fallback overrides from %s", path)
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable fallback file %s: %s", path, exc)
    return table


FALLBACK_MODELS: dict[str, dict[str, str]] = _load_fallbacks()

_IPHONE_RE = re.compile(r"iphone\s*(1[2-6])\b")
_GALAXY_RE = re.compile(r"(?:galaxy\s*)?\bs\s?(2[0-4])\b")


def resolve_fallback(query: str) -> dict[str, str] | None:
    """Map a free-text query to a cached model entry, keeping variant info in the title."""
    q = re.sub(r"\s+", " ", query.lower().replace("_", " ").replace("-", " ")).strip()
    entry_key: str | None = None
    title: str | None = None

    m = _IPHONE_RE.search(q)
    if m:
        entry_key = f"iphone {m.group(1)}"
        variant = " Pro Max" if re.search(r"\bpro max\b", q) else " Pro" if re.search(r"\bpro\b", q) else ""
        title = f"Apple iPhone {m.group(1)}{variant}"
    elif "samsung" in q or "galaxy" in q or _GALAXY_RE.search(q):
        m = _GALAXY_RE.search(q)
        if m:
            entry_key = f"galaxy s{m.group(1)}"
            variant = " Ultra" if "ultra" in q else " Plus" if ("plus" in q or "+" in query) else ""
            title = f"Samsung Galaxy S{m.group(1)}{variant}"
    if entry_key is None:
        if "zero" in q:
            entry_key = "coke zero"
        elif "diet" in q:
            entry_key = "diet coke"
        elif "pepsi" in q:
            entry_key = "pepsi"
        elif "sprite" in q:
            entry_key = "sprite"
        elif "coca" in q or "coke" in q or "cola" in q:
            entry_key = "coca-cola"
        elif any(w in q for w in ("phone", "iphone", "galaxy", "samsung", "apple", "smartphone", "pixel", "android")) \
                or re.search(r"\bs\s?2\d\b", q):
            entry_key = "_generic_phone"
            m = re.search(r"\bs\s?(2\d)\b", q)
            if m:
                variant = " Ultra" if "ultra" in q else " Plus" if ("plus" in q or "+" in query) else ""
                title = f"Samsung Galaxy S{m.group(1)}{variant}"
        elif any(w in q for w in ("can", "soda", "drink", "beverage")):
            entry_key = "_generic_can"
    if entry_key is None or entry_key not in FALLBACK_MODELS:
        return None
    entry = dict(FALLBACK_MODELS[entry_key])
    if title and entry_key == "_generic_phone":
        entry["title"] = title
    elif title and entry_key.startswith(("iphone", "galaxy")) and entry["title"].startswith(("Apple iPhone", "Samsung Galaxy")):
        entry["title"] = title
    return entry


# --------------------------------------------------------------------------------------
# Online providers
# --------------------------------------------------------------------------------------

_online_cache: dict[str, tuple[float, dict[str, str]]] = {}
_online_cache_lock = threading.Lock()


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: float = HTTP_TIMEOUT) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pick_thumbnail(thumbnails: dict[str, Any] | None) -> str:
    images = (thumbnails or {}).get("images") or []
    usable = [i for i in images if i.get("url")]
    if not usable:
        return ""
    wide = sorted((i for i in usable if int(i.get("width", 0)) >= 200), key=lambda i: int(i.get("width", 0)))
    return (wide[0] if wide else max(usable, key=lambda i: int(i.get("width", 0))))["url"]


def _search_sketchfab(query: str, deadline: float) -> dict[str, str] | None:
    token = os.environ.get("SKETCHFAB_TOKEN")
    if not token:
        return None
    params = urllib.parse.urlencode({"type": "models", "q": query, "downloadable": "true", "count": 6})
    data = _http_json(f"https://api.sketchfab.com/v3/search?{params}", {"Authorization": f"Token {token}"})
    for model in data.get("results", []):
        if time.monotonic() > deadline:
            break
        uid = model.get("uid")
        if not uid:
            continue
        try:
            dl = _http_json(f"https://api.sketchfab.com/v3/models/{uid}/download", {"Authorization": f"Token {token}"})
        except (urllib.error.URLError, OSError, ValueError):
            continue
        glb = (dl.get("glb") or {}).get("url")
        if glb:
            return {
                "title": str(model.get("name", query)),
                "model_url": glb,
                "thumbnail_url": _pick_thumbnail(model.get("thumbnails")),
            }
    return None


def _search_poly_pizza(query: str, deadline: float) -> dict[str, str] | None:
    key = os.environ.get("POLY_PIZZA_API_KEY")
    if not key:
        return None
    url = f"https://api.poly.pizza/v1.1/search/{urllib.parse.quote(query)}"
    data = _http_json(url, {"x-auth-token": key})
    for item in data.get("results", []):
        glb = item.get("Download") or item.get("download")
        if glb:
            return {
                "title": str(item.get("Title") or item.get("title") or query),
                "model_url": glb,
                "thumbnail_url": str(item.get("Thumbnail") or item.get("thumbnail") or ""),
            }
    return None


def _search_online(query: str) -> dict[str, str] | None:
    key = query.lower().strip()
    now = time.monotonic()
    with _online_cache_lock:
        hit = _online_cache.get(key)
        if hit and now - hit[0] < ONLINE_CACHE_TTL:
            return dict(hit[1])
    deadline = now + 2.5 * HTTP_TIMEOUT
    for provider in (_search_sketchfab, _search_poly_pizza):
        try:
            result = provider(query, deadline)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            log.warning("%s failed for %r: %s", provider.__name__, query, exc)
            continue
        if result:
            with _online_cache_lock:
                _online_cache[key] = (time.monotonic(), result)
            return dict(result)
    return None


# --------------------------------------------------------------------------------------
# MCP tools
# --------------------------------------------------------------------------------------


def lookup_3d_asset(query: str, format: str = "glb") -> dict:
    """Resolve a query to {"title", "model_url", "thumbnail_url", "status"} without downloading anything."""
    query = (query or "").strip()
    if format.lower() != "glb":
        log.info("Requested format %r unsupported; serving glb", format)
    online = _search_online(query) if query else None
    if online:
        return {**online, "status": "online"}
    cached = resolve_fallback(query)
    if cached:
        return {**cached, "status": "cached"}
    log.warning("No online or cached model for %r", query)
    return {"title": query or "unknown", "model_url": "", "thumbnail_url": "", "status": "cached"}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:60] or "asset"


def local_glb_path(slug: str) -> Path:
    return MODELS_DIR / f"{slug}.glb"


def is_valid_glb(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"glTF" and path.stat().st_size > 12
    except OSError:
        return False


_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def download_glb(url: str, slug: str) -> Path | None:
    """Download a GLB into the model cache (atomic, magic-checked). Returns the file path or None."""
    dest = local_glb_path(slug)
    with _download_locks_guard:
        lock = _download_locks.setdefault(slug, threading.Lock())
    with lock:
        if is_valid_glb(dest):
            return dest
        tmp = dest.with_suffix(".part")
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, open(tmp, "wb") as out:
                total = 0
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_GLB_BYTES:
                        raise ValueError("GLB exceeds size limit")
                    out.write(chunk)
            if not is_valid_glb(tmp):
                raise ValueError("response is not a binary glTF (GLB)")
            os.replace(tmp, dest)
            log.info("Cached %s (%d KB) from %s", dest.name, dest.stat().st_size // 1024, url)
            return dest
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            log.warning("GLB download failed for %s: %s", url, exc)
            try:
                tmp.unlink()
            except OSError:
                pass
            return None


@mcp.tool()
def search_3d_asset(query: str, format: str = "glb") -> dict:
    """Find and cache a 3D model for a product query (e.g. "Apple iPhone 15 Pro", "coke_can").

    Tries Sketchfab then Poly Pizza (when API credentials are configured) and falls back to a
    built-in table of public GLB URLs so live demos survive network failures and rate limits.
    The chosen GLB is downloaded once into static/models/<query-slug>.glb.
    Returns {"title", "model_url", "thumbnail_url", "status": "online" | "cached"} plus
    "glb_url" (the locally served /static/models/... path) when the file is cached on disk.
    Only GLB output is supported; other formats are served as GLB.
    """
    entry = lookup_3d_asset(query, format)
    slug = slugify(query)
    if is_valid_glb(local_glb_path(slug)):
        return {**entry, "glb_url": f"/static/models/{slug}.glb"}
    candidates = [entry]
    fallback = resolve_fallback(query or "")
    if fallback and fallback["model_url"] != entry["model_url"]:
        candidates.append({**fallback, "status": "cached"})
    for cand in candidates:
        if cand["model_url"] and download_glb(cand["model_url"], slug):
            return {**cand, "glb_url": f"/static/models/{slug}.glb"}
    return entry


class _RouterClient:
    """Persistent MessagePack-RPC client for the Arduino Router Unix socket."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._sock: socket.socket | None = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        if not hasattr(socket, "AF_UNIX"):
            raise OSError("AF_UNIX sockets are not available on this platform")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(RPC_TIMEOUT)
            sock.connect(self.path)
        except OSError:
            sock.close()
            raise
        return sock

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _roundtrip(self, sock: socket.socket, method: str, params: list[Any]) -> bool:
        msg_id = next(self._ids) & 0x7FFFFFFF
        sock.sendall(msgpack.packb([0, msg_id, method, params], use_bin_type=True))
        unpacker = msgpack.Unpacker(raw=False, max_buffer_size=65536)
        deadline = time.monotonic() + RPC_TIMEOUT
        while time.monotonic() < deadline:
            data = sock.recv(4096)
            if not data:
                raise ConnectionError("router closed the connection")
            unpacker.feed(data)
            for msg in unpacker:
                if isinstance(msg, list) and len(msg) == 4 and msg[0] == 1 and msg[1] == msg_id:
                    if msg[2] is not None:
                        log.error("RPC %s failed: %r", method, msg[2])
                        return False
                    return True
        raise TimeoutError(f"no reply to {method}")

    def call(self, method: str, params: list[Any]) -> bool:
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._sock is None:
                        self._sock = self._connect()
                    return self._roundtrip(self._sock, method, params)
                except (OSError, ConnectionError, TimeoutError, msgpack.UnpackException) as exc:
                    self._close()
                    if attempt == 2:
                        log.warning("Router RPC %s failed: %s", method, exc)
                        return False
            return False


_router = _RouterClient(ROUTER_SOCK_PATH)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def set_leds(r: int, g: int, b: int, brightness: int, count: int) -> bool:
    """Drive the Modulino Pixels via MessagePack-RPC `set_pixels`; True once the router acknowledges."""
    params = [_clamp(r, 0, 255), _clamp(g, 0, 255), _clamp(b, 0, 255), _clamp(brightness, 0, 100), _clamp(count, 0, 8)]
    return _router.call("set_pixels", params)


@mcp.tool()
def actuate_modulino(r: int, g: int, b: int, brightness: int, count: int) -> dict:
    """Set the 8-pixel Modulino bar (r/g/b 0-255, brightness 0-100, count 0-8 LEDs lit).

    Returns {"success": bool, "r", "g", "b", "brightness", "count"} with the clamped values sent.
    """
    params = {
        "r": _clamp(r, 0, 255), "g": _clamp(g, 0, 255), "b": _clamp(b, 0, 255),
        "brightness": _clamp(brightness, 0, 100), "count": _clamp(count, 0, 8),
    }
    return {"success": set_leds(**params), **params}


@mcp.tool()
def trigger_guidance(reason: str, prompt_user: str) -> dict:
    """Light 4 amber LEDs and tell the user how to reposition the object."""
    ok = set_leds(255, 120, 0, 40, 4)
    log.info("Guidance (%s): %s [leds=%s]", reason, prompt_user, "ok" if ok else "unavailable")
    return {"action": "reorient", "message": prompt_user}


def _receiver_call(method: str, path: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{RECEIVER_URL}{path}", data=data, method=method,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read().decode("utf-8"))
        except ValueError:
            return {"ok": False, "error": f"receiver returned HTTP {exc.code}"}
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        return {"ok": False, "error": f"receiver unreachable at {RECEIVER_URL}: {exc}"}


@mcp.tool()
def enroll_object(label: str) -> dict:
    """Few-shot enroll the object currently in front of the camera under `label` (e.g. "coke_can").

    Captures the next 12 live frames from the receiver, stores their L2-normalised centroid in the
    prototype bank and persists it. Blocks until done. Returns {"ok", "label", "frames", "consistency", ...}.
    """
    return _receiver_call("POST", "/api/enroll", {"label": label, "wait": True}, timeout=45.0)


@mcp.tool()
def get_guidance() -> dict:
    """Return the live guidance prompt: {"state", "message", "action", "label", "confidence", "margin"}."""
    state = _receiver_call("GET", "/api/state", timeout=5.0)
    if state.get("ok") is False and "error" in state:
        return state
    state_name = state.get("state", "SEARCHING")
    return {
        "state": state_name,
        "message": state.get("guidance", ""),
        "action": {"IDENTIFIED": "none", "GUIDING": "reorient"}.get(state_name, "present_object"),
        "label": state.get("label"),
        "confidence": state.get("confidence"),
        "margin": state.get("margin"),
    }


@mcp.tool()
def enroll_multiview(raw_name: str, category: str = "", brand: str = "", variant: str = "", package_type: str = "",
                     container_volume: str = "", frames_per_view: int = 6, pause_s: float = 4.0) -> dict:
    """Guided 4-angle enrollment (front, back, side, detail) of the object in front of the camera.

    Starts a session with the structured metadata, then captures `frames_per_view` frames per view, waiting
    `pause_s` seconds before each so the object can be rotated. Returns the final enrollment status.
    """
    body = {"raw_name": raw_name, "category": category, "brand": brand, "variant": variant,
            "package_type": package_type, "container_volume": container_volume, "frames_per_view": frames_per_view}
    status = _receiver_call("POST", "/api/enroll/start", {k: v for k, v in body.items() if v not in ("", None)}, timeout=10.0)
    if not status.get("ok"):
        return status
    for view in ("front", "back", "side", "detail"):
        time.sleep(max(0.0, pause_s))
        status = _receiver_call("POST", "/api/enroll/capture", {"view": view, "wait": True, "timeout": 30}, timeout=45.0)
        if not status.get("ok"):
            _receiver_call("POST", "/api/enroll/cancel", {}, timeout=5.0)
            return status
    return _receiver_call("GET", "/api/enroll/status", timeout=5.0)


@mcp.tool()
def fetch_cad_asset(raw_name: str, category: str = "", brand: str = "", variant: str = "", force: bool = False) -> dict:
    """Find, download and cache a 3D GLB model for a product (Sketchfab -> Poly Pizza -> GitHub -> procedural fallback).

    Returns {"title", "model_url", "thumbnail_url", "status", "source", "glb_url"} where glb_url is served from /static/models.
    """
    import cad_retriever
    import taxonomy

    meta = taxonomy.parse_metadata({k: v for k, v in {"raw_name": raw_name, "category": category, "brand": brand,
                                                       "variant": variant}.items() if v})
    return cad_retriever.retrieve(taxonomy.label_from_metadata(meta), meta, force=force)


@mcp.tool()
def get_pose_telemetry() -> dict:
    """Latest estimated physical pose {"pitch", "yaw", "roll"} (degrees) of the object held in front of the camera."""
    return _receiver_call("GET", "/api/pose", timeout=5.0)


if __name__ == "__main__":
    mcp.run(transport="stdio")
