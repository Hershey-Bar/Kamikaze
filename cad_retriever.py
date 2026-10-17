"""Autonomous 3D CAD / GLB retrieval for enrolled objects.

Source order for an object described by its enrollment metadata:
  1. Sketchfab Data API   (needs SKETCHFAB_TOKEN for the direct GLB download)
  2. Poly Pizza API       (needs POLY_PIZZA_API_KEY)
  3. Open GitHub glTF hubs (KhronosGroup/glTF-Sample-Assets model index, token-matched on the query)
  4. Offline procedural model (brand-coloured can/bottle, tier-aware phone) - always succeeds
The first GLB that downloads and passes the glTF magic check is cached at static/models/<slug>.glb.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import urllib.error

import mcp_server as mcp
import procedural_glb
import taxonomy

log = logging.getLogger("kamikaze.cad")

KHRONOS_INDEX = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/model-index.json"
KHRONOS_RAW = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
STOP_WORDS = {"glb", "3d", "model", "the", "a", "aluminium", "aluminum", "plastic", "smartphone", "can"}
RETRIEVE_BUDGET_S = float(os.environ.get("KAMIKAZE_CAD_BUDGET", "15"))
_index_lock = threading.Lock()
_index_cache: list | None = None


def _github_index() -> list:
    global _index_cache
    with _index_lock:
        if _index_cache is None:
            data = mcp._http_json(KHRONOS_INDEX, timeout=mcp.HTTP_TIMEOUT * 2)
            _index_cache = data if isinstance(data, list) else []
        return _index_cache


def search_github(query: str, deadline: float) -> dict | None:
    tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t and t not in STOP_WORDS]
    if not tokens:
        return None
    best, best_score = None, 0
    need = min(2, len(tokens))
    for entry in _github_index():
        text = " ".join([str(entry.get("name", "")), str(entry.get("label", "")), " ".join(entry.get("tags", []) or [])]).lower()
        score = sum(1 for t in tokens if t in text)
        if score >= need and score > best_score and "glTF-Binary" in (entry.get("variants") or {}):
            best, best_score = entry, score
    if not best:
        return None
    name = best["name"]
    return {
        "title": str(best.get("label") or name),
        "model_url": f"{KHRONOS_RAW}/{name}/glTF-Binary/{best['variants']['glTF-Binary']}",
        "thumbnail_url": f"{KHRONOS_RAW}/{name}/{best.get('screenshot', 'screenshot/screenshot.jpg')}",
    }


def _write_procedural(slug: str, meta: dict) -> bool:
    try:
        data = procedural_glb.build_for(meta)
        mcp.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = mcp.local_glb_path(slug)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        return mcp.is_valid_glb(dest)
    except Exception:
        log.exception("Procedural model generation failed for %s", slug)
        return False


def retrieve(label: str, meta: dict | None = None, force: bool = False, offline: bool = False) -> dict:
    """Resolve and cache a GLB for `label`. Returns {"title","model_url","thumbnail_url","status","source","glb_url"}."""
    slug = mcp.slugify(label)
    dest = mcp.local_glb_path(slug)
    glb_url = f"/static/models/{slug}.glb"
    if force:
        try:
            dest.unlink()
        except OSError:
            pass
    elif mcp.is_valid_glb(dest):
        return {"title": label, "model_url": "", "thumbnail_url": "", "status": "cached", "source": "local-cache", "glb_url": glb_url}
    if meta is None:
        found = mcp.search_3d_asset(label)
        return {**found, "source": found.get("status", "cached")}

    queries = taxonomy.search_queries(meta)
    deadline = time.monotonic() + RETRIEVE_BUDGET_S
    providers = () if offline else (
        ("sketchfab", mcp._search_sketchfab), ("polypizza", mcp._search_poly_pizza), ("github", search_github),
    )
    for name, provider in providers:
        for query in queries[:3]:
            if time.monotonic() > deadline:
                break
            try:
                hit = provider(query, deadline)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError, KeyError) as exc:
                log.warning("CAD source %s failed for %r: %s", name, query, exc)
                break
            if hit and hit.get("model_url") and mcp.download_glb(hit["model_url"], slug):
                log.info("CAD for %s from %s (%r)", label, name, query)
                return {**hit, "status": "online", "source": name, "query": query, "glb_url": glb_url}
    if _write_procedural(slug, meta):
        log.info("CAD for %s: offline procedural model", label)
        return {"title": taxonomy.display_name(meta), "model_url": "", "thumbnail_url": "", "status": "cached",
                "source": "procedural", "glb_url": glb_url}
    found = mcp.search_3d_asset(taxonomy.display_name(meta))
    return {**found, "source": found.get("status", "cached")}
