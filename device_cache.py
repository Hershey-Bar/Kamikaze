"""Offline phone CAD cache: models/devices_db.json (GSMArena-verified dimensions) + pre-built GLBs.

scripts/populate_device_cache.py fills `cache_dir()` with `{device_id}.glb` (metres, Y-up, screen +Z, centre of mass at
the origin) and `{device_id}.json` (the attribution record it was built from: a ranked 1:1 community dummy that passed
the +/-2 mm gate, or the spec-generated fallback flagged as estimated). cad_harvester.harvest() consults this cache
before any network query, so a cached device installs with a file copy (no parsing / validation on the hot path).
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "models" / "devices_db.json"
PHONE_TIERS = ("pro max", "pro", "plus", "ultra", "mini")
DIM_TOL_MM = 2.0

_db_lock = threading.Lock()
_db: dict | None = None
_db_mtime = 0.0


def cache_dir() -> Path:
    """KAMIKAZE_CAD_CACHE, default <project>/models/cad_cache (= /home/arduino/hackathon_iot/models/cad_cache)."""
    return Path(os.environ.get("KAMIKAZE_CAD_CACHE") or ROOT / "models" / "cad_cache")


def load_db(path: Path | None = None) -> dict:
    """{"devices": {id: entry}} (reloaded when the file changes)."""
    global _db, _db_mtime
    path = Path(path or DB_PATH)
    if path != DB_PATH:
        return json.loads(path.read_text(encoding="utf-8"))
    with _db_lock:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return {"devices": {}}
        if _db is None or mtime != _db_mtime:
            _db = json.loads(path.read_text(encoding="utf-8"))
            _db_mtime = mtime
        return _db


def devices() -> dict[str, dict]:
    return load_db().get("devices", {})


def _tokens(text: str) -> list[str]:
    import cad_dummies

    return cad_dummies.tokens(text)


def device_id_for(text: str) -> str | None:
    """Free text / class id -> devices_db id: "Samsung Galaxy S24+" -> galaxy_s24_plus, "iphone_15_pro_max",
    "Galaxy Z Fold6" -> galaxy_z_fold_6. None when the text names no listed device (FE / Edge / Flip ...)."""
    toks = _tokens(text)
    if not toks or {"fe", "edge", "flip", "lite"} & set(toks):
        return None

    def after(word: str) -> str | None:
        for i, t in enumerate(toks[:-1]):
            if t == word and toks[i + 1].isdigit():
                return toks[i + 1]
        return None

    words = set(toks)
    dev = None
    if "iphone" in words and (gen := after("iphone")):
        tier = "_pro_max" if {"pro", "max"} <= words else "_pro" if "pro" in words else \
            "_plus" if "plus" in words else "_mini" if "mini" in words else ""
        dev = f"iphone_{gen}{tier}"
    elif "fold" in words and (gen := after("fold")):
        dev = f"galaxy_z_fold_{gen}"
    elif (gen := after("s")) and ({"galaxy", "samsung"} & words or toks[0] == "s"):
        tier = "_ultra" if "ultra" in words else "_plus" if "plus" in words else ""
        dev = f"galaxy_s{gen}{tier}"
    return dev if dev in devices() else None


def match_product(product) -> str | None:
    """devices_db id for a CanonicalProduct smartphone (brand + model, then class id), else None."""
    if getattr(product, "category", "") != "smartphone":
        return None
    for text in (f"{getattr(product, 'brand', '')} {getattr(product, 'model_or_variant', '')}",
                 getattr(product, "canonical_name", ""), getattr(product, "class_id", "")):
        dev = device_id_for(text or "")
        if dev:
            return dev
    return None


def spec_for(dev_id: str) -> dict:
    """spec_search.phone_specs-shaped verified spec from the DB entry (no network)."""
    e = devices()[dev_id]
    return {
        "verified": True, "category": "smartphone", "maker": e["brand"], "name": f"{e['brand']} {e['name']}",
        "height_mm": e["height_mm"], "width_mm": e["width_mm"], "depth_mm": e["depth_mm"],
        "rear_cameras": e["rear_cameras"], "weight_g": e.get("weight_g"), "island_type": e.get("island_type"),
        "sources": e.get("sources", []),
        "approximations": ["corner radius = 11% of width (not published)",
                           f"camera island drawn in the {e.get('island_type', 'maker')} design family at "
                           "proportional positions; lens count is exact, lens coordinates are not published"],
        **({"unfolded_mm": e["unfolded_mm"]} if e.get("unfolded_mm") else {}),
    }


def glb_file(dev_id: str) -> Path:
    return cache_dir() / f"{dev_id}.glb"


def sidecar_file(dev_id: str) -> Path:
    return cache_dir() / f"{dev_id}.json"


def cached(dev_id: str) -> dict | None:
    """Cached attribution record (with "glb" path) when both files exist."""
    glb, side = glb_file(dev_id), sidecar_file(dev_id)
    if not (glb.is_file() and side.is_file()):
        return None
    try:
        attr = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {**attr, "glb": str(glb)}


def install(dev_id: str, dest: Path) -> dict | None:
    """Copy the cached GLB to `dest`; returns the cached attribution record (None when not cached)."""
    rec = cached(dev_id)
    if rec is None:
        return None
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    shutil.copyfile(rec.pop("glb"), tmp)
    os.replace(tmp, dest)
    return rec


def status() -> dict:
    devs = devices()
    have = {d: cached(d) for d in devs}
    kinds: dict[str, int] = {}
    for rec in have.values():
        if rec:
            kinds[rec.get("source_key", "?")] = kinds.get(rec.get("source_key", "?"), 0) + 1
    return {"devices": len(devs), "cached": sum(1 for r in have.values() if r), "by_source": kinds,
            "dir": str(cache_dir()), "missing": sorted(d for d, r in have.items() if not r)}


# --------------------------------------------------------------------------------------
# GLB geometry helpers (in-memory)
# --------------------------------------------------------------------------------------

_GLB_MAGIC, _JSON, _BIN = b"glTF", 0x4E4F534A, 0x004E4942
_COMP = {5126: ("<f4", 4), 5125: ("<u4", 4), 5123: ("<u2", 2), 5121: ("u1", 1)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def parse_glb(data: bytes) -> tuple[dict, bytearray]:
    if len(data) < 20 or data[:4] != _GLB_MAGIC:
        raise ValueError("not a GLB")
    total = struct.unpack("<I", data[8:12])[0]
    off, doc, binary = 12, None, bytearray()
    while off + 8 <= min(total, len(data)):
        length, ctype = struct.unpack("<II", data[off:off + 8])
        payload = data[off + 8:off + 8 + length]
        if ctype == _JSON:
            doc = json.loads(payload.decode("utf-8"))
        elif ctype == _BIN and not binary:
            binary = bytearray(payload)
        off += 8 + length
    if doc is None:
        raise ValueError("GLB has no JSON chunk")
    return doc, binary


def pack_glb(doc: dict, binary: bytes) -> bytes:
    js = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * (-len(js) % 4)
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    body = struct.pack("<II", len(js), _JSON) + js + (struct.pack("<II", len(binary), _BIN) + binary if binary else b"")
    return _GLB_MAGIC + struct.pack("<II", 2, 12 + len(body)) + body


def _accessor(doc: dict, binary: bytearray, idx: int) -> tuple[np.ndarray, int, int]:
    """(values (count, n), byte offset, stride) of a tightly packed or strided accessor."""
    acc = doc["accessors"][idx]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype, size = _COMP[acc["componentType"]]
    n = _NCOMP[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride") or size * n
    count = acc["count"]
    buf = np.frombuffer(bytes(binary), dtype=np.uint8)
    sel = start + np.arange(count)[:, None] * stride + np.arange(size * n)[None, :]
    vals = np.ascontiguousarray(buf[sel]).view(dtype).reshape(count, n)
    return vals.astype(np.float64), start, stride


def _has_transforms(doc: dict) -> bool:
    return any(k in node for node in doc.get("nodes", []) for k in ("matrix", "translation", "rotation", "scale"))


def triangles(doc: dict, binary: bytearray) -> list[tuple[np.ndarray, int | None]]:
    """[(triangles (n, 3, 3) in mesh space, material index)] of every TRIANGLES primitive."""
    out = []
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4 or "POSITION" not in prim.get("attributes", {}):
                continue
            pos = _accessor(doc, binary, prim["attributes"]["POSITION"])[0]
            idx = _accessor(doc, binary, prim["indices"])[0].reshape(-1).astype(np.int64) \
                if "indices" in prim else np.arange(len(pos))
            idx = idx[: len(idx) // 3 * 3]
            out.append((pos[idx].reshape(-1, 3, 3), prim.get("material")))
    return out


def centre_of_mass(tris: np.ndarray) -> np.ndarray:
    """Solid centroid (signed tetrahedra); area-weighted surface centroid for open / degenerate meshes."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total = vol.sum()
    span = float(np.linalg.norm(np.ptp(tris.reshape(-1, 3), axis=0))) or 1.0
    if abs(total) > 1e-4 * span ** 3:
        return ((a + b + c) / 4.0 * vol[:, None]).sum(0) / total
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0
    return ((a + b + c) / 3.0 * area[:, None]).sum(0) / max(area.sum(), 1e-12)


def recenter(data: bytes) -> tuple[bytes, list[float]]:
    """Shift every POSITION accessor so the model's centre of mass sits at (0, 0, 0). Returns (GLB, old centre)."""
    doc, binary = parse_glb(data)
    if _has_transforms(doc):
        raise ValueError("node transforms present - recentering expects baked geometry")
    tris = np.concatenate([t for t, _ in triangles(doc, binary)] or [np.zeros((0, 3, 3))])
    if not len(tris):
        raise ValueError("GLB has no triangles")
    com = centre_of_mass(tris)
    done = set()
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            ai = prim.get("attributes", {}).get("POSITION")
            if ai is None or ai in done:
                continue
            done.add(ai)
            acc = doc["accessors"][ai]
            if acc["componentType"] != 5126 or acc["type"] != "VEC3":
                raise ValueError("POSITION accessor is not float32 VEC3")
            vals, start, stride = _accessor(doc, binary, ai)
            vals -= com
            sel = start + np.arange(len(vals))[:, None] * stride + np.arange(12)[None, :]
            np.frombuffer(binary, dtype=np.uint8)[sel] = np.ascontiguousarray(vals.astype("<f4")).view(np.uint8)
            acc["min"] = [float(x) for x in vals.min(0).astype(np.float32)]
            acc["max"] = [float(x) for x in vals.max(0).astype(np.float32)]
    return pack_glb(doc, binary), [round(float(x), 6) for x in com]


def extents_mm(data: bytes) -> dict:
    """Axis-aligned extent (mm) and centre-of-mass offset (mm) of a GLB in metres."""
    doc, binary = parse_glb(data)
    tris = np.concatenate([t for t, _ in triangles(doc, binary)])
    pts = tris.reshape(-1, 3)
    ext = np.ptp(pts, axis=0) * 1000.0
    com = centre_of_mass(tris) * 1000.0
    return {"x": round(float(ext[0]), 2), "y": round(float(ext[1]), 2), "z": round(float(ext[2]), 2),
            "com_mm": [round(float(v), 3) for v in com]}


def dimension_error(ext: dict, entry: dict) -> dict:
    """Height (Y) / width (X) error vs the DB spec; depth (Z) may exceed it by the camera bump."""
    dh = ext["y"] - entry["height_mm"]
    dw = ext["x"] - entry["width_mm"]
    dd = ext["z"] - entry["depth_mm"]
    ok = abs(dh) <= DIM_TOL_MM and abs(dw) <= DIM_TOL_MM and -DIM_TOL_MM <= dd <= 7.0
    return {"ok": ok, "height_mm": round(dh, 2), "width_mm": round(dw, 2), "depth_mm": round(dd, 2),
            "abs_mm": round(abs(dh) + abs(dw), 2)}
