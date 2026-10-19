"""Local 3D model library: operator-owned GLBs (hand scans, purchased / licensed models) kept on the device.

Layout: <models_dir>/library/<id>.glb + <id>.json (attribution + match key). Library files are never committed (see
.gitignore) and never leave the device. A library model is used for an enrolled class when its match key
(category, brand, flavour / model) equals the class's - before any online search and also in Offline mode - by
copying it into the per-class cache (static/models/<class>.glb) with an attribution sidecar (source_key "library").
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import shutil
import threading
from pathlib import Path

import cad_harvester
import entity_resolver
import glb_orient
import mcp_server as mcp
import taxonomy

log = logging.getLogger("kamikaze.library")

_lock = threading.RLock()
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,95}$")
DEFAULT_AUTHOR = os.environ.get("KAMIKAZE_LIBRARY_AUTHOR", "Hershey-Bar")
DEFAULT_LICENSE = "owner-provided scan (all rights reserved by the author)"


def library_dir() -> Path:
    return cad_harvester.models_dir() / "library"


def valid_id(entry_id: str) -> str:
    entry_id = str(entry_id or "").strip().lower()
    if not _ID_RE.match(entry_id):
        raise ValueError(f"invalid library id {entry_id!r}")
    return entry_id


def _glb(entry_id: str) -> Path:
    return library_dir() / f"{entry_id}.glb"


def _json(entry_id: str) -> Path:
    return library_dir() / f"{entry_id}.json"


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split())


def match_key(product) -> dict | None:
    """{"category", "brand", "variant"} identifying which classes a library model depicts, or None when the product is
    too vague to bind a model to (e.g. a brand with no flavour / model). Brand compares on its first word, so
    "Kirkland" == "Kirkland Signature"; cans compare on flavour (else variant minus drink type), phones on model."""
    brand = _norm(getattr(product, "brand", "")).split()
    if not brand:
        return None
    category = getattr(product, "category", "") or ""
    if category == "smartphone":
        variant = _norm(product.model_or_variant)
    else:
        variant = _norm(product.flavor)
        if not variant:
            variant = _norm(product.sub_tier)
            for word in _norm(product.drink_type).split():
                variant = " ".join(w for w in variant.split() if w != word)
    if not variant:
        return None
    return {"category": category, "brand": brand[0], "variant": variant}


def _product(name: str, category: str = ""):
    try:
        return entity_resolver.resolve(name, category or "")
    except (taxonomy.MetadataError, ValueError) as exc:
        raise ValueError(f"cannot interpret {name!r} as a product: {exc}") from exc


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _read(entry_id: str) -> dict | None:
    rec = cad_harvester.read_json(_json(entry_id))
    if not isinstance(rec, dict) or not mcp.is_valid_glb(_glb(entry_id)):
        return None
    return rec


def _linked(entry_id: str) -> list[str]:
    out = []
    for rec in cad_harvester.credits():
        if rec.get("source_key") == "library" and rec.get("library_id") == entry_id:
            out.append(rec["class_id"])
    return out


def _public(entry_id: str, rec: dict, linked: list[str] | None = None) -> dict:
    path = _glb(entry_id)
    try:
        st = path.stat()
        size, version = st.st_size, st.st_mtime_ns
    except OSError:
        size, version = 0, 0
    try:
        rel = path.relative_to(cad_harvester.models_dir().parent)
        url = "/static/" + rel.as_posix() + f"?v={version}"
    except ValueError:
        url = ""
    return {**rec, "id": entry_id, "glb_url": url, "size_mb": round(size / 1e6, 2), "credit": cad_harvester.attribution_line(rec),
            "linked_classes": _linked(entry_id) if linked is None else linked}


def add(data: bytes | None = None, name: str = "", *, src: Path | None = None, filename: str = "", category: str = "",
        author: str = "", license: str = "", source: str = "", notes: str = "", entry_id: str = "",
        replace: bool = False) -> dict:
    """Validate (cad_harvester quality gate), auto-orient and store a GLB in the library. Returns the public entry.
    Raises ValueError (rejected / bad name) or FileExistsError (id taken and replace=False)."""
    name = str(name or "").strip() or Path(filename or "").stem.replace("_", " ").replace("-", " ").strip()
    if not name:
        raise ValueError("a product name is required (e.g. 'Kirkland Sparkling Water Lemon')")
    product = _product(name, category)
    key = match_key(product)
    entry_id = valid_id(entry_id or mcp.slugify(product.canonical_name)[:96])
    dest = _glb(entry_id)
    with _lock:
        if dest.exists() and not replace:
            raise FileExistsError(f"library model {entry_id!r} already exists")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".glb.part")
        if src is not None:
            shutil.copyfile(src, tmp)
        elif data:
            tmp.write_bytes(data)
        else:
            raise ValueError("empty upload")
        ok, why, info = cad_harvester.validate_glb(tmp)
        if not ok:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"library model rejected: {why}")
        orient = None
        try:
            orient = glb_orient.auto_orient(tmp, product.category)
        except (OSError, ValueError) as exc:
            log.warning("auto-orient skipped for library model %s: %s", entry_id, exc)
        os.replace(tmp, dest)
        author = author or DEFAULT_AUTHOR
        rec = {
            "title": f"{product.canonical_name} (hand scan)" if not source else product.canonical_name,
            "name": name, "author": author, "author_url": f"https://github.com/{author}" if re.match(r"^[\w-]+$", author) else "",
            "license": license or DEFAULT_LICENSE, "license_url": "", "source": source or "Hand scan (local library)",
            "source_key": "library", "source_url": "", "thumbnail_url": "", "modified": False,
            "category": product.category, "brand": product.brand, "variant": product.model_or_variant,
            "flavor": product.flavor, "match_key": key, "filename": Path(filename or (src.name if src else "")).name,
            "quality": info, "notes": notes, "scale": "as scanned (model-viewer frames it automatically)",
            "added_at": _now(),
        }
        if orient:
            rec["orientation_fix"] = {k: orient[k] for k in ("rotation", "description", "note", "identity")}
        if not source:
            rec["attribution_text"] = (f"\u201c{rec['title']}\u201d - 3D hand scan by {author}"
                                       + (f" ({rec['author_url']})" if rec["author_url"] else "")
                                       + " - owner-provided, stored on this device only")
        cad_harvester.write_json(_json(entry_id), rec)
    log.info("Library model %s added: %s (%s vertices, %.1f MB)%s", entry_id, rec["title"], info.get("vertices"),
             info.get("size_mb", 0), "" if key else " - no match key (too vague to auto-bind to a class)")
    return _public(entry_id, rec, [])


def entries() -> list[dict]:
    root = library_dir()
    if not root.is_dir():
        return []
    linked: dict[str, list[str]] = {}
    for rec in cad_harvester.credits():
        if rec.get("source_key") == "library" and rec.get("library_id"):
            linked.setdefault(rec["library_id"], []).append(rec["class_id"])
    out = []
    for js in sorted(root.glob("*.json")):
        rec = _read(js.stem)
        if rec is not None:
            out.append(_public(js.stem, rec, linked.get(js.stem, [])))
    return out


def get(entry_id: str) -> dict | None:
    entry_id = valid_id(entry_id)
    rec = _read(entry_id)
    return _public(entry_id, rec) if rec is not None else None


def delete(entry_id: str) -> dict:
    """Remove a library model (GLB + sidecar). Returns {"ok", "id", "linked_classes"}: the per-class copies made from
    it, which the caller should also drop (AssetManager.forget) so no class keeps showing a deleted model."""
    entry_id = valid_id(entry_id)
    with _lock:
        linked = _linked(entry_id)
        removed = False
        for path in (_glb(entry_id), _json(entry_id)):
            try:
                path.unlink()
                removed = removed or path.suffix == ".glb"
            except OSError:
                pass
    if removed:
        log.info("Library model %s deleted (linked classes: %s)", entry_id, ", ".join(linked) or "none")
    return {"ok": removed, "id": entry_id, "linked_classes": linked}


def find(product) -> dict | None:
    """The library entry whose match key equals the product's, or None."""
    key = match_key(product)
    if key is None:
        return None
    for entry in entries():
        if entry.get("match_key") == key:
            return entry
    return None


def install(entry_id: str, class_id: str) -> dict:
    """Copy library model `entry_id` into the per-class cache of `class_id` and write its attribution sidecar.
    Returns the sidecar. Caller holds no harvester lock (cad_harvester.harvest calls this under its own lock)."""
    entry_id = valid_id(entry_id)
    rec = _read(entry_id)
    if rec is None:
        raise FileNotFoundError(f"library model {entry_id!r} not found")
    dest = cad_harvester.glb_path(class_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".lib.part")
    shutil.copyfile(_glb(entry_id), tmp)
    os.replace(tmp, dest)
    attr = {k: rec.get(k) for k in ("title", "author", "author_url", "license", "license_url", "source", "source_url",
                                    "thumbnail_url", "quality", "orientation_fix", "notes", "attribution_text")}
    attr = {k: v for k, v in attr.items() if v is not None}
    attr.update({"source_key": "library", "library_id": entry_id, "class_id": class_id, "match": "library",
                 "badge": "Local library", "modified": False, "retrieved_at": _now()})
    cad_harvester.write_json(cad_harvester.sidecar_path(class_id), attr)
    cad_harvester.write_credits_md()
    log.info("CAD for %s: local library model %s (%s)", class_id, entry_id, rec.get("title"))
    return attr
