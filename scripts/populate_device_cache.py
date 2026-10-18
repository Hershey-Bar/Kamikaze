"""Pre-build the offline phone CAD cache (models/cad_cache/{device_id}.glb + .json) and phone_embeddings.npz.

For every device in models/devices_db.json (GSMArena-verified dimensions):
  1. 1:1 community dummies are searched on Printables (+ Thingiverse when THINGIVERSE_TOKEN is set), ranked by title
     relevance, popularity (likes / downloads) and licence (cad_dummies.search). The top candidates are downloaded
     and converted (metres, height +Y, screen +Z, camera bump -Z at the top). Every converted mesh is measured and
     must match the published height and width within +/-2.0 mm (depth: published -2 .. +7 mm for the camera bump).
     The back must also carry camera relief (device_cache.back_relief: lens rings / island standing >= 0.35 mm off
     the back glass); blank size-tester slabs are rejected. Among the passing files the most accurate + best regarded
     wins: rank = candidate score - 3 x |error mm|. CC non-commercial prints are only considered with --allow-nc
     (local, attributed, non-commercial display).
  2. Otherwise the spec-generated body (procedural_glb, exact published dimensions, design-family camera island,
     flagged ESTIMATED with its approximations) is cached so the device still loads offline.
  3. The GLB is re-centred so its centre of mass is at (0, 0, 0) and written with an attribution sidecar.
Then the camera-island renders are embedded with DINOv2 (when the exported model is present) into
models/phone_embeddings.npz for phone_classifier.py.

    python scripts/populate_device_cache.py [--only id,id] [--force] [--retry-estimated] [--budget 120]
                                            [--allow-nc] [--offline] [--no-embeddings]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import re
import time
import urllib.error
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import cad_dummies  # noqa: E402
import cad_harvester  # noqa: E402
import device_cache  # noqa: E402
import phone_classifier  # noqa: E402
import procedural_glb  # noqa: E402

log = logging.getLogger("populate")
MAX_CANDIDATES = 10
MAX_DOWNLOADS = 12
ENOUGH_PASSING = 3
# Prints titled as accessories are skipped before download: a case or holder is ~phone-sized and could pass the
# +/-2 mm gate without being a 1:1 replica. A title that also says dummy/replica/template is still considered.
ACCESSORY_WORDS = {"stand", "case", "cases", "mount", "holder", "dock", "raiser", "riser", "adapter", "controller",
                   "joycon", "cover", "bumper", "grip", "tripod", "clip", "wallet", "charger", "cradle", "sleeve"}
DUMMY_WORDS = {"dummy", "replica", "mockup", "mock", "template", "1:1", "blank", "body"}
# pre-launch models drawn from rumours / leaks / concept renders are not verified geometry
SPECULATIVE_WORDS = {"rumor", "rumored", "rumors", "rumour", "rumoured", "rumours", "leak", "leaked", "leaks",
                     "concept", "speculative", "unofficial"}


def is_accessory(title: str) -> bool:
    words = set(re.findall(r"[a-z0-9:]+", (title or "").lower()))
    return bool(words & ACCESSORY_WORDS) and not words & DUMMY_WORDS


def is_speculative(title: str) -> bool:
    return bool(set(re.findall(r"[a-z0-9:]+", (title or "").lower())) & SPECULATIVE_WORDS)


def product_for(dev_id: str, e: dict) -> SimpleNamespace:
    return SimpleNamespace(brand=e["brand"], model_or_variant=e["name"], canonical_name=f"{e['brand']} {e['name']}",
                           category="smartphone", class_id=dev_id, raw_query=e["name"], container_volume="")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def best_dummy(dev_id: str, e: dict, budget_s: float) -> tuple[dict, bytes, dict] | None:
    """(candidate + report, centred GLB, ranking info) of the best passing 1:1 dummy, or None."""
    product = product_for(dev_id, e)
    spec = device_cache.spec_for(dev_id)
    providers = ("printables", "thingiverse") if cad_dummies._thingiverse_token() else ("printables",)
    deadline = time.monotonic() + budget_s
    try:
        cands = cad_dummies.search(product, providers)[:MAX_CANDIDATES]
    except (urllib.error.URLError, OSError, ValueError, cad_dummies.DummyError) as exc:
        log.warning("%s: dummy search failed: %s", dev_id, exc)
        return None
    ident = cad_dummies.identity(product)
    passing, rejected, downloads = [], [], 0
    for cand in cands:
        if is_accessory(cand["title"]):
            rejected.append({"uid": cand["uid"], "why": f"accessory title: {cand['title']!r}"})
            continue
        if is_speculative(cand["title"]):
            rejected.append({"uid": cand["uid"], "why": f"pre-release / speculative model: {cand['title']!r}"})
            continue
        if time.monotonic() > deadline or downloads >= MAX_DOWNLOADS or len(passing) >= ENOUGH_PASSING:
            break
        _, list_files, download = cad_dummies.PROVIDERS[cand["provider"]]
        try:
            files = list_files(cand)
        except (urllib.error.URLError, OSError, ValueError, cad_dummies.DummyError) as exc:
            rejected.append({"uid": cand["uid"], "why": f"file list: {exc}"})
            continue
        picks = [f for f in files if cad_dummies.file_matches(f["name"], cand["title"], ident)[0]
                 and (not f.get("size") or f["size"] <= cad_dummies.MAX_MESH_BYTES)]
        picks.sort(key=lambda f: (not f["name"].lower().endswith(".stl"),
                                  -len(set(cad_dummies.tokens(f["name"])) & ident["tiers"])))
        for f in picks[:2]:
            if time.monotonic() > deadline or downloads >= MAX_DOWNLOADS:
                break
            downloads += 1
            try:
                data = download(cand, f)
                glb, report = cad_dummies.convert(data, f["name"], spec, {"source_url": cand["source_url"]},
                                                  allow_simplify=not cad_dummies._no_derivatives(cand["license"]))
                glb, com = device_cache.recenter(glb)
                ext = device_cache.extents_mm(glb)
                err = device_cache.dimension_error(ext, e)
            except (urllib.error.URLError, OSError, ValueError, cad_dummies.DummyError, zipfile.BadZipFile,
                    ET.ParseError) as exc:
                rejected.append({"uid": cand["uid"], "file": f["name"], "why": str(exc)[:160]})
                continue
            if not err["ok"]:
                rejected.append({"uid": cand["uid"], "file": f["name"],
                                 "why": f"+/-{device_cache.DIM_TOL_MM} mm gate: {err}"})
                continue
            # a right-sized slab without lens rings / island (size testers, blank shells) is not a 1:1 replica
            relief = device_cache.back_relief(glb)
            if relief["features"] == 0:
                rejected.append({"uid": cand["uid"], "file": f["name"],
                                 "why": f"no camera relief on the back (max {relief['max_mm']} mm): blank slab"})
                continue
            rank = cand["score"] - 3.0 * err["abs_mm"]
            passing.append(({**cand, "file": f["name"], "report": report}, glb,
                            {"rank": round(rank, 2), "error_mm": err, "extent_mm": ext, "com_shift_m": com,
                             "camera_bump": True, "camera_relief": relief}))
            log.info("%s: pass %r / %s  err %s  rank %.1f", dev_id, cand["title"], f["name"], err, rank)
            break  # one passing file per print is enough
    if not passing:
        log.info("%s: no dummy passed (%d candidates, %d downloads)", dev_id, len(cands), downloads)
        return None
    passing.sort(key=lambda p: -p[2]["rank"])
    cand, glb, info = passing[0]
    info["considered"] = len(cands)
    info["downloads"] = downloads
    info["passing"] = [{"uid": c["uid"], "title": c["title"], "rank": i["rank"], "error_mm": i["error_mm"]["abs_mm"]}
                       for c, _, i in passing]
    info["rejected"] = rejected[:12]
    return cand, glb, info


def estimated(dev_id: str, e: dict) -> tuple[dict, bytes]:
    spec = device_cache.spec_for(dev_id)
    data = procedural_glb.build_from_spec(spec, e["name"], {"class_id": dev_id, "device_id": dev_id})
    glb, com = device_cache.recenter(data)
    src = spec["sources"][0]
    attr = {
        "title": f"{e['brand']} {e['name']} (dimensionally calibrated)", "author": "Kamikaze spec generator",
        "author_url": cad_harvester.PROJECT_URL, "license": "MIT (project licence)",
        "license_url": f"{cad_harvester.PROJECT_URL}/blob/main/LICENSE", "source": "Spec search",
        "source_key": "spec_generated", "source_url": src["url"], "thumbnail_url": "", "modified": False,
        "match": "spec", "badge": cad_harvester.SPEC_BADGE, "spec": spec,
        "approximations": spec["approximations"],
        "attribution_text": (f"Generated by Kamikaze from published dimensions ({e['height_mm']} x {e['width_mm']} x "
                             f"{e['depth_mm']} mm, {e['rear_cameras']} rear cameras, {e['island_type']} island) - "
                             f"{src['publisher']} ({src['url']})"),
        "com_shift_m": com,
    }
    return attr, glb


def write_entry(dev_id: str, attr: dict, glb: bytes) -> dict:
    d = device_cache.cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    ext = device_cache.extents_mm(glb)
    attr = {**attr, "device_id": dev_id, "built_at": _now(), "extent_mm": ext,
            "dimension_error_mm": device_cache.dimension_error(ext, device_cache.devices()[dev_id])}
    tmp = device_cache.glb_file(dev_id).with_suffix(".part")
    tmp.write_bytes(glb)
    tmp.replace(device_cache.glb_file(dev_id))
    device_cache.sidecar_file(dev_id).write_text(json.dumps(attr, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return attr


def _has_relief(dev_id: str) -> bool:
    try:
        return device_cache.back_relief(device_cache.glb_file(dev_id).read_bytes())["features"] > 0
    except (OSError, ValueError):
        return False


def populate(only: set[str] | None, force: bool, retry_estimated: bool, budget_s: float, offline: bool) -> dict:
    stats = {"dummy": [], "estimated": [], "kept": [], "failed": []}
    for dev_id, e in device_cache.devices().items():
        if only and dev_id not in only:
            continue
        have = device_cache.cached(dev_id)
        if have and not force and not (retry_estimated and have.get("source_key") == "spec_generated"):
            stats["kept"].append(dev_id)
            continue
        t0 = time.monotonic()
        found = None if offline else best_dummy(dev_id, e, budget_s)
        try:
            if found:
                cand, glb, info = found
                attr = {**cad_harvester._dummy_attr(cand, device_cache.spec_for(dev_id)), "ranking": info}
                stats["dummy"].append(dev_id)
            elif (have and have.get("source_key") != "spec_generated" and _has_relief(dev_id)
                  and not is_speculative(have.get("title", ""))):
                stats["kept"].append(dev_id)  # never replace a verified dummy with an estimate
                continue
            else:
                attr, glb = estimated(dev_id, e)
                stats["estimated"].append(dev_id)
            rec = write_entry(dev_id, attr, glb)
        except (OSError, ValueError) as exc:
            log.error("%s: %s", dev_id, exc)
            stats["failed"].append(dev_id)
            continue
        print(f"{dev_id:22s} {rec['source_key']:15s} err {rec['dimension_error_mm']}  "
              f"{time.monotonic() - t0:5.1f}s  {rec.get('title', '')[:60]}", flush=True)
    return stats


def build_embeddings(path: Path = phone_classifier.EMB_PATH) -> int:
    """DINOv2 camera-island embeddings of every cached GLB -> phone_embeddings.npz. 0 when DINOv2 is unavailable."""
    try:
        import npu
    except ImportError as exc:
        print(f"embeddings skipped: {exc}")
        return 0
    emb = npu.make_pose_embedder()
    if emb is None:
        print("embeddings skipped: DINOv2 model not exported on this machine (export_dinov2.py on the board)")
        return 0
    ids, vecs, isl, front = [], [], [], []
    for dev_id, e in device_cache.devices().items():
        f = device_cache.glb_file(dev_id)
        if not f.is_file():
            continue
        img = phone_classifier.render_view(f.read_bytes(), "back")
        x0, y0, x1, y1 = phone_classifier.phone_bbox(img)
        crop = phone_classifier.island_crop(img[y0:y1, x0:x1])
        v = emb.embed_rgb01(phone_classifier.to_nhwc01(crop))
        ids.append(dev_id)
        vecs.append(np.asarray(v, np.float32).reshape(-1))
        isl.append(e.get("island_type", ""))
        front.append(e.get("front_type", ""))
    if not ids:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids), island=np.stack(vecs), island_type=np.array(isl),
                        front_type=np.array(front), embedder=np.array(npu.DINO_ID if hasattr(npu, "DINO_ID") else "dinov2"))
    print(f"{len(ids)} camera-island embeddings ({emb.device}) -> {path}")
    return len(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated device ids")
    ap.add_argument("--force", action="store_true", help="rebuild cached devices")
    ap.add_argument("--retry-estimated", action="store_true", help="search dummies again for spec-generated entries")
    ap.add_argument("--budget", type=float, default=120.0, help="seconds per device for the dummy search")
    ap.add_argument("--allow-nc", action="store_true",
                    help="also accept CC non-commercial prints (attributed, local non-commercial display)")
    ap.add_argument("--offline", action="store_true", help="no network: spec-generated fallbacks only")
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None
    if args.allow_nc:
        cad_harvester.ALLOW_NONCOMMERCIAL = True
    stats = populate(only, args.force, args.retry_estimated, args.budget, args.offline)
    print(json.dumps({k: len(v) for k, v in stats.items()}), "| dummies:", ", ".join(stats["dummy"]) or "-")
    if not args.no_embeddings:
        build_embeddings()
    st = device_cache.status()
    print(f"cache: {st['cached']}/{st['devices']} devices in {st['dir']} {st['by_source']}")
    return 0 if not stats["failed"] and st["cached"] == st["devices"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
