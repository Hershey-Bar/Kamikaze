"""Build models/devices_db.json: the 38 pre-cached phones with GSMArena-verified outer dimensions.

Dimensions (height x width x depth mm, Z Fold folded + unfolded) and the rear camera module count are scraped from each
device's GSMArena spec page (URL recorded per device). The categorical design-family fields (front cut-out type, rear
camera island type) are curated product facts used by phone_classifier.py; they are not measurements. Corner radius
and lens centre coordinates are NOT published by GSMArena, so they are stored as null (the procedural fallback flags
its approximations; nothing is invented as if measured).

    python scripts/build_devices_db.py [--only galaxy_z_fold_6,...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spec_search  # noqa: E402

DB_PATH = ROOT / "models" / "devices_db.json"

# id, brand, GSMArena model name, display name, front cut-out, rear island family
DEVICES = [
    ("iphone_13_mini", "Apple", "iPhone 13 mini", "iPhone 13 mini", "notch", "diagonal_dual"),
    ("iphone_13", "Apple", "iPhone 13", "iPhone 13", "notch", "diagonal_dual"),
    ("iphone_13_pro", "Apple", "iPhone 13 Pro", "iPhone 13 Pro", "notch", "triple_square"),
    ("iphone_13_pro_max", "Apple", "iPhone 13 Pro Max", "iPhone 13 Pro Max", "notch", "triple_square"),
    ("iphone_14", "Apple", "iPhone 14", "iPhone 14", "notch", "diagonal_dual"),
    ("iphone_14_plus", "Apple", "iPhone 14 Plus", "iPhone 14 Plus", "notch", "diagonal_dual"),
    ("iphone_14_pro", "Apple", "iPhone 14 Pro", "iPhone 14 Pro", "dynamic_island", "triple_square"),
    ("iphone_14_pro_max", "Apple", "iPhone 14 Pro Max", "iPhone 14 Pro Max", "dynamic_island", "triple_square"),
    ("iphone_15", "Apple", "iPhone 15", "iPhone 15", "dynamic_island", "diagonal_dual"),
    ("iphone_15_plus", "Apple", "iPhone 15 Plus", "iPhone 15 Plus", "dynamic_island", "diagonal_dual"),
    ("iphone_15_pro", "Apple", "iPhone 15 Pro", "iPhone 15 Pro", "dynamic_island", "triple_square"),
    ("iphone_15_pro_max", "Apple", "iPhone 15 Pro Max", "iPhone 15 Pro Max", "dynamic_island", "triple_square"),
    ("iphone_16", "Apple", "iPhone 16", "iPhone 16", "dynamic_island", "vertical_pill"),
    ("iphone_16_plus", "Apple", "iPhone 16 Plus", "iPhone 16 Plus", "dynamic_island", "vertical_pill"),
    ("iphone_16_pro", "Apple", "iPhone 16 Pro", "iPhone 16 Pro", "dynamic_island", "triple_square"),
    ("iphone_16_pro_max", "Apple", "iPhone 16 Pro Max", "iPhone 16 Pro Max", "dynamic_island", "triple_square"),
    ("galaxy_s20", "Samsung", "Galaxy S20", "Galaxy S20", "punch_hole", "rect_housing"),
    ("galaxy_s20_plus", "Samsung", "Galaxy S20+", "Galaxy S20+", "punch_hole", "rect_housing"),
    ("galaxy_s20_ultra", "Samsung", "Galaxy S20 Ultra", "Galaxy S20 Ultra", "punch_hole", "rect_housing"),
    ("galaxy_s21", "Samsung", "Galaxy S21", "Galaxy S21", "punch_hole", "contour_housing"),
    ("galaxy_s21_plus", "Samsung", "Galaxy S21+", "Galaxy S21+", "punch_hole", "contour_housing"),
    ("galaxy_s21_ultra", "Samsung", "Galaxy S21 Ultra", "Galaxy S21 Ultra", "punch_hole", "contour_housing"),
    ("galaxy_s22", "Samsung", "Galaxy S22", "Galaxy S22", "punch_hole", "contour_island"),
    ("galaxy_s22_plus", "Samsung", "Galaxy S22+", "Galaxy S22+", "punch_hole", "contour_island"),
    ("galaxy_s22_ultra", "Samsung", "Galaxy S22 Ultra", "Galaxy S22 Ultra", "punch_hole", "floating_rings"),
    ("galaxy_s23", "Samsung", "Galaxy S23", "Galaxy S23", "punch_hole", "floating_rings"),
    ("galaxy_s23_plus", "Samsung", "Galaxy S23+", "Galaxy S23+", "punch_hole", "floating_rings"),
    ("galaxy_s23_ultra", "Samsung", "Galaxy S23 Ultra", "Galaxy S23 Ultra", "punch_hole", "floating_rings"),
    ("galaxy_s24", "Samsung", "Galaxy S24", "Galaxy S24", "punch_hole", "floating_rings"),
    ("galaxy_s24_plus", "Samsung", "Galaxy S24+", "Galaxy S24+", "punch_hole", "floating_rings"),
    ("galaxy_s24_ultra", "Samsung", "Galaxy S24 Ultra", "Galaxy S24 Ultra", "punch_hole", "floating_rings"),
    ("galaxy_s25", "Samsung", "Galaxy S25", "Galaxy S25", "punch_hole", "floating_rings"),
    ("galaxy_s25_plus", "Samsung", "Galaxy S25+", "Galaxy S25+", "punch_hole", "floating_rings"),
    ("galaxy_s25_ultra", "Samsung", "Galaxy S25 Ultra", "Galaxy S25 Ultra", "punch_hole", "floating_rings"),
    ("galaxy_z_fold_3", "Samsung", "Galaxy Z Fold3", "Galaxy Z Fold 3", "punch_hole", "vertical_housing"),
    ("galaxy_z_fold_4", "Samsung", "Galaxy Z Fold4", "Galaxy Z Fold 4", "punch_hole", "vertical_housing"),
    ("galaxy_z_fold_5", "Samsung", "Galaxy Z Fold5", "Galaxy Z Fold 5", "punch_hole", "floating_rings"),
    ("galaxy_z_fold_6", "Samsung", "Galaxy Z Fold6", "Galaxy Z Fold 6", "punch_hole", "floating_rings"),
]


def build(only: set[str] | None = None, delay_s: float = 2.5) -> dict:
    db = json.loads(DB_PATH.read_text(encoding="utf-8")) if DB_PATH.exists() else {"devices": {}}
    devices = db.get("devices", {})
    failures = []
    for i, (dev_id, brand, gsm_name, display, front, island) in enumerate(DEVICES):
        if only and dev_id not in only:
            continue
        spec = spec_search.phone_specs(brand, gsm_name)
        if not spec.get("verified"):
            failures.append(f"{dev_id}: {spec.get('reason')}")
            print(f"FAIL {dev_id}: {spec.get('reason')}", flush=True)
            continue
        h, w, d = spec["height_mm"], spec["width_mm"], spec["depth_mm"]
        entry = {
            "brand": brand, "name": display, "gsmarena_name": spec["name"], "category": "smartphone",
            "form": "foldable" if "fold" in dev_id else "slab",
            "height_mm": h, "width_mm": w, "depth_mm": d, "weight_g": spec.get("weight_g"),
            "rear_cameras": spec["rear_cameras"], "aspect": round(h / w, 3),
            "front_type": front, "island_type": island,
            "corner_radius_mm": None, "lenses": None,
            "sources": spec["sources"],
        }
        if spec.get("unfolded_mm"):
            u = spec["unfolded_mm"]
            entry["unfolded_mm"] = u
            entry["unfolded_aspect"] = round(u["height_mm"] / u["width_mm"], 3)
        devices[dev_id] = entry
        print(f"ok   {dev_id:22s} {h} x {w} x {d} mm, {spec['rear_cameras']} rear", flush=True)
        if i < len(DEVICES) - 1:
            time.sleep(delay_s)
    db = {
        "schema": 1,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "units": "mm; height x width x depth of the body (GSMArena, camera bump excluded); foldables: folded body, "
                 "thickest point, unfolded_mm separately",
        "notes": ["front_type / island_type are curated design-family facts (not measurements)",
                  "corner_radius_mm and lenses are null: not published by GSMArena, never estimated here"],
        "devices": {k: devices[k] for k in [d[0] for d in DEVICES] if k in devices},
    }
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(db['devices'])}/{len(DEVICES)} devices -> {DB_PATH}")
    if failures:
        print("failures:\n  " + "\n  ".join(failures))
    return db


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated device ids")
    args = ap.parse_args()
    db = build({s.strip() for s in args.only.split(",") if s.strip()} or None)
    return 0 if len(db["devices"]) == len(DEVICES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
