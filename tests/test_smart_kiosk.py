"""Smart Kiosk v2: citrus colour gating, offline device CAD cache, fine-grained phone cues, live pose orientation.

The device-cache test validates the real cache (models/cad_cache, or $KAMIKAZE_CAD_CACHE) that
scripts/populate_device_cache.py builds: every devices_db.json id must have a GLB whose size matches GSMArena within
+-2 mm. No camera, board or network needed.
"""

import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from PIL import Image

import citrus_classifier as cc
import device_cache
import phone_classifier as pc
import procedural_glb
import receiver

TARGETS = (
    [f"iphone_{g}{t}" for g in (13, 14, 15, 16) for t in ("", "_plus" if g > 13 else "_mini", "_pro", "_pro_max")]
    + [f"galaxy_s{g}{t}" for g in (20, 21, 22, 23, 24, 25) for t in ("", "_plus", "_ultra")]
    + [f"galaxy_z_fold_{g}" for g in (3, 4, 5, 6)]
)


def hsv_rgb(h, s, v):
    if cv2 is not None:
        return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0, 0])
    hf = (h % 180) / 30.0
    sf = s / 255.0
    vf = v / 255.0
    c = vf * sf
    x = c * (1.0 - abs((hf % 2.0) - 1.0))
    m = vf - c
    if hf < 1: rgb = (c, x, 0)
    elif hf < 2: rgb = (x, c, 0)
    elif hf < 3: rgb = (0, c, x)
    elif hf < 4: rgb = (0, x, c)
    elif hf < 5: rgb = (x, 0, c)
    else: rgb = (c, 0, x)
    return tuple(int(round((val + m) * 255)) for val in rgb)


def scene_with_can(print_rgb, size=(640, 480), box=(260, 110, 380, 370), background=(70, 75, 80)):
    """Camera-like frame: a silver can with a flavour-print band, specular stripe and dark text, on a background."""
    img = np.full((size[1], size[0], 3), background, np.uint8)
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = (210, 212, 216)
    band = (y1 - y0) // 3
    img[y0 + band:y0 + 2 * band, x0:x1] = print_rgb
    xm = (x0 + x1) // 2
    img[y0:y1, xm - 4:xm + 4] = (252, 252, 252)
    img[y0 + band + 10:y0 + band + 16, x0 + 20:x1 - 20] = (15, 15, 18)
    rng = np.random.default_rng(1)
    img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    w, h = size
    target = {"label": "beverage can", "score": 0.9, "box": [x0 / w, y0 / h, x1 / w, y1 / h]}
    return Image.fromarray(img), target


class CitrusKioskTests(unittest.TestCase):
    AMBIGUOUS = {"kirkland", "lemon", "limon", "sparkling"}

    def decide(self, rgb, tokens):
        image, target = scene_with_can(rgb)
        ev = cc.color_evidence(receiver.citrus_crop(image, target, receiver.center_crop_box(*image.size)))
        return cc.fuse(ev, cc.ocr_scores(recent=tokens)), ev

    def test_green_crop_is_lime_even_with_ambiguous_ocr(self):
        for h in (40, 55, 70):
            dec, ev = self.decide(hsv_rgb(h, 210, 200), self.AMBIGUOUS)
            self.assertEqual(ev["verdict"], "lime")
            self.assertEqual(dec["flavor"], "lime")
            self.assertIn("lemon", dec["suppressed"])
            self.assertEqual(cc.gate_tokens(self.AMBIGUOUS, ev), {"kirkland", "limon", "sparkling"})

    def test_yellow_crop_is_lemon(self):
        dec, ev = self.decide(hsv_rgb(27, 220, 240), {"kirkland"})
        self.assertEqual((ev["verdict"], dec["flavor"]), ("lemon", "lemon"))

    def test_pink_coral_crop_is_grapefruit(self):
        for rgb in ((240, 110, 120), (250, 128, 114), hsv_rgb(172, 190, 210)):
            dec, ev = self.decide(rgb, {"lime"})
            self.assertEqual((ev["verdict"], dec["flavor"]), ("grapefruit", "grapefruit"), rgb)
            self.assertNotEqual(dec["flavor"], "lime")

    def test_detector_box_excludes_coloured_background(self):
        image, target = scene_with_can(hsv_rgb(27, 220, 240), background=hsv_rgb(60, 220, 180))
        ev = cc.color_evidence(receiver.citrus_crop(image, target, receiver.center_crop_box(*image.size)))
        self.assertEqual(ev["verdict"], "lemon")

    def test_only_lime_enrolled_lemon_can_never_identifies(self):
        image, target = scene_with_can(hsv_rgb(27, 220, 240))
        ev = cc.color_evidence(receiver.citrus_crop(image, target, receiver.center_crop_box(*image.size)))
        ranked = [("kirkland_sparkling_water_lime_can", 0.9, "front")]
        fused = {"kirkland_sparkling_water_lime_can": 0.88}
        info = cc.arbitrate(ranked, fused, lambda lab: cc.flavor_of({}, lab), ev, {"lime"}, {}, None,
                            receiver.calibrate, margin=receiver.MIN_MARGIN, demote_to=receiver.C_IDENT - 0.25)
        self.assertEqual(info["action"], "demoted")
        self.assertLess(fused["kirkland_sparkling_water_lime_can"], receiver.C_IDENT)


class DeviceDbTests(unittest.TestCase):
    def test_target_matrix_complete(self):
        devs = device_cache.devices()
        self.assertEqual(len(TARGETS), 38)
        self.assertEqual(sorted(devs), sorted(TARGETS))
        for dev, e in devs.items():
            for k in ("height_mm", "width_mm", "depth_mm"):
                self.assertGreater(e[k], 0, (dev, k))
            self.assertTrue(e.get("sources"), dev)
            self.assertIn(e["front_type"], ("notch", "dynamic_island", "punch_hole"))
        for fold in ("galaxy_z_fold_3", "galaxy_z_fold_4", "galaxy_z_fold_5", "galaxy_z_fold_6"):
            self.assertEqual(devs[fold]["form"], "foldable")
            self.assertGreater(devs[fold]["aspect"], 2.24)
            self.assertTrue(devs[fold]["unfolded_mm"])

    def test_design_families(self):
        devs = device_cache.devices()
        self.assertEqual(devs["iphone_13"]["front_type"], "notch")
        self.assertEqual(devs["iphone_14"]["front_type"], "notch")
        self.assertEqual(devs["iphone_14_pro"]["front_type"], "dynamic_island")
        self.assertEqual(devs["iphone_15"]["front_type"], "dynamic_island")
        self.assertEqual(devs["iphone_16"]["island_type"], "vertical_pill")
        self.assertEqual(devs["iphone_15"]["island_type"], "diagonal_dual")
        self.assertEqual(devs["galaxy_s21"]["island_type"], "contour_housing")
        self.assertEqual(devs["galaxy_s24"]["island_type"], "floating_rings")

    def test_device_id_for(self):
        f = device_cache.device_id_for
        self.assertEqual(f("Apple iPhone 15 Pro Max"), "iphone_15_pro_max")
        self.assertEqual(f("iphone_13_mini"), "iphone_13_mini")
        self.assertEqual(f("Samsung Galaxy S24+"), "galaxy_s24_plus")
        self.assertEqual(f("galaxy_s25_ultra"), "galaxy_s25_ultra")
        self.assertEqual(f("Galaxy Z Fold6"), "galaxy_z_fold_6")
        self.assertIsNone(f("Galaxy S23 FE"))
        self.assertIsNone(f("Coca-Cola can"))


class DeviceCacheTests(unittest.TestCase):
    def test_every_target_cached_within_2mm(self):
        st = device_cache.status()
        self.assertEqual(st["missing"], [], f"run scripts/populate_device_cache.py (cache {st['dir']})")
        devs = device_cache.devices()
        for dev in TARGETS:
            rec = device_cache.cached(dev)
            self.assertIsNotNone(rec, dev)
            ext = device_cache.extents_mm(Path(rec["glb"]).read_bytes())
            err = device_cache.dimension_error(ext, devs[dev])
            self.assertTrue(err["ok"], (dev, ext, err))
            self.assertLessEqual(abs(err["height_mm"]), device_cache.DIM_TOL_MM, dev)
            self.assertLessEqual(abs(err["width_mm"]), device_cache.DIM_TOL_MM, dev)
            self.assertGreater(ext["y"], ext["x"], f"{dev} not Y-up")  # metres, Y-up: height is the Y extent
            self.assertTrue(80 < ext["y"] < 200, f"{dev} not in metres ({ext})")
            self.assertLess(max(abs(v) for v in ext["com_mm"]), 0.5, f"{dev} not centred ({ext['com_mm']})")
            self.assertTrue(rec.get("source_key") and (rec.get("author") or rec.get("source")), dev)
            relief = device_cache.back_relief(Path(rec["glb"]).read_bytes())
            self.assertGreater(relief["features"], 0, f"{dev}: blank back, no camera relief ({rec.get('title')})")

    def test_s24_ultra_camera_layout(self):
        """S24 Ultra: three large rings in a column + the smaller periscope / flash beside them (not 4 in a row)."""
        if device_cache.cached("galaxy_s24_ultra") is None:
            self.skipTest("cache not populated")
        rec = device_cache.cached("galaxy_s24_ultra")
        assert_ultra_layout(self, device_cache.back_relief(Path(rec["glb"]).read_bytes()))

    def test_install_is_instant_and_offline(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for dev in ("iphone_15_pro", "galaxy_s24_ultra", "galaxy_z_fold_6"):
                if device_cache.cached(dev) is None:
                    self.skipTest("cache not populated")
                t = time.perf_counter()
                rec = device_cache.install(dev, tmp / f"{dev}.glb")
                ms = (time.perf_counter() - t) * 1000.0
                self.assertIsNotNone(rec)
                self.assertLess(ms, 50.0, f"{dev} install took {ms:.1f} ms")
                self.assertTrue((tmp / f"{dev}.glb").is_file())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def assert_ultra_layout(tc, relief):
    tc.assertGreaterEqual(relief["features"], 4, relief)
    big, small = relief["centroids"][:3], relief["centroids"][3:]
    areas = relief["areas_px"]
    tc.assertLess(areas[3], 0.7 * areas[2], f"4th module is not smaller than the main column: {relief}")
    xs = [c[0] for c in big]
    tc.assertLess(max(xs) - min(xs), 0.05, f"large rings not in one column: {relief}")
    ys = sorted(c[1] for c in big)
    tc.assertGreater(min(b - a for a, b in zip(ys, ys[1:])), 0.03, f"large rings overlap: {relief}")
    tc.assertTrue(any(c[0] > max(xs) + 0.05 and ys[0] - 0.05 <= c[1] <= ys[-1] for c in small),
                  f"no periscope / flash beside the column: {relief}")


class CameraReliefTests(unittest.TestCase):
    def test_blank_slab_has_no_relief(self):
        m = procedural_glb._Mesh()
        procedural_glb.rounded_prism(m, (0, 0, 0), 0.079, 0.1623, 0.0086, 0.008, procedural_glb.NEUTRAL, 0.8, 0.35)
        self.assertEqual(device_cache.back_relief(m.to_glb({}))["features"], 0)

    def test_estimated_ultra_is_column_plus_side(self):
        spec = device_cache.spec_for("galaxy_s24_ultra")
        self.assertEqual(spec["rear_cameras"], 4)
        assert_ultra_layout(self, device_cache.back_relief(procedural_glb.build_from_spec(spec, "galaxy_s24_ultra")))

    def test_estimated_base_model_keeps_three_in_a_column(self):
        r = device_cache.back_relief(procedural_glb.build_from_spec(device_cache.spec_for("galaxy_s24"), "galaxy_s24"))
        self.assertEqual(r["features"], 3)
        self.assertLess(max(c[0] for c in r["centroids"]) - min(c[0] for c in r["centroids"]), 0.05)

    def test_speculative_and_accessory_titles_rejected(self):
        import importlib.util
        path = Path(__file__).resolve().parent / "scripts" / "populate_device_cache.py"
        if not path.is_file():
            path = Path(__file__).resolve().parent.parent / "scripts" / "populate_device_cache.py"
        spec = importlib.util.spec_from_file_location("populate_device_cache", path)
        pop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pop)
        self.assertTrue(pop.is_speculative("3D model of Rumored Samsung galaxy S20 Ultra"))
        self.assertTrue(pop.is_speculative("iPhone 17 concept (leaked)"))
        self.assertFalse(pop.is_speculative("Samsung galaxy S24 Ultra phone model"))
        self.assertFalse(pop.is_speculative("Apple iPhone 14 Replica Mockup Template Blank Smartphone"))


class PhoneCueTests(unittest.TestCase):
    """Structural cues on deterministic spec renders (procedural_glb from devices_db), no DINOv2."""

    @classmethod
    def setUpClass(cls):
        cls.clf = pc.PhoneClassifier(index=None, embed=None)
        cls.renders = {}
        for dev in ("iphone_15", "iphone_16", "galaxy_s21", "galaxy_s24", "galaxy_z_fold_5"):
            spec = device_cache.spec_for(dev)
            glb, _ = device_cache.recenter(procedural_glb.build_from_spec(spec, dev, {"class_id": dev}))
            img = pc.render_view(glb, "back", 448)
            x0, y0, x1, y1 = pc.phone_bbox(img)
            cls.renders[dev] = img[y0:y1, x0:x1]

    def cues(self, dev):
        crop = self.renders[dev]
        return self.clf.cues(crop, "back", crop.shape[0] / crop.shape[1])

    def test_diagonal_vs_vertical_pill(self):
        self.assertEqual(self.cues("iphone_15")["layout"], "diagonal")
        self.assertEqual(self.cues("iphone_16")["layout"], "vertical_pair")

    def test_samsung_column(self):
        self.assertEqual(self.cues("galaxy_s24")["layout"], "column")

    def test_fold_aspect_separates_slabs(self):
        fold, slab = self.cues("galaxy_z_fold_5"), self.cues("galaxy_s24")
        devs = device_cache.devices()
        self.assertGreater(pc.aspect_score(fold["aspect"], devs["galaxy_z_fold_5"]),
                           pc.aspect_score(fold["aspect"], devs["galaxy_s24"]))
        self.assertGreater(pc.aspect_score(slab["aspect"], devs["galaxy_s24"]),
                           pc.aspect_score(slab["aspect"], devs["galaxy_z_fold_5"]))

    def test_classify_ranks_right_family(self):
        top = self.clf.classify(self.renders["iphone_16"], "back", None)["top"][0]["id"]
        self.assertEqual(device_cache.devices()[top]["island_type"], "vertical_pill")
        top = self.clf.classify(self.renders["iphone_15"], "back", None)["top"][0]["id"]
        self.assertEqual(device_cache.devices()[top]["island_type"], "diagonal_dual")
        r = self.clf.classify(self.renders["galaxy_z_fold_5"], "back", self.cues("galaxy_z_fold_5")["aspect"])
        self.assertEqual(device_cache.devices()[r["top"][0]["id"]].get("form"), "foldable")

    def test_scan_accumulator_and_conflicts(self):
        acc = pc.ScanAccumulator()
        for _ in range(3):
            acc.add(self.clf.classify(self.renders["iphone_16"], "back", None))
        s = acc.summary("iphone_15")
        self.assertEqual(s["frames"], 3)
        self.assertFalse(s["agrees"])
        self.assertTrue(any("layout" in c for c in s["conflicts"]))

    def test_device_matcher_thread(self):
        m = pc.DeviceMatcher(self.clf, hz=50)
        self.assertTrue(m.submit(self.renders["iphone_16"], "back"))
        deadline = time.time() + 5
        while m.runs == 0 and time.time() < deadline:
            time.sleep(0.02)
        s = m.summary("iphone_16")
        self.assertIsNotNone(s)
        self.assertEqual(device_cache.devices()[s["id"]]["island_type"], "vertical_pill")
        m.reset()
        self.assertIsNone(m.summary())


class OrientationTests(unittest.TestCase):
    def test_model_viewer_attribute(self):
        o = receiver.model_orientation({"yaw": 90.0, "pitch": 10.0, "roll": -5.0, "valid": True, "source": "dinov2-npu"})
        self.assertEqual(o["attribute"], "5.0deg 10.0deg -90.0deg")  # "roll pitch yaw"
        self.assertEqual(o["smoothing_alpha"], 0.25)
        self.assertIsNone(receiver.model_orientation({"valid": False})["attribute"])
        self.assertFalse(receiver.model_orientation(None)["valid"])

    def test_dashboard_has_auto_orient_hud(self):
        html = (Path(receiver.__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")
        for needle in ('id="oriSwitch"', 'id="snapBtn"', "ORI_ALPHA = 0.25", "setAttribute('orientation'",
                       "removeAttribute('camera-controls')"):
            self.assertIn(needle, html)
        names = re.findall(r"\bfunction\s+(\w+)\s*\(", html)
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [], "duplicate function declarations shadow each other")


if __name__ == "__main__":
    unittest.main()
