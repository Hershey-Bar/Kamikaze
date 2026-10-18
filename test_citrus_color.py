"""citrus_classifier: colour + OCR flavour disambiguation on synthetic can crops (no camera, no board, no network)."""

import unittest

import cv2
import numpy as np

import citrus_classifier as cc


def hsv_rgb(h, s, v):
    """OpenCV-scale HSV (h 0-180, s / v 0-255) -> RGB tuple."""
    px = np.uint8([[[h, s, v]]])
    return tuple(int(x) for x in cv2.cvtColor(px, cv2.COLOR_HSV2RGB)[0, 0])


def can(print_rgb, print_frac=0.35, h=160, w=80, seed=0, glare=True, ink=True, noise=6.0):
    """Synthetic can crop: silver-white body, a horizontal flavour-print band, a specular glare stripe, black text."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (208, 210, 214), np.float32)
    band = int(h * print_frac)
    y0 = (h - band) // 2
    img[y0:y0 + band, :] = print_rgb
    if glare:
        img[:, w // 2 - 3:w // 2 + 3] = (252, 252, 252)
    if ink:
        img[y0 + band // 3:y0 + band // 3 + 4, w // 4:3 * w // 4] = (12, 12, 14)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


GREENS = [hsv_rgb(h, s, v) for h in (38, 45, 55, 65, 75, 80) for s in (140, 200, 250) for v in (150, 220)]
YELLOWS = [hsv_rgb(h, s, v) for h in (23, 27, 31, 34) for s in (150, 200, 250) for v in (190, 245)]
PINKS = [hsv_rgb(h, s, v) for h in (162, 168, 174, 178, 2, 6, 9) for s in (120, 170, 230) for v in (170, 235)]


class ColorEvidenceTests(unittest.TestCase):
    def test_green_cans_are_lime_never_lemon(self):
        for i, rgb in enumerate(GREENS):
            ev = cc.color_evidence(can(rgb, seed=i))
            self.assertEqual(ev["verdict"], "lime", (rgb, ev))
            self.assertLessEqual(ev["scores"]["lemon"], cc.CONFLICT_MAX, (rgb, ev))

    def test_yellow_cans_are_lemon(self):
        for i, rgb in enumerate(YELLOWS):
            ev = cc.color_evidence(can(rgb, seed=i))
            self.assertEqual(ev["verdict"], "lemon", (rgb, ev))
            self.assertNotEqual(ev["verdict"], "lime")

    def test_pink_ruby_coral_cans_are_grapefruit_never_lime(self):
        for i, rgb in enumerate(PINKS):
            ev = cc.color_evidence(can(rgb, seed=i))
            self.assertEqual(ev["verdict"], "grapefruit", (rgb, ev))
            self.assertEqual(ev["scores"]["lime"], 0.0, (rgb, ev))

    def test_saturated_orange(self):
        self.assertEqual(cc.color_evidence(can(hsv_rgb(17, 240, 245)))["verdict"], "orange")

    def test_silver_can_and_glare_give_no_verdict(self):
        ev = cc.color_evidence(can((208, 210, 214)))
        self.assertIsNone(ev["verdict"])
        self.assertEqual(ev["scores"], {f: 0.0 for f in cc.FLAVORS})
        glare = np.full((60, 40, 3), 250, np.uint8)
        self.assertIsNone(cc.color_evidence(glare)["verdict"])
        ink = np.full((60, 40, 3), 20, np.uint8)
        hsv, _lab, area = cc.surface_pixels(ink)
        self.assertEqual((len(hsv), area), (0, 0))

    def test_background_outside_box_and_ellipse_is_ignored(self):
        scene = np.zeros((240, 320, 3), np.uint8)
        scene[:] = hsv_rgb(60, 230, 200)  # green table / wall all around the can
        crop = can(YELLOWS[5], h=160, w=80)
        scene[40:200, 120:200] = crop
        self.assertEqual(cc.color_evidence(scene, (120, 40, 200, 200))["verdict"], "lemon")
        norm = (120 / 320, 40 / 240, 200 / 320, 200 / 240)
        self.assertEqual(cc.color_evidence(scene, norm)["verdict"], "lemon")
        # a loose box that includes background margins: the inner 60 % ellipse still sees only the can
        self.assertEqual(cc.color_evidence(scene, (104, 30, 216, 210))["verdict"], "lemon")

    def test_small_print_fraction_on_silver_can(self):
        ev = cc.color_evidence(can(GREENS[7], print_frac=0.12))
        self.assertEqual(ev["verdict"], "lime")

    def test_deterministic(self):
        img = can(GREENS[3], seed=4)
        self.assertEqual(cc.color_evidence(img), cc.color_evidence(img))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            cc.color_evidence(np.zeros((10, 10), np.uint8))
        with self.assertRaises(ValueError):
            cc.color_evidence(np.zeros((50, 50, 3), np.uint8), (10, 10, 12, 12))


class FusionTests(unittest.TestCase):
    def test_ocr_lemon_on_green_can_is_suppressed_and_resolves_lime(self):
        ev = cc.color_evidence(can(hsv_rgb(65, 220, 200)))
        dec = cc.fuse(ev, cc.ocr_scores(recent={"kirkland", "lemon"}))
        self.assertEqual(dec["flavor"], "lime")
        self.assertTrue(dec["conflict"])
        self.assertEqual(dec["suppressed"], ["lemon"])
        self.assertEqual(dec["ocr"]["lemon"], 0.0)

    def test_ocr_lime_on_pink_can_is_suppressed(self):
        ev = cc.color_evidence(can(hsv_rgb(172, 180, 220)))
        dec = cc.fuse(ev, cc.ocr_scores(recent={"lime"}, rolling={"lime", "sparkling"}))
        self.assertEqual(dec["flavor"], "grapefruit")
        self.assertEqual(dec["suppressed"], ["lime"])

    def test_ocr_agrees(self):
        ev = cc.color_evidence(can(YELLOWS[2]))
        dec = cc.fuse(ev, cc.ocr_scores(recent={"lemon"}))
        self.assertEqual(dec["flavor"], "lemon")
        self.assertFalse(dec["conflict"])
        self.assertAlmostEqual(dec["p"]["lemon"], cc.W_COLOR * ev["scores"]["lemon"] + cc.W_OCR * 1.0, places=3)

    def test_ocr_only_when_colour_is_silent(self):
        dec = cc.fuse(cc.color_evidence(can((208, 210, 214))), cc.ocr_scores(recent={"grapefruit"}))
        self.assertEqual(dec["flavor"], "grapefruit")
        self.assertEqual(cc.fuse(None, cc.ocr_scores(rolling={"lime"}))["flavor"], "lime")
        self.assertIsNone(cc.fuse(None, None)["flavor"])

    def test_ocr_scores_recent_vs_rolling(self):
        s = cc.ocr_scores(recent=["LIME"], rolling={"lemon": 3})
        self.assertEqual((s["lime"], s["lemon"], s["grapefruit"]), (1.0, 0.6, 0.0))

    def test_gate_tokens(self):
        ev = cc.color_evidence(can(hsv_rgb(60, 220, 200)))
        self.assertEqual(cc.gate_tokens({"kirkland", "lemon", "lime"}, ev), {"kirkland", "lime"})
        self.assertEqual(cc.gate_tokens({"kirkland": 2, "lemons": 1}, ev), {"kirkland": 2})
        self.assertEqual(cc.gate_tokens({"lemon"}, None), {"lemon"})

    def test_flavor_of(self):
        self.assertEqual(cc.flavor_of({}, "kirkland_sparkling_water_lime_can"), "lime")
        self.assertEqual(cc.flavor_of({"flavor": "Grapefruit"}, "kirkland_can"), "grapefruit")
        self.assertIsNone(cc.flavor_of({}, "sprite_lemon_lime_can"))
        self.assertIsNone(cc.flavor_of({}, "coca_cola_classic_can"))


class ArbitrationTests(unittest.TestCase):
    @staticmethod
    def flavor(lab):
        return cc.flavor_of({}, lab)

    def test_only_lime_enrolled_lemon_can_is_demoted(self):
        ranked = [("kirkland_lime_can", 0.9, "front"), ("cola_can", 0.4, "front")]
        fused = {"kirkland_lime_can": 0.85, "cola_can": 0.3}
        ev = cc.color_evidence(can(YELLOWS[4]))
        info = cc.arbitrate(ranked, fused, self.flavor, ev, {"lime"}, {}, None, demote_to=0.45)
        self.assertEqual(info["action"], "demoted")
        self.assertEqual(info["suppressed"], ["lime"])
        self.assertLessEqual(fused["kirkland_lime_can"], 0.45)

    def test_lime_can_with_lime_class_is_untouched(self):
        ranked = [("kirkland_lime_can", 0.9, "front")]
        fused = {"kirkland_lime_can": 0.85}
        info = cc.arbitrate(ranked, fused, self.flavor, cc.color_evidence(can(GREENS[4])), {"lemon"}, {})
        self.assertIsNone(info["action"])
        self.assertEqual(fused["kirkland_lime_can"], 0.85)

    def test_weak_colour_never_demotes(self):
        ranked = [("kirkland_lime_can", 0.9, "front")]
        fused = {"kirkland_lime_can": 0.85}
        info = cc.arbitrate(ranked, fused, self.flavor, None, set(), {"lemon": 2})
        self.assertEqual(info["flavor"], "lemon")
        self.assertIsNone(info["action"])
        self.assertEqual(fused["kirkland_lime_can"], 0.85)

    def test_promotes_grapefruit_class(self):
        ranked = [("kirkland_lime_can", 0.8, "front"), ("kirkland_grapefruit_can", 0.78, "front")]
        fused = {"kirkland_lime_can": 0.8, "kirkland_grapefruit_can": 0.7}
        info = cc.arbitrate(ranked, fused, self.flavor, cc.color_evidence(can(PINKS[3])), set(), {}, margin=0.15)
        self.assertEqual(info["action"], "promoted")
        self.assertEqual(ranked[0][0], "kirkland_grapefruit_can")
        self.assertGreaterEqual(fused["kirkland_grapefruit_can"] - fused["kirkland_lime_can"], 0.15)


class KirklandFixedColorTests(unittest.TestCase):
    def test_mapping_overrides_ocr_and_large_visual_gap(self):
        for rgb, wanted in ((YELLOWS[4], "lemon"), (PINKS[3], "grapefruit"), (GREENS[4], "lime")):
            for wrong in ("lemon", "grapefruit", "lime"):
                if wrong == wanted:
                    continue
                with self.subTest(wanted=wanted, wrong=wrong):
                    old, new = f"kirkland_{wrong}_can", f"kirkland_{wanted}_can"
                    ranked = [(old, 0.95, "front"), (new, 0.3, "front")]
                    fused = {old: 0.95, new: 0.3}
                    result = cc.arbitrate(ranked, fused, ArbitrationTests.flavor, cc.color_evidence(can(rgb)),
                                          {wrong}, {wrong: 10}, lexicon_label=old)
                    self.assertEqual(ranked[0][0], new)
                    self.assertEqual(result["flavor"], wanted)
                    self.assertLessEqual(fused[old], 0.45)

    def test_metadata_brand_works_for_custom_class_ids(self):
        ranked = [("first", 0.9, "front"), ("second", 0.2, "back")]
        fused = {}
        metadata = {"first": {"brand": "Kirkland Signature"}, "second": {"brand": "Kirkland"}}
        cc.arbitrate(ranked, fused, lambda lab: {"first": "lemon", "second": "lime"}[lab],
                     cc.color_evidence(can(GREENS[4])), metadata_of_label=metadata.get)
        self.assertEqual(ranked[0][0], "second")

    def test_does_not_promote_a_different_brands_flavor(self):
        ranked = [("kirkland_lemon_can", 0.95, "front"), ("lacroix_lime_can", 0.2, "front")]
        fused = {"kirkland_lemon_can": 0.95, "lacroix_lime_can": 0.2}
        cc.arbitrate(ranked, fused, ArbitrationTests.flavor, cc.color_evidence(can(GREENS[4])))
        self.assertEqual(fused["lacroix_lime_can"], 0.2)
        self.assertLessEqual(fused["kirkland_lemon_can"], 0.45)

    def test_non_kirkland_keeps_lexicon_and_confidence_gates(self):
        ranked = [("lacroix_lemon_can", 0.95, "front"), ("lacroix_lime_can", 0.2, "front")]
        fused = {"lacroix_lemon_can": 0.95, "lacroix_lime_can": 0.2}
        cc.arbitrate(ranked, fused, ArbitrationTests.flavor, cc.color_evidence(can(GREENS[4])),
                     lexicon_label="lacroix_lemon_can")
        self.assertEqual(ranked[0][0], "lacroix_lemon_can")

    def test_neutral_ambiguous_and_other_colors_do_not_force_a_flavor(self):
        for ev in (None, cc.color_evidence(can((210, 210, 210))),
                   {"verdict": None, "scores": {"lime": 0.5, "lemon": 0.49}},
                   cc.color_evidence(can(hsv_rgb(17, 240, 245)))):
            self.assertIsNone(cc.kirkland_flavor(ev))


if __name__ == "__main__":
    unittest.main()
