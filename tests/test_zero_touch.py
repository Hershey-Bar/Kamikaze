"""Autonomous scans, without a camera, network, enrollment or persistent image files."""

import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from PIL import Image

import cad_harvester
import citrus_classifier
import entity_resolver
import mcp_server
import ocr_fusion
import phone_classifier
import pose
import receiver
import smart_identity
import smart_scan
from test_kamikaze import FakeAssets, FakeExtractor
from test_smart_citrus import Clock
from test_citrus_color import can


def read(text, run):
    return {"texts": [text], "corrected": [text.lower()], "tokens": text.lower().split(), "runs": run}


class EvidenceTests(unittest.TestCase):
    def evidence(self, text, color=None):
        evidence = smart_identity.ScanEvidence()
        patch = np.full((180, 80, 3), color, np.uint8) if color else None
        c = citrus_classifier.color_evidence(patch) if patch is not None else None
        for i in range(1, 4):
            evidence.add(read(text, i), c, patch)
        return evidence

    def test_new_citrus_without_enrollment(self):
        for rgb, flavor in [((110, 200, 40), "lime"), ((230, 200, 40), "lemon"), ((230, 80, 110), "grapefruit")]:
            with self.subTest(flavor=flavor):
                e = self.evidence("Kirkland sparkling water lemon lime 12 FL OZ", rgb)
                result = e.resolve("beverage_can")
                self.assertIn(flavor, result["label"])
                self.assertEqual(result["volume"], "355ml")
                self.assertTrue(result["metadata"]["autonomous"])
                self.assertEqual(result["metadata"]["brand"], "Kirkland")

    def test_unenrolled_red_bull_and_bottle(self):
        e = self.evidence("Red Bull energy drink 250 ml")
        result = e.resolve("beverage_can")
        self.assertEqual(result["label"], "red_bull_250ml_can")
        self.assertEqual(e.resolve("beverage_bottle")["category"], "beverage_bottle")

    def test_fractional_and_dotted_volumes(self):
        for text, volume in [("Red Bull 0.5l", "500ml"), ("Pepsi 12 fl. oz", "355ml")]:
            self.assertEqual(self.evidence(text).resolve("beverage_can")["volume"], volume)

    def test_kirkland_fixed_color_beats_competing_ocr_even_with_mixed_print(self):
        e = self.evidence("Kirkland sparkling water lime 355ml")
        e.colors.append({"verdict": "lemon", "scores": {"lemon": 0.55, "lime": 0.25}})
        self.assertEqual(e.resolve("beverage_can")["label"], "kirkland_lemon_sparkling_water_355ml_can")

    def test_known_kirkland_appearance_can_select_flavor_without_ocr(self):
        e = smart_identity.ScanEvidence()
        e.colors.append(citrus_classifier.color_evidence(np.full((80, 80, 3), (230, 80, 110), np.uint8)))
        self.assertIsNone(e.resolve("beverage_can"))
        result = e.resolve("beverage_can", known_kirkland_line=True)
        self.assertIn("grapefruit", result["label"])
        self.assertEqual(result["source"], "vision_color")

    def test_visible_competing_brand_is_not_overridden_by_visual_brand(self):
        e = self.evidence("Pepsi 355ml", (110, 200, 40))
        result = e.resolve("beverage_can", known_kirkland_line=True)
        self.assertEqual(result["metadata"]["brand"], "Pepsi")

    def test_pale_artwork_overrides_stale_text_but_not_other_kirkland_lines(self):
        e = self.evidence("Kirkland sparkling water lime 355ml", (244, 212, 225))
        self.assertIn("grapefruit", e.resolve("beverage_can")["label"])
        for text, category in (("Kirkland lime juice", "beverage_bottle"),
                               ("Kirkland lime hard seltzer", "beverage_can")):
            e = self.evidence(text, (244, 212, 225))
            result = e.resolve(category, known_kirkland_line=True)
            self.assertNotIn("sparkling_water", result["label"])

    def test_ambiguous_frames_age_out_old_flavor(self):
        e = self.evidence("Kirkland sparkling water lime", (110, 200, 40))
        blank = citrus_classifier.color_evidence(np.full((80, 80, 3), 225, np.uint8))
        for i in range(12):
            e.add(read("Kirkland sparkling water lime", 10 + i), blank)
        self.assertIsNone(e.resolve("beverage_can"))
        self.assertIsNone(citrus_classifier.kirkland_flavor(e.color()))

    def test_phone_text_matches_without_prototypes(self):
        self.assertEqual(self.evidence("Apple iPhone 14 Pro").resolve("smartphone")["label"], "iphone_14_pro")
        self.assertEqual(self.evidence("Samsung Galaxy S24 Ultra").resolve("smartphone")["label"], "galaxy_s24_ultra")

    def test_no_brand_invention_or_cross_brand_match(self):
        self.assertIsNone(self.evidence("sparkling water lime", (100, 200, 30)).resolve("beverage_can"))
        self.assertIsNone(self.evidence("Kirkland Pepsi lime").resolve("beverage_can"))
        self.assertIsNone(self.evidence("Kirkland 355ml 500ml lime").resolve("beverage_can"))

    def test_one_ocr_pass_cannot_repeat_itself_into_confirmation(self):
        e = smart_identity.ScanEvidence()
        for _ in range(20):
            e.add(read("Red Bull 250ml", 1), None)
        self.assertEqual(e.ocr_passes, 1)
        self.assertIsNone(e.resolve("beverage_can"))
        for i in range(2, 6):
            e.add({**read("Red Bull 250ml", i), "last_read_at": 123.0}, None)
        self.assertEqual(e.ocr_passes, 2)

    def test_phone_family_or_relative_score_is_not_exact_identity(self):
        e = smart_identity.ScanEvidence()
        summary = {"id": "iphone_14_pro", "score": 0.99, "frames": 8, "dino": True, "rear_frames": 8,
                   "raw_similarity": 0.3, "raw_margin": 0.2, "cues": {"layout": "triangle"},
                   "runner_up": [{"id": "iphone_15_pro", "score": 0.7}]}
        self.assertIsNone(e.resolve("smartphone", summary))
        summary.update(raw_similarity=0.85, raw_margin=0.01)
        self.assertIsNone(e.resolve("smartphone", summary))
        summary.update(raw_margin=0.08)
        self.assertEqual(e.resolve("smartphone", summary)["label"], "iphone_14_pro")
        summary.update(rear_frames=0)
        self.assertIsNone(e.resolve("smartphone", summary))

    def test_ram_bounded_and_reset_clears_all_evidence(self):
        e = smart_identity.ScanEvidence()
        crop = np.random.default_rng(1).integers(0, 255, (480, 240, 3), dtype=np.uint8)
        for i in range(40):
            e.add(read("Kirkland lime", i), None, crop, ["front", "side", "back"][i % 3])
        self.assertLessEqual(e.snapshot()["crop_bytes"], 3 * 320 * 320 * 3)
        self.assertEqual(e.snapshot()["crops_in_ram"], 3)
        e.reset()
        self.assertEqual(e.snapshot()["crop_bytes"], 0)
        self.assertIsNone(e.resolve("beverage_can"))

    def test_ocr_drops_text_from_before_scan(self):
        worker = ocr_fusion.OcrWorker()
        now = time.monotonic()
        worker._reads.extend([(now - 2, "Pepsi", 0.99), (now, "Kirkland", 0.9)])
        recent = worker.recent(since=now - 1)
        self.assertEqual(recent["texts"], ["Kirkland"])

    def test_raw_phone_margins_are_reported(self):
        acc = phone_classifier.ScanAccumulator()
        for _ in range(3):
            acc.add({"dino": True, "raw_dino": {"iphone_14_pro": 0.81, "iphone_15_pro": 0.80},
                     "top": [{"id": "iphone_14_pro", "score": 0.9}, {"id": "iphone_15_pro", "score": 0.8}],
                     "cues": {"layout": "triangle"}})
        summary = acc.summary()
        self.assertEqual(summary["rear_frames"], 3)
        self.assertAlmostEqual(summary["raw_margin"], 0.01)


class ZeroTouchPipelineTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.assets = FakeAssets()
        self.pipe = receiver.Pipeline(FakeExtractor(), receiver.EmbeddingBank(None), mock.Mock(), self.assets,
                                      detector=mock.Mock())
        self.pipe.view = "smart"
        self.pipe.devices = None
        self.pipe.smart = smart_scan.SmartScan(clock=self.clock)
        self.pipe._maybe_auto_label_scan = mock.Mock()
        self.pipe._detection = {"target": {"label": "beverage can", "score": 0.9}, "crop": None}
        self.frame = receiver.FrameScore(None, 0.0)
        self.pipe._citrus = citrus_classifier.color_evidence(np.full((80, 80, 3), (110, 200, 40), np.uint8))

    def step(self, run=1, text="Kirkland sparkling water lemon 355ml"):
        return self.pipe._smart_step(self.frame, None, None, {}, {"ocr": read(text, run)}, None, False)

    def test_empty_bank_auto_starts_and_locks_exact_identity_at_five_seconds(self):
        self.assertEqual(self.pipe.bank.labels(), [])
        self.assertEqual(self.step(), (255, 120, 0, 40, 1))
        for i in range(1, 11):
            self.clock.t += 0.5
            self.step(i)
        self.assertEqual(self.pipe.smart.status, "locked")
        self.assertIn("lime", self.pipe._viewer_label)
        self.assertEqual(self.pipe.bank.labels(), [])
        self.assertEqual(self.pipe.smart.snapshot()["elapsed"], None)
        self.assertEqual(self.pipe._metadata(self.pipe._viewer_label)["brand"], "Kirkland")

    def test_unknown_geometry_still_scans_but_does_not_invent_identity(self):
        self.step(text="")
        for i in range(11):
            self.clock.t += 0.5
            self.step(i, text="nutrition facts")
        self.assertEqual(self.pipe.smart.status, "inconclusive")
        self.assertEqual(self.pipe.smart_status()["category"], "beverage_can")
        self.assertIsNone(self.pipe._viewer_label)

    def test_wrong_detector_category_cannot_arm_scan(self):
        self.pipe._detection["target"] = {"label": "person", "score": 0.99}
        self.step()
        self.assertEqual(self.pipe.smart.status, "idle")

    def test_missing_detection_cancels_despite_stale_prototype_match(self):
        self.step()
        self.frame = receiver.FrameScore("old_can", 0.99)
        self.pipe._detection["target"] = None
        self.clock.t += smart_scan.LOST_S + 0.1
        self.step()
        self.assertEqual(self.pipe.smart.status, "idle")
        self.assertIsNone(self.pipe._viewer_label)

    def test_known_kirkland_locks_color_flavor_without_text(self):
        self.frame = receiver.FrameScore("kirkland_lemon_can", 0.95)
        self.step(text="")
        for i in range(10):
            self.clock.t += 0.5
            self.step(i, text="")
        self.assertEqual(self.pipe.smart.status, "locked")
        self.assertIn("lime", self.pipe.smart.label)

    def test_asset_readiness_not_scan_completion_controls_green_led(self):
        self.pipe.smart.status = "locked"
        self.pipe.smart.label = "red_bull_can"
        self.pipe._discovered["red_bull_can"] = {}
        with mock.patch.object(self.assets, "peek", return_value={"status": "resolving"}):
            self.assertEqual(self.step()[:3], (255, 120, 0))
            self.assertEqual(self.pipe.smart_status()["status"], "resolving")

    def test_new_identity_releases_without_stale_enrollment_votes(self):
        self.pipe.smart.status = "locked"
        self.pipe.smart.label = "red_bull_can"
        self.pipe.smart.last_seen = self.clock.t
        self.pipe._discovered["red_bull_can"] = {}
        self.pipe._viewer_label = "red_bull_can"
        self.pipe._detection["target"] = None
        self.clock.t += smart_scan.RELEASE_S + 0.1
        self.step()
        self.assertEqual(self.pipe.smart.status, "idle")
        self.assertIsNone(self.pipe._viewer_label)

    def test_locked_kirkland_rechecks_color_and_releases_on_sustained_swap(self):
        self.frame = receiver.FrameScore("kirkland_lime_can", 0.95)
        self.step(text="")
        for i in range(10):
            self.clock.t += 0.5
            self.step(i, text="")
        self.assertEqual(self.pipe.smart.status, "locked")
        lime = self.pipe._citrus
        pink = citrus_classifier.color_evidence(can((244, 212, 225)))
        self.pipe._citrus = pink
        self.step(text="")
        self.assertEqual(self.pipe.smart.status, "locked")  # one frame cannot swap a product
        self.pipe._citrus = lime
        self.step(text="")
        self.pipe._citrus = pink
        for _ in range(smart_scan.SWAP_FRAMES):
            self.clock.t += 0.05
            self.step(text="")
        self.assertEqual(self.pipe.smart.status, "idle")
        self.assertIsNone(self.pipe._viewer_label)
        self.step(text="")
        for i in range(10):
            self.clock.t += 0.5
            self.step(i, text="")
        self.assertEqual(self.pipe.smart.status, "locked")
        self.assertIn("grapefruit", self.pipe._viewer_label)

    def test_no_artwork_cannot_lock_just_the_enrolled_body(self):
        self.frame = receiver.FrameScore("kirkland_lime_can", 0.95)
        self.pipe._citrus = citrus_classifier.color_evidence(np.full((80, 80, 3), 225, np.uint8))
        self.step(text="")
        for i in range(10):
            self.clock.t += 0.5
            self.step(i, text="")
        self.assertEqual(self.pipe.smart.status, "inconclusive")
        self.assertIsNone(self.pipe._viewer_label)

    def test_smart_detection_id_changes_only_on_each_lock(self):
        initial = self.pipe.snapshot()["detection_id"]
        self.step()
        self.assertEqual(self.pipe.snapshot()["detection_id"], initial)
        for i in range(10):
            self.clock.t += 0.5
            self.step(i)
        first = self.pipe.snapshot()["detection_id"]
        self.assertNotEqual(first, initial)
        for _ in range(5):
            self.step()
        self.assertEqual(self.pipe.snapshot()["detection_id"], first)
        self.pipe._detection["target"] = None
        self.clock.t += smart_scan.RELEASE_S + 0.1
        self.step()
        self.pipe._detection["target"] = {"label": "beverage can", "score": 0.9}
        self.step()
        for i in range(10):
            self.clock.t += 0.5
            self.step(i)
        self.assertEqual(self.pipe.smart.status, "locked")
        self.assertNotEqual(self.pipe.snapshot()["detection_id"], first)


@unittest.skipIf(receiver.torch is None, "PyTorch not installed")
class KirklandOutputTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FakeExtractor()
        self.extractor.vec = receiver.torch.ones(16) / 4
        bank = receiver.EmbeddingBank(None)
        self.lime = "kirkland_sparkling_water_lime_can"
        bank.add_multiview(self.lime, {"front": [self.extractor.vec]}, {
            "brand": "Kirkland", "raw_name": "Kirkland Sparkling Water Lime", "category": "beverage_can",
            "variant": "sparkling water lime", "package_type": "can", "container_volume": "355ml"})
        self.pipe = receiver.Pipeline(self.extractor, bank, mock.Mock(), FakeAssets(), detector=mock.Mock())
        self.pipe.view = "debug"
        self.pipe.devices = None
        self.pipe._detection = {"target": {"label": "beverage can", "score": 0.9, "box": [0, 0, 1, 1]},
                                "crop": None}
        self.pipe._locate = mock.Mock(return_value=(None, 0))

    def frame(self, rgb):
        buf = io.BytesIO()
        Image.fromarray(can(rgb, print_frac=0.5)).save(buf, format="JPEG")
        with mock.patch.object(receiver, "SCENE_GATE", False), mock.patch.object(receiver, "LOG_FRAMES", False), \
                mock.patch.object(self.pipe, "_announce"):
            return self.pipe.process_jpeg(buf.getvalue())

    def test_only_lime_enrolled_corrects_debug_label_confirmed_label_and_asset(self):
        for rgb, flavor in (((244, 212, 225), "grapefruit"), ((242, 239, 209), "lemon"),
                            ((216, 241, 212), "lime"), ((230, 40, 50), "grapefruit")):
            with self.subTest(flavor=flavor):
                for _ in range(receiver.IDENT_FRAMES + 1):
                    result = self.frame(rgb)
                snap = self.pipe.snapshot()
                self.assertEqual(result["state"], receiver.IDENTIFIED)
                for key in ("label", "confirmed_label", "viewer_label", "glb_url"):
                    self.assertIn(flavor, snap[key], (key, snap[key]))
                self.assertEqual(snap["category"], "beverage_can")
                self.assertEqual(self.pipe.bank.labels(), [self.lime])

    def test_family_evidence_is_required_not_just_color(self):
        self.pipe._citrus = citrus_classifier.color_evidence(can((244, 212, 225)))
        ranked = [(self.lime, 0.5, "front")]
        self.assertEqual(self.pipe._kirkland_candidate(ranked, {})["action"], "awaiting_brand")
        self.pipe._ocr_tokens = ({"pepsi"}, {})
        self.assertEqual(self.pipe._kirkland_candidate([(self.lime, 0.95, "front")], {})["action"], "awaiting_brand")
        self.pipe._ocr_tokens = ({"kirkland", "juice"}, {})
        self.assertEqual(self.pipe._kirkland_candidate([(self.lime, 0.95, "front")], {})["action"], "awaiting_brand")

    def test_unreadable_artwork_does_not_return_lime_label(self):
        for _ in range(receiver.GUIDE_FRAMES + 1):
            result = self.frame((225, 225, 225))
        self.assertIsNone(result["label"])
        self.assertIsNone(self.pipe.snapshot()["confirmed_label"])

    def test_stale_lime_model_clears_when_pink_family_match_is_uncertain(self):
        for _ in range(receiver.IDENT_FRAMES + 1):
            self.frame((110, 200, 40))
        self.assertEqual(self.pipe._viewer_label, self.lime)
        with mock.patch.object(self.pipe.bank, "rank", return_value=[(self.lime, 0.6, "front")]):
            for _ in range(receiver.GUIDE_FRAMES + 1):
                self.frame((244, 212, 225))
        snap = self.pipe.snapshot()
        self.assertIsNone(snap["viewer_label"])
        self.assertIsNone(snap["confirmed_label"])
        self.assertIsNone(snap["label"])

    def test_debug_detection_id_changes_on_same_product_reconfirmation(self):
        initial = self.pipe.snapshot()["detection_id"]
        for _ in range(receiver.IDENT_FRAMES + 1):
            self.frame((110, 200, 40))
        first = self.pipe.snapshot()["detection_id"]
        self.assertIsInstance(first, str)
        self.assertNotEqual(first, initial)
        for _ in range(10):
            self.frame((110, 200, 40))
        self.assertEqual(self.pipe.snapshot()["detection_id"], first)
        for _ in range(receiver.GUIDE_FRAMES + 1):
            self.frame((225, 225, 225))
        self.assertEqual(self.pipe.snapshot()["detection_id"], first)
        for _ in range(receiver.IDENT_FRAMES + 1):
            self.frame((110, 200, 40))
        self.assertNotEqual(self.pipe.snapshot()["detection_id"], first)


class ProvenanceUpgradeTests(unittest.TestCase):
    def test_estimate_is_replaced_by_owned_scan_not_relabelled(self):
        product = entity_resolver.resolve("Kirkland lime can")
        scan = {"source_key": "library", "class_id": product.class_id, "match": "library",
                "library_id": "owned_lime", "title": "Lime hand scan"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mcp_server, "MODELS_DIR", Path(tmp)), \
                mock.patch.object(mcp_server, "is_valid_glb", return_value=True), \
                mock.patch.object(cad_harvester, "read_attribution", return_value={"source_key": "spec_generated"}), \
                mock.patch.object(cad_harvester, "_library_match", return_value={"id": "owned_lime"}), \
                mock.patch.object(cad_harvester.model_library, "install", return_value=scan) as install:
            result = cad_harvester.harvest(product, offline=True, source="local")
        install.assert_called_once_with("owned_lime", product.class_id)
        self.assertEqual(result["attribution"]["source_key"], "library")
        self.assertEqual(cad_harvester.provenance(scan)["type"], "verified")


class RelativePoseTests(unittest.TestCase):
    def test_stationary_surface_does_not_rotate_and_translation_tracks(self):
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 255, (192, 128, 3), dtype=np.uint8)
        tracker = pose.RelativeYaw()
        self.assertEqual(tracker.update(rgb), 0.0)
        self.assertAlmostEqual(tracker.update(rgb), 0.0, delta=0.1)
        moved = cv2.warpAffine(rgb, np.float32([[1, 0, 3], [0, 1, 0]]), (128, 192))
        self.assertLess(tracker.update(moved), -1)
        tracker.reset()
        self.assertEqual(tracker.update(moved), 0.0)


if __name__ == "__main__":
    unittest.main()
