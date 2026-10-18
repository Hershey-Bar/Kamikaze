"""Phase 4 tests: metadata parsing, 4-angle enrollment, pose estimation, CAD fallback and LED confidence mapping.

Run: python -m unittest test_comprehensive -v
Hardware tests run only when /var/run/arduino-router.sock exists (i.e. on the VENTUNO Q).
"""
import io
import json
import os
import statistics
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import cad_retriever
import mcp_server
import netguard
import pose
import receiver
import taxonomy

try:
    import torch
except ImportError:
    torch = None

needs_torch = unittest.skipIf(torch is None, "PyTorch not installed")
ROUTER_SOCK = "/var/run/arduino-router.sock"


def jpeg(size=(320, 240)):
    """Textured frame (a solid colour would be gated as an empty reticle)."""
    arr = np.full((size[1], size[0], 3), (120, 80, 40), dtype=np.uint8)
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    mask = ((yy // 12 + xx // 12) % 2).astype(bool)
    arr[mask] = 255 - arr[mask]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG")
    return buf.getvalue()


def rect_image(w, h, angle=0.0, size=(320, 240)):
    """Dark rectangle on a light background, centred (the 60% crop contains it)."""
    img = Image.new("RGB", size, (225, 225, 225))
    obj = Image.new("RGB", (w, h), (30, 40, 90))
    mask = Image.new("L", (w, h), 255)
    if angle:
        obj, mask = obj.rotate(angle, expand=True), mask.rotate(angle, expand=True)
    img.paste(obj, ((size[0] - obj.width) // 2, (size[1] - obj.height) // 2), mask)
    return img


class MetadataTests(unittest.TestCase):
    def test_scene_gate_keeps_dim_smooth_objects(self):
        """A dark, untextured phone on a dark desk has a low Laplacian variance but a clear outline: not empty."""
        rng = np.random.default_rng(3)
        base = np.full((480, 640), 20.0)
        empty = Image.fromarray(np.clip(base + rng.normal(0, 0.6, base.shape), 0, 255).astype(np.uint8)).convert("RGB")
        phone = base.copy()
        phone[150:330, 270:370] = 38.0  # 18 grey levels of contrast, no texture
        phone = Image.fromarray(np.clip(phone + rng.normal(0, 0.6, base.shape), 0, 255).astype(np.uint8)).convert("RGB")
        e, p = receiver.scene_saliency(empty), receiver.scene_saliency(phone)
        self.assertTrue(e["empty"], e)
        self.assertLess(p["laplacian_var"], receiver.LAPLACIAN_MIN, p)  # the Laplacian alone would call it empty
        self.assertGreaterEqual(p["edge_density"], receiver.EDGE_MIN, p)
        self.assertFalse(p["empty"], p)
        self.assertFalse(receiver.scene_saliency(rect_image(90, 150))["empty"])

    def test_can_metadata(self):
        meta = taxonomy.parse_metadata({
            "raw_name": "Coca Cola Original Taste Can 330ml", "category": "beverage_can", "brand": "Coca-Cola",
            "variant": "original", "package_type": "can", "container_volume": "330ml"})
        self.assertEqual(meta["category"], "beverage_can")
        self.assertEqual(meta["brand"], "Coca-Cola")
        self.assertEqual(meta["container_volume"], "330ml")
        self.assertEqual(taxonomy.label_from_metadata(meta), "coca_cola_330ml_can")

    def test_inference_from_name_only(self):
        s24 = taxonomy.parse_metadata({"raw_name": "Samsung Galaxy S24 Ultra"})
        self.assertEqual(s24["category"], "smartphone")
        self.assertEqual(taxonomy.label_from_metadata(s24), "galaxy_s24_ultra")
        self.assertEqual(taxonomy.phone_profile(s24)["tier"], "ultra")
        pro = taxonomy.parse_metadata({"raw_name": "iPhone 15 Pro Max"})
        self.assertEqual(taxonomy.label_from_metadata(pro), "iphone_15_pro_max")
        self.assertGreaterEqual(taxonomy.phone_profile(pro)["lenses"], 3)

    def test_rejects_bad_input(self):
        with self.assertRaises(taxonomy.MetadataError):
            taxonomy.parse_metadata({"raw_name": ""})
        with self.assertRaises(taxonomy.MetadataError):
            taxonomy.parse_metadata({"raw_name": "x", "category": "spaceship"})

    def test_search_queries_and_steps(self):
        meta = taxonomy.parse_metadata({"raw_name": "Samsung Galaxy S24 Ultra"})
        queries = taxonomy.search_queries(meta)
        self.assertTrue(any("S24 Ultra" in q and "glb" in q.lower() for q in queries), queries)
        for view in taxonomy.VIEWS:
            title, instruction = taxonomy.view_step("smartphone", view)
            self.assertTrue(title and instruction)

    def test_wizard_steps_are_category_specific(self):
        self.assertEqual(taxonomy.wizard_views("smartphone"), ("front", "back", "right", "left", "top"))
        self.assertEqual(taxonomy.wizard_views("beverage_can"), ("front", "back", "right", "top"))
        self.assertEqual(taxonomy.wizard_views(None), taxonomy.wizard_views("beverage_can"))
        for cat in taxonomy.CATEGORIES:
            for view in taxonomy.wizard_views(cat):
                title, prompt, target = taxonomy.view_detail(cat, view)
                self.assertTrue(title and prompt and target)
                self.assertNotIn("?", title)  # degree signs, not mojibake
        self.assertEqual(taxonomy.view_step("beverage_can", "right")[0], "Side seam (90°)")
        self.assertIn("power and volume", taxonomy.view_step("smartphone", "right", "Samsung")[1])
        self.assertIn("Action button", taxonomy.view_step("smartphone", "left", "Apple")[1])
        self.assertIn("pull-tab", taxonomy.view_step("beverage_can", "top")[1])


class ColorSignatureTests(unittest.TestCase):
    @staticmethod
    def solid(rgb, size=(48, 32)):
        return np.full((size[1], size[0], 3), rgb, np.uint8)

    def test_signature_separates_brand_colours(self):
        import color_sig
        red = color_sig.signature(self.solid((200, 20, 30)))
        red2 = color_sig.signature(self.solid((180, 30, 25)))
        blue = color_sig.signature(self.solid((20, 40, 190)))
        green = color_sig.signature(self.solid((30, 170, 60)))
        self.assertAlmostEqual(float(red.sum()), 1.0, places=5)
        self.assertEqual(color_sig.dominant(red), "red")
        self.assertEqual(color_sig.dominant(blue), "blue")
        self.assertGreater(color_sig.score(red2, {"front": red}), 0.9)
        self.assertLess(color_sig.score(blue, {"front": red}), 0.2)
        self.assertLess(color_sig.score(green, [red]), 0.2)
        self.assertIsNone(color_sig.score(red, {}))
        self.assertEqual(color_sig.score(blue, {"front": red, "back": blue}), 1.0)  # best view wins
        crop = receiver.color_signature(Image.fromarray(self.solid((200, 20, 30), (320, 240))),
                                        receiver.center_crop_box(320, 240))
        self.assertEqual(color_sig.dominant(crop), "red")


class PoseTests(unittest.TestCase):
    BOX = receiver.center_crop_box(320, 240)

    def test_top_view_pitches_toward_camera(self):
        est = pose.PoseEstimator(smoothing=1.0)
        out = est.update(rect_image(60, 80), self.BOX, {"front": 0.2, "top": 0.95}, "c", "beverage_can")
        self.assertTrue(out["valid"])
        self.assertGreater(out["pitch"], 20.0)  # positive = top tilted toward the camera

    def test_roll_is_recovered(self):
        est = pose.PoseEstimator(smoothing=1.0)
        out = est.update(rect_image(40, 110, angle=-20), self.BOX, {"front": 0.9, "right": 0.3}, "p", "smartphone")
        self.assertTrue(out["valid"])
        self.assertAlmostEqual(abs(out["roll"]), 20, delta=6)

    def test_view_matching_drives_yaw(self):
        est = pose.PoseEstimator(smoothing=1.0)
        img = rect_image(40, 110)
        front = est.update(img, self.BOX, {"front": 0.95, "right": 0.2, "back": 0.1, "top": 0.1}, "a", "beverage_can")
        est.reset()
        side = est.update(img, self.BOX, {"front": 0.1, "right": 0.95, "back": 0.2, "top": 0.1}, "b", "beverage_can")
        est.reset()
        back = est.update(img, self.BOX, {"front": 0.1, "right": 0.2, "back": 0.95, "top": 0.1}, "c", "beverage_can")
        self.assertLess(abs(front["yaw"]), 25)
        self.assertGreater(abs(side["yaw"]), 60)
        self.assertGreater(abs(back["yaw"]), 150)

    def test_valid_angles_under_10ms(self):
        est = pose.PoseEstimator()
        img = rect_image(50, 120, angle=-8, size=(640, 480))
        box = receiver.center_crop_box(640, 480)
        sims = {"front": 0.8, "back": 0.2, "right": 0.4, "top": 0.1}
        est.update(img, box, sims, "p", "smartphone")
        times, last = [], None
        for _ in range(30):
            last = est.update(img, box, sims, "p", "smartphone")
            times.append(last["ms"])
        self.assertTrue(last["valid"])
        for key, lim in (("pitch", 90.0), ("yaw", 180.0), ("roll", 90.0)):
            self.assertTrue(np.isfinite(last[key]) and abs(last[key]) <= lim, (key, last))
        self.assertLess(statistics.median(times), 10.0, times)

    def test_empty_scene_is_invalid(self):
        est = pose.PoseEstimator()
        out = est.update(Image.new("RGB", (320, 240), (200, 200, 200)), self.BOX, {}, "x", None)
        self.assertFalse(out["valid"])


class CadFallbackTests(unittest.TestCase):
    def test_offline_retrieve_never_invents_a_model(self):
        for raw in ("Samsung Galaxy S24 Ultra", "iPhone 15 Pro", "Coca Cola Original Taste Can 330ml", "Pepsi Zero Can"):
            meta = taxonomy.parse_metadata({"raw_name": raw})
            label = taxonomy.label_from_metadata(meta)
            with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mcp_server, "MODELS_DIR", Path(tmp)):
                info = cad_retriever.retrieve(label, meta, offline=True)
                self.assertEqual((info["status"], info["glb_url"]), ("offline_missing", ""), raw)
                self.assertTrue(info["reason"], raw)
                self.assertEqual(list(Path(tmp).iterdir()), [], raw)

    def test_unparseable_label_is_not_found(self):
        info = cad_retriever.retrieve("zz_mystery_object", None, offline=True)
        self.assertEqual(info["status"], "not_found")


class LedMappingTests(unittest.TestCase):
    def test_confidence_to_led_params(self):
        self.assertEqual(receiver.led_params(receiver.IDENTIFIED, 1.0), (0, 255, 0, 50, 8))
        self.assertEqual(receiver.led_params(receiver.IDENTIFIED, 0.5), (0, 255, 0, 50, 4))
        self.assertEqual(receiver.led_params(receiver.GUIDING, 0.6), (255, 120, 0, 40, 4))
        self.assertEqual(receiver.led_params(receiver.SEARCHING, 0.0)[4], 1)

    def test_phone_hints(self):
        self.assertEqual(receiver.phone_hint("smartphone", "front", 0.05), "flip")
        self.assertEqual(receiver.phone_hint("smartphone", "back", 0.05), "tilt")
        self.assertEqual(receiver.phone_hint("smartphone", "front", 0.5), "flip")  # the screen never verifies a phone
        self.assertIsNone(receiver.phone_hint("smartphone", "back", 0.5))
        self.assertIsNone(receiver.phone_hint("smartphone", "front", 0.5, front=False))
        self.assertIsNone(receiver.phone_hint("beverage_can", "front", 0.0))
        self.assertEqual(receiver.guidance_message(receiver.GUIDING, "x", 0.5, "flip"), taxonomy.MSG_PHONE_FLIP)
        self.assertEqual(receiver.guidance_message(receiver.GUIDING, "x", 0.5, "tilt"), taxonomy.MSG_PHONE_TILT)
        self.assertEqual(receiver.guidance_message(receiver.GUIDING, "x", 0.5, "tilt", prompt="custom"), "custom")
        self.assertEqual(receiver.guidance_message(receiver.GUIDING, "x", 0.5, "verify"), receiver.MSG_VERIFY)

    @unittest.skipUnless(os.path.exists(ROUTER_SOCK), "arduino-router socket not present (not on the board)")
    def test_hardware_rpc_roundtrip(self):
        for params in (receiver.led_params(receiver.IDENTIFIED, 1.0), receiver.led_params(receiver.GUIDING, 0.5),
                       receiver.led_params(receiver.SEARCHING, 0.0)):
            self.assertTrue(mcp_server.actuate_modulino(*params), params)


class FakeExtractor:
    name = "fake"

    def __init__(self):
        self.vec = None

    def embed_array(self, arr):
        return self.vec


class FakeAssets:
    def peek(self, label):
        return {"glb_url": f"/static/models/{label}.glb", "status": "ready", "title": label}

    get = peek


def unit(seed, dim=32):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(dim, generator=g), dim=0)


@needs_torch
class MultiViewFlowTests(unittest.TestCase):
    def setUp(self):
        self.ext = FakeExtractor()
        self.bank = receiver.EmbeddingBank(None)
        self.log = []
        self.leds = receiver.LedDriver(lambda *p: self.log.append(p) or True)
        self.pipe = receiver.Pipeline(self.ext, self.bank, self.leds, FakeAssets())
        self.server = receiver.make_server(self.pipe, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.conn = HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)
        self.patches = [mock.patch("builtins.print"), mock.patch.object(receiver, "LOG_FRAMES", False)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.server.shutdown()
        self.server.server_close()
        self.conn.close()

    def call(self, method, path, body=None):
        payload = json.dumps(body).encode() if isinstance(body, dict) else body
        self.conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
        resp = self.conn.getresponse()
        data = resp.read()
        return resp.status, (json.loads(data) if resp.getheader("Content-Type", "").startswith("application/json") else data)

    def frame(self, vec):
        self.ext.vec = vec
        status, data = self.call("POST", "/api/frame", jpeg())
        self.assertEqual(status, 200, data)
        return data

    def enroll_phone(self, name, base, noise=0.02):
        status, st = self.call("POST", "/api/enroll/start", {"raw_name": name, "frames_per_view": 6})
        self.assertEqual(status, 200, st)
        views = {}
        for i, view in enumerate(taxonomy.VIEWS):
            status, st = self.call("POST", "/api/enroll/capture", {"view": view})
            self.assertEqual(status, 200, st)
            views[view] = torch.nn.functional.normalize(base + 0.6 * unit(100 + i), dim=0)
            for k in range(6):
                self.frame(torch.nn.functional.normalize(views[view] + noise * unit(500 + k), dim=0))
        return views

    def test_four_angle_enrollment_flow(self):
        status, st = self.call("POST", "/api/enroll/start", {
            "raw_name": "Coca Cola Original Taste Can 330ml", "category": "beverage_can", "brand": "Coca-Cola",
            "variant": "original", "package_type": "can", "container_volume": "330ml", "frames_per_view": 6})
        self.assertEqual((status, st["phase"]), (200, "ready"))
        self.assertEqual(st["current_view"], "front")
        steps = taxonomy.wizard_views("beverage_can")
        self.assertEqual(st["steps"], list(steps))
        base = {v: unit(i + 1) for i, v in enumerate(steps)}
        for i, view in enumerate(steps):
            self.assertEqual(self.call("POST", "/api/enroll/capture", {})[1]["current_view"], view)
            for k in range(6):
                self.frame(torch.nn.functional.normalize(base[view] + 0.02 * unit(900 + k), dim=0))
            _, st = self.call("GET", "/api/enroll/status")
            if i < len(steps) - 1:
                self.assertEqual(st["phase"], "ready")
                self.assertEqual(st["views"][i]["state"], "done")
        self.assertEqual(st["phase"], "done", st)
        label = "coca_cola_330ml_can"
        entry = self.bank.info()[0]
        self.assertEqual(entry["label"], label)
        self.assertEqual(set(entry["views"]), set(steps))
        self.assertEqual(set(entry["color_views"]), set(steps))
        self.assertEqual(entry["metadata"]["category"], "beverage_can")
        sims = self.bank.view_sims(label, base["right"])
        self.assertEqual(max(sims, key=sims.get), "right")
        for _ in range(4):
            out = self.frame(base["back"])
        self.assertEqual((out["state"], out["label"]), (receiver.IDENTIFIED, label))
        self.assertEqual(self.pipe.snapshot()["best_view"], "back")
        self.assertEqual(self.log[-1][:4], (0, 255, 0, 50))
        self.assertGreaterEqual(self.log[-1][4], 6)

    def test_busy_and_cancel(self):
        self.call("POST", "/api/enroll/start", {"raw_name": "Pepsi Can"})
        self.call("POST", "/api/enroll/capture", {})
        self.assertEqual(self.call("POST", "/api/enroll/capture", {})[0], 409)
        self.assertEqual(self.call("POST", "/api/enroll/cancel", {})[0], 200)
        self.assertEqual(self.call("POST", "/api/enroll/start", {"raw_name": "", "category": "x"})[0], 400)

    def test_finish_early_requires_a_view(self):
        self.call("POST", "/api/enroll/start", {"raw_name": "Sprite Can", "frames_per_view": 3})
        self.assertEqual(self.call("POST", "/api/enroll/finish", {})[0], 400)
        self.call("POST", "/api/enroll/capture", {})
        for k in range(3):
            self.frame(unit(7 + k))
        status, st = self.call("POST", "/api/enroll/finish", {})
        self.assertEqual((status, st["phase"]), (200, "done"))
        self.assertEqual(list(self.bank.info()[0]["views"]), ["front"])

    def test_phone_ambiguity_gives_amber_and_flip_hint(self):
        base = unit(1)
        self.enroll_phone("Samsung Galaxy S24", base)
        twin = self.enroll_phone("Samsung Galaxy S24 Ultra", torch.nn.functional.normalize(base + 0.15 * unit(2), dim=0))
        self.assertEqual(len(self.bank.labels()), 2)
        for _ in range(8):
            out = self.frame(twin["front"])
        snap = self.pipe.snapshot()
        self.assertEqual(snap["state"], receiver.GUIDING, out)
        self.assertEqual(snap["hint"], "flip")
        self.assertEqual(snap["guidance"], taxonomy.MSG_PHONE_FLIP)
        self.assertEqual(self.log[-1], (255, 120, 0, 40, 4))

    def test_pose_in_state_and_endpoint(self):
        self.enroll_phone("Samsung Galaxy S24", unit(3))
        front = self.bank.views[self.bank.labels()[0]]["front"]
        buf = io.BytesIO()
        rect_image(60, 120, size=(320, 240)).save(buf, "JPEG")
        self.ext.vec = front
        times = []
        for _ in range(8):
            self.conn.request("POST", "/api/frame", body=buf.getvalue(), headers={"Content-Type": "image/jpeg"})
            self.conn.getresponse().read()
            _, snap = self.call("GET", "/api/pose")
            if snap["pose"].get("valid"):
                times.append(snap["pose"]["ms"])
        status, data = self.call("GET", "/api/pose")
        self.assertEqual(status, 200)
        p = data["pose"]
        self.assertTrue(p["valid"], data)
        # strict <10 ms budget is enforced in PoseTests; here the estimator shares a busy test process
        self.assertLess(min(times), 30.0, times)
        self.assertIn("pose", self.pipe.snapshot())

    def test_dinov2_pose_views_enrolled_and_drive_yaw(self):
        class FakeDino:
            device = "npu"
            desc = None

            def embed_rgb01(self, nhwc):
                return self.desc

            def info(self):
                return {"model": "fake-dinov2", "device": "npu"}

        dino = FakeDino()
        self.pipe.pose_worker = receiver.PoseWorker(dino, self.bank, hz=1000)
        steps = taxonomy.wizard_views("beverage_can")
        descs = {v: np.eye(8, dtype=np.float32)[i] for i, v in enumerate(steps)}
        self.call("POST", "/api/enroll/start", {"raw_name": "Sprite Can", "category": "beverage_can", "frames_per_view": 3})
        base = {v: unit(40 + i) for i, v in enumerate(steps)}
        for view in steps:
            self.call("POST", "/api/enroll/capture", {"view": view})
            dino.desc = descs[view]
            for _ in range(3):
                self.frame(base[view])
        _, st = self.call("GET", "/api/enroll/status")
        self.assertEqual(st["phase"], "done", st)
        label = self.bank.labels()[0]
        self.assertEqual(self.bank.info()[0]["pose_views"], sorted(steps))
        buf = io.BytesIO()
        rect_image(60, 120, size=(320, 240)).save(buf, "JPEG")
        self.ext.vec, dino.desc = base["right"], descs["right"]
        for _ in range(12):
            self.conn.request("POST", "/api/frame", body=buf.getvalue(), headers={"Content-Type": "image/jpeg"})
            self.conn.getresponse().read()
            time.sleep(0.02)  # let the pose worker publish
        pose = self.pipe.snapshot()["pose"]
        self.assertEqual(self.pipe.snapshot()["label"], label)
        self.assertEqual(pose["source"], "dinov2-npu", pose)
        self.assertEqual(pose["view"], "right")
        self.assertGreater(abs(pose["yaw"]), 60.0, pose)
        models = self.pipe.telemetry_snapshot()["inference"]["models"]
        self.assertEqual(models["pose"]["model"], "fake-dinov2")

    def test_db_save_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.bank.path = Path(tmp) / "enrolled_classes.pt"
            self.enroll_phone("iPhone 15 Pro", unit(5))
            self.assertEqual(self.call("POST", "/api/db/save", {})[0], 200)
            status, blob = self.call("GET", "/api/db/export")
            self.assertEqual(status, 200)
            saved = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            proto = next(iter(saved["prototypes"].values()))
            self.assertEqual(set(proto["views"]), set(taxonomy.VIEWS))
            self.assertIn("global_centroid", proto)
            self.assertIn("cad_asset", proto)
            fresh = receiver.EmbeddingBank(self.bank.path)
            self.assertEqual(fresh.load(), 1)
            self.assertEqual(set(fresh.views[fresh.labels()[0]]), set(taxonomy.VIEWS))


class LightingGateTests(unittest.TestCase):
    @staticmethod
    def frame(arr):
        return Image.fromarray(arr.astype(np.uint8))

    def test_dark_glare_and_ok(self):
        dark = receiver.lighting_check(self.frame(np.full((240, 320, 3), 15)))
        self.assertEqual((dark["ok"], dark["issue"], dark["message"]), (False, "low_light", receiver.MSG_LOW_LIGHT))
        glare = np.full((240, 320, 3), 120)
        glare[40:200, 60:260] = 255
        out = receiver.lighting_check(self.frame(glare))
        self.assertEqual((out["ok"], out["issue"], out["message"]), (False, "glare", receiver.MSG_GLARE))
        yy, xx = np.mgrid[0:240, 0:320]
        tex = np.stack([90 + 60 * (((yy // 10 + xx // 10) % 2))] * 3, axis=-1)
        good = receiver.lighting_check(self.frame(tex))
        self.assertTrue(good["ok"], good)
        self.assertIsNone(good["issue"])
        with mock.patch.object(receiver, "LIGHT_GATE", False):
            self.assertTrue(receiver.lighting_check(self.frame(np.full((240, 320, 3), 15)))["ok"])


class OcrFusionTests(unittest.TestCase):
    COKE_ZERO = {"brand": "Coca-Cola", "variant": "zero", "raw_name": "coke zero can 330ml", "category": "beverage_can"}
    PEPSI = {"brand": "Pepsi", "variant": "", "raw_name": "pepsi can", "category": "beverage_can"}

    def test_text_score_and_fusion(self):
        import ocr_fusion
        zero = ocr_fusion.class_words(self.COKE_ZERO)
        pepsi = ocr_fusion.class_words(self.PEPSI)
        seen = ocr_fusion.tokens_of("Coca-Cola ZERO SUGAR")
        self.assertEqual(ocr_fusion.text_score(zero, seen), 1.0)
        self.assertEqual(ocr_fusion.text_score(pepsi, seen), 0.0)  # another brand's name is in view
        self.assertEqual(ocr_fusion.text_score(zero, ocr_fusion.tokens_of("Coca-Cola")), 0.6)
        self.assertIsNone(ocr_fusion.text_score(zero, set()))
        self.assertIsNone(ocr_fusion.text_score(zero, ocr_fusion.tokens_of("330 ml")))
        self.assertAlmostEqual(ocr_fusion.fuse(0.8, 1.0), (0.60 * 0.8 + 0.25) / 0.85)
        self.assertEqual(ocr_fusion.fuse(0.8, None), 0.8)
        self.assertLessEqual(ocr_fusion.fuse(0.0, 1.0), 0.35)  # text alone can never identify
        # Tri-modal: absent modalities drop out, below the gate vision passes through untouched.
        self.assertAlmostEqual(ocr_fusion.fuse_calibrated(0.8, 1.0, 1.0), 0.6 * 0.8 + 0.25 + 0.15)
        self.assertAlmostEqual(ocr_fusion.fuse_calibrated(0.8, None, 0.0), 0.6 * 0.8 / 0.75)
        self.assertEqual(ocr_fusion.fuse_calibrated(0.2, 1.0, 1.0, gate=0.25), 0.2)

    def test_corrector_fixes_ocr_glyph_errors(self):
        import ocr_fusion
        c = ocr_fusion.Corrector()
        c.set_extra(["Kirkland", "grapefruit"])
        for raw, want in (("c0ke", "coke"), ("samsunq", "samsung"), ("S2S", "s25"), ("ZER0", "zero"),
                          ("kirklond", "kirkland"), ("grapefrult", "grapefruit"), ("PEPSI", "pepsi")):
            self.assertEqual(c.word(raw), want, raw)
        self.assertEqual(c.word("cake"), "cake")  # short, non-glyph difference: never forced onto "coke"
        self.assertIsNone(c.word("ml"))  # too short to be a token
        words, fixes = c.text("C0KE ZER0 sugar")
        self.assertEqual(words, ["coke", "zero", "sugar"])
        self.assertEqual(fixes, [("c0ke", "coke"), ("zer0", "zero")])
        seen = set(ocr_fusion.tokens_from_words(words))
        zero = ocr_fusion.class_words(self.COKE_ZERO)
        self.assertEqual(ocr_fusion.text_score(zero, seen), 1.0)

    def test_worker_reads_into_recent_window(self):
        import ocr_fusion

        def engine(bgr, use_cls=False):
            return [[None, "PEPSI", 0.93], [None, "noise", 0.2]], None

        worker = ocr_fusion.OcrWorker(engine, hz=100)
        worker.submit(np.zeros((400, 300, 3), np.uint8))
        deadline = time.monotonic() + 5
        while worker.recent()["runs"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        rec = worker.recent()
        self.assertTrue(rec["enabled"])
        self.assertEqual(rec["texts"], ["PEPSI"])  # low-confidence reads are dropped
        self.assertIn("pepsi", rec["tokens"])
        self.assertEqual(worker.recent(window_s=0.0)["texts"], [])
        self.assertFalse(ocr_fusion.OcrWorker(None).recent()["enabled"])


class Yaw360Tests(unittest.TestCase):
    def test_cardinal_views_are_unambiguous(self):
        yaw = pose.PoseEstimator._view_yaw
        cases = {"front": 0.0, "right": 90.0, "back": 180.0, "left": -90.0}
        for view, want in cases.items():
            sims = {v: (0.9 if v == view else 0.1) for v in ("front", "right", "back", "left", "top")}
            got, top_w, best = yaw(sims, 0.0)
            self.assertEqual(best, view)
            self.assertAlmostEqual(abs(((got - want) + 180) % 360 - 180), 0.0, delta=2.0, msg=(view, got))
        between, _, _ = yaw({"front": 0.9, "right": 0.9, "back": 0.1, "left": 0.1}, 0.0)
        self.assertAlmostEqual(between, 45.0, delta=2.0)
        _, top_w, best = yaw({"front": 0.1, "right": 0.1, "back": 0.1, "left": 0.1, "top": 0.9}, 0.0)
        self.assertEqual(best, "top")
        self.assertGreater(top_w, 0.9)

    def test_legacy_side_view_keeps_last_sign(self):
        yaw = pose.PoseEstimator._view_yaw
        sims = {"front": 0.1, "side": 0.9, "back": 0.1}
        self.assertAlmostEqual(yaw(sims, 0.0)[0], 90.0, delta=2.0)
        self.assertAlmostEqual(yaw(sims, -60.0)[0], -90.0, delta=2.0)


@needs_torch
class CentredPoseSimTests(unittest.TestCase):
    def test_shared_identity_component_is_removed(self):
        bank = receiver.EmbeddingBank(None)
        shared = np.ones(8, np.float32) * 3.0
        descs = {v: shared + np.eye(8, dtype=np.float32)[i] for i, v in enumerate(taxonomy.VIEWS)}
        bank.add_multiview("x_can", {v: [unit(60 + i)] for i, v in enumerate(taxonomy.VIEWS)}, {"category": "beverage_can"},
                           {v: [d] for v, d in descs.items()})
        raw = {v: float(np.dot(d / np.linalg.norm(d), descs["left"] / np.linalg.norm(descs["left"]))) for v, d in descs.items()}
        self.assertGreater(min(raw.values()), 0.9)  # uncentred cosines barely separate the views
        sims = bank.pose_view_sims("x_can", descs["left"])
        self.assertEqual(max(sims, key=sims.get), "left")
        self.assertAlmostEqual(sims["left"], 1.0, places=4)
        self.assertLess(sims["right"], 0.0)


@needs_torch
class EnrollRejectTests(unittest.TestCase):
    def test_lighting_rejects_abort_capture(self):
        enr = receiver.MultiViewEnroller(receiver.EmbeddingBank(None))
        enr.start(taxonomy.parse_metadata({"raw_name": "Pepsi Can"}), frames_per_view=3)
        self.assertFalse(enr.reject("too dark"))  # not capturing yet
        enr.capture()
        limit = max(3 * enr.per_view, 30)
        for _ in range(limit - 1):
            self.assertFalse(enr.reject(receiver.MSG_LOW_LIGHT))
        self.assertEqual(enr.phase, "capturing")
        self.assertTrue(enr.reject(receiver.MSG_LOW_LIGHT))
        self.assertEqual(enr.phase, "ready")
        self.assertEqual(enr.error, f"capture failed: {receiver.MSG_LOW_LIGHT}")
        enr.capture()
        self.assertIsNone(enr.error)
        self.assertEqual(enr.rejected, 0)


@needs_torch
class CadModeEndpointTests(unittest.TestCase):
    """Offline/online mode, NOT_FOUND, candidate browsing, upload and spec-generation endpoints (no network)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(mcp_server, "MODELS_DIR", Path(self.tmp.name)),
                        mock.patch("builtins.print"), mock.patch.object(receiver, "LOG_FRAMES", False)]
        for p in self.patches:
            p.start()
        self.bank = receiver.EmbeddingBank(None)
        self.assets = receiver.AssetManager(offline=True, source="auto")
        self.pipe = receiver.Pipeline(FakeExtractor(), self.bank, receiver.LedDriver(lambda *a: True), self.assets)
        self.server = receiver.make_server(self.pipe, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.conn = HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=15)
        meta = taxonomy.parse_metadata({"raw_name": "Coca Cola Zero Can 330ml"})
        self.label = taxonomy.label_from_metadata(meta)
        self.bank.add_multiview(self.label, {"front": [unit(1)]}, meta)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.conn.close()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()
        netguard.set_offline(False)  # process-global: later tests need the network

    def call(self, method, path, body=None, headers=None):
        payload = json.dumps(body).encode() if isinstance(body, dict) else body
        self.conn.request(method, path, body=payload, headers=headers or {"Content-Type": "application/json"})
        resp = self.conn.getresponse()
        data = resp.read()
        return resp.status, json.loads(data) if "json" in resp.getheader("Content-Type", "") else data

    def wait_status(self, want, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.assets.peek(self.label)
            if info["status"] == want:
                return info
            time.sleep(0.02)
        self.fail(f"asset never reached {want}: {self.assets.peek(self.label)}")

    def test_mode_toggle_and_validation(self):
        status, out = self.call("GET", "/api/settings/mode")
        self.assertEqual((status, out["mode"], out["offline"]), (200, "offline", True))
        self.assertEqual((out["sources"], out["source"]), (list(receiver.cad_harvester.OFFLINE_SOURCES), "local"))
        self.assertEqual(out["all_sources"], list(cad_harvester_sources()))
        self.assertTrue(out["network"]["offline"])
        self.assertTrue(netguard.is_offline())
        status, out = self.call("POST", "/api/settings/mode", {"source": "sketchfab"})
        self.assertEqual(status, 400, out)  # online repositories are refused while offline
        self.assertEqual(self.call("POST", "/api/settings/mode", {"source": "upload"})[1]["source"], "upload")
        status, out = self.call("POST", "/api/settings/mode", {"mode": "online"})
        self.assertEqual((status, out["source"], out["sources"]), (200, "auto", list(cad_harvester_sources())), out)
        self.assertFalse(netguard.is_offline())
        status, out = self.call("POST", "/api/settings/mode", {"mode": "online", "source": "polypizza"})
        self.assertEqual((status, out["mode"], out["source"]), (200, "online", "polypizza"))
        self.assertFalse(self.assets.offline)
        self.assertEqual(self.call("POST", "/api/settings/mode", {"mode": "sideways"})[0], 400)
        self.assertEqual(self.call("POST", "/api/settings/mode", {"source": "turbosquid"})[0], 400)
        status, out = self.call("POST", "/api/settings/mode", {"mode": "offline"})
        self.assertEqual((out["mode"], out["source"]), ("offline", "local"))
        self.assertEqual(self.call("POST", "/api/settings/mode", {"mode": "online"})[1]["source"], "polypizza")

    def test_offline_class_is_offline_missing_and_no_request_is_made(self):
        before = netguard.stats()["blocked"]
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used offline")) as net:
            self.assets.get(self.label)
            info = self.wait_status("offline_missing")
        net.assert_not_called()
        self.assertEqual(netguard.stats()["blocked"], before)  # nothing even tried to leave the box
        self.assertEqual(info["glb_url"], "")
        self.assertIn("offline", info["reason"])
        self.assertEqual(list(Path(self.tmp.name).glob("*.glb")), [])

    def test_candidates_and_actions_guarded(self):
        self.assertEqual(self.call("GET", "/api/classes/ghost_can/cad_candidates")[0], 404)
        status, out = self.call("GET", f"/api/classes/{self.label}/cad_candidates")
        self.assertEqual((status, out["offline"]), (409, True))
        self.assertEqual(self.call("GET", f"/api/classes/{self.label}/cad_candidates?source=nope")[0], 400)
        self.assertEqual(self.call("POST", f"/api/classes/{self.label}/generate_spec", {})[0], 409)  # offline
        self.assertEqual(self.call("POST", f"/api/classes/{self.label}/cad_select", {"uid": "x"})[0], 409)
        self.assertEqual(self.call("POST", "/api/classes/ghost_can/generate_spec", {})[0], 404)
        self.call("POST", "/api/settings/mode", {"mode": "online"})
        self.assertEqual(self.call("POST", f"/api/classes/{self.label}/cad_select", {})[0], 400)
        status, out = self.call("POST", f"/api/classes/{self.label}/cad_select", {"uid": "never-browsed"})
        self.assertEqual(status, 409, out)

    def test_upload_rejects_bad_glb_and_accepts_real_one(self):
        import procedural_glb
        hdr = {"Content-Type": "model/gltf-binary", "X-Filename": "mine.glb"}
        status, out = self.call("POST", f"/api/classes/{self.label}/upload_cad", b"<html>nope</html>", hdr)
        self.assertEqual(status, 422, out)
        self.assertEqual(self.call("POST", "/api/classes/ghost_can/upload_cad", b"glTF", hdr)[0], 404)
        good = procedural_glb.build_can_spec(66.0, 115.2, 54.0)
        status, out = self.call("POST", f"/api/classes/{self.label}/upload_cad?filename=coke.glb", good, hdr)
        self.assertEqual((status, out["ok"], out["status"]), (200, True, "cached"), out)
        asset = out["asset"]
        self.assertEqual(asset["status"], "ready")
        self.assertTrue(asset["glb_url"].startswith(f"/static/models/{self.label}.glb"))
        self.assertEqual(asset["attribution"]["filename"], "coke.glb")
        self.assertEqual(self.bank.info()[0]["cad_status"], "ready")

    def test_generate_spec_online(self):
        self.call("POST", "/api/settings/mode", {"mode": "online"})
        status, out = self.call("POST", f"/api/classes/{self.label}/generate_spec", {})
        self.assertEqual((status, out["status"]), (200, "generated"), out)
        self.assertEqual(out["asset"]["badge"], receiver.cad_harvester.SPEC_BADGE)
        self.assertEqual(out["result"]["spec"]["diameter_mm"], 66.0)

    def test_cad_orient_endpoint(self):
        import procedural_glb
        self.assertEqual(self.call("POST", f"/api/classes/{self.label}/cad_orient", {"auto": True})[0], 404)  # no GLB yet
        self.assertEqual(self.call("POST", "/api/classes/ghost_can/cad_orient", {"auto": True})[0], 404)
        hdr = {"Content-Type": "model/gltf-binary"}
        self.call("POST", f"/api/classes/{self.label}/upload_cad", procedural_glb.build_can_spec(66.0, 115.2, 54.0), hdr)
        url0 = self.assets.peek(self.label)["glb_url"]
        self.assertEqual(self.call("POST", f"/api/classes/{self.label}/cad_orient", {"axis": "q"})[0], 400)
        self.assertEqual(self.call("POST", f"/api/classes/{self.label}/cad_orient", {"axis": "x", "deg": 1e9})[0], 400)
        time.sleep(0.01)
        status, out = self.call("POST", f"/api/classes/{self.label}/cad_orient", {"axis": "x", "deg": 90})
        self.assertEqual(status, 200, out)
        self.assertFalse(out["orientation"]["identity"])
        self.assertFalse(out["asset"]["orientation"]["identity"])
        self.assertNotEqual(out["asset"]["glb_url"], url0)  # cache-busted
        status, out = self.call("POST", f"/api/classes/{self.label}/cad_orient", {"auto": True})
        self.assertTrue(out["orientation"]["identity"])  # upright can: auto removes the manual tilt
        status, out = self.call("POST", f"/api/classes/{self.label}/cad_orient", {})
        self.assertEqual((status, out["orientation"]["identity"]), (200, True))



class FakeV4L2:
    """In-memory V4L2 device: QUERYCTRL / G_CTRL / S_CTRL over the real struct layouts."""

    def __init__(self, controls):
        import v4l2_ctrl as v
        self.v = v
        self.ctrls = {c.cid: c for c in controls}
        self.values = {c.cid: c.default for c in controls}
        self.writes = []

    def __call__(self, fd, req, buf):
        import struct
        v = self.v
        if req == v.VIDIOC_QUERYCTRL:
            cid = struct.unpack_from("I", buf)[0]
            if cid & v.CTRL_FLAG_NEXT_CTRL:
                nxt = sorted(c for c in self.ctrls if c > cid & ~v.CTRL_FLAG_NEXT_CTRL)
                if not nxt:
                    raise OSError(22, "EINVAL")
                cid = nxt[0]
            c = self.ctrls.get(cid)
            if c is None:
                raise OSError(22, "EINVAL")
            buf[:] = struct.pack(v.QUERYCTRL_FMT, c.cid, c.type, c.name.encode(), c.minimum, c.maximum, c.step,
                                 c.default, c.flags, 0, 0)
        elif req in (v.VIDIOC_G_CTRL, v.VIDIOC_S_CTRL):
            cid, val = struct.unpack(v.CONTROL_FMT, buf)
            if cid not in self.ctrls:
                raise OSError(22, "EINVAL")
            if req == v.VIDIOC_S_CTRL:
                self.values[cid] = val
                self.writes.append((cid, val))
            else:
                buf[:] = struct.pack(v.CONTROL_FMT, cid, self.values[cid])
        else:
            raise OSError(25, "ENOTTY")
        return 0


def brio_controls(focus=True):
    import v4l2_ctrl as v
    out = [v.Control(0x00980900, v.CTRL_TYPE_INTEGER, "Brightness", 0, 255, 1, 128, 0)]
    if focus:
        out += [v.Control(v.CID_FOCUS_ABSOLUTE, v.CTRL_TYPE_INTEGER, "Focus, Absolute", 0, 255, 5, 0, 0),
                v.Control(v.CID_FOCUS_AUTO, v.CTRL_TYPE_BOOLEAN, "Focus, Automatic Continuous", 0, 1, 1, 1, 0)]
    return out


def focus_for(dev):
    import v4l2_ctrl as v
    return v.FocusControl(v.V4L2Controls("/dev/video0", ioctl=dev, opener=lambda path: 3, closer=lambda fd: None))


class V4L2FocusTests(unittest.TestCase):
    def test_query_list_and_clamp(self):
        import v4l2_ctrl as v
        dev = FakeV4L2(brio_controls())
        ctl = v.V4L2Controls("/dev/video0", ioctl=dev, opener=lambda path: 3, closer=lambda fd: None)
        self.assertEqual([c.name for c in ctl.list()], ["Brightness", "Focus, Absolute", "Focus, Automatic Continuous"])
        absolute = ctl.query(v.CID_FOCUS_ABSOLUTE)
        self.assertEqual((absolute.minimum, absolute.maximum, absolute.step), (0, 255, 5))
        self.assertEqual([absolute.clamp(x) for x in (-9, 12, 13, 999)], [0, 10, 15, 255])
        self.assertIsNone(ctl.query(0x12345678))
        self.assertEqual(v.VIDIOC_QUERYCTRL, 0xC0445624)  # the kernel's _IOWR('V', 36, struct v4l2_queryctrl)
        self.assertEqual((v.VIDIOC_G_CTRL, v.VIDIOC_S_CTRL), (0xC008561B, 0xC008561C))

    def test_manual_focus_disables_af_and_snaps(self):
        import v4l2_ctrl as v
        dev = FakeV4L2(brio_controls())
        focus = focus_for(dev)
        st = focus.status()
        self.assertEqual((st["supported"], st["auto_supported"], st["auto"], st["max"]), (True, True, True, 255))
        st = focus.set(auto=False, value=122)
        self.assertEqual((st["auto"], st["value"]), (False, 120))
        self.assertEqual(dev.writes, [(v.CID_FOCUS_AUTO, 0), (v.CID_FOCUS_ABSOLUTE, 120)])
        self.assertTrue(focus.set(auto=True)["auto"])

    def test_fixed_focus_camera_reports_unsupported(self):
        focus = focus_for(FakeV4L2(brio_controls(focus=False)))
        st = focus.status()
        self.assertEqual((st["supported"], st["auto_supported"]), (False, False))
        self.assertIn("fixed-focus", st["reason"])
        with self.assertRaises(LookupError):
            focus.set(auto=False, value=100)

    def test_unavailable_host_and_env_parsing(self):
        import v4l2_ctrl as v
        ctl = v.V4L2Controls("/dev/video0", ioctl=None)
        ctl._ioctl = None
        self.assertIn("unavailable", v.FocusControl(ctl).status()["reason"])
        self.assertEqual([v.parse_focus_env(x) for x in ("", "auto", " 40 ")], [(None, None), (True, None), (False, 40)])
        with self.assertRaises(ValueError):
            v.parse_focus_env("sharp")


class NetguardTests(unittest.TestCase):
    def tearDown(self):
        netguard.set_offline(False)

    def test_blocks_external_hosts_only_when_offline(self):
        netguard.set_offline(False)
        netguard.check("https://api.sketchfab.com/v3/search")
        netguard.set_offline(True)
        before = netguard.stats()["blocked"]
        for url in ("http://127.0.0.1:5000/api/state", "http://localhost/x", "http://[::1]:8080/"):
            netguard.check(url)
        for url in ("https://api.sketchfab.com/v3/search", "https://api.poly.pizza/v1.1/search", "http://10.0.0.2/"):
            with self.assertRaises(netguard.NetworkBlocked):
                netguard.check(url)
        st = netguard.stats()
        self.assertEqual((st["offline"], st["blocked"] - before), (True, 3))
        self.assertEqual(st["last_blocked"]["host"], "10.0.0.2")

    def test_offline_mcp_and_spec_lookups_never_open_sockets(self):
        import spec_search
        netguard.set_offline(True)
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("socket opened offline")) as net:
            with self.assertRaises(Exception):
                mcp_server._http_json("https://api.sketchfab.com/v3/search?q=x")
            with self.assertRaises(Exception):
                spec_search._get("https://www.gsmarena.com/x.php")
        net.assert_not_called()


@needs_torch
class FocusEndpointTests(unittest.TestCase):
    def setUp(self):
        self.pipe = receiver.Pipeline(FakeExtractor(), receiver.EmbeddingBank(None), receiver.LedDriver(lambda *a: True),
                                      receiver.AssetManager(offline=True))
        self.server = receiver.make_server(self.pipe, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.conn = HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.conn.close()
        netguard.set_offline(False)

    def call(self, method, body=None):
        self.conn.request(method, "/api/camera/focus", body=json.dumps(body) if body is not None else None)
        resp = self.conn.getresponse()
        return resp.status, json.loads(resp.read())

    def attach(self, controls):
        cam = mock.Mock(name="cam", live=False, device="/dev/video0")
        cam.name = "Brio 101"
        cam.focus = focus_for(FakeV4L2(controls))
        self.pipe.attach_camera(cam)

    def test_http_ingest_has_no_focus(self):
        self.assertEqual(self.call("GET")[0], 409)
        self.assertEqual(self.call("POST", {"auto": True})[0], 409)

    def test_focus_camera(self):
        self.attach(brio_controls())
        status, out = self.call("GET")
        self.assertEqual((status, out["camera"], out["supported"]), (200, "Brio 101", True))
        status, out = self.call("POST", {"auto": False, "value": 63})
        self.assertEqual((status, out["auto"], out["value"]), (200, False, 65))
        self.assertEqual(self.call("POST", {"value": 5000})[0], 400)
        self.assertEqual(self.call("POST", {"auto": "yes"})[0], 400)
        self.assertEqual(self.call("POST", {})[0], 400)

    def test_fixed_focus_camera_is_422_with_status(self):
        self.attach(brio_controls(focus=False))
        status, out = self.call("GET")
        self.assertEqual((status, out["supported"]), (200, False))
        status, out = self.call("POST", {"auto": False, "value": 100})
        self.assertEqual((status, out["ok"], out["supported"]), (422, False, False), out)
        self.assertIn("fixed-focus", out["reason"])


class BankDbTypesTests(unittest.TestCase):
    @needs_torch
    def test_cad_info_with_numpy_values_round_trips_weights_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = receiver.EmbeddingBank(Path(tmp) / "db.pt")
            bank.add_multiview("pepsi_can", {"front": [unit(1)]}, {"category": "beverage_can"})
            bank.set_cad("pepsi_can", {"glb_url": "/static/models/pepsi_can.glb",
                                       "attribution": {"orientation_fix": {"rotation": [np.float64(0.5)] * 4,
                                                                           "identity": np.bool_(False)}}})
            data = torch.load(Path(tmp) / "db.pt", map_location="cpu", weights_only=True)
            fix = data["prototypes"]["pepsi_can"]["cad_asset"]["attribution"]["orientation_fix"]
            self.assertEqual((fix["rotation"], fix["identity"]), ([0.5] * 4, False))
            self.assertIs(type(fix["rotation"][0]), float)


def cad_harvester_sources():
    return receiver.cad_harvester.SOURCES


if __name__ == "__main__":
    unittest.main()
