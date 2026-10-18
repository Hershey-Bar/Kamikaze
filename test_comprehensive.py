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


class PoseTests(unittest.TestCase):
    BOX = receiver.center_crop_box(320, 240)

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
                self.assertEqual((info["status"], info["glb_url"]), ("not_found", ""), raw)
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
        base = {v: unit(i + 1) for i, v in enumerate(taxonomy.VIEWS)}
        for i, view in enumerate(taxonomy.VIEWS):
            self.assertEqual(self.call("POST", "/api/enroll/capture", {})[1]["current_view"], view)
            for k in range(6):
                self.frame(torch.nn.functional.normalize(base[view] + 0.02 * unit(900 + k), dim=0))
            _, st = self.call("GET", "/api/enroll/status")
            if i < len(taxonomy.VIEWS) - 1:
                self.assertEqual(st["phase"], "ready")
                self.assertEqual(st["views"][i]["state"], "done")
        self.assertEqual(st["phase"], "done", st)
        label = "coca_cola_330ml_can"
        entry = self.bank.info()[0]
        self.assertEqual(entry["label"], label)
        self.assertEqual(set(entry["views"]), set(taxonomy.VIEWS))
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
        descs = {v: np.eye(8, dtype=np.float32)[i] for i, v in enumerate(taxonomy.VIEWS)}
        self.call("POST", "/api/enroll/start", {"raw_name": "Sprite Can", "category": "beverage_can", "frames_per_view": 3})
        base = {v: unit(40 + i) for i, v in enumerate(taxonomy.VIEWS)}
        for view in taxonomy.VIEWS:
            self.call("POST", "/api/enroll/capture", {"view": view})
            dino.desc = descs[view]
            for _ in range(3):
                self.frame(base[view])
        _, st = self.call("GET", "/api/enroll/status")
        self.assertEqual(st["phase"], "done", st)
        label = self.bank.labels()[0]
        self.assertEqual(self.bank.info()[0]["pose_views"], sorted(taxonomy.VIEWS))
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
        self.assertAlmostEqual(ocr_fusion.fuse(0.8, 1.0), 0.65 * 0.8 + 0.35)
        self.assertEqual(ocr_fusion.fuse(0.8, None), 0.8)
        self.assertLessEqual(ocr_fusion.fuse(0.0, 1.0), 0.35)  # text alone can never identify

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
        self.assertEqual(out["sources"], list(cad_harvester_sources()))
        status, out = self.call("POST", "/api/settings/mode", {"mode": "online", "source": "polypizza"})
        self.assertEqual((status, out["mode"], out["source"]), (200, "online", "polypizza"))
        self.assertFalse(self.assets.offline)
        self.assertEqual(self.call("POST", "/api/settings/mode", {"mode": "sideways"})[0], 400)
        self.assertEqual(self.call("POST", "/api/settings/mode", {"source": "thingiverse"})[0], 400)
        self.assertEqual(self.call("POST", "/api/settings/mode", {"mode": "offline"})[1]["mode"], "offline")

    def test_offline_class_is_not_found_and_no_file_is_written(self):
        self.assets.get(self.label)
        info = self.wait_status("not_found")
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


def cad_harvester_sources():
    return receiver.cad_harvester.SOURCES


if __name__ == "__main__":
    unittest.main()
