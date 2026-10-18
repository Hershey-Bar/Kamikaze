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
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, "JPEG")
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
        out = est.update(rect_image(40, 110, angle=-20), self.BOX, {"front": 0.9, "side": 0.3}, "p", "smartphone")
        self.assertTrue(out["valid"])
        self.assertAlmostEqual(abs(out["roll"]), 20, delta=6)

    def test_view_matching_drives_yaw(self):
        est = pose.PoseEstimator(smoothing=1.0)
        img = rect_image(40, 110)
        front = est.update(img, self.BOX, {"front": 0.95, "side": 0.2, "back": 0.1, "top": 0.1}, "a", "beverage_can")
        est.reset()
        side = est.update(img, self.BOX, {"front": 0.1, "side": 0.95, "back": 0.2, "top": 0.1}, "b", "beverage_can")
        est.reset()
        back = est.update(img, self.BOX, {"front": 0.1, "side": 0.2, "back": 0.95, "top": 0.1}, "c", "beverage_can")
        self.assertLess(abs(front["yaw"]), 25)
        self.assertGreater(abs(side["yaw"]), 60)
        self.assertGreater(abs(back["yaw"]), 150)

    def test_valid_angles_under_10ms(self):
        est = pose.PoseEstimator()
        img = rect_image(50, 120, angle=-8, size=(640, 480))
        box = receiver.center_crop_box(640, 480)
        sims = {"front": 0.8, "back": 0.2, "side": 0.4, "top": 0.1}
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
    def test_offline_procedural_models_are_valid_glb(self):
        for raw in ("Samsung Galaxy S24 Ultra", "iPhone 15 Pro", "Coca Cola Original Taste Can 330ml", "Pepsi Zero Can"):
            meta = taxonomy.parse_metadata({"raw_name": raw})
            label = taxonomy.label_from_metadata(meta)
            with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mcp_server, "MODELS_DIR", Path(tmp)):
                info = cad_retriever.retrieve(label, meta, offline=True)
                self.assertTrue(mcp_server.is_valid_glb(Path(tmp) / f"{mcp_server.slugify(label)}.glb"), raw)
                self.assertEqual(info["source"], "procedural")
                self.assertEqual(cad_retriever.retrieve(label, meta, offline=True)["source"], "local-cache")


class LedMappingTests(unittest.TestCase):
    def test_confidence_to_led_params(self):
        self.assertEqual(receiver.led_params(receiver.IDENTIFIED, 1.0), (0, 255, 0, 50, 8))
        self.assertEqual(receiver.led_params(receiver.IDENTIFIED, 0.5), (0, 255, 0, 50, 4))
        self.assertEqual(receiver.led_params(receiver.GUIDING, 0.6), (255, 120, 0, 40, 4))
        self.assertEqual(receiver.led_params(receiver.SEARCHING, 0.0)[4], 1)

    def test_phone_hints(self):
        self.assertEqual(receiver.phone_hint("smartphone", "front", 0.05), "flip")
        self.assertEqual(receiver.phone_hint("smartphone", "back", 0.05), "tilt")
        self.assertIsNone(receiver.phone_hint("smartphone", "front", 0.5))
        self.assertIsNone(receiver.phone_hint("beverage_can", "front", 0.0))
        self.assertEqual(receiver.guidance_message(receiver.GUIDING, "x", 0.5, "flip"), taxonomy.MSG_PHONE_FLIP)
        self.assertEqual(receiver.guidance_message(receiver.GUIDING, "x", 0.5, "tilt"), taxonomy.MSG_PHONE_TILT)

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
            if i < 3:
                self.assertEqual(st["phase"], "ready")
                self.assertEqual(st["views"][i]["state"], "done")
        self.assertEqual(st["phase"], "done", st)
        label = "coca_cola_330ml_can"
        entry = self.bank.info()[0]
        self.assertEqual(entry["label"], label)
        self.assertEqual(set(entry["views"]), set(taxonomy.VIEWS))
        self.assertEqual(entry["metadata"]["category"], "beverage_can")
        sims = self.bank.view_sims(label, base["side"])
        self.assertEqual(max(sims, key=sims.get), "side")
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


if __name__ == "__main__":
    unittest.main()
