import io
import json
import os
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

import msgpack
import numpy as np
from PIL import Image

import mcp_server
import receiver

try:
    import torch
except ImportError:
    torch = None

needs_torch = unittest.skipIf(torch is None, "PyTorch not installed")


def jpeg(color=(120, 80, 40), size=(320, 240)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def unit(seed, dim=16):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(dim, generator=g), dim=0)


class FakeRouterSock:
    def __init__(self, sent):
        self.sent = sent

    def sendall(self, data):
        self.sent.append(msgpack.unpackb(data, raw=False))

    def recv(self, n):
        return msgpack.packb([1, self.sent[-1][1], None, None])

    def close(self):
        pass


class FallbackTests(unittest.TestCase):
    def test_every_target_has_cached_model(self):
        for label in mcp_server.TARGET_LABELS:
            entry = mcp_server.resolve_fallback(label)
            self.assertIsNotNone(entry, label)
            self.assertTrue(entry["model_url"].endswith(".glb"), label)

    def test_variant_kept_in_title(self):
        self.assertEqual(mcp_server.resolve_fallback("iPhone 15 Pro Max")["title"], "Apple iPhone 15 Pro Max")
        self.assertEqual(mcp_server.resolve_fallback("Samsung Galaxy S24 Ultra")["title"], "Samsung Galaxy S24 Ultra")

    def test_enrolled_class_labels_resolve(self):
        for label in ("coke_can", "pepsi_can", "s25_ultra"):
            entry = mcp_server.resolve_fallback(label)
            self.assertIsNotNone(entry, label)
            self.assertTrue(entry["model_url"].endswith(".glb"), label)
        self.assertEqual(mcp_server.resolve_fallback("s25_ultra")["title"], "Samsung Galaxy S25 Ultra")

    def test_lookup_uses_cache_without_credentials(self):
        env = {k: v for k, v in os.environ.items() if k not in ("SKETCHFAB_TOKEN", "POLY_PIZZA_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            res = mcp_server.lookup_3d_asset("Sprite")
        self.assertEqual(res["status"], "cached")
        self.assertEqual(set(res), {"title", "model_url", "thumbnail_url", "status"})

    def test_online_result_marked_online(self):
        fake = {"title": "T", "model_url": "u.glb", "thumbnail_url": "t.jpg"}
        with mock.patch.object(mcp_server, "_search_online", return_value=fake):
            self.assertEqual(mcp_server.lookup_3d_asset("Pepsi")["status"], "online")


class AssetCacheTests(unittest.TestCase):
    GLB = b"glTF" + b"\x02\x00\x00\x00" + b"\x00" * 64

    class FakeResp:
        def __init__(self, data):
            self.buf = io.BytesIO(data)

        def read(self, n=-1):
            return self.buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_search_downloads_and_caches_glb(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mcp_server, "MODELS_DIR", Path(tmp)), \
                mock.patch.object(mcp_server, "_search_online", return_value=None), \
                mock.patch.object(mcp_server.urllib.request, "urlopen", lambda *a, **k: self.FakeResp(self.GLB)) as _:
            res = mcp_server.search_3d_asset("coke_can")
            self.assertEqual(res["glb_url"], "/static/models/coke_can.glb")
            self.assertTrue((Path(tmp) / "coke_can.glb").is_file())
            self.assertEqual(set(res) & {"title", "model_url", "thumbnail_url", "status"}, {"title", "model_url", "thumbnail_url", "status"})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mcp_server, "MODELS_DIR", Path(tmp)), \
                mock.patch.object(mcp_server, "_search_online", return_value=None), \
                mock.patch.object(mcp_server.urllib.request, "urlopen", lambda *a, **k: self.FakeResp(b"<html>nope</html>")):
            res = mcp_server.search_3d_asset("coke_can")
            self.assertNotIn("glb_url", res)
            self.assertFalse((Path(tmp) / "coke_can.glb").exists())

    def test_slugify(self):
        self.assertEqual(mcp_server.slugify("Apple iPhone 15 Pro!"), "apple_iphone_15_pro")
        self.assertEqual(mcp_server.slugify("../../etc"), "etc")


class RpcTests(unittest.TestCase):
    def test_payload_and_failure(self):
        sent = []
        client = mcp_server._RouterClient("/nonexistent")
        client._sock = FakeRouterSock(sent)
        with mock.patch.object(mcp_server, "_router", client):
            res = mcp_server.actuate_modulino(0, 255, 0, 50, 8)
            self.assertEqual(res, {"success": True, "r": 0, "g": 255, "b": 0, "brightness": 50, "count": 8})
            self.assertEqual(sent[0][0], 0)
            self.assertEqual(sent[0][2:], ["set_pixels", [0, 255, 0, 50, 8]])
            self.assertEqual(mcp_server.trigger_guidance("test", "Rotate"), {"action": "reorient", "message": "Rotate"})
            self.assertEqual(sent[-1][3], [255, 120, 0, 40, 4])
            mcp_server.actuate_modulino(999, -5, 0, 500, 99)
            self.assertEqual(sent[-1][3], [255, 0, 0, 100, 8])

        dead = mcp_server._RouterClient("/nonexistent/sock")
        with mock.patch.object(mcp_server, "_router", dead):
            self.assertFalse(mcp_server.actuate_modulino(1, 2, 3, 4, 5)["success"])
            self.assertFalse(mcp_server.set_leds(1, 2, 3, 4, 5))


class McpReceiverToolTests(unittest.TestCase):
    def test_unreachable_receiver_is_reported(self):
        with mock.patch.object(mcp_server, "RECEIVER_URL", "http://127.0.0.1:9"):
            res = mcp_server.enroll_object("coke_can")
            self.assertFalse(res["ok"])
            self.assertIn("unreachable", res["error"])
            self.assertFalse(mcp_server.get_guidance().get("ok", True))


class StateMachineTests(unittest.TestCase):
    def feed(self, sm, frames):
        out = None
        for f in frames:
            out = sm.update(f)
        return out

    def ident(self, label="coke_can", s1=0.9, s2=0.1):
        return receiver.FrameScore(label, s1, s2, "other")

    def test_identified_needs_three_consecutive_frames(self):
        sm = receiver.StateMachine()
        self.assertEqual(self.feed(sm, [self.ident()] * 2)["state"], receiver.SEARCHING)
        out = self.feed(sm, [self.ident()])
        self.assertEqual(out["state"], receiver.IDENTIFIED)
        self.assertEqual(out["label"], "coke_can")
        self.assertAlmostEqual(out["score"], 0.9)

    def test_small_margin_never_identifies_and_guides_after_four(self):
        sm = receiver.StateMachine()
        amb = receiver.FrameScore("coke_can", 0.85, 0.80, "pepsi_can")
        self.assertEqual(self.feed(sm, [amb] * 3)["state"], receiver.SEARCHING)
        self.assertEqual(self.feed(sm, [amb])["state"], receiver.GUIDING)

    def test_mid_similarity_guides(self):
        sm = receiver.StateMachine()
        out = self.feed(sm, [receiver.FrameScore("s25_ultra", 0.55, 0.0)] * 4)
        self.assertEqual(out["state"], receiver.GUIDING)

    def test_low_similarity_searches(self):
        sm = receiver.StateMachine()
        self.feed(sm, [self.ident()] * 3)
        out = self.feed(sm, [receiver.FrameScore("coke_can", 0.2, 0.0)] * 3)
        self.assertEqual(out["state"], receiver.SEARCHING)
        self.assertIsNone(out["label"])

    def test_single_dropout_does_not_break_identified(self):
        sm = receiver.StateMachine()
        self.feed(sm, [self.ident()] * 3)
        out = sm.update(receiver.FrameScore("coke_can", 0.2, 0.0))
        self.assertEqual(out["state"], receiver.IDENTIFIED)

    def test_label_switch_requires_streak(self):
        sm = receiver.StateMachine()
        self.feed(sm, [self.ident("coke_can")] * 3)
        out = self.feed(sm, [self.ident("pepsi_can")] * 2)
        self.assertEqual(out["label"], "coke_can")
        out = self.feed(sm, [self.ident("pepsi_can")])
        self.assertEqual(out["label"], "pepsi_can")

    def test_empty_bank_is_searching(self):
        sm = receiver.StateMachine()
        self.assertEqual(self.feed(sm, [receiver.FrameScore(None, 0.0)] * 5)["state"], receiver.SEARCHING)


class GuidanceTests(unittest.TestCase):
    def test_messages(self):
        self.assertEqual(receiver.guidance_message(receiver.SEARCHING, None, 0), "Place target object inside the central alignment box.")
        self.assertEqual(receiver.guidance_message(receiver.IDENTIFIED, "coke_can", 0.912), "Object Verified: coke_can (91%)")
        self.assertEqual(receiver.guidance_message(receiver.GUIDING, "pepsi_can", 0.5), "Rotate can 90° slowly to align the front branding with the camera.")
        self.assertEqual(receiver.guidance_message(receiver.GUIDING, "s25_ultra", 0.5), "Flip phone over to reveal the rear camera island and lenses.")
        self.assertEqual(receiver.guidance_message(receiver.GUIDING, "mug", 0.5), "Center the item within the reticle and adjust distance for optimal lighting.")

    def test_led_params(self):
        self.assertEqual(receiver.led_params(receiver.IDENTIFIED, 0.93), (0, 255, 0, 50, 7))
        self.assertEqual(receiver.led_params(receiver.IDENTIFIED, 1.0), (0, 255, 0, 50, 8))
        self.assertEqual(receiver.led_params(receiver.IDENTIFIED, 0.05), (0, 255, 0, 50, 1))
        self.assertEqual(receiver.led_params(receiver.GUIDING, 0.5), (255, 120, 0, 40, 4))
        self.assertEqual(receiver.led_params(receiver.SEARCHING, 0.0), (0, 100, 150, 15, 1))

    def test_label_normalisation(self):
        self.assertEqual(receiver.normalize_label(" Coke-Can "), "coke_can")
        for bad in ("", "../x", "a" * 41, "-x"):
            with self.assertRaises(ValueError):
                receiver.normalize_label(bad)


class PreprocessTests(unittest.TestCase):
    def test_center_crop_is_sixty_percent(self):
        self.assertEqual(receiver.center_crop_box(640, 480), (288, 176, 96))

    def test_shape_dtype_contiguity(self):
        arr = receiver.preprocess(Image.new("RGB", (640, 480), (255, 255, 255)))
        self.assertEqual(arr.shape, (1, 3, 224, 224))
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(arr.flags["C_CONTIGUOUS"])
        self.assertAlmostEqual(float(arr[0, 0, 0, 0]), (1 - 0.485) / 0.229, places=4)


class LedDriverTests(unittest.TestCase):
    def test_sends_changes_only_and_retries_failures(self):
        calls, ok = [], [False]

        def setter(*p):
            calls.append(p)
            return ok[0]

        drv = receiver.LedDriver(setter)
        drv.set((0, 255, 0, 50, 8))
        time.sleep(0.2)
        self.assertEqual(drv.ok, False)
        ok[0] = True
        time.sleep(1.3)
        self.assertEqual(drv.ok, True)
        n = len(calls)
        drv.set((0, 255, 0, 50, 8))
        time.sleep(0.2)
        self.assertEqual(len(calls), n)
        drv.set((255, 120, 0, 40, 4))
        time.sleep(0.2)
        self.assertEqual(calls[-1], (255, 120, 0, 40, 4))


@needs_torch
class BankTests(unittest.TestCase):
    def test_scores_persistence_and_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.pt"
            bank = receiver.EmbeddingBank(path)
            a, b = unit(1), unit(2)
            info = bank.add("alpha", [a, a])
            self.assertEqual(info["vectors"], 2)
            self.assertAlmostEqual(info["consistency"], 1.0, places=4)
            bank.add("beta", [b])
            ranked = bank.scores(a)
            self.assertEqual(ranked[0][0], "alpha")
            self.assertAlmostEqual(ranked[0][1], 1.0, places=4)
            again = receiver.EmbeddingBank(path)
            self.assertEqual(again.load(), 2)
            self.assertEqual(again.scores(b)[0][0], "beta")
            self.assertTrue(again.remove("beta"))
            self.assertFalse(again.remove("beta"))
            self.assertEqual(receiver.EmbeddingBank(path).load(), 1)

    def test_append_extends_exemplars(self):
        bank = receiver.EmbeddingBank(None)
        bank.add("alpha", [unit(1)])
        self.assertEqual(bank.add("alpha", [unit(2), unit(3)], append=True)["vectors"], 3)
        self.assertEqual(bank.add("alpha", [unit(2)])["vectors"], 1)

    def test_empty_bank_scores(self):
        self.assertEqual(receiver.EmbeddingBank(None).scores(unit(1)), [])


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


@needs_torch
class PipelineHttpTests(unittest.TestCase):
    def setUp(self):
        self.ext = FakeExtractor()
        self.bank = receiver.EmbeddingBank(None)
        self.leds_log = []
        self.leds = receiver.LedDriver(lambda *p: self.leds_log.append(p) or True)
        self.pipe = receiver.Pipeline(self.ext, self.bank, self.leds, FakeAssets())
        self.server = receiver.make_server(self.pipe, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.conn = HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)
        self.patch = mock.patch("builtins.print")
        self.patch.start()
        self.log_patch = mock.patch.object(receiver, "LOG_FRAMES", False)
        self.log_patch.start()

    def tearDown(self):
        self.patch.stop()
        self.log_patch.stop()
        self.server.shutdown()
        self.server.server_close()
        self.conn.close()

    def call(self, method, path, body=None, ctype="application/json"):
        self.conn.request(method, path, body=body, headers={"Content-Type": ctype})
        resp = self.conn.getresponse()
        data = resp.read()
        return resp.status, data

    def frame(self, vec):
        self.ext.vec = vec
        status, data = self.call("POST", "/api/frame", jpeg(), "image/jpeg")
        self.assertEqual(status, 200, data)
        return json.loads(data)

    def enroll(self, label, vec, n=12):
        status, data = self.call("POST", "/api/enroll", json.dumps({"label": label, "frames": n}).encode())
        self.assertEqual(status, 202, data)
        self.assertTrue(json.loads(data)["ok"])
        for _ in range(n):
            self.frame(vec)
        return json.loads(self.call("GET", "/api/state")[1])

    def test_enroll_identify_guidance_and_asset(self):
        a, b = unit(1), unit(2)
        st = self.enroll("coke_can", a)
        self.assertTrue(st["enrollment"]["ok"])
        self.assertEqual(st["classes"], {"coke_can": 12})
        self.enroll("pepsi_can", b)

        for _ in range(3):
            verdict = self.frame(a)
        self.assertEqual((verdict["state"], verdict["label"]), ("IDENTIFIED", "coke_can"))
        time.sleep(0.3)
        self.assertEqual(self.leds_log[-1], (0, 255, 0, 50, 8))
        asset = json.loads(self.call("GET", "/api/asset")[1])
        self.assertEqual(asset["label"], "coke_can")
        self.assertEqual(asset["glb_url"], "/static/models/coke_can.glb")
        self.assertEqual(asset["camera_orbit"], "45deg 75deg 1.5m")
        self.assertEqual(asset["state"], "IDENTIFIED")

        rand = unit(3)
        rand = rand - (rand @ a) * a
        orth = torch.nn.functional.normalize(rand, dim=0)
        mixed = 0.6 * a + 0.8 * orth  # cosine 0.6 to coke_can: between T_SEARCH and T_IDENT
        for _ in range(6):
            verdict = self.frame(mixed)
        self.assertEqual(verdict["state"], "GUIDING")
        time.sleep(0.3)
        self.assertEqual(self.leds_log[-1], (255, 120, 0, 40, 4))
        state = json.loads(self.call("GET", "/api/state")[1])
        self.assertIn("Rotate can", state["guidance"])

        for _ in range(6):
            verdict = self.frame(-a)
        self.assertEqual(verdict["state"], "SEARCHING")

    def test_endpoints_and_validation(self):
        status, data = self.call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"model-viewer", data)
        self.assertEqual(self.call("GET", "/api/frame.jpg")[0], 404)
        self.frame(unit(1))
        status, data = self.call("GET", "/api/frame.jpg")
        self.assertEqual((status, data[:2]), (200, b"\xff\xd8"))
        self.assertEqual(self.call("POST", "/api/frame", b"junk", "image/jpeg")[0], 400)
        self.assertEqual(self.call("POST", "/api/enroll", b"{}")[0], 400)
        self.assertEqual(self.call("POST", "/api/enroll", b'{"label": "../x"}')[0], 400)
        self.assertEqual(self.call("POST", "/api/enroll", b"[]")[0], 400)
        self.assertEqual(self.call("GET", "/static/../receiver.py")[0], 404)
        self.assertEqual(self.call("GET", "/static/%2e%2e/receiver.py")[0], 404)
        self.assertEqual(self.call("GET", "/nope")[0], 404)
        self.assertEqual(self.call("POST", "/api/unenroll", b'{"label": "ghost"}')[0], 404)
        self.assertEqual(json.loads(self.call("GET", "/state")[1])["state"], "SEARCHING")

    def test_blocking_enroll_and_busy(self):
        a = unit(5)
        self.ext.vec = a

        def feeder():
            time.sleep(0.4)
            c = HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)
            for _ in range(3):
                c.request("POST", "/api/frame", body=jpeg(), headers={"Content-Type": "image/jpeg"})
                c.getresponse().read()
            c.close()

        t = threading.Thread(target=feeder)
        t.start()
        status, data = self.call("POST", "/api/enroll", json.dumps({"label": "thing", "frames": 3, "wait": True}).encode())
        t.join()
        body = json.loads(data)
        self.assertEqual(status, 200, body)
        self.assertEqual((body["label"], body["frames"]), ("thing", 3))
        self.assertGreater(body["consistency"], 0.99)

        self.assertEqual(self.call("POST", "/api/enroll", json.dumps({"label": "x", "frames": 5}).encode())[0], 202)
        self.assertEqual(self.call("POST", "/api/enroll", json.dumps({"label": "y", "frames": 5}).encode())[0], 409)

    def test_unenroll_resets_live_state(self):
        a = unit(1)
        self.enroll("coke_can", a)
        for _ in range(3):
            self.frame(a)
        self.assertEqual(json.loads(self.call("GET", "/api/state")[1])["state"], "IDENTIFIED")
        self.assertEqual(self.call("POST", "/api/unenroll", b'{"label": "coke_can"}')[0], 200)
        state = json.loads(self.call("GET", "/api/state")[1])
        self.assertEqual((state["state"], state["label"], state["classes"]), ("SEARCHING", None, {}))
        time.sleep(0.3)
        self.assertEqual(self.leds_log[-1], (0, 100, 150, 15, 1))

    def test_enroll_timeout_reports_error(self):
        status, data = self.call("POST", "/api/enroll", json.dumps({"label": "slow", "frames": 5, "wait": True, "timeout": 0.3}).encode())
        self.assertEqual(status, 504)
        self.assertIn("stream_cam", json.loads(data)["error"])


if __name__ == "__main__":
    unittest.main()
