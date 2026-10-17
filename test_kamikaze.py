import io
import json
import os
import threading
import time
import unittest
from http.client import HTTPConnection
from unittest import mock

import msgpack
from PIL import Image, ImageDraw

import mcp_server
import receiver


def can_image(color, size=224):
    img = Image.new("RGB", (size, size), (200, 200, 200))
    ImageDraw.Draw(img).rounded_rectangle((80, 30, 150, 200), radius=12, fill=color)
    return img


class FallbackTests(unittest.TestCase):
    def test_every_target_has_cached_model(self):
        for label in mcp_server.TARGET_LABELS:
            entry = mcp_server.resolve_fallback(label)
            self.assertIsNotNone(entry, label)
            self.assertTrue(entry["model_url"].endswith(".glb"), label)

    def test_variant_kept_in_title(self):
        self.assertEqual(mcp_server.resolve_fallback("iPhone 15 Pro Max")["title"], "Apple iPhone 15 Pro Max")
        self.assertEqual(mcp_server.resolve_fallback("Samsung Galaxy S24 Ultra")["title"], "Samsung Galaxy S24 Ultra")

    def test_search_uses_cache_without_credentials(self):
        env = {k: v for k, v in os.environ.items() if k not in ("SKETCHFAB_TOKEN", "POLY_PIZZA_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            res = mcp_server.search_3d_asset("Sprite")
        self.assertEqual(res["status"], "cached")
        self.assertEqual(set(res), {"title", "model_url", "thumbnail_url", "status"})

    def test_online_result_marked_online(self):
        fake = {"title": "T", "model_url": "u.glb", "thumbnail_url": "t.jpg"}
        with mock.patch.object(mcp_server, "_search_online", return_value=fake):
            self.assertEqual(mcp_server.search_3d_asset("Pepsi")["status"], "online")


class RpcTests(unittest.TestCase):
    def test_payload_and_failure(self):
        sent = []

        class FakeSock:
            def __init__(self):
                self.unpacker = None

            def sendall(self, data):
                sent.append(msgpack.unpackb(data, raw=False))

            def recv(self, n):
                return msgpack.packb([1, sent[-1][1], None, None])

            def close(self):
                pass

        client = mcp_server._RouterClient("/nonexistent")
        client._sock = FakeSock()
        with mock.patch.object(mcp_server, "_router", client):
            self.assertTrue(mcp_server.actuate_modulino(0, 255, 0, 50, 8))
            self.assertEqual(sent[0][0], 0)
            self.assertEqual(sent[0][2:], ["set_pixels", [0, 255, 0, 50, 8]])
            result = mcp_server.trigger_guidance("test", "Rotate")
            self.assertEqual(result, {"action": "reorient", "message": "Rotate"})
            self.assertEqual(sent[-1][3], [255, 120, 0, 40, 4])

        dead = mcp_server._RouterClient("/nonexistent/sock")
        with mock.patch.object(mcp_server, "_router", dead):
            self.assertFalse(mcp_server.actuate_modulino(1, 2, 3, 4, 5))


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.clf = receiver.ColorShapeClassifier()

    def test_cans(self):
        for color, label in (((200, 20, 20), "Coca-Cola Classic"), ((20, 60, 200), "Pepsi"), ((20, 170, 60), "Sprite")):
            det = self.clf.classify(can_image(color))
            self.assertEqual(det.label, label)
            self.assertGreaterEqual(det.confidence, receiver.CONF_HIGH)

    def test_blank_frame_is_not_a_target(self):
        det = self.clf.classify(Image.new("RGB", (224, 224), (128, 128, 128)))
        self.assertIsNone(det.label)
        self.assertLess(det.confidence, receiver.CONF_HIGH)

    def test_red_wall_is_not_confident(self):
        det = self.clf.classify(Image.new("RGB", (224, 224), (200, 20, 20)))
        self.assertLess(det.confidence, receiver.CONF_HIGH)

    def test_label_matching(self):
        self.assertEqual(receiver.match_label("It's an iPhone 15 Pro Max."), "Apple iPhone 15 Pro Max")
        self.assertEqual(receiver.match_label("galaxy s24+"), "Samsung Galaxy S24 Plus")
        self.assertEqual(receiver.match_label("Coke Zero"), "Coke Zero")
        self.assertIsNone(receiver.match_label("a banana"))

    def test_vlm_parsing(self):
        det = receiver.parse_vlm_output('noise {"label": "iPhone 14 Pro", "confidence": 0.9} tail')
        self.assertEqual((det.label, det.confidence), ("Apple iPhone 14 Pro", 0.9))
        self.assertIsNone(receiver.parse_vlm_output('{"label": "none", "confidence": 0.1}').label)


class OrchestratorTests(unittest.TestCase):
    def run_flow(self, detections):
        calls = {"leds": [], "search": [], "guidance": []}
        orch = receiver.Orchestrator()
        with mock.patch.object(receiver, "actuate_modulino", lambda *a: calls["leds"].append(a) or True), \
             mock.patch.object(receiver, "search_3d_asset", lambda q: calls["search"].append(q) or {"model_url": "x.glb"}), \
             mock.patch.object(receiver, "trigger_guidance",
                               lambda r, m: calls["guidance"].append((r, m)) or {"action": "reorient", "message": m}), \
             mock.patch("builtins.print"):
            for det in detections:
                orch.process(det)
            time.sleep(0.3)
        return orch, calls

    def test_lock_after_confirmation(self):
        det = receiver.Detection("Pepsi", "can", 0.95, "test")
        orch, calls = self.run_flow([det] * 5)
        self.assertEqual(calls["leds"], [(0, 255, 0, 50, 8)])
        self.assertEqual(calls["search"], ["Pepsi"])
        self.assertEqual(orch.latest_asset["asset"]["model_url"], "x.glb")

    def test_guidance_on_ambiguous_phone(self):
        det = receiver.Detection(None, "phone", 0.5, "test")
        orch, calls = self.run_flow([det] * 6)
        self.assertEqual(len(calls["guidance"]), 1)
        self.assertIn("camera island", calls["guidance"][0][1])
        self.assertEqual(calls["search"], [])

    def test_no_target_guidance(self):
        orch, calls = self.run_flow([receiver.NO_DETECTION] * 6)
        self.assertEqual(calls["guidance"][0][0], "no_target")


class ServerTests(unittest.TestCase):
    def test_http_roundtrip(self):
        server = receiver.ThreadingHTTPServer(("127.0.0.1", 0), receiver.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            buf = io.BytesIO()
            can_image((200, 20, 20)).save(buf, "JPEG")
            conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            with mock.patch.object(receiver, "orchestrator", receiver.Orchestrator()), \
                 mock.patch.object(receiver, "actuate_modulino", lambda *a: True), \
                 mock.patch.object(receiver, "search_3d_asset", lambda q: {"model_url": "x.glb"}), \
                 mock.patch("builtins.print"):
                for _ in range(4):
                    conn.request("POST", "/", body=buf.getvalue(), headers={"Content-Type": "image/jpeg"})
                    resp = conn.getresponse()
                    body = json.loads(resp.read())
                    self.assertEqual(resp.status, 200)
                self.assertEqual(body["state"], "locked")
                time.sleep(0.3)
                conn.request("GET", "/asset")
                resp = conn.getresponse()
                self.assertEqual(json.loads(resp.read())["label"], "Coca-Cola Classic")
                conn.request("POST", "/", body=b"junk")
                resp = conn.getresponse()
                resp.read()
                self.assertEqual(resp.status, 400)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
