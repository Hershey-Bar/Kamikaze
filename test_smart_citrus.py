"""SMART kiosk scan, provenance badges and the lime-vs-lemon fix (no network, no camera, no board)."""

import json
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

import numpy as np

import cad_harvester
import color_sig
import mcp_server
import ocr_fusion
import receiver
import smart_scan
from test_comprehensive import FakeExtractor


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class SmartScanTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.scan = smart_scan.SmartScan(window_s=5.0, detect_min=0.7, lock_conf=0.5, min_share=0.6, clock=self.clock)

    def feed(self, seconds, label="kirkland_lime_can", conf=0.8, det=0.9, step=0.1, tokens=()):
        events = []
        for _ in range(int(round(seconds / step))):
            self.clock.t += step
            ev = self.scan.update(det_score=det, label=label, confidence=conf, category="beverage_can", tokens=tokens)
            if ev:
                events.append(ev)
        return events

    def test_idle_until_strong_detection(self):
        self.assertIsNone(self.scan.update(det_score=0.5, label="x", confidence=0.9))
        self.assertEqual(self.scan.status, smart_scan.IDLE)
        self.assertIsNone(self.scan.led())
        self.assertEqual(self.scan.update(det_score=0.75, label="x", confidence=0.9), "started")
        self.assertEqual(self.scan.status, smart_scan.SCANNING)
        self.assertEqual(self.scan.scans, 1)

    def test_locks_after_window_with_amber_progress_then_green(self):
        self.scan.update(det_score=0.9, label="kirkland_lime_can", confidence=0.8)
        self.feed(2.5, tokens=("kirkland", "lime"))
        r, g, b, bri, n = self.scan.led()
        self.assertEqual((r, g, b, bri), (255, 120, 0, 40))
        self.assertIn(n, (4, 5))
        snap = self.scan.snapshot()
        self.assertEqual(snap["status"], "scanning")
        self.assertAlmostEqual(snap["progress"], 0.5, delta=0.03)
        self.assertEqual(self.feed(3.0), ["locked"])
        self.assertEqual(self.scan.status, smart_scan.LOCKED)
        self.assertEqual(self.scan.label, "kirkland_lime_can")
        self.assertEqual(self.scan.led(), (0, 255, 0, 50, 8))
        snap = self.scan.snapshot()
        self.assertEqual((snap["progress"], snap["category"]), (1.0, "beverage_can"))
        self.assertIn("lime", snap["tokens"])
        self.assertEqual(snap["votes"][0], {"label": "kirkland_lime_can", "share": 1.0})

    def test_split_votes_are_inconclusive_then_rearm(self):
        self.scan.update(det_score=0.9, label="a", confidence=0.8)
        for i in range(55):
            self.clock.t += 0.1
            self.scan.update(det_score=0.9, label="a" if i % 2 else "b", confidence=0.8)
        self.assertEqual(self.scan.status, smart_scan.INCONCLUSIVE)
        self.assertEqual(self.scan.led(), (255, 120, 0, 40, 4))
        self.assertIn("inconclusive", self.scan.message)
        self.clock.t += smart_scan.HOLD_S + 0.1
        self.scan.update(det_score=0.9, label="a", confidence=0.8)
        self.assertEqual(self.scan.status, smart_scan.IDLE)
        self.assertEqual(self.scan.update(det_score=0.9, label="a", confidence=0.8), "started")

    def test_low_confidence_never_locks(self):
        self.scan.update(det_score=0.9, label="a", confidence=0.3)
        self.feed(5.5, label="a", conf=0.3)
        self.assertEqual(self.scan.status, smart_scan.INCONCLUSIVE)

    def test_object_lost_cancels_scan(self):
        self.scan.update(det_score=0.9, label="a", confidence=0.8)
        self.feed(1.0, label="a")
        self.clock.t += smart_scan.LOST_S + 0.05
        self.assertEqual(self.scan.update(det_score=None, label=None, confidence=0.0), "lost")
        self.assertEqual(self.scan.status, smart_scan.IDLE)
        self.assertIn("left the reticle", self.scan.message)

    def test_locked_released_when_absent_or_swapped(self):
        self.scan.update(det_score=0.9, label="a", confidence=0.8)
        self.feed(5.2, label="a")
        self.assertEqual(self.scan.status, smart_scan.LOCKED)
        self.clock.t += smart_scan.RELEASE_S + 0.1
        self.assertEqual(self.scan.update(det_score=None, label=None, confidence=0.0), "released")
        self.scan.update(det_score=0.9, label="a", confidence=0.8)
        self.feed(5.2, label="a")
        self.assertEqual(self.scan.status, smart_scan.LOCKED)
        events = self.feed(0.1 * smart_scan.SWAP_FRAMES, label="b", conf=0.9)
        self.assertEqual(events, ["released"])


class ProvenanceTests(unittest.TestCase):
    def test_tiers(self):
        P = cad_harvester.provenance
        self.assertEqual(P({"source_key": "library"})["type"], "verified")
        self.assertIn("Hand-scanned", P({"source_key": "library"})["text"])
        self.assertEqual(P({"source_key": "upload"})["type"], "verified")
        self.assertEqual(P({"match": "library", "source_key": "local"})["type"], "verified")
        cad = P({"source_key": "printables", "author": "jdoe"})
        self.assertEqual(cad, {"type": "cad", "text": "CAD 1:1 \u00b7 jdoe (Printables)"})
        self.assertEqual(P({"source_key": "packaging"})["type"], "packaging")
        self.assertEqual(P({"source_key": "label_scan"})["type"], "scan")
        self.assertEqual(P({"source_key": "sketchfab"})["type"], "online")
        self.assertEqual(P({"source_key": "spec_generated"})["type"], "estimated")
        self.assertEqual(P(None), {"type": "", "text": ""})
        # only spec-generated geometry is ever called ESTIMATED
        self.assertEqual([k for k, (t, _) in cad_harvester.PROVENANCE.items() if t == "estimated"], ["spec_generated"])


def solid(rgb, size=64):
    return np.full((size, size, 3), rgb, np.uint8)


class CitrusTests(unittest.TestCase):
    LEMON, LIME, GRAPEFRUIT = (230, 200, 40), (120, 200, 40), (230, 80, 110)

    def test_hue_verdict(self):
        self.assertEqual(color_sig.citrus_verdict(color_sig.hue_stats(solid(self.LEMON))), "lemon")
        self.assertEqual(color_sig.citrus_verdict(color_sig.hue_stats(solid(self.LIME))), "lime")
        self.assertIsNone(color_sig.citrus_verdict(color_sig.hue_stats(solid(self.GRAPEFRUIT))))
        silver = color_sig.hue_stats(solid((200, 200, 205)))
        self.assertIsNone(silver["median_hue"])
        self.assertIsNone(color_sig.citrus_verdict(silver))
        self.assertIsNone(color_sig.citrus_verdict(None))

    def test_mostly_silver_can_with_lime_print(self):
        img = solid((205, 205, 210))
        img[20:40, 10:54] = self.LIME  # ~21% coloured print on a silver can
        self.assertEqual(color_sig.citrus_verdict(color_sig.hue_stats(img)), "lime")

    def flavor(self, label):
        return ocr_fusion.citrus_flavor({}, label)

    def test_flavor_and_conflict(self):
        self.assertEqual(self.flavor("kirkland_lime_can"), "lime")
        self.assertEqual(self.flavor("kirkland_lemon_can"), "lemon")
        self.assertIsNone(self.flavor("sprite_lemon_lime_can"))
        self.assertIsNone(self.flavor("kirkland_grapefruit_can"))
        self.assertTrue(ocr_fusion.citrus_conflict("lime", {"kirkland", "lemon"}))
        self.assertFalse(ocr_fusion.citrus_conflict("lime", {"lime", "lemon"}))
        self.assertFalse(ocr_fusion.citrus_conflict(None, {"lemon"}))

    def test_corrector_never_fuzzes_into_citrus(self):
        c = ocr_fusion.Corrector(["kirkland lime", "kirkland lemon"])
        self.assertEqual(c.word("limon"), "limon")
        self.assertEqual(c.word("lemmon"), "lemmon")
        self.assertEqual(c.word("lime"), "lime")

    def test_lexicon_excludes_other_citrus(self):
        meta = {"raw_name": "Kirkland Sparkling Water Lime", "brand": "Kirkland", "flavor": "lime"}
        lex = ocr_fusion.build_lexicon(meta, "kirkland_lime_can", {"lemon": 9, "sparkling": 9}, min_runs=2)
        self.assertIn("lime", lex)
        self.assertIn("sparkling", lex)
        self.assertNotIn("lemon", lex)

    def tiebreak(self, hue_rgb, lexicon=None, lemon_first=True, gap=0.05):
        ranked = [("kirkland_lemon_can", 0.80, "front"), ("kirkland_lime_can", 0.78, "front")]
        if not lemon_first:
            ranked.reverse()
        fused = {ranked[0][0]: 0.80, ranked[1][0]: 0.80 - gap}
        hue = color_sig.hue_stats(solid(hue_rgb)) if hue_rgb else None
        info = receiver.citrus_tiebreak(ranked, fused, self.flavor, hue, lexicon)
        return ranked, fused, info

    def test_tiebreak_swaps_lemon_to_lime_on_green_print(self):
        ranked, fused, info = self.tiebreak(self.LIME)
        self.assertTrue(info["swapped"])
        self.assertEqual(ranked[0][0], "kirkland_lime_can")
        self.assertEqual(info["winner"], "kirkland_lime_can")
        self.assertGreaterEqual(fused["kirkland_lime_can"] - fused["kirkland_lemon_can"], receiver.MIN_MARGIN)

    def test_tiebreak_keeps_lemon_on_yellow_print(self):
        ranked, fused, info = self.tiebreak(self.LEMON)
        self.assertFalse(info["swapped"])
        self.assertEqual(ranked[0][0], "kirkland_lemon_can")
        self.assertGreaterEqual(fused["kirkland_lemon_can"] - fused["kirkland_lime_can"], receiver.MIN_MARGIN)

    def test_tiebreak_does_not_apply(self):
        self.assertIsNone(self.tiebreak(self.LIME, lexicon="kirkland_lemon_can")[2])  # OCR already decided
        self.assertIsNone(self.tiebreak(self.LIME, gap=0.4)[2])  # vision is confident enough
        ranked, _, info = self.tiebreak(self.GRAPEFRUIT)  # no colour evidence: leave the ranking alone
        self.assertIsNone(info["verdict"])
        self.assertEqual(ranked[0][0], "kirkland_lemon_can")
        ranked = [("kirkland_grapefruit_can", 0.8, "front"), ("kirkland_lime_can", 0.79, "front")]
        self.assertIsNone(receiver.citrus_tiebreak(ranked, {}, self.flavor, None, None))


class ViewEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(mcp_server, "MODELS_DIR", Path(self.tmp.name)),
                        mock.patch("builtins.print"), mock.patch.object(receiver, "LOG_FRAMES", False)]
        for p in self.patches:
            p.start()
        self.pipe = receiver.Pipeline(FakeExtractor(), receiver.EmbeddingBank(None), receiver.LedDriver(lambda *a: True),
                                      receiver.AssetManager(offline=True, source="auto"))
        self.server = receiver.make_server(self.pipe, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.conn = HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=15)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.conn.close()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()
        receiver.netguard.set_offline(False)

    def call(self, method, path, body=None):
        self.conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json"})
        resp = self.conn.getresponse()
        return resp.status, json.loads(resp.read())

    def test_view_toggle(self):
        status, out = self.call("GET", "/api/settings/view")
        self.assertEqual((status, out["view"], out["views"]), (200, receiver.VIEW, ["smart", "debug"]))
        status, out = self.call("POST", "/api/settings/view", {"view": "debug"})
        self.assertEqual((status, out["view"]), (200, "debug"))
        self.assertEqual(self.pipe.view, "debug")
        self.assertEqual(self.call("POST", "/api/settings/view", {"view": "cinema"})[0], 400)
        self.assertEqual(self.call("POST", "/api/settings/view", {"view": "SMART"})[1]["view"], "smart")

    def test_scan_and_state_fields(self):
        status, out = self.call("GET", "/api/scan")
        self.assertEqual((status, out["status"], out["progress"]), (200, "idle", 0.0))
        self.assertIsNone(out["auto_label_scan"])
        status, st = self.call("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(st["smart_scan"]["status"], "idle")
        self.assertEqual(st["asset_provenance"], {"type": "", "text": ""})
        self.assertIn(st["view"], receiver.VIEWS)
        self.assertFalse(st["label_scan_auto"])

    def test_debug_view_never_runs_smart_scan(self):
        self.pipe.set_view("debug")
        self.pipe.smart.status = smart_scan.SCANNING
        frame = mock.Mock(label="a", c1=0.9)
        self.assertIsNone(self.pipe._smart_step(frame, "beverage_can", None, {}, {}, None, False))
        self.assertEqual(self.pipe.smart.status, smart_scan.IDLE)

    def test_auto_label_scan_skips_labelled_models(self):
        self.pipe.assets.offline = False
        with mock.patch.object(self.pipe, "_class_category", return_value="beverage_can"), \
                mock.patch.object(self.pipe.assets, "peek", return_value={"status": "ready", "source_key": "library"}), \
                mock.patch.object(threading, "Thread") as thread:
            self.pipe._maybe_auto_label_scan("kirkland_lime_can")
            thread.assert_not_called()
            self.pipe._maybe_auto_label_scan("kirkland_lime_can")  # tried once per label
        self.assertIn("kirkland_lime_can", self.pipe._auto_scan_tried)
        with mock.patch.object(self.pipe, "_class_category", return_value="smartphone"), \
                mock.patch.object(threading, "Thread") as thread:
            self.pipe._maybe_auto_label_scan("iphone_15_pro")
            thread.assert_not_called()

    def test_labelled_model_waits_for_resolution(self):
        seq = iter([{"status": "pending"}, {"status": "resolving"}, {"status": "ready", "source_key": "library"}])
        with mock.patch.object(self.pipe.assets, "peek", side_effect=lambda _l: next(seq)), \
                mock.patch.object(self.pipe.assets, "get") as get:
            self.assertTrue(self.pipe._labelled_model("kirkland_lime_can", wait_s=2.0))
            get.assert_called_once_with("kirkland_lime_can")
        with mock.patch.object(self.pipe.assets, "peek", return_value={"status": "ready", "source_key": "sketchfab"}):
            self.assertFalse(self.pipe._labelled_model("coke_can"))


class SmartPipelineTests(unittest.TestCase):
    """End to end over HTTP: frames of an enrolled can -> scan (amber progress) -> lock (green x8) -> released."""

    def setUp(self):
        import test_kamikaze as tk
        self.tk = tk
        self.ext = tk.FakeExtractor()
        self.bank = receiver.EmbeddingBank(None)
        self.leds_log = []
        self.pipe = receiver.Pipeline(self.ext, self.bank, receiver.LedDriver(lambda *p: self.leds_log.append(p) or True),
                                      tk.FakeAssets())
        self.pipe.view = "smart"
        self.pipe.smart = smart_scan.SmartScan(window_s=0.6)
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

    def call(self, method, path, body=None, ctype="application/json"):
        self.conn.request(method, path, body=body, headers={"Content-Type": ctype})
        resp = self.conn.getresponse()
        return resp.status, resp.read()

    def frame(self, vec, textured=True):
        self.ext.vec = vec
        status, data = self.call("POST", "/api/frame", self.tk.jpeg(textured=textured), "image/jpeg")
        self.assertEqual(status, 200, data)
        return json.loads(data)

    def test_scan_locks_then_releases(self):
        a = self.tk.unit(1)
        self.call("POST", "/api/enroll", json.dumps({"label": "coke_can", "frames": 6}).encode())
        for _ in range(5):
            self.frame(a)
        self.assertEqual(self.pipe.smart.status, smart_scan.IDLE)  # enrollment frames never arm the scan
        self.frame(a)
        self.frame(a)
        self.assertEqual(self.pipe.smart.status, smart_scan.SCANNING)
        time.sleep(0.3)
        self.assertEqual(self.leds_log[-1][:4], (255, 120, 0, 40))
        scan = json.loads(self.call("GET", "/api/scan")[1])
        self.assertEqual(scan["status"], "scanning")
        deadline = time.monotonic() + 5
        while self.pipe.smart.status == smart_scan.SCANNING and time.monotonic() < deadline:
            self.frame(a)
            time.sleep(0.05)
        self.assertEqual(self.pipe.smart.status, smart_scan.LOCKED)
        time.sleep(0.3)
        self.assertEqual(self.leds_log[-1], (0, 255, 0, 50, 8))
        st = json.loads(self.call("GET", "/api/state")[1])
        self.assertEqual((st["smart_scan"]["status"], st["smart_scan"]["label"], st["viewer_label"]),
                         ("locked", "coke_can", "coke_can"))
        self.assertTrue(st["smart_scan"]["display_name"])
        with mock.patch.object(smart_scan, "RELEASE_S", 0.2):
            deadline = time.monotonic() + 3
            while self.pipe.smart.status == smart_scan.LOCKED and time.monotonic() < deadline:
                self.frame(a, textured=False)  # empty reticle
                time.sleep(0.05)
        self.assertEqual(self.pipe.smart.status, smart_scan.IDLE)


if __name__ == "__main__":
    unittest.main()
