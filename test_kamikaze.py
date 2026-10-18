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
import onboard_camera
import receiver

try:
    import torch
except ImportError:
    torch = None

needs_torch = unittest.skipIf(torch is None, "PyTorch not installed")


def jpeg(color=(120, 80, 40), size=(320, 240), textured=True):
    """Solid-colour frames read as an empty reticle (scene gate); `textured` adds a checkerboard so the pipeline runs."""
    img = Image.new("RGB", size, color)
    if textured:
        arr = np.asarray(img).copy()
        yy, xx = np.mgrid[0:size[1], 0:size[0]]
        mask = ((yy // 12 + xx // 12) % 2).astype(bool)
        arr[mask] = 255 - arr[mask]
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, "JPEG")
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


class FakeV4L2:
    """Stand-in for cv2.VideoCapture: returns raw MJPG buffers (1xN uint8) or decoded BGR frames, optionally failing."""

    def __init__(self, frame=None, bgr=False, fail=False):
        self.frame, self.bgr, self.fail, self.props, self.released = frame or jpeg(size=(192, 108)), bgr, fail, {}, False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        cv2 = onboard_camera.cv2
        if cv2 is not None and prop == cv2.CAP_PROP_FOURCC:
            return float(int.from_bytes(b"MJPG", "little"))
        return float(self.props.get(prop, 0))

    def read(self):
        time.sleep(0.005)
        if self.fail:
            return False, None
        if self.bgr:
            return True, np.asarray(Image.open(io.BytesIO(self.frame)).convert("RGB"))[:, :, ::-1].copy()
        return True, np.frombuffer(self.frame, np.uint8).reshape(1, -1)

    def release(self):
        self.released = True


def wait_for(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class OnboardCameraTests(unittest.TestCase):
    def test_resolve_device(self):
        self.assertIsNone(onboard_camera.resolve_device("none"))
        self.assertIsNone(onboard_camera.resolve_device(""))
        self.assertEqual(onboard_camera.resolve_device("2"), "/dev/video2")
        self.assertEqual(onboard_camera.resolve_device("/dev/video0"), "/dev/video0")

    def test_find_camera_prefers_usb_capture_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usb = root / "devices" / "platform" / "usb1" / "1-1" / "1-1.0"
            isp = root / "devices" / "platform" / "camss"
            usb.mkdir(parents=True), isp.mkdir(parents=True)
            sysfs = root / "video4linux"
            for name, index, parent in (("video32", "0", isp), ("video1", "1", usb), ("video0", "0", usb)):
                (sysfs / name).mkdir(parents=True)
                (sysfs / name / "index").write_text(index + "\n")
                (sysfs / name / "name").write_text("Brio 101\n")
                try:
                    os.symlink(parent, sysfs / name / "device", target_is_directory=True)
                except OSError as exc:
                    self.skipTest(f"symlinks unavailable: {exc}")
            self.assertEqual(onboard_camera.find_camera(sysfs), "/dev/video0")
            self.assertEqual(onboard_camera.resolve_device("auto", sysfs), "/dev/video0")
            self.assertEqual(onboard_camera.device_name("/dev/video0", sysfs), "Brio 101")
            self.assertIsNone(onboard_camera.find_camera(root / "missing"))

    def test_mjpeg_passthrough_delivers_camera_bytes_unchanged(self):
        fake, got = FakeV4L2(), []
        cam = onboard_camera.LocalBrioCapture("/dev/video9", on_frame=got.append, opener=lambda d: fake).start()
        try:
            self.assertTrue(wait_for(lambda: len(got) >= 3))
            self.assertEqual(got[0], fake.frame)
            info = cam.info()
            self.assertTrue(info["live"])
            self.assertTrue(info["mjpeg_passthrough"])
            self.assertEqual(info["status"], "streaming")
            self.assertFalse(info["persisted_to_disk"])
            if onboard_camera.cv2 is not None:
                cv2 = onboard_camera.cv2
                self.assertEqual(fake.props[cv2.CAP_PROP_FRAME_WIDTH], 1920)
                self.assertEqual(fake.props[cv2.CAP_PROP_FRAME_HEIGHT], 1080)
                self.assertEqual(fake.props[cv2.CAP_PROP_CONVERT_RGB], 0)
                self.assertEqual(info["fourcc"], "MJPG")
                self.assertEqual(cam.read_latest().shape, (108, 192, 3))
        finally:
            cam.stop()
        self.assertFalse(cam.live)
        self.assertTrue(fake.released)

    @unittest.skipIf(onboard_camera.cv2 is None, "OpenCV not installed")
    def test_decoded_frames_are_reencoded_in_memory(self):
        got = []
        cam = onboard_camera.LocalBrioCapture("/dev/video9", on_frame=got.append, opener=lambda d: FakeV4L2(bgr=True)).start()
        try:
            self.assertTrue(wait_for(lambda: got))
            self.assertTrue(got[0].startswith(b"\xff\xd8"))
            self.assertFalse(cam.info()["mjpeg_passthrough"])
        finally:
            cam.stop()

    def test_failed_reads_trigger_reopen(self):
        opened = []
        cam = onboard_camera.LocalBrioCapture("/dev/video9", opener=lambda d: opened.append(d) or FakeV4L2(fail=True)).start()
        try:
            self.assertTrue(wait_for(lambda: cam.reconnects >= 1 and len(opened) >= 2, 5.0))
            self.assertFalse(cam.live)
        finally:
            cam.stop()

    def test_unopenable_device_reports_unavailable(self):
        def boom(device):
            raise OSError("no such device")
        cam = onboard_camera.LocalBrioCapture("/dev/video9", opener=boom)
        self.assertFalse(cam.open())
        self.assertEqual(cam.info()["status"], "unavailable")
        self.assertIn("no such device", cam.info()["error"])


class FallbackTests(unittest.TestCase):
    """The offline catalog only ever returns a verified model of the *same* product - never a generic stand-in."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.patches = [mock.patch.object(mcp_server, "MODELS_DIR", self.dir),
                        mock.patch.object(mcp_server, "FALLBACK_MODELS", {}),
                        mock.patch.object(mcp_server, "_search_online", return_value=None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def cache(self, stem, canonical):
        (self.dir / f"{stem}.glb").write_bytes(b"glTF" + b"\x02\x00\x00\x00" + b"\x00" * 64)
        (self.dir / f"{stem}.json").write_text(json.dumps({"canonical_name": canonical, "attribution_text": "by someone"}))

    def test_product_key_separates_tiers(self):
        key = mcp_server.product_key
        self.assertEqual(key("iPhone 15 Pro Max"), "iphone15promax")
        self.assertEqual(key("apple_iphone_15_pro"), "iphone15pro")
        self.assertNotEqual(key("iPhone 15 Pro"), key("iPhone 15 Pro Max"))
        self.assertEqual(key("Samsung Galaxy S24 Ultra"), "galaxys24ultra")
        self.assertEqual(key("s25_ultra"), "galaxys25ultra")
        self.assertEqual(key("Galaxy S22+"), "galaxys22plus")
        self.assertEqual(key("coke zero can"), "can:coke:zero")

    def test_unknown_product_has_no_stand_in(self):
        for label in ("Sprite", "Apple iPhone 15 Pro Max", "s25_ultra"):
            self.assertIsNone(mcp_server.resolve_fallback(label), label)
        res = mcp_server.lookup_3d_asset("Sprite")
        self.assertEqual(set(res), {"title", "model_url", "thumbnail_url", "status"})
        self.assertEqual((res["status"], res["model_url"]), ("not_found", ""))

    def test_cached_model_matches_only_same_product(self):
        self.cache("galaxy_s24_ultra", "Samsung Galaxy S24 Ultra")
        hit = mcp_server.resolve_fallback("Samsung Galaxy S24 Ultra")
        self.assertEqual(hit["model_url"], "/static/models/galaxy_s24_ultra.glb")
        self.assertEqual(mcp_server.resolve_fallback("s24_ultra")["title"], "Samsung Galaxy S24 Ultra")
        self.assertIsNone(mcp_server.resolve_fallback("Samsung Galaxy S24"))
        self.assertIsNone(mcp_server.resolve_fallback("Samsung Galaxy S25 Ultra"))
        self.assertEqual(mcp_server.lookup_3d_asset("s24 ultra")["status"], "cached")
        (self.dir / "galaxy_s24_ultra.json").unlink()  # no attribution sidecar: not a verified model
        self.assertIsNone(mcp_server.resolve_fallback("Samsung Galaxy S24 Ultra"))

    def test_curated_fallback_matches_same_product(self):
        entry = {"title": "S25 Ultra", "model_url": "https://example.org/s25u.glb", "thumbnail_url": "", "attribution": "x"}
        with mock.patch.dict(mcp_server.FALLBACK_MODELS, {"samsung galaxy s25 ultra": entry}):
            self.assertEqual(mcp_server.resolve_fallback("s25_ultra")["model_url"], entry["model_url"])
            self.assertIsNone(mcp_server.resolve_fallback("s24_ultra"))

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

    CURATED = {"coke can": {"title": "Coca-Cola can", "model_url": "https://example.org/coke.glb", "thumbnail_url": "",
                            "attribution": "\u201cCoca-Cola can\u201d by tester (CC-BY)"}}

    def test_search_downloads_and_caches_glb(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mcp_server, "MODELS_DIR", Path(tmp)), \
                mock.patch.object(mcp_server, "FALLBACK_MODELS", self.CURATED), \
                mock.patch.object(mcp_server, "_search_online", return_value=None), \
                mock.patch.object(mcp_server.urllib.request, "urlopen", lambda *a, **k: self.FakeResp(self.GLB)):
            res = mcp_server.search_3d_asset("coke_can")
            self.assertEqual(res["glb_url"], "/static/models/coke_can.glb")
            self.assertEqual(res["status"], "cached")
            self.assertTrue((Path(tmp) / "coke_can.glb").is_file())
            self.assertEqual(set(res) & {"title", "model_url", "thumbnail_url", "status"}, {"title", "model_url", "thumbnail_url", "status"})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mcp_server, "MODELS_DIR", Path(tmp)), \
                mock.patch.object(mcp_server, "FALLBACK_MODELS", self.CURATED), \
                mock.patch.object(mcp_server, "_search_online", return_value=None), \
                mock.patch.object(mcp_server.urllib.request, "urlopen", lambda *a, **k: self.FakeResp(b"<html>nope</html>")):
            res = mcp_server.search_3d_asset("coke_can")
            self.assertNotIn("glb_url", res)
            self.assertFalse((Path(tmp) / "coke_can.glb").exists())

    def test_search_without_verified_source_is_not_found(self):
        def never(*a, **k):
            raise AssertionError("nothing to download")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mcp_server, "MODELS_DIR", Path(tmp)), \
                mock.patch.object(mcp_server, "FALLBACK_MODELS", {}), \
                mock.patch.object(mcp_server, "_search_online", return_value=None), \
                mock.patch.object(mcp_server.urllib.request, "urlopen", never):
            res = mcp_server.search_3d_asset("Pepsi can")
            self.assertEqual((res["status"], res["model_url"]), ("not_found", ""))
            self.assertNotIn("glb_url", res)
            self.assertEqual(list(Path(tmp).iterdir()), [])

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
        self.assertAlmostEqual(out["score"], receiver.calibrate(0.9))

    def test_small_margin_never_identifies_and_guides_after_three(self):
        sm = receiver.StateMachine()
        amb = receiver.FrameScore("coke_can", 0.95, 0.90, "pepsi_can")  # calibrated 0.90 vs 0.79: margin 0.10 < 0.15
        self.assertLess(amb.margin, receiver.MIN_MARGIN)
        self.assertEqual(self.feed(sm, [amb] * 2)["state"], receiver.SEARCHING)
        self.assertEqual(self.feed(sm, [amb])["state"], receiver.GUIDING)

    def test_mid_similarity_guides(self):
        sm = receiver.StateMachine()
        out = self.feed(sm, [receiver.FrameScore("s25_ultra", 0.75, 0.0)] * 3)
        self.assertEqual(out["state"], receiver.GUIDING)

    def test_background_cosine_calibrates_low_and_smooth(self):
        # Measured on the board: idle room 0.40-0.48, empty scene 0.13-0.30 -> all well under SEARCHING.
        for raw in (-0.3, 0.0, 0.30, 0.38, 0.45):
            self.assertLess(receiver.calibrate(raw), 0.15)
        self.assertLess(receiver.calibrate(0.52), receiver.C_SEARCH)
        grid = [i / 100 for i in range(-30, 101)]
        cal = [receiver.calibrate(s) for s in grid]
        self.assertTrue(all(b >= a for a, b in zip(cal, cal[1:])))  # monotone
        lo, hi = receiver.CAL_KNOTS[0], receiver.CAL_KNOTS[2]
        active = [receiver.calibrate(s) for s in grid if lo <= s <= hi]
        self.assertTrue(all(b > a for a, b in zip(active, active[1:])))  # strictly increasing over the working range
        self.assertTrue(all(b - a < 0.03 for a, b in zip(cal, cal[1:])))  # no jumps between 0.01 cosine steps
        self.assertAlmostEqual(receiver.calibrate(1.0), receiver.CAL_CAP)  # never a literal 100%
        self.assertAlmostEqual(receiver.calibrate(receiver.T_IDENT), receiver.C_IDENT)
        self.assertAlmostEqual(receiver.calibrate(receiver.T_SEARCH), receiver.C_SEARCH)
        self.assertAlmostEqual(receiver.T_IDENT, 0.804, places=2)
        self.assertAlmostEqual(receiver.T_SEARCH, 0.560, places=2)
        for s in (0.6, 0.75, 0.85, 0.95):
            self.assertAlmostEqual(receiver.raw_threshold(receiver.calibrate(s)), s, places=6)
        # Presented objects (0.62-0.89 measured) span GUIDING -> IDENTIFIED instead of saturating.
        self.assertLess(receiver.calibrate(0.75), receiver.C_IDENT)
        self.assertGreater(receiver.calibrate(0.85), receiver.C_IDENT)
        sm = receiver.StateMachine()
        out = self.feed(sm, [receiver.FrameScore("coke_can", 0.45, 0.40, "pepsi_can")] * 5)
        self.assertEqual(out["state"], receiver.SEARCHING)
        self.assertLess(out["score"], receiver.C_SEARCH)

    def test_fused_confidence_overrides_vision(self):
        f = receiver.FrameScore("coke_can", 0.75, 0.5, "pepsi_can", f1=0.8, f2=0.3)
        self.assertAlmostEqual(f.c1, 0.8)
        self.assertAlmostEqual(f.c2, 0.3)
        self.assertAlmostEqual(f.margin, 0.5)
        capped = receiver.FrameScore("iphone_16", 0.97, f1=0.9, cap=0.35)
        self.assertAlmostEqual(capped.c1, 0.35)

    def test_reported_confidence_is_ema(self):
        sm = receiver.StateMachine()
        self.feed(sm, [self.ident(s1=0.90)] * 3)
        before = self.feed(sm, [self.ident(s1=0.90)])["score"]
        after = self.feed(sm, [self.ident(s1=0.95)])["score"]
        expected = receiver.EMA_ALPHA * receiver.calibrate(0.95) + (1 - receiver.EMA_ALPHA) * before
        self.assertAlmostEqual(after, expected, places=6)

    def test_empty_scene_reads_zero(self):
        sm = receiver.StateMachine()
        self.feed(sm, [self.ident()] * 3)
        out = self.feed(sm, [receiver.FrameScore(None, 0.0, empty=True)] * 3)
        self.assertEqual((out["state"], out["score"], out["margin"]), (receiver.SEARCHING, 0.0, 0.0))
        msg = receiver.guidance_message(out["state"], None, 0.0, empty=True)
        self.assertEqual(msg, "Awaiting object inside center reticle...")
        self.assertEqual(receiver.led_params(out["state"], out["score"]), (0, 100, 150, 15, 1))

    def test_front_cap_limits_confidence(self):
        sm = receiver.StateMachine()
        f = receiver.FrameScore("iphone_16", 0.97, 0.5, "galaxy_s25", hint="flip", cap=0.35)
        out = self.feed(sm, [f] * 3)
        self.assertEqual(out["state"], receiver.GUIDING)
        self.assertLessEqual(out["score"], 0.35)
        self.assertEqual(receiver.guidance_message(out["state"], out["label"], out["score"], "flip"),
                         "Front screen detected. Flip phone 180° to expose camera island.")

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
    def test_center_crop_is_seventy_five_percent(self):
        self.assertEqual(receiver.center_crop_box(640, 480), (360, 140, 60))
        size, left, top = receiver.center_crop_box(1920, 1080)
        self.assertEqual((left, top, left + size, top + size), (555, 135, 1365, 945))

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

    def test_dinov2_pose_views_persist_and_clear(self):
        def desc(i, dim=8):
            v = np.zeros(dim, np.float32)
            v[i] = 1.0
            return v

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.pt"
            bank = receiver.EmbeddingBank(path)
            info = bank.add_multiview("coke_can", {"front": [unit(1)] * 2, "right": [unit(2)] * 2, "back": [unit(3)] * 2},
                                      {"category": "beverage_can"},
                                      {"front": [desc(0)] * 2, "right": [desc(1)] * 2, "top": [desc(3)]})
            self.assertEqual(info["pose_views"], ["front", "right"])  # "top" has no MobileNet view: dropped
            self.assertTrue(bank.has_pose_views("coke_can"))
            again = receiver.EmbeddingBank(path)
            again.load()
            sims = again.pose_view_sims("coke_can", desc(1))
            self.assertAlmostEqual(sims["right"], 1.0, places=5)
            self.assertAlmostEqual(sims["front"], 0.0, places=5)
            self.assertEqual(again.info()[0]["pose_views"], ["front", "right"])
            again.add_multiview("pepsi_can", {"front": [unit(4)]}, {"category": "beverage_can"})
            self.assertFalse(again.has_pose_views("pepsi_can"))
            self.assertEqual(again.pose_view_sims("pepsi_can", desc(0)), {})
            data = torch.load(path, weights_only=True)
            data["pose_embedder"] = "some-other-model"
            torch.save(data, path)
            stale = receiver.EmbeddingBank(path)
            stale.load()
            self.assertFalse(stale.has_pose_views("coke_can"))  # descriptors from another model are never compared
            self.assertTrue(again.remove("coke_can"))
            self.assertFalse(again.has_pose_views("coke_can"))


class FakePoseEmbedder:
    device = "npu"

    def __init__(self, desc):
        self.desc = desc
        self.calls = 0

    def embed_rgb01(self, nhwc):
        assert nhwc.shape == (1, 224, 224, 3) and 0.0 <= float(nhwc.min()) and float(nhwc.max()) <= 1.0001
        self.calls += 1
        return self.desc

    def info(self):
        return {"model": "fake-dinov2", "device": self.device}


class PoseWorkerTests(unittest.TestCase):
    def test_rgb01_inverts_preprocess(self):
        img = Image.new("RGB", (300, 300), (200, 30, 90))
        back = receiver.rgb01_nhwc(receiver.preprocess(img, (300, 0, 0)))
        self.assertEqual(back.shape, (1, 224, 224, 3))
        np.testing.assert_allclose(back[0, 100, 100], np.array([200, 30, 90]) / 255.0, atol=1e-4)

    @needs_torch
    def test_worker_scores_latest_crop_against_view_centroids(self):
        bank = receiver.EmbeddingBank(None)
        right = np.array([0, 1, 0, 0], np.float32)
        bank.add_multiview("coke_can", {"front": [unit(1)], "right": [unit(2)], "back": [unit(3)]}, {"category": "beverage_can"},
                           {"front": [np.array([1, 0, 0, 0], np.float32)], "right": [right],
                            "back": [np.array([0, 0, 1, 0], np.float32)]})
        emb = FakePoseEmbedder(right)
        worker = receiver.PoseWorker(emb, bank, hz=1000)
        arr = receiver.preprocess(Image.new("RGB", (64, 64), (10, 20, 30)))
        self.assertIsNone(worker.latest("coke_can"))
        self.assertTrue(worker.submit("coke_can", arr))
        deadline = time.monotonic() + 5
        while worker.latest("coke_can") is None and time.monotonic() < deadline:
            time.sleep(0.01)
        sims = worker.latest("coke_can")
        self.assertEqual(max(sims, key=sims.get), "right")
        self.assertIsNone(worker.latest("pepsi_can"))  # results are tied to the class they were computed for
        yaw, _, best = receiver.pose_mod.PoseEstimator._view_yaw(sims, 0.0)
        self.assertEqual(best, "right")
        self.assertAlmostEqual(yaw, 90.0, delta=1.0)  # atan2(w_right - w_left, w_front - w_back)
        worker.reset()
        self.assertIsNone(worker.latest("coke_can"))
        slow = receiver.PoseWorker(emb, bank, hz=0.5)
        self.assertTrue(slow.submit("coke_can", arr))
        self.assertFalse(slow.submit("coke_can", arr))  # rate cap keeps NPU time for the detector


class FakeExtractor:
    name = "fake"

    def __init__(self):
        self.vec = None

    def embed_array(self, arr):
        return self.vec


class FakeAssets:
    def __init__(self):
        self.refetched = []

    def peek(self, label):
        return {"glb_url": f"/static/models/{label}.glb", "status": "ready", "title": label}

    get = peek

    def refetch(self, label):
        self.refetched.append(label)
        return True


class FakeDetector:
    device = "cpu"

    def __init__(self, label="cell phone", score=0.9):
        self.label, self.score = label, score

    def detect(self, image):
        return [receiver.npu_mod.Detection(self.label, self.score, (0.4, 0.3, 0.6, 0.7))] if self.label else []

    def info(self):
        return {"runtime": "fake"}


def basis(i, dim=16):
    v = torch.zeros(dim)
    v[i] = 1.0
    return v


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
        mixed = 0.75 * a + 0.6614 * orth  # cosine 0.75 to coke_can: calibrated 0.48, between C_SEARCH and C_IDENT
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

    def phone(self, label, front, back, extra=None):
        meta = {"category": "smartphone", "raw_name": label.replace("_", " "), "brand": "Apple" if "iphone" in label else "Samsung"}
        self.bank.add_multiview(label, {"front": [front] * 3, "back": [back] * 3, **(extra or {})}, meta)

    def test_empty_reticle_is_gated_to_zero(self):
        a = basis(0)
        self.enroll("coke_can", a)
        self.ext.vec = a
        for _ in range(3):
            status, data = self.call("POST", "/api/frame", jpeg(textured=False), "image/jpeg")
            verdict = json.loads(data)
        self.assertTrue(verdict["empty"])
        self.assertEqual((verdict["state"], verdict["confidence"], verdict["margin"]), ("SEARCHING", 0.0, 0.0))
        time.sleep(0.3)
        self.assertEqual(self.leds_log[-1], (0, 100, 150, 15, 1))
        state = json.loads(self.call("GET", "/api/state")[1])
        self.assertEqual(state["guidance"], "Awaiting object inside center reticle...")
        self.assertTrue(state["empty_scene"])
        self.assertLess(state["saliency"]["laplacian_var"], receiver.LAPLACIAN_MIN)
        self.assertEqual(state["detection"].get("skipped"), "empty_scene")
        self.assertEqual(self.frame(a)["empty"], False)

    def test_phone_front_capped_back_identified(self):
        self.phone("iphone_16", basis(0), basis(1))
        for _ in range(3):
            verdict = self.frame(basis(0))
        self.assertEqual(verdict["state"], "GUIDING")
        self.assertLessEqual(verdict["confidence"], 0.35)
        self.assertEqual(verdict["hint"], "flip")
        state = json.loads(self.call("GET", "/api/state")[1])
        self.assertEqual(state["guidance"], "Front screen detected. Flip phone 180° to expose camera island.")
        self.assertTrue(state["front_dominant"])
        time.sleep(0.3)
        self.assertEqual(self.leds_log[-1], (255, 120, 0, 40, 4))
        for _ in range(3):
            verdict = self.frame(basis(1))
        self.assertEqual((verdict["state"], verdict["label"]), ("IDENTIFIED", "iphone_16"))
        self.assertGreaterEqual(verdict["confidence"], 0.75)
        time.sleep(0.3)
        self.assertEqual(self.leds_log[-1][:3], (0, 255, 0))

    def test_close_phone_siblings_get_tier_prompt(self):
        self.phone("iphone_15", basis(0), basis(1))
        self.phone("iphone_15_pro", basis(2), basis(3))
        live = torch.nn.functional.normalize(basis(1) * 1.0 + basis(3) * 0.95, dim=0)
        for _ in range(3):
            verdict = self.frame(live)
        self.assertEqual(verdict["state"], "GUIDING")
        self.assertEqual(verdict["hint"], "tilt")
        state = json.loads(self.call("GET", "/api/state")[1])
        self.assertIn("LiDAR", state["guidance"])

    def test_detector_domain_gate_and_conflict(self):
        self.pipe.detector = FakeDetector("cell phone", 0.9)
        self.phone("galaxy_s25_ultra", basis(0), basis(1))
        self.bank.add_multiview("coke_can", {"front": [basis(5)] * 3}, {"category": "beverage_can", "raw_name": "coke can"})
        for _ in range(3):
            verdict = self.frame(basis(5))
        self.assertEqual(verdict["domain"], ["smartphone"])
        self.assertNotEqual(verdict["label"], "coke_can")
        self.assertEqual(verdict["state"], "GUIDING")
        state = json.loads(self.call("GET", "/api/state")[1])
        self.assertTrue(state["domain_conflict"])
        self.assertEqual(state["guidance"], "Verifying object geometry... Hold steady in center.")
        self.pipe.detector = FakeDetector("cup", 0.9)
        for _ in range(3):
            verdict = self.frame(basis(5))
        self.assertEqual((verdict["state"], verdict["label"]), ("IDENTIFIED", "coke_can"))
        self.pipe.detector = FakeDetector("cup", 0.3)  # below the 0.5 gate: no domain restriction
        self.assertIsNone(self.frame(basis(1))["domain"])

    def test_purge_deletes_db_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.bank.path = Path(tmp) / "enrolled_classes.pt"
            self.enroll("coke_can", basis(0))
            self.bank.save()
            self.assertTrue(self.bank.path.exists())
            status, data = self.call("POST", "/api/classes/purge", b'{"delete_cad": false}')
            out = json.loads(data)
            self.assertEqual(status, 200, data)
            self.assertEqual((out["removed"], out["labels"], out["db_file_exists"]), (1, ["coke_can"], False))
            self.assertFalse(self.bank.path.exists())
            listing = json.loads(self.call("GET", "/api/classes")[1])
            self.assertEqual((listing["count"], listing["classes"]), (0, {}))
            self.assertEqual(json.loads(self.call("GET", "/api/state")[1])["state"], "SEARCHING")

    def test_fetch_cad_endpoint(self):
        self.enroll("coke_can", basis(0))
        status, data = self.call("POST", "/api/classes/coke_can/fetch_cad", b"")
        self.assertEqual(status, 202, data)
        out = json.loads(data)
        self.assertTrue(out["started"])
        self.assertEqual(self.pipe.assets.refetched, ["coke_can"])
        self.assertEqual(json.loads(self.call("GET", "/api/asset")[1])["glb_url"], "/static/models/coke_can.glb")
        self.assertEqual(self.call("POST", "/api/classes/ghost/fetch_cad", b"")[0], 404)
        row = json.loads(self.call("GET", "/api/classes")[1])["prototypes"]
        self.assertIn("cad_path", row[0] if isinstance(row, list) else next(iter(row.values())))

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

    def test_onboard_camera_feeds_pipeline_and_owns_the_stream(self):
        a = unit(1)
        self.ext.vec = a
        cam = onboard_camera.LocalBrioCapture("/dev/video0", opener=lambda d: FakeV4L2(frame=jpeg()))
        self.pipe.attach_camera(cam)
        cam.start()
        try:
            self.assertTrue(wait_for(lambda: self.pipe.stream_stats()["processed"] >= 3))
            st = self.pipe.stream_stats()
            self.assertEqual((st["source"]["client"], st["source"]["format"]), ("onboard", "mjpeg-v4l2"))
            self.assertEqual(st["camera"]["device"], "/dev/video0")
            self.assertTrue(st["camera"]["live"])
            self.assertEqual(self.pipe.latest_jpeg, cam.latest_jpeg())  # preview == the camera's own JPEG
            status, data = self.call("POST", "/api/ingest", jpeg(), "image/jpeg")
            self.assertEqual(status, 409, data)
            self.assertIn("on-board camera", json.loads(data)["error"])
            state = json.loads(self.call("GET", "/api/state")[1])
            self.assertEqual(state["stream"]["camera"]["source"], "onboard-v4l2")
        finally:
            cam.stop()
        self.assertEqual(self.call("POST", "/api/frame", jpeg(), "image/jpeg")[0], 200)  # HTTP ingest is back

    def test_http_ingest_reports_no_camera(self):
        self.assertEqual(self.pipe.stream_stats()["camera"], {"source": "http-ingest", "device": None})


if __name__ == "__main__":
    unittest.main()
