"""Phase 5 full-suite tests (run on the VENTUNO Q for the hardware parts).

    cd ~/hackathon_iot && .venv/bin/python -m unittest test_full_suite -v

Covers: plaintext parsing of cans and phones, CAD ranking + harvesting + attribution into static/models,
vector-only enrollment (no image files on disk), class lifecycle against enrolled_classes.pt, real /proc hardware
stats, the /api/telemetry endpoint and a real LED RPC through /var/run/arduino-router.sock.
Online CAD tests need SKETCHFAB_TOKEN and network access and skip themselves otherwise.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.request
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

import cad_harvester
import entity_resolver
import mcp_server
import taxonomy
import telemetry

try:
    import torch
except ImportError:
    torch = None

needs_torch = unittest.skipIf(torch is None, "PyTorch not installed")
ROUTER_SOCK = "/var/run/arduino-router.sock"
ROOT = Path(__file__).resolve().parent
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic"}
TEST_PREFIX = "zz_test_"


def image_files(*roots: Path) -> set[str]:
    found = set()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "__pycache__", "node_modules", "vendor"}]
            found.update(os.path.join(dirpath, f) for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXT)
    return found


def online() -> bool:
    if not os.environ.get("SKETCHFAB_TOKEN"):
        return False
    try:
        urllib.request.urlopen("https://api.sketchfab.com/v3/models?count=1", timeout=6).close()
        return True
    except Exception:
        return False


def cleanup_models(*class_ids: str) -> None:
    for cid in class_ids:
        cad_harvester.delete_cached(cid)


class PlaintextParsingTests(unittest.TestCase):
    CANS = {
        "Coca Cola Original Taste Can 330ml": ("coca_cola_330ml_can", "Coca-Cola", "regular"),
        "coke zero can 330ml": ("coca_cola_zero_330ml_can", "Coca-Cola", "zero"),
        "Diet Coke 330ml": ("coca_cola_diet_330ml_can", "Coca-Cola", "diet"),
        "Pepsi Max 12oz can": ("pepsi_zero_12oz_can", "Pepsi", "zero"),
        "pepsi can": ("pepsi_can", "Pepsi", "regular"),
        "Sprite can 330ml": ("sprite_330ml_can", "Sprite", "regular"),
        "red bull can 250ml": ("red_bull_250ml_can", "Red Bull", "regular"),
    }
    PHONES = {
        "iPhone 15 Pro Max deep purple": ("iphone_15_pro_max", "Apple", "pro_max"),
        "iphone 12": ("iphone_12", "Apple", "standard"),
        "iPhone 14 Plus": ("iphone_14_plus", "Apple", "plus"),
        "Samsung Galaxy S24 Ultra": ("galaxy_s24_ultra", "Samsung", "ultra"),
        "galaxy s23+": ("galaxy_s23_plus", "Samsung", "plus"),
        "Samsung S22 Plus": ("galaxy_s22_plus", "Samsung", "plus"),
        "Galaxy S25 Ultra": ("galaxy_s25_ultra", "Samsung", "ultra"),
    }

    def test_cans(self):
        for text, (class_id, brand, tier) in self.CANS.items():
            with self.subTest(text=text):
                p = entity_resolver.resolve(text)
                self.assertEqual((p.class_id, p.brand, p.sub_tier, p.category), (class_id, brand, tier, "beverage_can"))
                self.assertTrue(p.cad_search_terms)

    def test_phones(self):
        for text, (class_id, brand, tier) in self.PHONES.items():
            with self.subTest(text=text):
                p = entity_resolver.resolve(text)
                self.assertEqual((p.class_id, p.brand, p.sub_tier, p.category), (class_id, brand, tier, "smartphone"))
                self.assertTrue(any("3D model" in t or "glb" in t.lower() or "model" in t.lower() for t in p.cad_search_terms))

    def test_colour_is_captured_but_not_part_of_the_class(self):
        p = entity_resolver.resolve("iPhone 15 Pro Max deep purple")
        self.assertEqual(p.color, "deep purple")
        self.assertEqual(p.metadata()["color"], "deep purple")
        self.assertNotIn("purple", p.class_id)

    def test_class_id_matches_taxonomy_label(self):
        for text in list(self.CANS) + list(self.PHONES):
            p = entity_resolver.resolve(text)
            self.assertEqual(p.class_id, taxonomy.label_from_metadata(p.metadata()), text)

    def test_structured_metadata_schema(self):
        meta = taxonomy.parse_metadata({
            "raw_name": "Coca Cola Original Taste Can 330ml", "category": "beverage_can", "brand": "Coca-Cola",
            "variant": "original", "package_type": "can", "container_volume": "330ml"})
        p = entity_resolver.resolve_metadata(meta)
        self.assertEqual(p.class_id, "coca_cola_330ml_can")
        self.assertEqual(p.container_volume, "330ml")

    def test_unknown_brand_is_rejected(self):
        with self.assertRaises(taxonomy.MetadataError):
            entity_resolver.resolve("mystery object")


class CadRankerTests(unittest.TestCase):
    @staticmethod
    def model(uid, name, lic="CC Attribution", faces=20000, likes=10, user="Alice", downloadable=True):
        return {"uid": uid, "name": name, "license": {"label": lic, "uid": "x"}, "faceCount": faces, "likeCount": likes,
                "isDownloadable": downloadable, "user": {"displayName": user, "profileUrl": f"https://sketchfab.com/{user}"},
                "thumbnails": {"images": []}}

    def rank(self, text, *models):
        return cad_harvester.rank_sketchfab(entity_resolver.resolve(text), list(models))

    def test_ultra_vs_non_ultra_and_other_generation(self):
        out = self.rank("Samsung Galaxy S24 Ultra",
                        self.model("a", "Samsung Galaxy S22 Ultra"), self.model("b", "Samsung Galaxy S24"),
                        self.model("c", "Samsung Galaxy S24 Ultra", user="Bob"))
        self.assertEqual([c.uid for c in out], ["c"])
        self.assertEqual(out[0].author, "Bob")

    def test_non_commercial_and_unknown_licences_rejected(self):
        self.assertEqual(self.rank("iphone 15 pro", self.model("a", "iPhone 15 Pro", lic="CC Attribution-NonCommercial")), [])
        self.assertEqual(self.rank("iphone 15 pro", self.model("a", "iPhone 15 Pro", lic="All Rights Reserved")), [])
        self.assertEqual(self.rank("iphone 15 pro", self.model("a", "iPhone 15 Pro", lic="")), [])
        std = self.rank("iphone 15 pro", self.model("a", "iPhone 15 Pro", lic="Standard"),
                        self.model("b", "iPhone 15 Pro", lic="CC Attribution"))
        self.assertEqual([c.uid for c in std], ["b", "a"])
        self.assertEqual(len(self.rank("iphone 15 pro", self.model("a", "iPhone 15 Pro", lic="CC0 Public Domain"))), 1)

    def test_scenes_accessories_and_heavy_meshes_rejected(self):
        for name in ("iPhone 15 Pro case cover", "iPhone 15 Pro collection", "iPhone 15 Pro box packaging"):
            self.assertEqual(self.rank("iphone 15 pro", self.model("a", name)), [], name)
        self.assertEqual(self.rank("iphone 15 pro", self.model("a", "iPhone 15 Pro", faces=9_000_000)), [])
        self.assertEqual(self.rank("iphone 15 pro", self.model("a", "iPhone 15 Pro", downloadable=False)), [])

    def test_can_variants_and_lookalikes(self):
        zero = self.rank("coke zero can", self.model("a", "Coca-Cola can"), self.model("b", "Coke Zero can"))
        self.assertEqual([c.uid for c in zero], ["b"])
        self.assertEqual(self.rank("sprite can", self.model("a", "Sprite Sonic fanart")), [])
        self.assertEqual(self.rank("sprite can", self.model("a", "Sprite 3D character")), [])
        self.assertEqual([c.uid for c in self.rank("sprite can", self.model("a", "Sprite soda can"))], ["a"])
        self.assertEqual(self.rank("pepsi can", self.model("a", "Pepsi bottle")), [])

    def test_better_licence_and_likes_rank_first(self):
        out = self.rank("pepsi can", self.model("a", "Pepsi can", lic="CC Attribution", likes=1),
                        self.model("b", "Pepsi can", lic="CC0 Public Domain", likes=1))
        self.assertEqual(out[0].uid, "b")


class CadCacheTests(unittest.TestCase):
    def setUp(self):
        self.ids = []

    def tearDown(self):
        cleanup_models(*self.ids)
        cad_harvester.write_credits_md()

    def product(self, text, suffix):
        cid = TEST_PREFIX + suffix
        self.ids.append(cid)
        return cad_harvester._with_class_id(entity_resolver.resolve(text), cid)

    def test_offline_fallback_is_cached_into_static_models_with_attribution(self):
        prod = self.product("coke zero can 330ml", "offline_can")
        first = cad_harvester.harvest(prod, offline=True)
        glb = cad_harvester.glb_path(prod.class_id)
        self.assertEqual(glb.parent.resolve(), (ROOT / "static" / "models").resolve() if "KAMIKAZE_MODELS_DIR" not in os.environ
                         else Path(os.environ["KAMIKAZE_MODELS_DIR"]).resolve())
        self.assertTrue(mcp_server.is_valid_glb(glb))
        self.assertEqual(first["glb_url"], f"/static/models/{prod.class_id}.glb")
        attr = cad_harvester.read_attribution(prod.class_id)
        self.assertTrue(attr and attr["author"] and attr["license"])
        self.assertIn("Kamikaze", attr["author"])
        second = cad_harvester.harvest(prod, offline=True)
        self.assertEqual((second["status"], second["source"], second["cached"]), ("cached", "local-cache", True))
        self.assertTrue(second["credit"])

    def test_delete_cached_removes_glb_and_sidecar(self):
        prod = self.product("iphone 13", "delete_cad")
        cad_harvester.harvest(prod, offline=True)
        self.assertTrue(cad_harvester.glb_path(prod.class_id).exists())
        self.assertTrue(cad_harvester.delete_cached(prod.class_id))
        self.assertFalse(cad_harvester.glb_path(prod.class_id).exists())
        self.assertIsNone(cad_harvester.read_attribution(prod.class_id))
        self.assertFalse(cad_harvester.delete_cached(prod.class_id))

    def test_glb_without_attribution_is_reharvested(self):
        prod = self.product("iphone 14", "no_sidecar")
        cad_harvester.harvest(prod, offline=True)
        cad_harvester.sidecar_path(prod.class_id).unlink()
        out = cad_harvester.harvest(prod, offline=True)
        self.assertFalse(out["cached"])
        self.assertTrue(cad_harvester.read_attribution(prod.class_id))

    def test_credits_listing_includes_author_and_licence(self):
        prod = self.product("pepsi can", "credits")
        cad_harvester.harvest(prod, offline=True)
        row = next(c for c in cad_harvester.credits() if c.get("class_id") == prod.class_id)
        self.assertTrue(row["author"] and row["license"])
        credits_md = cad_harvester.models_dir() / "CREDITS.md"
        self.assertTrue(credits_md.exists())

    @unittest.skipUnless(online(), "needs SKETCHFAB_TOKEN and network access")
    def test_live_sketchfab_download_records_original_author(self):
        prod = self.product("Samsung Galaxy S24 Ultra", "live_s24u")
        out = cad_harvester.harvest(prod, force=True)
        self.assertTrue(mcp_server.is_valid_glb(cad_harvester.glb_path(prod.class_id)))
        attr = out["attribution"]
        self.assertTrue(attr["author"] and attr["license"] and attr["source"], attr)
        if out["source"] == "sketchfab":
            self.assertIn("sketchfab.com", attr["source_url"])
            self.assertFalse(any(w in attr["license"].lower() for w in ("noncommercial", "non-commercial")))
            self.assertIn(attr["author"], out["credit"])
        print(f"\n  live CAD: {out['source']} | {out['credit']}")


@needs_torch
class EnrollmentAndLifecycleTests(unittest.TestCase):
    def setUp(self):
        import receiver
        import test_comprehensive as tc
        self.receiver, self.tc = receiver, tc
        self.tmp = Path(tempfile.mkdtemp(prefix="kz_full_"))
        self.db = self.tmp / "enrolled_classes.pt"
        self.ext = tc.FakeExtractor()
        self.bank = receiver.EmbeddingBank(self.db)
        self.log = []
        self.leds = receiver.LedDriver(lambda *a: self.log.append(a) or True)
        self.pipe = receiver.Pipeline(self.ext, self.bank, self.leds, tc.FakeAssets())
        self.server = receiver.make_server(self.pipe, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]
        self.conn = HTTPConnection("127.0.0.1", self.port, timeout=15)
        self.patches = [mock.patch("builtins.print"), mock.patch.object(receiver, "LOG_FRAMES", False)]
        for p in self.patches:
            p.start()
        self.before = image_files(ROOT, self.tmp, Path(tempfile.gettempdir()) / "kz_none")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.server.shutdown()
        self.server.server_close()
        self.conn.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, method, path, body=None):
        payload = json.dumps(body).encode() if isinstance(body, dict) else body
        self.conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
        resp = self.conn.getresponse()
        data = resp.read()
        return resp.status, (json.loads(data) if "json" in resp.getheader("Content-Type", "") else data)

    def enroll(self, name, seed, frames=8, **meta):
        tc = self.tc
        status, st = self.call("POST", "/api/enroll/start", {"raw_name": name, "frames_per_view": frames, **meta})
        self.assertEqual(status, 200, st)
        base = tc.unit(seed)
        for i, view in enumerate(taxonomy.VIEWS):
            self.assertEqual(self.call("POST", "/api/enroll/capture", {"view": view})[0], 200)
            vec = torch.nn.functional.normalize(base + 0.6 * tc.unit(seed * 10 + i), dim=0)
            for k in range(frames):
                self.ext.vec = torch.nn.functional.normalize(vec + 0.02 * tc.unit(900 + k), dim=0)
                status, data = self.call("POST", "/api/frame", tc.jpeg())
                self.assertEqual(status, 200, data)
        _, st = self.call("GET", "/api/enroll/status")
        self.assertEqual(st["phase"], "done", st)
        return st

    def test_views_are_front_back_side_top(self):
        self.assertEqual(tuple(taxonomy.VIEWS), ("front", "back", "side", "top"))

    def test_fresh_empty_database_serves_state_and_classes(self):
        status, state = self.call("GET", "/api/state")
        self.assertEqual((status, state["bank"], state["state"]), (200, [], "SEARCHING"))
        status, listing = self.call("GET", "/api/classes")
        self.assertEqual((status, listing["count"], listing["prototypes"]), (200, 0, []))
        self.assertEqual(self.call("GET", "/api/telemetry")[0], 200)

    def test_enrollment_persists_vectors_only(self):
        self.enroll("Samsung Galaxy S24 Ultra", 3, category="smartphone")
        self.bank.save()
        self.assertTrue(self.db.is_file())
        after = image_files(ROOT, self.tmp, Path(tempfile.gettempdir()) / "kz_none")
        self.assertEqual(after - self.before, set(), "enrollment wrote image files to disk")
        _, privacy = self.call("GET", "/api/telemetry")
        self.assertFalse(privacy["privacy"]["raw_frames_persisted"])
        self.assertFalse(privacy["privacy"]["images_written_to_disk"])
        data = torch.load(self.db, map_location="cpu", weights_only=True)
        proto = data["prototypes"]["galaxy_s24_ultra"]
        for key in ("class_id", "canonical_name", "metadata", "cad_model_url", "views", "global_centroid", "created_at"):
            self.assertIn(key, proto)
        self.assertEqual(set(proto["views"]), set(taxonomy.VIEWS))
        self.assertEqual(tuple(proto["global_centroid"].shape), (32,))
        self.assertAlmostEqual(float(proto["global_centroid"].norm()), 1.0, places=4)
        for vec in list(proto["views"].values()) + [proto["global_centroid"]]:
            self.assertIsInstance(vec, torch.Tensor)
        self.assertEqual(proto["canonical_name"], "Samsung Galaxy S24 Ultra")
        self.assertEqual(proto["metadata"]["category"], "smartphone")
        self.assertTrue(proto["created_at"])

    def test_delete_class_updates_enrolled_classes_pt(self):
        self.enroll("Samsung Galaxy S24 Ultra", 3)
        self.enroll("Coke Zero can 330ml", 5)
        self.bank.save()
        saved = torch.load(self.db, map_location="cpu", weights_only=True)
        self.assertEqual(set(saved["prototypes"]), {"galaxy_s24_ultra", "coca_cola_zero_330ml_can"})

        status, listing = self.call("GET", "/api/classes")
        self.assertEqual((status, listing["count"]), (200, 2))
        self.assertEqual({p["class_id"] for p in listing["prototypes"]}, set(saved["prototypes"]))

        status, out = self.call("DELETE", "/api/classes/galaxy_s24_ultra")
        self.assertEqual((status, out["ok"]), (200, True), out)
        on_disk = torch.load(self.db, map_location="cpu", weights_only=True)
        self.assertEqual(set(on_disk["prototypes"]), {"coca_cola_zero_330ml_can"})
        self.assertEqual(set(on_disk["centroids"]), {"coca_cola_zero_330ml_can"})
        self.assertEqual(set(on_disk["exemplars"]), {"coca_cola_zero_330ml_can"})

        reloaded = self.receiver.EmbeddingBank(self.db)
        self.assertEqual(reloaded.load(), 1)
        self.assertEqual(reloaded.labels(), ["coca_cola_zero_330ml_can"])
        self.assertEqual(self.call("DELETE", "/api/classes/galaxy_s24_ultra")[0], 404)
        self.assertEqual(self.call("DELETE", "/api/classes/..%2F..%2Fetc")[0] in (400, 404), True)

    def test_delete_with_cad_removes_cached_model(self):
        self.enroll("iphone 15 pro", 7)
        cid = "iphone_15_pro"
        keep = cad_harvester.glb_path(cid).exists()
        if keep:
            self.skipTest("a real cached model for iphone_15_pro exists; not deleting it in tests")
        try:
            cad_harvester.harvest(entity_resolver.resolve("iphone 15 pro"), offline=True)
            self.assertTrue(cad_harvester.glb_path(cid).exists())
            status, out = self.call("DELETE", f"/api/classes/{cid}?delete_cad=1")
            self.assertEqual((status, out["cad_deleted"]), (200, True), out)
            self.assertFalse(cad_harvester.glb_path(cid).exists())
        finally:
            cad_harvester.delete_cached(cid)
            cad_harvester.write_credits_md()

    def test_reset_empties_the_database_file(self):
        self.enroll("Pepsi can", 11)
        self.enroll("Sprite can", 13)
        self.bank.save()
        status, out = self.call("POST", "/api/classes/reset", {})
        self.assertEqual((status, out["ok"], sorted(out["removed"])), (200, True, ["pepsi_can", "sprite_can"]))
        self.assertEqual(self.call("GET", "/api/classes")[1]["count"], 0)
        reloaded = self.receiver.EmbeddingBank(self.db)
        reloaded.load()
        self.assertEqual(reloaded.labels(), [])

    def test_resolve_endpoint_and_cad_fetch_link_into_the_class(self):
        self.enroll("iPhone 13", 17)
        status, out = self.call("POST", "/api/resolve", {"raw_name": "galaxy s23+"})
        self.assertEqual((status, out["class_id"], out["sub_tier"]), (200, "galaxy_s23_plus", "plus"))
        cid = "iphone_13"
        had = cad_harvester.glb_path(cid).exists()
        try:
            with mock.patch.object(cad_harvester, "_try_sketchfab", return_value=None), \
                    mock.patch.object(cad_harvester, "_try_polypizza", return_value=None), \
                    mock.patch.object(cad_harvester, "_try_khronos", return_value=None):
                status, out = self.call("POST", "/api/cad/fetch", {"label": cid})
            self.assertEqual(status, 200, out)
            self.assertTrue(out["glb_url"].endswith(f"{cid}.glb"))
            self.assertTrue(out["credit"])
            row = next(p for p in self.call("GET", "/api/classes")[1]["prototypes"] if p["class_id"] == cid)
            self.assertEqual(row["cad_status"], "ready")
            self.assertEqual(row["cad_model_url"], out["glb_url"])
            self.bank.save()
            proto = torch.load(self.db, map_location="cpu", weights_only=True)["prototypes"][cid]
            self.assertEqual(proto["cad_model_url"], out["glb_url"])
        finally:
            if not had:
                cad_harvester.delete_cached(cid)
                cad_harvester.write_credits_md()

    def test_telemetry_endpoint_reports_real_stats(self):
        self.enroll("Pepsi can", 19, frames=3)
        status, t = self.call("GET", "/api/telemetry")
        self.assertEqual(status, 200)
        for key in ("soc", "memory", "cpu_load", "cpu_freq", "thermal", "uptime_s", "ipc", "modulino", "inference", "privacy", "mode"):
            self.assertIn(key, t)
        self.assertEqual(t["memory"]["nominal_gb"], 16)
        self.assertIn("bridge_rpc", t["ipc"])
        self.assertEqual(t["modulino"]["leds_total"], 8)
        self.assertGreater(t["inference"]["samples"], 0)
        self.assertIsNotNone(t["inference"]["embed_ms_median"])
        if os.path.exists("/proc/meminfo"):
            total_kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
            self.assertAlmostEqual(t["memory"]["total_mb"], total_kb / 1024, delta=2)
            self.assertGreater(t["memory"]["used_pct"], 0)
            self.assertEqual(len(t["cpu_load"]["per_core_pct"]), os.cpu_count())
            self.assertGreater(t["uptime_s"], 0)

    def test_telemetry_stream_emits_server_sent_events(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", "/api/telemetry/stream")
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            self.assertIn("text/event-stream", resp.getheader("Content-Type"))
            line = b""
            while not line.startswith(b"data:"):
                line = resp.readline()
                self.assertTrue(line, "stream closed before any event")
            self.assertIn("memory", json.loads(line[5:]))
        finally:
            conn.close()

    def test_dashboard_is_served_with_the_phase5_panels(self):
        status, body = self.call("GET", "/")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        for needle in ("Resolve &amp; Fetch CAD", "classRows", "ramBar", "model-viewer", "/api/telemetry/stream", "syncBtn"):
            self.assertIn(needle, text)

    def test_preview_frame_is_not_served_in_deployment_mode(self):
        with mock.patch.object(self.receiver, "PREVIEW_FRAME", False):
            self.assertEqual(self.call("GET", "/api/frame.jpg")[0], 404)


class HardwareStatsTests(unittest.TestCase):
    def test_hardware_snapshot_has_expected_shape(self):
        hw = telemetry.hardware()
        for key in ("soc", "thermal", "cpu_freq", "cpu_load", "memory", "uptime_s", "router", "serial"):
            self.assertIn(key, hw)
        json.dumps(hw)

    @unittest.skipUnless(os.path.exists("/proc/meminfo"), "Linux only")
    def test_memory_matches_proc_meminfo(self):
        info = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
        mem = telemetry.memory()
        self.assertAlmostEqual(mem["total_mb"], info["MemTotal"] / 1024, delta=2)
        self.assertAlmostEqual(mem["available_mb"], info["MemAvailable"] / 1024, delta=256)
        self.assertGreater(mem["total_mb"], 8 * 1024)
        self.assertEqual(mem["nominal_gb"], 16)

    @unittest.skipUnless(os.path.exists("/proc/stat"), "Linux only")
    def test_cpu_load_per_core_is_measured(self):
        telemetry.cpu_load()
        sum(i * i for i in range(300000))
        load = telemetry.cpu_load()
        self.assertEqual(len(load["per_core_pct"]), os.cpu_count())
        self.assertTrue(all(0.0 <= v <= 100.0 for v in load["per_core_pct"].values()))
        self.assertTrue(0.0 <= load["total_pct"] <= 100.0)

    @unittest.skipUnless(os.path.isdir("/sys/class/thermal"), "Linux only")
    def test_thermal_zones_are_sane(self):
        zones = telemetry.thermal()
        for z in zones:
            self.assertTrue(-40 < z["temp_c"] < 150, z)

    @unittest.skipUnless(os.path.exists("/proc/cpuinfo"), "Linux only")
    def test_soc_identity(self):
        soc = telemetry.soc_identity()
        self.assertEqual(soc["cores"], os.cpu_count())
        self.assertTrue(soc["kernel"])
        self.assertTrue(soc["arch"])

    def test_percentile(self):
        self.assertIsNone(telemetry.percentile([], 50))
        self.assertEqual(telemetry.percentile([1, 2, 3, 4, 5], 50), 3)


@unittest.skipUnless(os.path.exists(ROUTER_SOCK), "arduino-router socket not present (run on the VENTUNO Q)")
class HardwareLedTests(unittest.TestCase):
    def test_router_socket_is_connectable(self):
        state = telemetry.router_socket(max_age_s=0)
        self.assertTrue(state["exists"])
        self.assertTrue(state["connectable"], state)

    def test_set_pixels_rpc_round_trip(self):
        import socket
        import msgpack
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect(ROUTER_SOCK)
            s.sendall(msgpack.packb([0, 4242, "set_pixels", [0, 255, 0, 50, 8]]))
            reply = msgpack.unpackb(s.recv(1024))
        self.assertEqual(reply[:2], [1, 4242], reply)
        self.assertIsNone(reply[2], f"router/MCU reported an error: {reply}")

    def test_led_states_via_mcp_server(self):
        self.assertTrue(mcp_server.actuate_modulino(0, 255, 0, 50, 8))
        self.assertTrue(mcp_server.actuate_modulino(255, 120, 0, 40, 4))
        self.assertTrue(mcp_server.trigger_guidance("test", "Rotate the object")["action"] == "reorient")
        self.assertTrue(mcp_server.actuate_modulino(0, 100, 150, 15, 1))

    @needs_torch
    def test_led_driver_records_latency_and_status(self):
        import receiver
        drv = receiver.LedDriver(mcp_server.actuate_modulino, heartbeat_s=0)
        drv.set((0, 255, 0, 50, 8))
        deadline = threading.Event()
        for _ in range(50):
            if drv.status().get("rpc_ok", 0) >= 1:
                break
            deadline.wait(0.1)
        st = drv.status()
        self.assertGreaterEqual(st["rpc_ok"], 1, st)
        self.assertIsNotNone(st["latency_avg_ms"])
        self.assertEqual(st["rpc_fail"], 0)
        drv.set((0, 100, 150, 15, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
