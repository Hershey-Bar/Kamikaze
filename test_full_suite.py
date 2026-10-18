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
import time
import unittest
import urllib.parse
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
                terms = p.cad_search_terms
                self.assertTrue(terms[0].lower().replace(" ", "_").endswith(class_id), terms)
                self.assertEqual(terms[-1], "Smartphone")
                self.assertFalse(any("3d model" in t.lower() or "glb" in t.lower() for t in terms), terms)

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

        status, out = self.call("DELETE", "/api/classes/galaxy_s24_ultra?delete_cad=0")
        self.assertEqual((status, out["ok"], out["cad_deleted"]), (200, True, False), out)
        on_disk = torch.load(self.db, map_location="cpu", weights_only=True)
        self.assertEqual(set(on_disk["prototypes"]), {"coca_cola_zero_330ml_can"})
        self.assertEqual(set(on_disk["centroids"]), {"coca_cola_zero_330ml_can"})
        self.assertEqual(set(on_disk["exemplars"]), {"coca_cola_zero_330ml_can"})

        reloaded = self.receiver.EmbeddingBank(self.db)
        self.assertEqual(reloaded.load(), 1)
        self.assertEqual(reloaded.labels(), ["coca_cola_zero_330ml_can"])
        self.assertEqual(self.call("DELETE", "/api/classes/galaxy_s24_ultra?delete_cad=0")[0], 404)
        self.assertEqual(self.call("DELETE", "/api/classes/..%2F..%2Fetc?delete_cad=0")[0] in (400, 404), True)

    def test_delete_removes_cached_model_by_default(self):
        self.enroll("iphone 15 pro", 7)
        cid = "iphone_15_pro"
        keep = cad_harvester.glb_path(cid).exists()
        if keep:
            self.skipTest("a real cached model for iphone_15_pro exists; not deleting it in tests")
        try:
            cad_harvester.harvest(entity_resolver.resolve("iphone 15 pro"), offline=True)
            self.assertTrue(cad_harvester.glb_path(cid).exists())
            self.assertTrue(cad_harvester.sidecar_path(cid).exists())
            status, out = self.call("DELETE", f"/api/classes/{cid}")
            self.assertEqual((status, out["cad_deleted"]), (200, True), out)
            self.assertFalse(cad_harvester.glb_path(cid).exists())
            self.assertFalse(cad_harvester.sidecar_path(cid).exists())
        finally:
            cad_harvester.delete_cached(cid)
            cad_harvester.write_credits_md()

    def test_asset_manager_discards_harvest_that_lands_after_delete(self):
        cid = "zz_test_inflight_can"
        assets = self.receiver.AssetManager()
        started, release = threading.Event(), threading.Event()

        def slow_retrieve(label, meta=None, force=False, offline=False):
            started.set()
            release.wait(5)
            path = cad_harvester.glb_path(label)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"glTF" + b"\0" * 64)
            return {"glb_url": f"/static/models/{label}.glb", "title": label, "source": "test"}

        resolved = []
        assets.on_resolved = lambda label, info: resolved.append(label)
        try:
            with mock.patch.object(cad_harvester, "retrieve", side_effect=slow_retrieve):
                assets.get(cid)
                self.assertTrue(started.wait(5))
                assets.forget(cid)
                release.set()
                for _ in range(100):
                    with assets._lock:
                        if cid not in assets._inflight:
                            break
                    threading.Event().wait(0.02)
            self.assertEqual(resolved, [])
            self.assertFalse(cad_harvester.glb_path(cid).exists())
        finally:
            cad_harvester.delete_cached(cid)

    def test_reset_empties_the_database_file(self):
        self.enroll("Pepsi can", 11)
        self.enroll("Sprite can", 13)
        self.bank.save()
        status, out = self.call("POST", "/api/classes/reset", {"delete_cad": False})
        self.assertEqual((status, out["ok"], sorted(out["removed"]), out["cad_deleted"]),
                         (200, True, ["pepsi_can", "sprite_can"], []))
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
        for needle in ("Resolve &amp; Fetch CAD", "classRows", "ramBar", "model-viewer", "/api/events?telemetry=1", "streamFeed", "syncBtn"):
            self.assertIn(needle, text)

    def test_preview_frame_is_not_served_in_deployment_mode(self):
        with mock.patch.object(self.receiver, "PREVIEW_FRAME", False):
            self.assertEqual(self.call("GET", "/api/frame.jpg")[0], 404)

    def _until(self, cond, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.02)
        return False

    def test_ingest_answers_fast_and_worker_infers_newest_frame(self):
        self.assertEqual(self.call("POST", "/api/ingest", b"junk")[0], 400)
        frame = self.tc.jpeg()
        t0 = time.perf_counter()
        for _ in range(20):
            status, data = self.call("POST", "/api/ingest", frame)
            self.assertEqual(status, 200, data)
            self.assertTrue(data["queued"])
        per_post_ms = (time.perf_counter() - t0) / 20 * 1000
        self.assertTrue(self._until(lambda: (lambda s: s["processed"] + s["dropped"] == 20)(self.pipe.stream_stats())),
                        self.pipe.stream_stats())
        st = self.pipe.stream_stats()
        self.assertEqual((st["ingested"], st["decode_errors"]), (20, 0))
        self.assertGreaterEqual(st["processed"], 1)
        _, state = self.call("GET", "/api/state")
        self.assertTrue(state["stream"]["live"])
        self.assertGreater(state["ingest_fps"], 0)
        self.assertEqual(self.call("GET", "/api/frame.jpg")[1], frame)
        print(f"\n  /api/ingest: {per_post_ms:.2f} ms per POST (loopback), {st['processed']} inferred, {st['dropped']} superseded")

    def test_idle_stream_drops_preview_and_zeroes_fps(self):
        for _ in range(3):
            self.call("POST", "/api/ingest", self.tc.jpeg())
        self.assertEqual(self.call("GET", "/api/frame.jpg")[0], 200)
        with mock.patch.object(self.receiver, "STALE_S", 0.05):
            time.sleep(0.15)
            self.assertEqual(self.call("GET", "/api/frame.jpg")[0], 404)
            _, state = self.call("GET", "/api/state")
            self.assertFalse(state["stream"]["live"])
            self.assertEqual((state["fps"], state["ingest_fps"]), (0.0, 0.0))
            self.assertFalse(state["privacy"]["preview_frame_in_ram"])

    def test_mjpeg_stream_pushes_each_new_frame(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        frame = self.tc.jpeg()
        try:
            conn.request("GET", "/api/stream")
            resp = conn.getresponse()
            self.assertIn("multipart/x-mixed-replace", resp.getheader("Content-Type"))
            threading.Timer(0.2, lambda: self.pipe.ingest(frame)).start()
            self.assertEqual(resp.readline(), b"--kzframe\r\n")
            headers = {}
            while (line := resp.readline()) != b"\r\n":
                key, value = line.decode().split(":", 1)
                headers[key.strip().lower()] = value.strip()
            self.assertEqual(headers["content-type"], "image/jpeg")
            self.assertEqual(resp.read(int(headers["content-length"])), frame)
        finally:
            conn.close()


    def _post_raw(self, path, body, ctype):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("POST", path, body=body, headers={"Content-Type": ctype})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read())
        finally:
            conn.close()

    def test_every_ingest_alias_and_payload_format_reaches_the_live_preview(self):
        import base64
        frame = self.tc.jpeg()
        b64 = base64.b64encode(frame).decode()
        boundary = "kzTestBoundary"
        multipart = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"frame\"; filename=\"f.jpg\"\r\n"
                     f"Content-Type: image/jpeg\r\n\r\n").encode() + frame + f"\r\n--{boundary}--\r\n".encode()
        cases = [("/", frame, "image/jpeg", "jpeg"), ("/upload", frame, "application/octet-stream", "jpeg"),
                 ("/frame", multipart, f"multipart/form-data; boundary={boundary}", "multipart/jpeg"),
                 ("/ingest", json.dumps({"image": "data:image/jpeg;base64," + b64}).encode(), "application/json", "json-base64/jpeg"),
                 ("/api/upload", ("frame=" + urllib.parse.quote(b64)).encode(), "application/x-www-form-urlencoded", "form-base64/jpeg"),
                 ("/api/ingest/", b64.encode(), "text/plain", "base64/jpeg")]
        for i, (path, body, ctype, fmt) in enumerate(cases, 1):
            status, data = self._post_raw(path, body, ctype)
            self.assertEqual(status, 200, (path, data))
            self.assertTrue(data["queued"], path)
            st = self.pipe.stream_stats()
            self.assertEqual((st["ingested"], st["source"]["path"], st["source"]["format"]), (i, path.rstrip("/") or "/", fmt))
        self.assertEqual(self.call("GET", "/api/frame.jpg")[1], frame)
        status, data = self._post_raw("/api/frame", multipart, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(status, 200)
        self.assertNotIn("queued", data)  # /api/frame stays synchronous
        status, data = self._post_raw("/", b"<html>not a frame</html>", "text/html")
        self.assertEqual(status, 400)
        self.assertIn("not a JPEG frame", data["error"])
        self.assertEqual(self.pipe.stream_stats()["rejected"], 1)
        status, info = self.call("GET", "/api/ingest")
        self.assertEqual(status, 200)
        self.assertIn("/upload", info["async_routes"])
        self.assertEqual(info["stream"]["rejected"], 1)

    def test_events_stream_carries_state_and_telemetry_on_one_connection(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", "/api/events?telemetry=1")
            resp = conn.getresponse()
            self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
            self.assertEqual(resp.readline(), b"event: telemetry\n")
            self.assertIn("stream", json.loads(resp.readline()[len(b"data: "):]))
        finally:
            conn.close()


class RamPoolTests(unittest.TestCase):
    def test_static_cache_hits_revalidates_and_evicts(self):
        import receiver
        tmp = Path(tempfile.mkdtemp(prefix="kz_cache_"))
        try:
            a, b = tmp / "a.glb", tmp / "b.glb"
            a.write_bytes(b"x" * 600)
            b.write_bytes(b"y" * 600)
            cache = receiver.StaticCache(1000)
            self.assertEqual(cache.read(a), b"x" * 600)
            self.assertEqual(cache.read(a), b"x" * 600)
            self.assertEqual((cache.hits, cache.misses), (1, 1))
            a.write_bytes(b"z" * 500)
            self.assertEqual(cache.read(a), b"z" * 500)
            self.assertEqual(cache.misses, 2)
            cache.read(b)
            self.assertEqual(cache.info()["files"], 1)
            b.unlink()
            self.assertEqual(cache.info()["files"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_frame_ring_wraps_newest_first_and_is_off_by_default(self):
        import receiver
        from PIL import Image
        ring = receiver.FrameRing(4 * 4 * 3 * 3)
        for v in range(5):
            ring.push(Image.new("RGB", (4, 4), (v, v, v)))
        self.assertEqual(len(ring), 3)
        self.assertEqual([int(ring.get(i)[0, 0, 0]) for i in range(3)], [4, 3, 2])
        self.assertIsNone(ring.get(3))
        self.assertEqual((ring.info()["capacity"], ring.info()["frames"]), (3, 3))
        ring.clear()
        self.assertEqual(len(ring), 0)
        self.assertFalse(receiver.FrameRing(0).enabled)


class FramePayloadTests(unittest.TestCase):
    def test_extract_frame_normalises_common_streamer_payloads(self):
        import base64
        import io
        import receiver
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (32, 24), (200, 30, 30)).save(buf, "JPEG")
        jpeg = buf.getvalue()
        png = io.BytesIO()
        Image.new("RGB", (32, 24), (10, 200, 30)).save(png, "PNG")
        b64 = base64.b64encode(jpeg)
        self.assertEqual(receiver.extract_frame(jpeg), (jpeg, "jpeg"))
        self.assertEqual(receiver.extract_frame(b"--xyz\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n--xyz--\r\n"),
                         (jpeg, "multipart/jpeg"))
        self.assertEqual(receiver.extract_frame(b'{"frame": "' + b64 + b'"}', "application/json"), (jpeg, "json-base64/jpeg"))
        self.assertEqual(receiver.extract_frame(base64.urlsafe_b64encode(jpeg).rstrip(b"=")), (jpeg, "base64/jpeg"))
        out, fmt = receiver.extract_frame(png.getvalue(), "image/png")
        self.assertEqual((out[:3], fmt), (receiver.JPEG_SOI, "transcoded"))
        for bad in (b"", b"junk", b'{"nothing": 1}', b"--b\r\n\r\nno image\r\n--b--", b"\x89PNG\r\n\x1a\n" + b"0" * 40):
            with self.assertRaises(ValueError, msg=bad):
                receiver.extract_frame(bad)


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


class SearchQueryNormalizerTests(unittest.TestCase):
    def test_directive_examples(self):
        coke = cad_harvester.build_search_queries("coke zero can 330ml")
        self.assertEqual(coke[0], "Coca Cola Zero Can")
        self.assertIn("Coke Zero 330ml Can", coke)
        self.assertEqual(coke[-1], "Soda Can")
        s25 = cad_harvester.build_search_queries("s25 ultra")
        self.assertEqual((s25[0], s25[-1]), ("Samsung Galaxy S25 Ultra", "Smartphone"))
        self.assertIn("Samsung S25 Ultra", s25)
        self.assertIn("Galaxy S24 Ultra", s25)
        ip = cad_harvester.build_search_queries("iphone 15 pro max")
        self.assertEqual((ip[0], ip[-1]), ("iPhone 15 Pro Max", "Smartphone"))
        self.assertIn("iPhone 15 Pro", ip)

    def test_queries_are_unique_and_specific_first(self):
        for text in ("coke zero can 330ml", "s25 ultra", "iphone 15 pro max", "pepsi can", "galaxy s23+"):
            with self.subTest(text=text):
                q = cad_harvester.build_search_queries(text)
                self.assertEqual(len(q), len({x.lower() for x in q}), q)
                levels = [lvl for _, lvl in cad_harvester.search_plan(entity_resolver.resolve(text))]
                self.assertEqual(levels[0], "exact")
                self.assertEqual(levels[-1], "category")

    def test_match_score_levels(self):
        s24u = entity_resolver.resolve("Samsung Galaxy S24 Ultra")
        self.assertIsNotNone(cad_harvester.match_score(s24u, "Samsung Galaxy S24 Ultra", "exact"))
        self.assertIsNone(cad_harvester.match_score(s24u, "Samsung Galaxy S23", "exact"))
        self.assertIsNone(cad_harvester.match_score(s24u, "Galaxy S24 Ultra phone case", "exact"))
        for accessory in ("Samsung S24 Ultra S-pen", "Galaxy S24 Ultra Stylus", "Samsung Galaxy S24 Ultra S Pen"):
            for level in ("exact", "similar", "category"):
                self.assertIsNone(cad_harvester.match_score(s24u, accessory, level), (accessory, level))
        self.assertIsNotNone(cad_harvester.match_score(s24u, "Galaxy S23 Ultra", "similar"))
        self.assertIsNotNone(cad_harvester.match_score(s24u, "Smartphone", "category"))
        self.assertIsNone(cad_harvester.match_score(s24u, "Soda Can", "category"))
        coke = entity_resolver.resolve("coke zero can 330ml")
        self.assertIsNotNone(cad_harvester.match_score(coke, "Soda Can", "category"))
        self.assertIsNone(cad_harvester.match_score(coke, "Pepsi Can", "exact"))

    def test_polypizza_ranking_keeps_author_and_licence(self):
        product = entity_resolver.resolve("coke zero can 330ml")
        results = [
            {"ID": "abc123", "Title": "Soda Can", "Licence": "CC-BY 3.0",
             "Attribution": "Soda Can by Jane Doe [CC-BY] (https://creativecommons.org/licenses/by/3.0/) via Poly Pizza",
             "Download": "https://static.poly.pizza/abc123.glb", "Thumbnail": "https://static.poly.pizza/abc123.webp",
             "Tri Count": 500, "Creator": {"Username": "Jane Doe"}},
            {"ID": "nc1", "Title": "Soda Can", "Licence": "CC-BY-NC", "Download": "https://x/nc.glb",
             "Creator": {"Username": "nc"}},
            {"ID": "nodl", "Title": "Soda Can", "Licence": "CC0", "Creator": {"Username": "z"}},
        ]
        ranked = cad_harvester.rank_polypizza(product, results, "category")
        self.assertEqual([c.uid for c in ranked], ["abc123"])
        c = ranked[0]
        self.assertEqual((c.author, c.author_url), ("Jane Doe", "https://poly.pizza/u/Jane%20Doe"))
        self.assertEqual(c.license_url, "https://creativecommons.org/licenses/by/3.0/")
        self.assertEqual(c.source_url, "https://poly.pizza/m/abc123")
        self.assertIn("Jane Doe", c.attribution_text)

    def test_polypizza_endpoint_is_v11_path_search(self):
        self.assertEqual(cad_harvester.POLY_PIZZA_SEARCH.format(query="Soda%20Can"),
                         "https://api.poly.pizza/v1.1/search/Soda%20Can")


class NpuMathTests(unittest.TestCase):
    """Pure-NumPy parts of npu.py (anchors, decode, NMS, crop selection, tracking)."""

    @classmethod
    def setUpClass(cls):
        import npu
        cls.npu = npu

    def test_anchor_counts_match_efficientdet_lite(self):
        self.assertEqual(self.npu.generate_anchors(320).shape, (19206, 4))
        self.assertEqual(self.npu.generate_anchors(448).shape, (37629, 4))
        a = self.npu.generate_anchors(320)
        self.assertTrue(((a[:, :2] > 0) & (a[:, :2] < 1)).all())
        self.assertTrue((a[:, 2:] > 0).all())

    def test_decode_zero_encoding_returns_anchor_box(self):
        import numpy as np
        anchors = np.array([[0.5, 0.5, 0.2, 0.4]], dtype=np.float32)
        box = self.npu.decode_boxes(np.zeros((1, 4), dtype=np.float32), anchors)[0]
        np.testing.assert_allclose(box, [0.4, 0.3, 0.6, 0.7], atol=1e-6)

    def test_nms_suppresses_overlaps(self):
        import numpy as np
        boxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.11, 0.11, 0.51, 0.51], [0.6, 0.6, 0.9, 0.9]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        self.assertEqual(self.npu.nms(boxes, scores, 0.5), [0, 2])
        self.assertEqual(self.npu.nms(boxes, scores, 0.5, limit=1), [0])

    def test_target_classes_map_to_categories(self):
        D = self.npu.Detection
        self.assertEqual(D("cell phone", 0.9, (0, 0, 1, 1)).category, "smartphone")
        self.assertEqual(D("bottle", 0.9, (0, 0, 1, 1)).category, "beverage")
        self.assertIsNone(D("person", 0.9, (0, 0, 1, 1)).category)
        self.assertEqual(len(self.npu.COCO_LABELS), 90)
        self.assertEqual(len(self.npu.COCO80_LABELS), 80)
        c80 = self.npu.COCO80_LABELS
        self.assertEqual((c80[0], c80[39], c80[41], c80[65], c80[67], c80[79]),
                         ("person", "bottle", "cup", "remote", "cell phone", "toothbrush"))

    def test_default_detector_is_yolo(self):
        self.assertTrue(self.npu.DETECTOR_VARIANT.startswith("yolo") or "KAMIKAZE_DETECTOR" in os.environ)
        self.assertEqual(self.npu.detector_model_path("yolo26n").name, "yolo26n_float32.tflite")
        self.assertEqual(self.npu.detector_model_path("my_cans").name, "my_cans_float32.tflite")
        self.assertEqual(self.npu.detector_model_path("lite2").name, "efficientdet_lite2_int8.tflite")
        with self.assertRaises(FileNotFoundError):
            self.npu.ensure_model(self.npu.yolo_spec("definitely_missing"), Path(tempfile.mkdtemp(prefix="kz_y_")))

    def _yolo_head(self, rows, n=8400, nc=80, layout="channels_first"):
        import numpy as np
        out = np.zeros((4 + nc, n), dtype=np.float32)
        for i, (cx, cy, w, h, cls, score) in enumerate(rows):
            out[:4, i] = (cx, cy, w, h)
            out[4 + cls, i] = score
        return out[None] if layout == "channels_first" else out.T[None]

    def test_decode_yolo_pixels_and_layouts(self):
        import numpy as np
        ids = self.npu._target_ids(self.npu.COCO80_LABELS, False)
        self.assertEqual(sorted(self.npu.COCO80_LABELS[i] for i in ids), ["bottle", "cell phone", "cup", "remote"])
        rows = [(320, 320, 128, 256, 67, 0.9),    # phone
                (322, 318, 130, 250, 67, 0.7),    # duplicate -> NMS
                (100, 100, 64, 64, 0, 0.99),      # person: not a target class
                (500, 500, 40, 120, 39, 0.5),     # bottle
                (50, 600, 20, 20, 41, 0.1)]       # cup under threshold
        for layout in ("channels_first", "channels_last"):
            with self.subTest(layout=layout):
                dets = self.npu.decode_yolo(self._yolo_head(rows, layout=layout), ids, 0.3, 0.5, 640)
                self.assertEqual([(c, round(s, 2)) for c, s, _ in dets], [(67, 0.9), (39, 0.5)])
                np.testing.assert_allclose(dets[0][2], [0.4, 0.3, 0.6, 0.7], atol=1e-6)
        norm = [(0.5, 0.5, 0.2, 0.4, 67, 0.8)]
        dets = self.npu.decode_yolo(self._yolo_head(norm), ids, 0.3, 0.5, 640)
        np.testing.assert_allclose(dets[0][2], [0.4, 0.3, 0.6, 0.7], atol=1e-6)
        self.assertEqual(self.npu.decode_yolo(self._yolo_head([]), ids, 0.3, 0.5, 640), [])

    def test_letterbox_round_trip(self):
        from PIL import Image
        img = Image.new("RGB", (640, 480), (10, 200, 30))
        canvas, scale, left, top = self.npu.letterbox(img, 320)
        self.assertEqual(canvas.shape, (320, 320, 3))
        self.assertEqual((round(scale, 3), left, top), (0.5, 0, 40))
        self.assertEqual(tuple(canvas[0, 0]), (114, 114, 114))
        self.assertEqual(tuple(canvas[160, 160]), (10, 200, 30))
        # a box covering the whole image area inside the letterbox maps back to the full frame
        box = (0.0, 40 / 320, 1.0, 280 / 320)
        for got, want in zip(self.npu.unletterbox(box, 320, scale, left, top, 640, 480), (0.0, 0.0, 1.0, 1.0)):
            self.assertAlmostEqual(got, want, places=5)
        half = self.npu.unletterbox((0.25, 0.5, 0.75, 0.75), 320, scale, left, top, 640, 480)
        for got, want in zip(half, (0.25, (160 - 40) / 240, 0.75, (240 - 40) / 240)):
            self.assertAlmostEqual(got, want, places=5)

    def test_pick_target_prefers_centred_target_classes(self):
        D = self.npu.Detection
        dets = [D("person", 0.99, (0.3, 0.3, 0.7, 0.7)),
                D("cell phone", 0.95, (0.0, 0.0, 0.1, 0.1)),
                D("cell phone", 0.60, (0.4, 0.35, 0.6, 0.65)),
                D("bottle", 0.30, (0.45, 0.4, 0.55, 0.6))]
        best = self.npu.pick_target(dets, 640, 480)
        self.assertIsNotNone(best)
        self.assertEqual((best.label, best.score), ("cell phone", 0.60))
        self.assertIsNone(self.npu.pick_target(dets[:2], 640, 480))

    def test_square_crop_stays_inside_frame(self):
        D = self.npu.Detection
        for box in [(0.4, 0.3, 0.6, 0.7), (0.0, 0.0, 0.05, 0.05), (0.9, 0.8, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)]:
            with self.subTest(box=box):
                size, left, top = self.npu.square_crop(D("bottle", 0.9, box), 640, 480)
                self.assertGreaterEqual(size, int(0.25 * 480))
                self.assertLessEqual(size, 480)
                self.assertTrue(0 <= left <= 640 - size and 0 <= top <= 480 - size, (size, left, top))

    def test_crop_tracker_smooths_holds_and_releases(self):
        t = self.npu.CropTracker(alpha=0.5, hold_frames=2, jump=0.5)
        self.assertEqual(t.update((200, 100, 100), 640, 480), (200, 100, 100))
        size, left, top = t.update((200, 120, 100), 640, 480)
        self.assertEqual((size, left, top), (200, 110, 100))
        self.assertEqual(t.update(None, 640, 480), (200, 110, 100))
        self.assertEqual(t.update(None, 640, 480), (200, 110, 100))
        self.assertIsNone(t.update(None, 640, 480))
        t.update((200, 0, 0), 640, 480)
        self.assertEqual(t.update((200, 400, 200), 640, 480), (200, 400, 200))

    def test_npu_available_reports_reason(self):
        ok, reason = self.npu.npu_available()
        self.assertIsInstance(ok, bool)
        self.assertTrue(reason)

    def test_telemetry_npu_status_shape(self):
        st = telemetry.npu_status([{"type": "nsp0-thermal", "temp_c": 41.0}, {"type": "cpu0", "temp_c": 60.0}])
        self.assertEqual(set(st), {"fastrpc_cdsp", "cdsp", "nsp_temp_c", "nsp_zones"})
        self.assertEqual((st["nsp_temp_c"], st["nsp_zones"]), (41.0, 1))
        self.assertIn("npu", telemetry.hardware())


@needs_torch
class EmbedderMismatchTests(unittest.TestCase):
    def test_bank_from_another_embedder_is_backed_up_not_mixed(self):
        import receiver
        tmp = Path(tempfile.mkdtemp(prefix="kz_emb_"))
        try:
            db = tmp / "enrolled_classes.pt"
            old = receiver.EmbeddingBank(db, "mobilenet_v3_large-fc1280")
            old.add("pepsi_can", [torch.nn.functional.normalize(torch.ones(8), dim=0)])
            old.save()
            fresh = receiver.EmbeddingBank(db, "mediapipe-mobilenet_v3_large-embedder-1280")
            self.assertEqual(fresh.load(), 0)
            self.assertEqual(fresh.labels(), [])
            backups = list(tmp.glob("enrolled_classes.*.bak.pt"))
            self.assertEqual(len(backups), 1, backups)
            same = receiver.EmbeddingBank(db, "mobilenet_v3_large-fc1280")
            self.assertEqual(same.load(), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _board_npu() -> tuple[bool, str]:
    try:
        import npu
    except Exception as exc:
        return False, str(exc)
    ok, why = npu.npu_available()
    if not ok:
        return False, why
    files = [npu.MODELS_DIR / npu.EMBEDDER_SPEC.file, npu.detector_model_path()]
    missing = [f.name for f in files if not f.is_file()]
    return (not missing), (f"models missing: {missing}" if missing else "ok")


_NPU_OK, _NPU_WHY = _board_npu()


@unittest.skipUnless(_NPU_OK, f"Hexagon NPU not available: {_NPU_WHY}")
class HexagonNpuLiveTests(unittest.TestCase):
    """Runs the real TFLite models through the QNN HTP delegate on the VENTUNO Q."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import npu
        cls.np, cls.npu = np, npu
        cls.embedder = npu.NpuEmbedder()
        cls.detector = npu.make_detector()

    def test_embedder_runs_on_npu_and_is_unit_norm(self):
        import time
        np = self.np
        self.assertEqual(self.embedder.device, "npu", self.embedder.info())
        x = np.random.default_rng(0).random((1, 224, 224, 3), dtype=np.float32)
        v = self.embedder.embed_rgb01(x)
        self.assertEqual(v.shape, (1280,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=4)
        t0 = time.perf_counter()
        for _ in range(20):
            self.embedder.embed_rgb01(x)
        ms = (time.perf_counter() - t0) * 50
        print(f"\n  NPU embedder: {ms:.2f} ms/frame")
        self.assertLess(ms, 15.0)

    def test_embedder_is_deterministic_and_discriminative(self):
        np = self.np
        rng = np.random.default_rng(1)
        a, b = rng.random((1, 224, 224, 3), dtype=np.float32), np.zeros((1, 224, 224, 3), dtype=np.float32)
        va1, va2, vb = self.embedder.embed_rgb01(a), self.embedder.embed_rgb01(a), self.embedder.embed_rgb01(b)
        self.assertGreater(float(va1 @ va2), 0.999)
        self.assertLess(float(va1 @ vb), 0.99)

    def test_detector_runs_on_npu(self):
        import time
        np = self.np
        self.assertEqual(self.detector.device, "npu", self.detector.info())
        frame = np.full((self.detector.size, self.detector.size, 3), 127, dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(10):
            dets = self.detector.detect_array(frame)
        ms = (time.perf_counter() - t0) * 100
        print(f"\n  NPU detector ({self.detector.variant}): {ms:.2f} ms/frame, {len(dets)} detections on grey")
        self.assertIsInstance(dets, list)
        self.assertLess(ms, 40.0)

    def test_yolo_detects_on_ultralytics_sample(self):
        import glob
        import importlib.util
        from PIL import Image
        if not self.npu.DETECTOR_VARIANT.startswith("yolo"):
            self.skipTest("detector is not YOLO")
        spec = importlib.util.find_spec("ultralytics")
        if spec is None:
            self.skipTest("ultralytics not installed")
        bus = glob.glob(str(Path(spec.origin).parent / "assets" / "bus.jpg"))
        if not bus:
            self.skipTest("ultralytics sample image missing")
        det = self.npu.YoloDetector(all_classes=True)
        self.assertEqual(det.device, "npu", det.info())
        img = Image.open(bus[0]).convert("RGB")
        found = det.detect(img)
        labels = {d.label for d in found}
        self.assertIn("person", labels, found)
        self.assertIn("bus", labels, found)
        for d in found:
            self.assertTrue(0.0 <= d.box[0] < d.box[2] <= 1.0 and 0.0 <= d.box[1] < d.box[3] <= 1.0, d)
        best_bus = max((d for d in found if d.label == "bus"), key=lambda d: d.score)
        self.assertGreater(best_bus.box[2] - best_bus.box[0], 0.7)  # the bus spans most of the 810x1080 frame width


if __name__ == "__main__":
    unittest.main(verbosity=2)
