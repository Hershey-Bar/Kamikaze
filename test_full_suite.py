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
import struct
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
import device_taxonomy
import entity_resolver
import mcp_server
import procedural_glb
import spec_search
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
        with self.assertRaises(taxonomy.MetadataError):
            entity_resolver.resolve("grapefruit seltzer")  # a flavour alone names no product

    def test_store_brand_flavour_resolution(self):
        p = entity_resolver.resolve("Kirkland Grapefruit")
        self.assertEqual((p.class_id, p.brand, p.category, p.flavor), ("kirkland_grapefruit_can", "Kirkland",
                                                                       "beverage_can", "grapefruit"))
        plan = cad_harvester.search_plan(p)
        self.assertEqual(plan[0], ("Kirkland Grapefruit Can", "exact"))
        self.assertIn(("Kirkland Can", "similar"), plan)
        self.assertIn(("Grapefruit Seltzer Can", "similar"), plan)
        self.assertEqual(plan[-1], ("Soda Can", "category"))
        self.assertFalse(any("coca" in q.lower() or "coke" in q.lower() for q, _ in plan), plan)
        stored = {"raw_name": "Kirkland Sparkling Water Grapefruit", "category": "beverage_can", "brand": "Kirkland",
                  "variant": "sparkling water grapefruit", "package_type": "can", "container_volume": ""}
        s = entity_resolver.resolve_metadata(stored)
        self.assertEqual((s.class_id, s.drink_type), ("kirkland_sparkling_water_grapefruit_can", "sparkling water"))
        self.assertEqual(cad_harvester.search_plan(s)[0], ("Kirkland Grapefruit Sparkling Water Can", "exact"))
        self.assertEqual(entity_resolver.resolve("Pepsi Lime").class_id, "pepsi_lime_can")
        self.assertEqual(entity_resolver.resolve("Sprite lemon lime can").class_id, "sprite_can")
        self.assertEqual(entity_resolver.resolve("La Croix lime").brand, "La Croix")


class CadRankerTests(unittest.TestCase):
    @staticmethod
    def model(uid, name, lic="CC Attribution", faces=20000, likes=10, user="Alice", downloadable=True, views=0,
              staff=False, tags=(), description=""):
        return {"uid": uid, "name": name, "license": {"label": lic, "uid": "x"}, "faceCount": faces, "likeCount": likes,
                "viewCount": views, "staffpickedAt": "2020-01-01" if staff else None,
               "tags": [{"name": t, "slug": t} for t in tags], "description": description,
                "isDownloadable": downloadable, "user": {"displayName": user, "profileUrl": f"https://sketchfab.com/{user}"},
                "thumbnails": {"images": []}}

    def rank(self, text, *models, level="exact"):
        return cad_harvester.rank_sketchfab(entity_resolver.resolve(text), list(models), level)

    def test_flavoured_store_brand_never_takes_another_brand_or_flavour(self):
        live = [  # the real Sketchfab "soda can" results (likes / views as of 2026-10)
            self.model("orange", "Orange Soda", likes=306, views=15460, staff=True, faces=61878),
            self.model("cola", "Simple Cola Can", likes=556, views=60626, faces=3290),
            self.model("coke", "Coca Cola Soda Can", likes=88, views=19251, faces=2406),
            self.model("soda", "Soda Can", likes=105, views=21151, faces=490),
            self.model("pack", "Soda Cans 3D pack", likes=223, views=9396, faces=110000),
            self.model("diet", "Diet Soda", likes=105, views=21151, faces=490),
        ]
        out = self.rank("Kirkland Grapefruit", *live, level="category")
        self.assertEqual(out[0].uid, "soda", [(c.uid, round(c.score, 1)) for c in out])
        self.assertNotIn("coke", [c.uid for c in out])
        self.assertNotIn("cola", [c.uid for c in out])  # a cola livery is a different drink, not a sparkling water
        self.assertNotIn("pack", [c.uid for c in out])
        self.assertEqual(out[0].extra["likes"], 105)
        # the same pool for Coke Zero: the same-brand can ranks as "similar"; cola is its own drink type so the popular
        # cola can beats the (staff-picked) orange soda
        coke = [c.uid for c in self.rank("coke zero can", *live, level="category")]
        self.assertEqual(coke[:2], ["coke", "cola"])
        self.assertLess(coke.index("cola"), coke.index("orange"))
        exact = self.rank("Kirkland Grapefruit", self.model("a", "Kirkland Lime Sparkling Water Can"),
                          self.model("b", "Kirkland Grapefruit Sparkling Water Can"))
        self.assertEqual([c.uid for c in exact], ["b"])
        similar = self.rank("Kirkland Grapefruit", self.model("a", "Kirkland Lime Sparkling Water Can"),
                            self.model("g", "Grapefruit Seltzer Can"), self.model("c", "Grapefruit Cola Can"), level="similar")
        self.assertEqual(sorted(c.uid for c in similar), ["a", "g"])

    def test_brand_hidden_in_tags_or_description_is_rejected(self):
        live = [  # real Sketchfab "soda can" results whose generic titles hide a branded texture
            self.model("junhyde", "Soda Can", likes=105, views=21151, faces=490, description="A simple old Pepsi Soda can"),
            self.model("aisaac", "Soda Can", likes=80, views=9000, faces=1200, tags=("pepsi", "mtdew", "dew", "can")),
            self.model("youssef", "Soda can", likes=45, views=3000, faces=2000,
                       tags=("can", "soda", "sodacan", "softdrink", "crushed")),
            self.model("sunich", "Soda Can", likes=57, views=4000, faces=1500, tags=("orange", "lemon", "can")),
            self.model("sweetwood", "Soda Can (High Poly)", likes=221, views=30000, faces=2894,
                       tags=("high", "pop", "can", "cola", "soda")),
            self.model("plain", "Soda Can", likes=30, views=2000, faces=1000, tags=("can", "aluminium", "drink")),
        ]
        out = [c.uid for c in self.rank("Kirkland Grapefruit", *live, level="category")]
        for bad in ("junhyde", "aisaac", "youssef", "sweetwood"):
            self.assertNotIn(bad, out)
        self.assertEqual(out, ["plain", "sunich"])
        # the product's own brand in the metadata is fine, and a title naming the brand is trusted over its description
        pepsi = [c.uid for c in self.rank("pepsi can", *live, level="category")]
        self.assertIn("junhyde", pepsi)
        self.assertIn("sweetwood", [c.uid for c in self.rank("coke zero can", *live, level="category")])
        own = self.rank("pepsi can", self.model("p", "Pepsi Can", description="Pepsi vs Coca-Cola comparison"))
        self.assertEqual([c.uid for c in own], ["p"])
        phone = self.rank("iphone 15 pro", self.model("x", "Smartphone", tags=("samsung", "galaxy")),
                          self.model("y", "Smartphone", tags=("iphone", "apple")), level="category")
        self.assertNotIn("x", [c.uid for c in phone])
        poly = cad_harvester.rank_polypizza(entity_resolver.resolve("Kirkland Grapefruit"), [
            {"ID": "p1", "Title": "Soda Can", "Licence": "CC0", "Download": "https://x/p1.glb", "Tags": ["pepsi"]},
            {"ID": "p2", "Title": "Soda Can", "Licence": "CC0", "Download": "https://x/p2.glb", "Tags": ["can"]},
        ], level="category")
        self.assertEqual([c.uid for c in poly], ["p2"])

    def test_popularity_is_log_scaled_and_capped(self):
        pop = cad_harvester.popularity
        self.assertLess(pop(10, 1000), pop(100, 1000))
        self.assertLess(pop(100, 1000) - pop(10, 1000), 6.0)
        self.assertGreater(pop(10, 1000, True), pop(10, 1000))
        self.assertEqual(pop(10 ** 9, 10 ** 12, True), cad_harvester.POPULARITY_CAP)
        out = self.rank("pepsi can", self.model("a", "Pepsi can", likes=2, views=50),
                        self.model("b", "Pepsi can", likes=400, views=40000))
        self.assertEqual([c.uid for c in out], ["b", "a"])

    def test_level_pool_downloads_best_ranked_across_providers(self):
        prod = entity_resolver.resolve("pepsi can")
        sf = [cad_harvester.Candidate("sketchfab", "Pepsi can", 70.0, uid="s1"),
              cad_harvester.Candidate("sketchfab", "Pepsi can", 40.0, uid="s2")]
        pp = [cad_harvester.Candidate("polypizza", "Pepsi Can", 55.0, uid="p1", download_url="https://x/p1.glb")]
        tried = []
        with mock.patch.object(cad_harvester, "_search_sketchfab", return_value=sf), \
                mock.patch.object(cad_harvester, "_search_polypizza", return_value=pp), \
                mock.patch.object(cad_harvester, "_download_and_validate",
                                  side_effect=lambda c, slug: tried.append(c.uid) or c.uid == "p1"):
            cand, source = cad_harvester._try_online(prod, "x", time.monotonic() + 30, ["Pepsi Can"], "exact")
        self.assertEqual((cand.uid, source, tried), ("p1", "Poly Pizza", ["s1", "p1"]))

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


def glb_doc(data: bytes) -> dict:
    n = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + n])


def check_glb_bytes(data: bytes) -> tuple[bool, str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.glb"
        path.write_bytes(data)
        return cad_harvester.validate_glb(path)


class SpecGlbTests(unittest.TestCase):
    """Generated models only exist for verified published dimensions; primitive placeholders fail the quality gate."""
    S25U = {"verified": True, "category": "smartphone", "maker": "Samsung", "name": "Samsung Galaxy S25 Ultra",
            "height_mm": 162.8, "width_mm": 77.6, "depth_mm": 8.2, "rear_cameras": 4,
            "sources": [{"url": "https://www.gsmarena.com/samsung_galaxy_s25_ultra-13322.php", "publisher": "GSMArena"}]}

    def test_can_from_crown_dimensions(self):
        spec = spec_search.can_specs("coke zero can 330ml")
        data = procedural_glb.build_from_spec(spec, "coke zero can 330ml")
        ok, why, info = check_glb_bytes(data)
        self.assertTrue(ok, why)
        extras = glb_doc(data)["asset"]["extras"]
        self.assertEqual(extras["kind"], procedural_glb.SPEC_TAG)
        self.assertEqual((extras["dimensions_mm"]["diameter"], extras["dimensions_mm"]["height"]), (66.0, 115.2))
        self.assertTrue(extras["sources"][0]["url"].startswith("https://www.crowncork.com/"))

    def test_phone_from_gsmarena_dimensions(self):
        data = procedural_glb.build_from_spec(self.S25U, "galaxy s25 ultra")
        ok, why, info = check_glb_bytes(data)
        self.assertTrue(ok, why)
        extras = glb_doc(data)["asset"]["extras"]
        self.assertEqual((extras["rear_cameras"], extras["dimensions_mm"]["height"]), (4, 162.8))

    def test_primitive_box_and_non_glb_are_rejected(self):
        m = procedural_glb._Mesh()
        m.box((0, 0, 0), (0.07, 0.15, 0.008), procedural_glb.NEUTRAL)
        ok, why, _ = check_glb_bytes(m.to_glb({}))
        self.assertFalse(ok)
        self.assertIn("vertices", why)
        self.assertFalse(check_glb_bytes(b"<html>not a model</html>")[0])

    def test_unverified_spec_never_builds(self):
        with self.assertRaises(ValueError):
            procedural_glb.build_from_spec({"verified": False, "category": "beverage_can", "reason": "volume unknown"})


class SpecSearchTests(unittest.TestCase):
    PAGE = ('<td class="nfo" data-spec="dimensions">162.8 x 77.6 x 8.2 mm (6.41 x 3.06 x 0.32 in)</td>'
            '<td class="nfo" data-spec="weight">218 g (7.69 oz)</td>'
            '<td class="nfo" data-spec="cam1modules">200 MP, f/1.7 (wide)<br>50 MP, f/1.9 (periscope telephoto)<br>'
            '10 MP, f/2.4 (telephoto)<br>50 MP, f/1.9 (ultrawide)</td>')
    INDEX = ({9: "Samsung", 48: "Apple"},
             [[9, 13322, "Galaxy S25 Ultra", ""], [9, 13324, "Galaxy S25", ""], [48, 13064, "iPhone 16 Pro Max", ""]])

    def test_parse_phone_page(self):
        spec = spec_search.parse_phone_page(self.PAGE)
        self.assertEqual((spec["height_mm"], spec["width_mm"], spec["depth_mm"], spec["rear_cameras"], spec["weight_g"]),
                         (162.8, 77.6, 8.2, 4, 218.0))
        with self.assertRaises(spec_search.SpecError):
            spec_search.parse_phone_page("<html>no specs</html>")

    def test_find_phone_exact_tier_only(self):
        self.assertEqual(spec_search.find_phone("Samsung", "Galaxy S25 Ultra", self.INDEX), ("Samsung", 13322, "Galaxy S25 Ultra"))
        self.assertEqual(spec_search.find_phone("Samsung", "Galaxy S25", self.INDEX)[1], 13324)
        self.assertEqual(spec_search.find_phone("Apple", "iPhone 16 Pro Max", self.INDEX)[1], 13064)
        self.assertIsNone(spec_search.find_phone("Apple", "iPhone 16 Pro", self.INDEX))
        self.assertIsNone(spec_search.find_phone("Samsung", "Galaxy S24 Ultra", self.INDEX))
        self.assertEqual(spec_search.phone_page_url("Samsung", 13322, "Galaxy S25 Ultra"),
                         "https://www.gsmarena.com/samsung_galaxy_s25_ultra-13322.php")

    def test_volume_parsing(self):
        vol = spec_search.volume_ml
        self.assertEqual((vol("330ml"), vol("12 oz"), vol("0.5l"), vol("33cl")), (330, 355, 500, 330))
        self.assertIsNone(vol("400ml"))
        self.assertIsNone(vol("coke zero can"))

    def test_can_specs(self):
        std = spec_search.can_specs("coke zero can 330ml")
        self.assertTrue(std["verified"])
        self.assertEqual((std["diameter_mm"], std["height_mm"], std["neck_diameter_mm"], std["profile_assumed"]),
                         (66.0, 115.2, 54.0, True))
        self.assertEqual(spec_search.can_specs("330ml sleek can")["diameter_mm"], 58.0)
        self.assertFalse(spec_search.can_specs("coke zero can")["verified"])
        self.assertIn("volume", spec_search.can_specs("coke zero can")["reason"])
        self.assertFalse(spec_search.can_specs("330ml slim can")["verified"])

    def test_phone_specs_verified_only_from_gsmarena(self):
        with mock.patch.object(spec_search, "_quicksearch_index", return_value=self.INDEX), \
                mock.patch.object(spec_search, "_get", return_value=self.PAGE.encode()):
            spec = spec_search.phone_specs("Samsung", "Galaxy S25 Ultra")
            self.assertTrue(spec["verified"], spec)
            self.assertEqual(spec["sources"][0]["url"], "https://www.gsmarena.com/samsung_galaxy_s25_ultra-13322.php")
            self.assertFalse(spec_search.phone_specs("Samsung", "Galaxy S24 Ultra")["verified"])
            self.assertFalse(spec_search.phone_specs("Google", "Pixel 9")["verified"])


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

    GOOD = None

    @classmethod
    def setUpClass(cls):
        cls.GOOD = procedural_glb.build_can_spec(66.0, 115.2, 54.0)

    def assert_no_model(self, cid):
        self.assertFalse(cad_harvester.glb_path(cid).exists())
        self.assertIsNone(cad_harvester.read_attribution(cid))

    def test_offline_without_cache_is_not_found(self):
        prod = self.product("coke zero can 330ml", "offline_can")
        out = cad_harvester.harvest(prod, offline=True)
        self.assertEqual((out["status"], out["glb_url"]), ("not_found", ""))
        self.assertIn("offline", out["reason"])
        self.assert_no_model(prod.class_id)
        self.assertEqual(cad_harvester.harvest(prod, source="local")["status"], "not_found")

    def test_upload_is_cached_with_attribution(self):
        prod = self.product("iphone 13", "upload")
        out = cad_harvester.upload_glb(prod.class_id, self.GOOD, "my_iphone13.glb", "Apple iPhone 13")
        self.assertEqual((out["status"], out["source"], out["glb_url"]), ("cached", "upload", f"/static/models/{prod.class_id}.glb"))
        attr = cad_harvester.read_attribution(prod.class_id)
        self.assertEqual((attr["author"], attr["filename"]), ("user upload", "my_iphone13.glb"))
        glb = cad_harvester.glb_path(prod.class_id)
        self.assertEqual(glb.parent.resolve(), (ROOT / "static" / "models").resolve() if "KAMIKAZE_MODELS_DIR" not in os.environ
                         else Path(os.environ["KAMIKAZE_MODELS_DIR"]).resolve())
        again = cad_harvester.harvest(prod, offline=True)
        self.assertEqual((again["status"], again["source"], again["cached"]), ("cached", "local-cache", True))
        self.assertTrue(again["credit"])

    def test_bad_upload_is_rejected_and_keeps_existing_model(self):
        prod = self.product("iphone 13", "bad_upload")
        cad_harvester.upload_glb(prod.class_id, self.GOOD, "good.glb")
        before = cad_harvester.glb_path(prod.class_id).read_bytes()
        with self.assertRaises(ValueError):
            cad_harvester.upload_glb(prod.class_id, b"<html>nope</html>", "bad.glb")
        m = procedural_glb._Mesh()
        m.box((0, 0, 0), (0.07, 0.15, 0.008), procedural_glb.NEUTRAL)
        with self.assertRaisesRegex(ValueError, "vertices"):
            cad_harvester.upload_glb(prod.class_id, m.to_glb({}), "box.glb")
        self.assertEqual(cad_harvester.glb_path(prod.class_id).read_bytes(), before)
        self.assertEqual(cad_harvester.read_attribution(prod.class_id)["filename"], "good.glb")

    def test_delete_cached_removes_glb_and_sidecar(self):
        prod = self.product("iphone 13", "delete_cad")
        cad_harvester.upload_glb(prod.class_id, self.GOOD, "x.glb")
        self.assertTrue(cad_harvester.glb_path(prod.class_id).exists())
        self.assertTrue(cad_harvester.delete_cached(prod.class_id))
        self.assert_no_model(prod.class_id)
        self.assertFalse(cad_harvester.delete_cached(prod.class_id))

    def test_glb_without_attribution_is_never_shown(self):
        prod = self.product("iphone 14", "no_sidecar")
        cad_harvester.upload_glb(prod.class_id, self.GOOD, "x.glb")
        cad_harvester.sidecar_path(prod.class_id).unlink()
        out = cad_harvester.harvest(prod, offline=True)
        self.assertEqual(out["status"], "not_found")
        self.assertFalse(cad_harvester.glb_path(prod.class_id).exists())

    def test_credits_listing_includes_author_and_licence(self):
        prod = self.product("pepsi can", "credits")
        cad_harvester.upload_glb(prod.class_id, self.GOOD, "pepsi.glb")
        row = next(c for c in cad_harvester.credits() if c.get("class_id") == prod.class_id)
        self.assertTrue(row["author"] and row["license"])
        self.assertTrue((cad_harvester.models_dir() / "CREDITS.md").exists())

    def test_generate_from_verified_specs_only(self):
        prod = self.product("coke zero can 330ml", "spec_can")
        out = cad_harvester.generate_from_specs(prod)
        self.assertEqual((out["status"], out["badge"]), ("generated", cad_harvester.SPEC_BADGE))
        self.assertTrue(cad_harvester.validate_glb(cad_harvester.glb_path(prod.class_id))[0])
        attr = cad_harvester.read_attribution(prod.class_id)
        self.assertIn("crowncork.com", attr["source_url"])
        self.assertEqual(cad_harvester.harvest(prod, offline=True)["badge"], cad_harvester.SPEC_BADGE)
        vague = self.product("coke zero can", "spec_vague")
        out = cad_harvester.generate_from_specs(vague)
        self.assertEqual(out["status"], "not_found")
        self.assertIn("volume", out["reason"])
        self.assert_no_model(vague.class_id)

    @unittest.skipUnless(online(), "needs SKETCHFAB_TOKEN and network access")
    def test_live_sketchfab_download_records_original_author(self):
        prod = self.product("Samsung Galaxy S24 Ultra", "live_s24u")
        out = cad_harvester.harvest(prod, force=True)
        if out["status"] == "not_found":
            self.skipTest(f"no exact-match model online right now: {out['reason']}")
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

    def test_views_are_front_right_back_left_top(self):
        self.assertEqual(tuple(taxonomy.VIEWS), ("front", "right", "back", "left", "top"))
        self.assertEqual(taxonomy.canonical_view("side"), "right")
        self.assertEqual(taxonomy.canonical_view("detail"), "top")

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
            cad_harvester.upload_glb(cid, procedural_glb.build_can_spec(66.0, 115.2, 54.0), "test.glb")
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

        def slow_retrieve(label, meta=None, force=False, offline=False, source="auto"):
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
        if had:
            self.skipTest("a real iphone_13 model is cached in this checkout")
        cand = cad_harvester.Candidate(provider="sketchfab", title="Apple iPhone 13", score=99.0, author="Test Author",
                                       author_url="https://sketchfab.com/test", license="CC Attribution",
                                       source_url="https://sketchfab.com/3d-models/x", uid="abc")

        def found(product, slug, deadline, queries, level, providers=cad_harvester.PROVIDERS):
            mcp_server.local_glb_path(slug).write_bytes(procedural_glb.build_can_spec(66.0, 115.2, 54.0))
            return cand, "Sketchfab"

        try:
            with mock.patch.object(cad_harvester, "_try_online", return_value=None):
                status, out = self.call("POST", "/api/cad/fetch", {"label": cid})
            self.assertEqual(status, 200, out)
            self.assertEqual((out["status"], out["glb_url"]), ("not_found", ""), out)
            self.assertTrue(out["reason"])
            row = next(p for p in self.call("GET", "/api/classes")[1]["prototypes"] if p["class_id"] == cid)
            self.assertEqual(row["cad_status"], "missing")
            with mock.patch.object(cad_harvester, "_try_online", side_effect=found):
                status, out = self.call("POST", "/api/cad/fetch", {"label": cid})
            self.assertEqual((status, out["status"]), (200, "online"), out)
            self.assertTrue(out["glb_url"].endswith(f"{cid}.glb"))
            self.assertIn("Test Author", out["credit"])
            row = next(p for p in self.call("GET", "/api/classes")[1]["prototypes"] if p["class_id"] == cid)
            self.assertEqual(row["cad_status"], "ready")
            self.assertEqual(row["cad_model_url"], out["glb_url"])
            self.bank.save()
            proto = torch.load(self.db, map_location="cpu", weights_only=True)["prototypes"][cid]
            self.assertEqual(proto["cad_model_url"], out["glb_url"])
        finally:
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
        self.assertEqual(self.npu.detector_model_path(self.npu.WORLD_VARIANT).name, "yolov8s-worldv2-kz_float32.tflite")
        self.assertEqual(self.npu.detector_model_path("my_cans").name, "my_cans_float32.tflite")
        self.assertEqual(self.npu.detector_model_path("lite2").name, "efficientdet_lite2_int8.tflite")
        with self.assertRaises(FileNotFoundError):
            self.npu.ensure_model(self.npu.yolo_spec("definitely_missing"), Path(tempfile.mkdtemp(prefix="kz_y_")))

    def test_detection_labels_use_domain_vocabulary(self):
        D, w, h = self.npu.Detection, 1920, 1080
        self.assertEqual(D("cup", 0.9, (0.4, 0.3, 0.5, 0.6)).to_dict(w, h)["label"], "beverage can")
        self.assertEqual(D("cup", 0.9, (0.4, 0.3, 0.5, 0.6)).to_dict(w, h)["raw_label"], "cup")
        self.assertEqual(D("soda can", 0.9, (0.4, 0.3, 0.5, 0.6)).to_dict(w, h)["label"], "beverage can")
        self.assertEqual(D("cell phone", 0.9, (0.4, 0.3, 0.5, 0.6)).to_dict(w, h)["label"], "smartphone")
        # bottle box 100 px wide: 200 px tall (H/W 2.0) is can-shaped, 400 px tall (H/W 4.0) stays a bottle
        self.assertEqual(D("bottle", 0.9, (0.5, 0.3, 0.5 + 100 / w, 0.3 + 200 / h)).to_dict(w, h)["label"], "beverage can")
        self.assertEqual(D("bottle", 0.9, (0.5, 0.3, 0.5 + 100 / w, 0.3 + 400 / h)).to_dict(w, h)["label"], "bottle")
        self.assertEqual(D("bottle", 0.9, (0.5, 0.3, 0.6, 0.5)).to_dict()["label"], "bottle")  # no frame size: no remap
        for raw in ("cup", "bottle", "cell phone"):
            self.assertNotEqual(device_taxonomy.domain_for(D(raw, 0.9, (0.4, 0.3, 0.5, 0.6)).to_dict(w, h)), None)

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
        det = self.npu.YoloDetector(self.npu.FALLBACK_DETECTOR, all_classes=True)  # COCO-80 model (person, bus)
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

    def test_yolo_world_vocabulary_on_npu(self):
        path = self.npu.detector_model_path(self.npu.WORLD_VARIANT)
        if not self.npu._valid_tflite(path):
            self.skipTest("YOLO-World export missing (./export_yolo.sh world)")
        det = self.npu.YoloDetector(self.npu.WORLD_VARIANT)
        self.assertEqual(det.device, "npu", det.info())
        self.assertEqual(tuple(det.names), self.npu.WORLD_CLASSES)
        self.assertEqual(sorted(det.info()["classes"]), sorted(self.npu.WORLD_CLASSES))

    def test_dinov2_pose_embedder_on_npu(self):
        import time
        np = self.np
        if not self.npu._valid_tflite(self.npu.MODELS_DIR / self.npu.DINO_SPEC.file):
            self.skipTest("DINOv2 export missing (export_dinov2.py)")
        dino = self.npu.DinoPoseEmbedder()
        self.assertEqual(dino.device, "npu")
        x = np.random.default_rng(2).random((1, 224, 224, 3), dtype=np.float32)
        v = dino.embed_rgb01(x)
        self.assertEqual(v.shape, (768,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=3)
        self.assertGreater(float(v @ dino.embed_rgb01(x)), 0.999)
        t0 = time.perf_counter()
        for _ in range(10):
            dino.embed_rgb01(x)
        ms = (time.perf_counter() - t0) * 100
        print(f"\n  NPU DINOv2 pose: {ms:.2f} ms/frame")
        self.assertLess(ms, 60.0)


class DeviceTaxonomyTests(unittest.TestCase):
    """Phone tiers, rear-weighted view reliability, front-screen detection and the detector domain gate."""

    def test_view_reliability_ranks_back_first(self):
        rel = {v: device_taxonomy.view_reliability(v) for v in ("back", "right", "top", "front")}
        self.assertEqual(rel["back"], 1.0)
        self.assertGreater(rel["right"], rel["top"])
        self.assertEqual(device_taxonomy.view_reliability("left"), rel["right"])
        self.assertAlmostEqual(rel["top"], rel["front"])
        self.assertEqual(device_taxonomy.view_reliability(None), 1.0)
        self.assertEqual(device_taxonomy.view_reliability("global"), 1.0)

    def test_weighted_phone_score_prefers_back(self):
        score, view = device_taxonomy.weighted_phone_score({"back": 0.80, "front": 0.85})
        self.assertEqual(view, "back")
        self.assertAlmostEqual(score, 0.80)
        score, view = device_taxonomy.weighted_phone_score({"back": 0.40, "front": 0.90})
        self.assertEqual(view, "front")
        self.assertLess(score, 0.90)
        self.assertAlmostEqual(device_taxonomy.weighted_average({"back": 1.0, "front": 0.0}), 0.65 / 0.75)

    def test_front_dominates(self):
        self.assertTrue(device_taxonomy.front_dominates({"front": 0.8, "back": 0.6}))
        self.assertFalse(device_taxonomy.front_dominates({"front": 0.6, "back": 0.8}))
        self.assertFalse(device_taxonomy.front_dominates({"front": 0.9}))

    def test_tier_for(self):
        t = device_taxonomy.tier_for
        self.assertEqual(t({"category": "smartphone", "brand": "Apple", "raw_name": "iphone 15 pro max"}), "iphone_pro")
        self.assertEqual(t({"category": "smartphone", "raw_name": "iphone 13"}), "iphone_standard")
        self.assertEqual(t({"category": "smartphone", "raw_name": "s24 ultra"}), "samsung_ultra")
        self.assertEqual(t({"category": "smartphone", "brand": "Samsung", "raw_name": "galaxy s22+"}), "samsung_standard_plus")
        self.assertIsNone(t({"category": "beverage_can", "raw_name": "coke"}))
        self.assertIsNone(t({"category": "smartphone", "raw_name": "pixel 8"}))
        self.assertIsNone(t(None))

    def test_disambiguation_prompt(self):
        pro = {"category": "smartphone", "raw_name": "iphone 15 pro"}
        base = {"category": "smartphone", "raw_name": "iphone 15"}
        base2 = {"category": "smartphone", "raw_name": "iphone 12"}
        ultra = {"category": "smartphone", "raw_name": "galaxy s24 ultra"}
        plus = {"category": "smartphone", "raw_name": "galaxy s24 plus"}
        self.assertIn("LiDAR", device_taxonomy.disambiguation_prompt(pro, base))
        self.assertIn("periscope", device_taxonomy.disambiguation_prompt(ultra, plus))
        self.assertEqual(device_taxonomy.disambiguation_prompt(base, base2), device_taxonomy.MSG_IPHONE_LAYOUT)
        self.assertEqual(device_taxonomy.disambiguation_prompt(base, plus), device_taxonomy.MSG_CROSS_BRAND)
        self.assertIsNone(device_taxonomy.disambiguation_prompt({"category": "beverage_can"}, None))

    def test_domain_gate(self):
        d = device_taxonomy.domain_for
        self.assertEqual(d({"label": "cell phone", "score": 0.7}), device_taxonomy.PHONE_DOMAIN)
        self.assertEqual(d({"label": "cup", "score": 0.5}), device_taxonomy.BEVERAGE_DOMAIN)
        self.assertEqual(d({"label": "bottle", "score": 0.9}), device_taxonomy.BEVERAGE_DOMAIN)
        self.assertIsNone(d({"label": "cell phone", "score": 0.49}))
        self.assertIsNone(d({"label": "remote", "score": 0.9}))
        self.assertIsNone(d({"label": "cup", "score": "bad"}))
        self.assertIsNone(d(None))
        self.assertEqual(device_taxonomy.describe("samsung_ultra")["lens_count"], 4)
        self.assertIsNone(device_taxonomy.describe("nope"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
