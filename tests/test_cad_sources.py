"""Manufacturing / packaging CAD sources: 1:1 phone dummies (cad_dummies.py), Open Food Facts packaging photos
(packaging.py), live 360-degree label stitching (label_stitch.py), cited phone camera layouts (procedural_glb.py),
the category-aware harvest order (cad_harvester.py), the receiver label-scan route and the MCP tools.

All tests are offline: HTTP is mocked and every mesh / photo is synthesised in memory (nothing is written except
GLBs under a temporary KAMIKAZE models directory).
"""

from __future__ import annotations

import io
import json
import math
import struct
import tempfile
import threading
import unittest
import zipfile
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

import cad_dummies
import cad_harvester
import entity_resolver
import label_stitch
import mcp_server
import packaging
import procedural_glb
import spec_search

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

needs_torch = unittest.skipIf(torch is None, "PyTorch not installed")

PHONE_SPEC = {"verified": True, "category": "smartphone", "maker": "Apple", "name": "Apple iPhone 15 Pro Max",
              "height_mm": 159.9, "width_mm": 76.7, "depth_mm": 8.25, "rear_cameras": 3,
              "sources": [{"what": "dimensions", "url": "https://www.gsmarena.com/x.php", "publisher": "GSMArena"}],
              "approximations": []}


def glb_json(data: bytes) -> dict:
    assert data[:4] == b"glTF", "not a GLB"
    n = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + n])


def box_tris(w: float, h: float, d: float, c=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Closed axis-aligned box (12 triangles) centred at c."""
    x, y, z = w / 2, h / 2, d / 2
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]]) + np.asarray(c)
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
    return v[np.asarray(f)]


def phone_dummy_tris(spec: dict = PHONE_SPEC) -> np.ndarray:
    """Phone body + a camera bump on +Z at the top-left as seen by a viewer on the +Z side, in millimetres."""
    w, h, d = spec["width_mm"], spec["height_mm"], spec["depth_mm"]
    bump = box_tris(30.0, 30.0, 2.0, (-(w / 2 - 20.0), h / 2 - 20.0, d / 2 + 1.0))
    return np.concatenate([box_tris(w, h, d), bump])


def subdivide(tris: np.ndarray, levels: int = 2) -> np.ndarray:
    """Split every triangle into 4, `levels` times (same shape, realistic vertex count for the GLB quality gate)."""
    for _ in range(levels):
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tris = np.concatenate([np.stack(t, 1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return tris


def binary_stl(tris: np.ndarray) -> bytes:
    rec = np.zeros(len(tris), dtype=[("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = tris
    return b"\0" * 80 + struct.pack("<I", len(tris)) + rec.tobytes()


def ascii_stl(tris: np.ndarray) -> bytes:
    out = ["solid t"]
    for t in tris:
        out += ["facet normal 0 0 0", "outer loop"] + [f"vertex {a} {b} {c}" for a, b, c in t] + ["endloop", "endfacet"]
    return ("\n".join(out + ["endsolid t"])).encode()


def obj_bytes(tris: np.ndarray) -> bytes:
    lines = [f"v {a} {b} {c}" for a, b, c in tris.reshape(-1, 3)]
    lines += [f"f {3 * i + 1} {3 * i + 2} {3 * i + 3}" for i in range(len(tris))]
    return "\n".join(lines).encode()


def threemf_bytes(tris: np.ndarray, scale: float = 1.0) -> bytes:
    v = tris.reshape(-1, 3)
    verts = "".join(f'<vertex x="{a}" y="{b}" z="{c}"/>' for a, b, c in v)
    faces = "".join(f'<triangle v1="{3 * i}" v2="{3 * i + 1}" v3="{3 * i + 2}"/>' for i in range(len(tris)))
    s = scale
    model = (f'<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>'
             f'<object id="1" type="model"><mesh><vertices>{verts}</vertices><triangles>{faces}</triangles></mesh></object>'
             f'</resources><build><item objectid="1" transform="{s} 0 0 0 {s} 0 0 0 {s} 0 0 0"/></build></model>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("3D/3dmodel.model", model)
    return buf.getvalue()


def stripe_label(w: int = 720, h: int = 240) -> np.ndarray:
    """Distinctive, non-periodic synthetic label (BGR) for stitching / unwrap tests."""
    rng = np.random.default_rng(7)
    lab = np.zeros((h, w, 3), np.uint8)
    x0 = 0
    while x0 < w:
        bw = int(rng.integers(8, 40))
        lab[:, x0:x0 + bw] = rng.integers(30, 230, 3)
        x0 += bw
    for _ in range(40):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(lab, (cx, cy), int(rng.integers(6, 22)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    cv2.putText(lab, "KIRKLAND LIME", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
    return lab


# --------------------------------------------------------------------------------------
# 1:1 phone dummies
# --------------------------------------------------------------------------------------


class DummyMatchingTests(unittest.TestCase):
    def setUp(self):
        self.ident = cad_dummies.identity(entity_resolver.resolve("iphone 15 pro max"))

    def test_tokens_split_camel_case_digits_and_samsung_suffixes(self):
        self.assertEqual(cad_dummies.tokens("iPhone15ProMax_v2.stl")[:4], ["iphone", "15", "pro", "max"])
        self.assertIn("ultra", cad_dummies.tokens("Galaxy S24U dummy"))
        self.assertIn("plus", cad_dummies.tokens("S23+"))

    def test_identity_and_title_relevance(self):
        self.assertEqual((self.ident["gen"], self.ident["tiers"]), ("15", {"pro", "max"}))
        dummy = cad_dummies.title_relevance("iPhone 15 Pro Max 1:1 dummy", self.ident)
        case = cad_dummies.title_relevance("iPhone 15 Pro Max case", self.ident)
        self.assertIsNotNone(dummy)
        self.assertGreater(dummy, case)
        self.assertIsNone(cad_dummies.title_relevance("iPhone 14 Pro Max dummy", self.ident))
        self.assertIsNone(cad_dummies.title_relevance("Pixel 15 dummy", self.ident))
        with self.assertRaises(cad_dummies.DummyError):
            cad_dummies.identity(entity_resolver.resolve("coca cola can"))

    def test_file_matches_rejects_other_variants_and_parts(self):
        t = "iPhone 15 Pro / Pro Max dummies"
        self.assertTrue(cad_dummies.file_matches("iphone_15_pro_max.stl", t, self.ident)[0])
        self.assertFalse(cad_dummies.file_matches("iphone_15_pro.stl", t, self.ident)[0])
        self.assertFalse(cad_dummies.file_matches("iphone_14_pro_max.stl", t, self.ident)[0])
        self.assertFalse(cad_dummies.file_matches("15_pro_max_case.stl", t, self.ident)[0])
        self.assertFalse(cad_dummies.file_matches("iphone_15_pro_max.png", t, self.ident)[0])
        self.assertTrue(cad_dummies.file_matches("dummy.stl", "iPhone 15 Pro Max dummy", self.ident)[0])
        self.assertFalse(cad_dummies.file_matches("dummy.stl", t, self.ident)[0])  # ambiguous: which tier?

    def test_license_link_and_derivatives(self):
        self.assertEqual(cad_dummies.license_link("Creative Commons - Attribution - Share Alike"),
                         "https://creativecommons.org/licenses/by-sa/4.0/")
        self.assertEqual(cad_dummies.license_link("Creative Commons \u2014 Public Domain"),
                         "https://creativecommons.org/publicdomain/zero/1.0/")
        self.assertEqual(cad_dummies.license_link("Creative Commons - Attribution - NoDerivatives"),
                         "https://creativecommons.org/licenses/by-nd/4.0/")
        self.assertEqual(cad_dummies.license_link("All rights reserved"), "")
        self.assertFalse(cad_dummies.license_ok("Standard Digital File License"))
        self.assertTrue(cad_dummies._no_derivatives("CC BY-ND"))
        self.assertFalse(cad_dummies._no_derivatives("CC BY-SA"))

    def test_search_printables_ranks_and_filters(self):
        items = [
            {"id": 1, "name": "iPhone 15 Pro Max case", "slug": "case", "likesCount": 900, "downloadCount": 9000,
             "license": {"name": "Creative Commons - Attribution"}, "user": {"handle": "a", "publicUsername": "A"}},
            {"id": 2, "name": "iPhone 15 Pro Max 1:1 dummy", "slug": "dummy", "likesCount": 40, "downloadCount": 300,
             "license": {"name": "Creative Commons - Attribution"}, "user": {"handle": "b", "publicUsername": "B"},
             "image": {"filePath": "media/x.jpg"}},
            {"id": 3, "name": "iPhone 13 dummy", "slug": "old", "likesCount": 5, "downloadCount": 5,
             "license": {"name": "Creative Commons - Attribution"}, "user": {}},
            {"id": 4, "name": "iPhone 15 Pro Max dummy", "slug": "paid", "likesCount": 5, "downloadCount": 5,
             "license": {"name": "Standard Digital File License"}, "user": {}},
        ]
        with mock.patch.object(cad_dummies, "_graphql", return_value={"searchPrints2": {"items": items}}):
            out = cad_dummies.search_printables(entity_resolver.resolve("iphone 15 pro max"))
        self.assertEqual([c["id"] for c in out], ["2", "1"])
        self.assertEqual(out[0]["author_url"], "https://www.printables.com/@b")
        self.assertTrue(out[0]["thumbnail_url"].startswith(cad_dummies.PRINTABLES_MEDIA))

    def test_thingiverse_requires_a_token(self):
        with mock.patch.dict("os.environ", {"THINGIVERSE_TOKEN": ""}):
            with self.assertRaises(cad_dummies.DummyError):
                cad_dummies._tv_json("/search/x")


class DummyGeometryTests(unittest.TestCase):
    def test_loaders_agree(self):
        tris = phone_dummy_tris()
        for data, name in ((binary_stl(tris), "a.stl"), (ascii_stl(tris), "a.stl"), (obj_bytes(tris), "a.obj"),
                           (threemf_bytes(tris), "a.3mf")):
            got = cad_dummies.load_mesh(data, name)
            self.assertEqual(got.shape, tris.shape, name)
            np.testing.assert_allclose(np.ptp(got.reshape(-1, 3), 0), np.ptp(tris.reshape(-1, 3), 0), atol=1e-3)

    def test_3mf_entities_refused(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("3D/3dmodel.model", '<!DOCTYPE m [<!ENTITY a "b">]><model/>')
        with self.assertRaises(cad_dummies.DummyError):
            cad_dummies.load_3mf(buf.getvalue())

    def test_dimension_gate_units_and_rejection(self):
        ok, scale, _ = cad_dummies.dimension_gate(np.array([15.99, 7.67, 0.95]), PHONE_SPEC)  # cm file
        self.assertEqual((ok, scale), (True, 10.0))
        ok, _, why = cad_dummies.dimension_gate(np.array([150.0, 76.7, 8.3]), PHONE_SPEC)  # wrong model height
        self.assertFalse(ok)
        self.assertIn("height", why)

    def test_orient_puts_bump_on_back_top_left(self):
        tris = phone_dummy_tris()
        rot = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)  # authored lying on its side
        out, rep = cad_dummies.orient_phone(tris @ rot.T, PHONE_SPEC)
        pts = out.reshape(-1, 3)
        ext = np.ptp(pts, 0)
        self.assertAlmostEqual(ext[0], 76.7, 1)
        self.assertAlmostEqual(ext[1], 159.9, 1)
        back = pts[pts[:, 2] < pts[:, 2].min() + 0.3]
        self.assertGreater(back[:, 1].mean(), 40.0, rep)  # bump at the top
        self.assertGreater(back[:, 0].mean(), 0.0, rep)  # top-left seen from the back = +X

    def test_convert_wrong_phone_is_rejected(self):
        small = dict(PHONE_SPEC, height_mm=146.7, width_mm=71.5)
        with self.assertRaises(cad_dummies.DummyError):
            cad_dummies.convert(binary_stl(phone_dummy_tris()), "x.stl", small)

    def test_convert_to_glb_and_decimate(self):
        glb, rep = cad_dummies.convert(binary_stl(phone_dummy_tris()), "x.stl", PHONE_SPEC, {"source_url": "u"})
        doc = glb_json(glb)
        self.assertEqual(doc["asset"]["extras"]["kind"], "kamikaze-dummy-v1")
        pos = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
        self.assertAlmostEqual(pos["max"][1] - pos["min"][1], 0.1599, 3)  # metres
        self.assertEqual(rep["triangles"], 24)
        g = np.linspace(-1, 1, 60)
        xx, yy = np.meshgrid(g, g)
        quads = np.stack([xx[:-1, :-1], yy[:-1, :-1], xx[1:, 1:], yy[1:, 1:]], -1).reshape(-1, 4)
        dense = np.concatenate([np.stack([np.c_[q[:, 0], q[:, 1], 0 * q[:, 0]], np.c_[q[:, 2], q[:, 1], 0 * q[:, 0]],
                                          np.c_[q[:, 2], q[:, 3], np.sin(q[:, 0])]], 1) for q in [quads]])
        self.assertLessEqual(len(cad_dummies.decimate(dense, 1000)), 1000)

    def test_harvest_dummy_picks_first_passing_file(self):
        prod = entity_resolver.resolve("iphone 15 pro max")
        cands = [{"provider": "printables", "title": "iPhone 15 Pro Max dummy", "license": "CC BY", "source_url": "u1"},
                 {"provider": "printables", "title": "iPhone 15 Pro Max 1:1", "license": "CC BY", "source_url": "u2"}]
        files = {"u1": [{"name": "iphone_15_pro_max.stl", "size": 1000}],
                 "u2": [{"name": "iphone_15_pro_max.stl", "size": 1000}]}
        meshes = {"u1": binary_stl(phone_dummy_tris(dict(PHONE_SPEC, height_mm=150.0))),
                  "u2": binary_stl(phone_dummy_tris())}
        prov = (None, lambda c: files[c["source_url"]], lambda c, f: meshes[c["source_url"]])
        with mock.patch.dict(cad_dummies.PROVIDERS, {"printables": prov}):
            cand, glb = cad_dummies.harvest_dummy(prod, PHONE_SPEC, float("inf"), ("printables",), cands)
        self.assertEqual(cand["source_url"], "u2")
        self.assertIn("vs published", cand["report"]["dimension_check"])
        self.assertEqual(glb[:4], b"glTF")


# --------------------------------------------------------------------------------------
# Open Food Facts packaging photos
# --------------------------------------------------------------------------------------


def can_photo(spec: dict, label: np.ndarray, px_per_mm: float = 3.0, tilt: float = 0.0) -> np.ndarray:
    """Synthetic upright can on a light background: the label rendered through the cylinder projection."""
    w = int(spec["diameter_mm"] * px_per_mm)
    h = int(spec["height_mm"] * px_per_mm)
    can = label_stitch.render_cylinder(label, 0.0, w, h)
    img = np.full((h + 160, w + 260, 3), 235, np.uint8)
    img[80:80 + h, 130:130 + w] = can
    if tilt:
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), tilt, 1.0)
        img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderValue=(235, 235, 235))
    return img


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.product = entity_resolver.resolve("kirkland lime sparkling water can")
        self.spec = spec_search.can_specs("kirkland lime", "355ml")

    def hit(self, **kw):
        base = {"code": "1", "product_name": "Sparkling Water Lime", "brands": "Kirkland Signature",
                "quantity": "355 ml", "image_front_url": "https://images.openfoodfacts.org/x/front.400.jpg"}
        return {**base, **kw}

    def test_product_gate(self):
        ok = packaging.product_gate
        self.assertTrue(ok(self.product, self.hit())[0])
        self.assertFalse(ok(self.product, self.hit(brands="Coca-Cola", product_name="Coke Lime"))[0])
        self.assertFalse(ok(self.product, self.hit(product_name="Sparkling Water Grapefruit"))[0])
        self.assertFalse(ok(self.product, self.hit(product_name="Sparkling Water Lime Lemon"))[0])
        self.assertFalse(ok(self.product, self.hit(product_name="Sparkling Water Lime bottle"))[0])
        self.assertTrue(ok(self.product, self.hit(quantity="500 ml"))[0])  # no volume in the class: OFF size used
        sized = entity_resolver.resolve("kirkland lime sparkling water can 355ml")
        self.assertFalse(ok(sized, self.hit(quantity="500 ml"))[0])
        self.assertTrue(ok(sized, self.hit(quantity="12 fl oz"))[0])
        self.assertFalse(ok(self.product, self.hit(image_front_url=""))[0])
        zero = entity_resolver.resolve("coke zero can")
        self.assertFalse(ok(zero, self.hit(brands="Coca-Cola", product_name="Coca-Cola Original"))[0])

    def test_full_image_url_and_query(self):
        self.assertTrue(packaging.full_image_url("https://i/x/front_en.12.400.jpg").endswith(".full.jpg"))
        self.assertIn('brands:"kirkland"', packaging.lucene_query(self.product))

    def test_find_products_sorts_by_scans(self):
        hits = [self.hit(code="a", unique_scans_n=2), self.hit(code="b", unique_scans_n=50),
                self.hit(code="c", product_name="Sparkling Water Grapefruit")]
        with mock.patch.object(packaging, "_get", return_value=json.dumps({"hits": hits}).encode()):
            out = packaging.find_products(self.product)
        self.assertEqual([h["code"] for h in out], ["b", "a"])
        self.assertEqual(out[0]["volume_ml"], 355)

    def test_check_photo_and_unwrap(self):
        label = stripe_label()
        photo = can_photo(self.spec, label)
        box, rep, mask = packaging.check_photo(photo, self.spec)
        self.assertLess(abs(rep["aspect"] - rep["expected_aspect"]) / rep["expected_aspect"], packaging.ASPECT_TOL)
        tex, info = packaging.unwrap_front(photo, box, self.spec, mask)
        self.assertEqual(tex.shape[1], packaging.TEX_W)
        self.assertAlmostEqual(info["photographed_fraction"], 140 / 360, 2)
        # the photographed front (u in 0.4..0.6) matches the source label's centre
        th, tw = tex.shape[:2]
        ref = cv2.resize(label, (tw, th))
        sl = slice(int(0.42 * tw), int(0.58 * tw))
        err = np.abs(tex[th // 8:-th // 8, sl].astype(float) - ref[th // 8:-th // 8, sl].astype(float)).mean()
        self.assertLess(err, 30.0)

    def test_check_photo_rejects_tilted_and_cropped(self):
        label = stripe_label()
        with self.assertRaises(packaging.PackagingError):
            packaging.check_photo(can_photo(self.spec, label, tilt=12.0), self.spec)
        photo = can_photo(self.spec, label)
        with self.assertRaises(packaging.PackagingError):
            packaging.check_photo(photo[:, :photo.shape[1] // 2], self.spec)

    def test_detector_hook_is_used(self):
        photo = can_photo(self.spec, stripe_label())
        h, w = photo.shape[:2]
        det = mock.Mock(return_value=[(130 / w, 80 / h, (130 + 198) / w, (80 + 366) / h, "beverage_can")])
        packaging.set_detector(det)
        try:
            _, rep, _ = packaging.can_box(photo)
        finally:
            packaging.set_detector(None)
        det.assert_called_once()
        self.assertEqual(rep["segmentation"], "detector+grabcut")

    def test_build_end_to_end_with_mocked_download(self):
        ok, jpg = cv2.imencode(".jpg", can_photo(self.spec, stripe_label()))
        hit = {**self.hit(), "volume_ml": 355}
        with mock.patch.object(packaging, "_get", return_value=jpg.tobytes()):
            attr, glb = packaging.build(self.product, {"class_id": "kz"}, hits=[hit])
        doc = glb_json(glb)
        self.assertEqual(len(doc.get("textures", [])), 1)
        self.assertEqual(attr["license"], "CC BY-SA 3.0")
        self.assertIn("Open Food Facts", attr["attribution_text"])
        self.assertTrue(any("not photographed" in a for a in attr["approximations"]))


# --------------------------------------------------------------------------------------
# Live label stitching
# --------------------------------------------------------------------------------------


class LabelStitchTests(unittest.TestCase):
    def run_turn(self, step_deg: float, frames: int, label=None):
        label = stripe_label() if label is None else label
        st = label_stitch.LabelStitcher("kz_can")
        out = None
        for i in range(frames):
            out = st.feed(label_stitch.render_cylinder(label, math.radians(step_deg * i), 160, 300))
            if out["complete"]:
                break
        return st, out, label

    def test_full_turn_completes_both_directions(self):
        for step in (7.0, -7.0):
            st, out, label = self.run_turn(step, 70)
            self.assertTrue(out["complete"], out)
            self.assertGreater(out["rotation_deg"], 270.0)  # + the visible band = 360
            tex = st.texture(720)
            ref = cv2.resize(label, (720, tex.shape[0]))
            best = min(np.abs(np.roll(tex, s, 1).astype(float) - ref.astype(float)).mean() for s in range(-12, 13, 2))
            self.assertLess(best, 25.0, step)

    def test_partial_scan_is_refused_unless_preview(self):
        st, out, _ = self.run_turn(7.0, 10)
        self.assertFalse(out["complete"])
        with self.assertRaises(label_stitch.StitchError):
            st.texture_jpeg()
        self.assertEqual(st.texture_jpeg(80, allow_partial=True)[:2], b"\xff\xd8")

    def test_rejects_scale_jumps_and_empty_crops(self):
        label = stripe_label()
        st = label_stitch.LabelStitcher()
        st.feed(label_stitch.render_cylinder(label, 0.0, 160, 300))
        out = st.feed(label_stitch.render_cylinder(label, 0.1, 230, 300))
        self.assertEqual(out["rejected"], 1)
        self.assertIn("crop width", out["reason"])
        with self.assertRaises(label_stitch.StitchError):
            st.feed(np.zeros((4, 4, 3), np.uint8))
        with self.assertRaises(label_stitch.StitchError):
            label_stitch.LabelStitcher().texture()


# --------------------------------------------------------------------------------------
# Cited phone layouts + harvest order
# --------------------------------------------------------------------------------------


LAYOUT = {"source": "test fixture (measured)", "island": {"x": 20, "y": 20, "w": 34, "h": 34, "r": 8, "bump": 1.5},
          "lenses": [{"x": 12, "y": 12, "r": 6, "h": 1.2}, {"x": 12, "y": 28, "r": 6, "h": 1.2},
                     {"x": 28, "y": 20, "r": 6, "h": 1.2}]}


class LayoutAndOrderTests(unittest.TestCase):
    def test_validate_layout(self):
        procedural_glb.validate_layout(LAYOUT, 76.7, 159.9)
        for bad in ({**LAYOUT, "source": ""}, {**LAYOUT, "lenses": []},
                    {**LAYOUT, "lenses": [{"x": 90, "y": 10, "r": 5}]}, {**LAYOUT, "lenses": [{"x": 10, "y": 10, "r": 40}]}):
            with self.assertRaises(ValueError):
                procedural_glb.validate_layout(bad, 76.7, 159.9)

    def test_layout_changes_geometry_and_is_recorded(self):
        plain = glb_json(procedural_glb.build_from_spec(PHONE_SPEC, "iphone"))
        laid = glb_json(procedural_glb.build_from_spec(PHONE_SPEC, "iphone", layout=LAYOUT))
        self.assertNotEqual(plain["accessors"], laid["accessors"])  # same part count, different lens positions
        self.assertEqual(laid["asset"]["extras"]["layout_source"], LAYOUT["source"])

    def test_textured_can(self):
        ok, jpg = cv2.imencode(".jpg", stripe_label())
        spec = spec_search.can_specs("x", "355ml")
        doc = glb_json(procedural_glb.build_from_spec(spec, "x", label=jpg.tobytes()))
        self.assertEqual(len(doc["textures"]), 1)
        self.assertIn("TEXCOORD_0", doc["meshes"][0]["primitives"][0]["attributes"])

    def test_harvest_steps_order(self):
        phone = entity_resolver.resolve("galaxy s24 ultra")
        can = entity_resolver.resolve("pepsi can")
        P = cad_harvester.PROVIDERS
        self.assertEqual([s for s, _ in cad_harvester.harvest_steps(phone, P)], ["dummies", "game_art"])
        self.assertEqual([s for s, _ in cad_harvester.harvest_steps(can, P)], ["game_art", "packaging"])
        self.assertEqual(cad_harvester.harvest_steps(phone, ("sketchfab",)), [("game_art", ("sketchfab",))])
        self.assertEqual(cad_harvester.harvest_steps(can, ("printables",)), [])
        self.assertIn("packaging", cad_harvester.SOURCES)


class HarvestIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(mcp_server, "MODELS_DIR", Path(self.tmp.name))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_dummy_wins_for_phones_and_is_credited(self):
        prod = entity_resolver.resolve("iphone 15 pro max")
        glb, rep = cad_dummies.convert(binary_stl(subdivide(phone_dummy_tris())), "d.stl", PHONE_SPEC)
        cand = {"provider": "printables", "uid": "printables:9", "title": "iPhone 15 Pro Max dummy", "author": "Maker",
                "author_url": "https://www.printables.com/@m", "license": "Creative Commons - Attribution",
                "source_url": "https://www.printables.com/model/9", "file": "d.stl", "report": rep, "score": 50.0}
        with mock.patch.object(spec_search, "product_specs", return_value=PHONE_SPEC), \
                mock.patch.object(cad_dummies, "harvest_dummy", return_value=(cand, glb)), \
                mock.patch.object(cad_harvester, "_try_online") as online, \
                mock.patch.object(cad_harvester, "_library_match", return_value=None):
            out = cad_harvester.harvest(prod, force=True)
        online.assert_not_called()
        self.assertEqual((out["status"], out["source"]), ("online", "printables"))
        self.assertIn("Maker", out["credit"])
        self.assertEqual(out["attribution"]["license_url"], "https://creativecommons.org/licenses/by/4.0/")
        self.assertEqual(out["badge"], cad_harvester.DUMMY_BADGE)

    def test_unverified_phone_specs_skip_dummies(self):
        prod = entity_resolver.resolve("iphone 15 pro max")
        with mock.patch.object(spec_search, "product_specs", return_value={"verified": False, "reason": "x"}), \
                mock.patch.object(cad_dummies, "harvest_dummy") as hd:
            self.assertIsNone(cad_harvester._try_dummies(prod, float("inf"), ("printables",)))
        hd.assert_not_called()

    def test_packaging_after_game_art_for_cans(self):
        prod = entity_resolver.resolve("kirkland lime sparkling water can")
        spec = spec_search.can_specs("x", "355ml")
        ok, jpg = cv2.imencode(".jpg", stripe_label())
        glb = procedural_glb.build_from_spec(spec, "x", label=jpg.tobytes())
        attr = {"title": "Kirkland lime (355 ml, packaging photo)", "author": "Open Food Facts contributors",
                "author_url": "https://world.openfoodfacts.org/product/1", "license": "CC BY-SA 3.0",
                "license_url": packaging.OFF_LICENSE_URL, "source_url": "https://world.openfoodfacts.org/product/1",
                "thumbnail_url": "", "attribution_text": "Label photo: Open Food Facts contributors", "modified": True}
        order = []
        with mock.patch.object(cad_harvester, "_try_online", side_effect=lambda *a, **k: order.append("art")), \
                mock.patch.object(packaging, "build", side_effect=lambda *a, **k: order.append("off") or (attr, glb)), \
                mock.patch.object(cad_harvester, "_library_match", return_value=None):
            out = cad_harvester.harvest(prod, force=True)
        self.assertEqual(order, ["art", "off"])
        self.assertEqual((out["status"], out["source"]), ("online", "packaging"))
        self.assertEqual(out["badge"], cad_harvester.PACKAGING_BADGE)

    def test_install_label_scan_and_generate_with_layout(self):
        with self.assertRaises(ValueError):  # no volume -> no verified dimensions -> refused
            cad_harvester.label_scan_spec(entity_resolver.resolve("kirkland lime sparkling water can"))
        prod = entity_resolver.resolve("kirkland lime sparkling water can 355ml")
        ok, jpg = cv2.imencode(".jpg", stripe_label())
        out = cad_harvester.install_label_scan(prod, jpg.tobytes(), {"frames": 40, "coverage": 1.0})
        self.assertTrue(out["glb_url"])
        self.assertEqual(out["attribution"]["source_key"], "label_scan")
        self.assertIn("40 frames", out["credit"])
        phone = entity_resolver.resolve("iphone 15 pro max")
        with mock.patch.object(spec_search, "product_specs", return_value=PHONE_SPEC):
            out = cad_harvester.generate_from_specs(phone, layout=LAYOUT)
            self.assertIn("camera layout: test fixture", out["credit"])
            with self.assertRaises(ValueError):
                cad_harvester.generate_from_specs(phone, layout={**LAYOUT, "source": ""})

    def test_mcp_tools(self):
        cand = {"uid": "printables:1", "provider": "printables", "title": "iPhone 15 Pro Max dummy", "author": "A",
                "license": "Creative Commons - Attribution", "source_url": "u", "score": 12.345}
        with mock.patch.object(cad_dummies, "search", return_value=[cand]), \
                mock.patch.object(spec_search, "product_specs", return_value=PHONE_SPEC):
            out = mcp_server.search_cad_dummies("iphone 15 pro max")
        self.assertEqual(out["candidates"][0]["rank_score"], 12.35)
        self.assertEqual(out["candidates"][0]["license_url"], "https://creativecommons.org/licenses/by/4.0/")
        self.assertFalse(mcp_server.search_cad_dummies("iphone 15", providers="sketchfab")["ok"])
        hit = {"code": "1", "product_name": "Sparkling Water Lime", "brands": "Kirkland Signature", "quantity": "355 ml",
               "volume_ml": 355, "image_front_url": "https://images.openfoodfacts.org/a.400.jpg"}
        with mock.patch.object(packaging, "find_products", return_value=[hit]):
            out = mcp_server.get_product_packaging("kirkland lime sparkling water")
        self.assertEqual(out["products"][0]["page"], "https://world.openfoodfacts.org/product/1")
        self.assertTrue(out["products"][0]["image_full_url"].endswith(".full.jpg"))
        with mock.patch.object(spec_search, "product_specs", return_value=PHONE_SPEC):
            out = mcp_server.assemble_phone_mesh("iphone 15 pro max", json.dumps(LAYOUT), "zz_test_phone")
        self.assertEqual(out["class_id"], "zz_test_phone")
        self.assertTrue(out["glb_url"])
        self.assertEqual(mcp_server.assemble_phone_mesh("iphone 15 pro max", "[1]")["status"], "error")
        with mock.patch.object(mcp_server, "_receiver_call", return_value={"ok": True}) as rc:
            mcp_server.stitch_camera_texture("START", "kz_can")
        rc.assert_called_once_with("POST", "/api/label_scan", {"action": "start", "label": "kz_can"}, timeout=10.0)


# --------------------------------------------------------------------------------------
# Receiver label-scan session + route
# --------------------------------------------------------------------------------------


@needs_torch
class ReceiverLabelScanTests(unittest.TestCase):
    def setUp(self):
        import receiver
        import taxonomy
        from test_comprehensive import FakeExtractor, unit

        self.receiver = receiver
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(mcp_server, "MODELS_DIR", Path(self.tmp.name)),
                        mock.patch.object(receiver, "LOG_FRAMES", False)]
        for p in self.patches:
            p.start()
        self.bank = receiver.EmbeddingBank(None)
        self.assets = receiver.AssetManager(offline=False, source="auto")
        self.leds = receiver.LedDriver(lambda *a: True)
        self.pipe = receiver.Pipeline(FakeExtractor(), self.bank, self.leds, self.assets)
        self.server = receiver.make_server(self.pipe, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.conn = HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=30)
        meta = taxonomy.parse_metadata({"raw_name": "Kirkland Signature Sparkling Water Lime Can 355ml"})
        self.label = taxonomy.label_from_metadata(meta)
        self.bank.add_multiview(self.label, {"front": [unit(1)]}, meta)
        pmeta = taxonomy.parse_metadata({"raw_name": "Apple iPhone 15 Pro Max"})
        self.phone = taxonomy.label_from_metadata(pmeta)
        self.bank.add_multiview(self.phone, {"front": [unit(2)]}, pmeta)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.conn.close()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def call(self, method, path, body=None):
        self.conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json"})
        resp = self.conn.getresponse()
        data = resp.read()
        return resp.status, json.loads(data) if "json" in resp.getheader("Content-Type", "") else data

    def test_scan_lifecycle(self):
        self.assertEqual(self.call("GET", "/api/label_scan")[1], {"ok": True, "active": False})
        self.assertEqual(self.call("POST", "/api/label_scan", {"action": "start", "label": self.phone})[0], 400)
        self.assertEqual(self.call("POST", "/api/label_scan", {"action": "start", "label": "ghost_can"})[0], 404)
        self.assertEqual(self.call("POST", "/api/label_scan", {"action": "finish"})[0], 409)
        status, out = self.call("POST", "/api/label_scan", {"action": "start", "label": self.label})
        self.assertEqual((status, out["active"], out["coverage"]), (200, True, 0.0))
        self.assertEqual(self.call("GET", "/api/label_scan/texture.jpg")[0], 404)  # no frames yet
        # no can detected: frame is not fed
        img = Image.new("RGB", (640, 480))
        self.pipe._detection = {"target": None}
        self.assertIn("no beverage can", self.pipe._feed_label_scan(self.pipe._label_scan, img)["reason"])
        # feed a turning can through the detector-box path
        label = stripe_label()
        canvas = np.zeros((480, 640, 3), np.uint8)
        box = [0.4, 0.2, 0.4 + 160 / 640, 0.2 + 300 / 480]
        self.pipe._detection = {"target": {"label": "beverage can", "box": box}}
        st = None
        for i in range(70):
            canvas[96:396, 256:416] = label_stitch.render_cylinder(label, math.radians(7.0 * i), 160, 300)
            st = self.pipe._feed_label_scan(self.pipe._label_scan, Image.fromarray(canvas[:, :, ::-1].copy()))
            if st["complete"]:
                break
        self.assertTrue(st["complete"], st)
        self.assertEqual(self.receiver.scan_led_params(st["coverage"], True), (0, 255, 0, 50, 8))
        self.assertEqual(self.receiver.scan_led_params(0.5, False)[4], 4)
        self.assertIsNotNone(self.pipe.snapshot()["label_scan"])
        status, jpg = self.call("GET", "/api/label_scan/texture.jpg")
        self.assertEqual((status, jpg[:2]), (200, b"\xff\xd8"))
        status, out = self.call("POST", "/api/label_scan", {"action": "finish"})
        self.assertEqual(status, 200, out)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["result"]["attribution"]["source_key"], "label_scan")
        self.assertEqual(self.call("GET", "/api/label_scan")[1]["active"], False)
        self.assertIsNone(self.pipe.snapshot()["label_scan"])

    def test_cancel(self):
        self.call("POST", "/api/label_scan", {"action": "start", "label": self.label})
        self.assertTrue(self.call("POST", "/api/label_scan", {"action": "cancel"})[1]["cancelled"])
        self.assertEqual(self.call("POST", "/api/label_scan", {"action": "bogus"})[0], 409)



_CAD_CACHE_ENV = None


def setUpModule():
    """Isolate from the populated offline device cache: these tests expect "no CAD exists" unless they create it."""
    global _CAD_CACHE_ENV
    import os as _os
    import tempfile as _tempfile
    _CAD_CACHE_ENV = _os.environ.get("KAMIKAZE_CAD_CACHE")
    _os.environ["KAMIKAZE_CAD_CACHE"] = _tempfile.mkdtemp(prefix="kz_empty_cad_cache_")


def tearDownModule():
    import os as _os
    import shutil as _shutil
    _shutil.rmtree(_os.environ.pop("KAMIKAZE_CAD_CACHE"), ignore_errors=True)
    if _CAD_CACHE_ENV is not None:
        _os.environ["KAMIKAZE_CAD_CACHE"] = _CAD_CACHE_ENV


if __name__ == "__main__":
    unittest.main()
