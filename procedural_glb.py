"""Offline procedural GLB generator: brand-coloured soda can / bottle and tier-aware smartphone meshes.

Used as the last-resort CAD fallback so the 3D viewer always shows a category-appropriate model, even with
no network. Pure Python (struct + json); output is a valid glTF 2.0 binary with PBR materials.
"""

from __future__ import annotations

import json
import math
import struct

BRAND_COLORS = {
    "coca-cola": ((0.80, 0.05, 0.07), (0.95, 0.95, 0.95)),
    "pepsi": ((0.05, 0.22, 0.60), (0.80, 0.10, 0.15)),
    "sprite": ((0.05, 0.55, 0.20), (0.85, 0.95, 0.30)),
    "red bull": ((0.15, 0.20, 0.45), (0.85, 0.75, 0.10)),
}
SILVER = (0.78, 0.79, 0.82)


class _Mesh:
    def __init__(self) -> None:
        self.prims: list[dict] = []

    def add(self, pos: list, nrm: list, idx: list, color: tuple, metallic: float, rough: float) -> None:
        self.prims.append({"pos": pos, "nrm": nrm, "idx": idx, "color": color, "metallic": metallic, "rough": rough})

    def box(self, c: tuple, s: tuple, color: tuple, metallic: float = 0.2, rough: float = 0.5) -> None:
        cx, cy, cz = c
        hx, hy, hz = s[0] / 2, s[1] / 2, s[2] / 2
        faces = [
            ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
            ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
            ((1, 0, 0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
            ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
            ((0, 1, 0), [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)]),
            ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ]
        pos, nrm, idx = [], [], []
        for normal, verts in faces:
            base = len(pos)
            for vx, vy, vz in verts:
                pos.append((cx + vx, cy + vy, cz + vz))
                nrm.append(normal)
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
        self.add(pos, nrm, idx, color, metallic, rough)

    def cylinder(self, c: tuple, r: float, h: float, color: tuple, axis: str = "y", segs: int = 48,
                 metallic: float = 0.6, rough: float = 0.4, r_top: float | None = None) -> None:
        cx, cy, cz = c
        r_top = r if r_top is None else r_top
        pos, nrm, idx = [], [], []

        def place(a: float, b: float, along: float) -> tuple:
            return (cx + a, cy + along, cz + b) if axis == "y" else (cx + a, cy + b, cz + along)

        def direction(x: float, z: float, ny: float) -> tuple:
            return (x, ny, z) if axis == "y" else (x, z, ny)

        for i in range(segs + 1):
            t = 2 * math.pi * i / segs
            x, z = math.cos(t), math.sin(t)
            slope = (r - r_top) / h if h else 0.0
            for along, rad in ((-h / 2, r), (h / 2, r_top)):
                pos.append(place(x * rad, z * rad, along))
                nrm.append(direction(x, z, slope))
        for i in range(segs):
            a = i * 2
            idx += [a, a + 2, a + 1, a + 1, a + 2, a + 3]
        for along, rad, ny in ((h / 2, r_top, 1.0), (-h / 2, r, -1.0)):
            base = len(pos)
            pos.append(place(0, 0, along))
            nrm.append(direction(0, 0, ny))
            for i in range(segs + 1):
                t = 2 * math.pi * i / segs
                pos.append(place(math.cos(t) * rad, math.sin(t) * rad, along))
                nrm.append(direction(0, 0, ny))
            for i in range(segs):
                tri = [base, base + 1 + i, base + 2 + i]
                idx += tri if ny > 0 else [tri[0], tri[2], tri[1]]
        self.add(pos, nrm, idx, color, metallic, rough)

    def to_glb(self) -> bytes:
        bin_data = bytearray()
        views, accessors, meshes_prims, materials = [], [], [], []
        for prim in self.prims:
            pos = [c for v in prim["pos"] for c in v]
            nrm = [c for v in prim["nrm"] for c in v]
            idx = prim["idx"]
            entries = []
            for kind, data, fmt, target, ctype, count in (
                ("pos", pos, "f", 34962, 5126, len(prim["pos"])),
                ("nrm", nrm, "f", 34962, 5126, len(prim["nrm"])),
                ("idx", idx, "I", 34963, 5125, len(idx)),
            ):
                while len(bin_data) % 4:
                    bin_data.append(0)
                offset = len(bin_data)
                bin_data += struct.pack(f"<{len(data)}{fmt}", *data)
                views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(bin_data) - offset, "target": target})
                acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": count,
                       "type": "VEC3" if kind != "idx" else "SCALAR"}
                if kind == "pos":
                    xs, ys, zs = pos[0::3], pos[1::3], pos[2::3]
                    acc["min"], acc["max"] = [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]
                accessors.append(acc)
                entries.append(len(accessors) - 1)
            materials.append({
                "pbrMetallicRoughness": {"baseColorFactor": [*prim["color"], 1.0], "metallicFactor": prim["metallic"],
                                         "roughnessFactor": prim["rough"]},
                "doubleSided": True,
            })
            meshes_prims.append({"attributes": {"POSITION": entries[0], "NORMAL": entries[1]}, "indices": entries[2],
                                 "material": len(materials) - 1})
        doc = {
            "asset": {"version": "2.0", "generator": "kamikaze-procedural"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": meshes_prims}], "materials": materials,
            "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(bin_data)}],
        }
        js = json.dumps(doc, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        bin_data += b"\x00" * (-len(bin_data) % 4)
        total = 12 + 8 + len(js) + 8 + len(bin_data)
        return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
                + struct.pack("<II", len(bin_data), 0x004E4942) + bytes(bin_data))


def _brand_colors(brand: str) -> tuple:
    return BRAND_COLORS.get((brand or "").lower(), ((0.55, 0.55, 0.60), (0.9, 0.9, 0.9)))


def build_can(brand: str = "", variant: str = "original", bottle: bool = False) -> bytes:
    body, accent = _brand_colors(brand)
    if variant in ("diet", "zero"):
        body = (0.10, 0.10, 0.12) if variant == "zero" else (0.72, 0.72, 0.76)
    m = _Mesh()
    if bottle:
        m.cylinder((0, 0.085, 0), 0.032, 0.17, (0.85, 0.92, 0.96), metallic=0.0, rough=0.08)
        m.cylinder((0, 0.100, 0), 0.0325, 0.07, body, metallic=0.1, rough=0.5)
        m.cylinder((0, 0.19, 0), 0.016, 0.04, (0.85, 0.92, 0.96), metallic=0.0, rough=0.08, r_top=0.013)
        m.cylinder((0, 0.218, 0), 0.0155, 0.018, accent, metallic=0.1, rough=0.45)
        return m.to_glb()
    m.cylinder((0, 0.0575, 0), 0.033, 0.115, body, metallic=0.7, rough=0.32)
    m.cylinder((0, 0.0575, 0), 0.0333, 0.028, accent, metallic=0.3, rough=0.45)
    m.cylinder((0, 0.1165, 0), 0.0285, 0.003, SILVER, metallic=0.95, rough=0.25, r_top=0.026)
    m.cylinder((0, 0.1185, 0), 0.025, 0.001, SILVER, metallic=0.95, rough=0.3)
    m.cylinder((0, 0.0015, 0), 0.026, 0.003, SILVER, metallic=0.95, rough=0.3)
    m.box((0, 0.1203, 0.004), (0.014, 0.0012, 0.008), (0.55, 0.56, 0.6), metallic=0.95, rough=0.35)
    return m.to_glb()


def build_phone(maker: str = "Apple", tier: str = "standard", lenses: int = 2) -> bytes:
    apple = maker.lower() == "apple"
    w, h, d = (0.0715, 0.1467, 0.0078) if apple else (0.0766, 0.1623 if tier == "ultra" else 0.1470, 0.0086)
    body = (0.22, 0.23, 0.26) if tier in ("pro", "ultra") else (0.45, 0.55, 0.70)
    m = _Mesh()
    m.box((0, 0, 0), (w, h, d), body, metallic=0.85 if tier != "standard" else 0.4, rough=0.35)
    m.box((0, 0, d / 2 + 0.0002), (w - 0.004, h - 0.004, 0.0004), (0.02, 0.02, 0.03), metallic=0.1, rough=0.05)
    lens_r = 0.0085 if tier != "standard" else 0.0065
    rear = -d / 2 - 0.0012
    if apple:
        island = (0.034, 0.034, 0.0022)
        ix, iy = -w / 2 + 0.0225, h / 2 - 0.0225
        m.box((ix, iy, rear + 0.0001), island, (0.12, 0.12, 0.13) if tier == "pro" else (0.35, 0.45, 0.6), metallic=0.5, rough=0.3)
        spots = ([(-0.009, 0.009), (0.009, 0.009), (0.0, -0.009)] if lenses >= 3 else [(-0.008, 0.008), (0.008, -0.008)])
        for dx, dy in spots:
            m.cylinder((ix + dx, iy + dy, rear - 0.0014), lens_r, 0.0022, (0.03, 0.03, 0.04), axis="z", metallic=0.9, rough=0.1)
        if tier == "pro":
            m.cylinder((ix + 0.011, iy - 0.011, rear - 0.0014), 0.0028, 0.002, (0.02, 0.02, 0.02), axis="z", metallic=0.2, rough=0.2)
    else:
        n = 4 if tier == "ultra" else 3
        y0 = h / 2 - 0.016
        for i in range(n):
            y = y0 - i * (0.0205 if tier == "ultra" else 0.0185)
            m.cylinder((-w / 2 + 0.0135, y, rear - 0.0012), 0.0072 if tier == "ultra" else 0.0068, 0.0022, (0.03, 0.03, 0.04),
                       axis="z", metallic=0.9, rough=0.1)
        if tier == "ultra":
            m.box((w / 2 - 0.016, h / 2 - 0.012, rear), (0.01, 0.008, 0.0012), (0.1, 0.1, 0.1), metallic=0.3, rough=0.3)
    return m.to_glb()


def build_for(meta: dict) -> bytes:
    """Pick a procedural model for an enrollment metadata dict."""
    category = meta.get("category")
    if category == "smartphone":
        from taxonomy import phone_profile

        p = phone_profile(meta)
        return build_phone(p["maker"], p["tier"], p["lenses"])
    return build_can(meta.get("brand", ""), meta.get("variant", "original"), bottle=category == "beverage_bottle")
