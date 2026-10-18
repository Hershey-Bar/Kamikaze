"""Dimension-calibrated GLB builder for products whose real measurements were verified by spec_search.py.

There is no generic / placeholder / guessed model here: build_phone_spec() and build_can_spec() only take measured
dimensions (GSMArena for phones, Crown Holdings for cans), and are only called on an explicit user request for a class
that has no downloadable CAD. Pure Python (struct + json); output is a valid glTF 2.0 binary with PBR materials, in
metres, Y up, the phone screen / can label facing +Z (model-viewer's default front view).
"""

from __future__ import annotations

import json
import math
import re
import struct

NEUTRAL = (0.62, 0.64, 0.68)  # grey-silver PBR baseline (metallic 0.8 / roughness 0.35) when no colour is named
ALUMINIUM = (0.80, 0.81, 0.84)
GLASS = (0.02, 0.02, 0.03)
LENS = (0.03, 0.03, 0.04)
SPEC_TAG = "kamikaze-spec-v1"
# colour words a user may put in the class name ("s25 ultra titanium black"); only used to tint the frame
NAMED_COLORS = {
    "black": (0.08, 0.08, 0.09), "graphite": (0.20, 0.21, 0.23), "titanium": (0.55, 0.55, 0.57),
    "silver": (0.80, 0.81, 0.84), "white": (0.90, 0.90, 0.91), "gray": (0.45, 0.46, 0.48), "grey": (0.45, 0.46, 0.48),
    "blue": (0.20, 0.33, 0.55), "navy": (0.10, 0.14, 0.30), "green": (0.25, 0.42, 0.32), "pink": (0.88, 0.70, 0.74),
    "purple": (0.48, 0.40, 0.60), "violet": (0.48, 0.40, 0.60), "gold": (0.85, 0.74, 0.52), "red": (0.70, 0.08, 0.10),
    "yellow": (0.95, 0.85, 0.40), "cream": (0.92, 0.89, 0.80), "lavender": (0.72, 0.68, 0.85),
}


class _Mesh:
    def __init__(self) -> None:
        self.prims: list[dict] = []

    def add(self, pos: list, nrm: list, idx: list, color: tuple, metallic: float, rough: float,
            uv: list | None = None, texture: bytes | None = None) -> None:
        """One primitive; `uv` (glTF TEXCOORD_0, v=0 at the image top) + `texture` (JPEG/PNG bytes) map an image."""
        if (uv is None) != (texture is None):
            raise ValueError("uv and texture must be given together")
        self.prims.append({"pos": pos, "nrm": nrm, "idx": idx, "color": color, "metallic": metallic, "rough": rough,
                           "uv": uv, "texture": texture})

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
                 metallic: float = 0.6, rough: float = 0.4, r_top: float | None = None, caps: bool = True,
                 texture: bytes | None = None, v_range: tuple[float, float] = (0.0, 1.0)) -> None:
        """Cylinder / frustum along `axis`. With `texture` the side wall is UV-mapped as an unwrapped label: u = 0.5 is
        the +Z (front) meridian, u grows clockwise seen from above (left-to-right reading order seen from the front),
        the seam is at the back; v runs from v_range[0] at the top rim to v_range[1] at the bottom rim."""
        cx, cy, cz = c
        r_top = r if r_top is None else r_top
        pos, nrm, idx, uv = [], [], [], []

        def place(a: float, b: float, along: float) -> tuple:
            return (cx + a, cy + along, cz + b) if axis == "y" else (cx + a, cy + b, cz + along)

        def direction(x: float, z: float, ny: float) -> tuple:
            return (x, ny, z) if axis == "y" else (x, z, ny)

        slope = (r - r_top) / h if h else 0.0
        for i in range(segs + 1):
            u = i / segs
            t = 1.5 * math.pi - 2 * math.pi * u if texture is not None else 2 * math.pi * i / segs
            x, z = math.cos(t), math.sin(t)
            for along, rad, v in ((-h / 2, r, v_range[1]), (h / 2, r_top, v_range[0])):
                pos.append(place(x * rad, z * rad, along))
                nrm.append(direction(x, z, slope))
                uv.append((u, v))
        for i in range(segs):
            a = i * 2
            idx += [a, a + 2, a + 1, a + 1, a + 2, a + 3]
        if texture is not None:
            self.add(pos, nrm, idx, (1.0, 1.0, 1.0), metallic, rough, uv, texture)
            pos, nrm, idx = [], [], []
        if caps:
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
        if idx:
            self.add(pos, nrm, idx, color, metallic, rough)

    def to_glb(self, extras: dict | None = None) -> bytes:
        bin_data = bytearray()
        views, accessors, meshes_prims, materials, images, textures = [], [], [], [], [], []
        image_index: dict[int, int] = {}

        def put(blob: bytes, target: int | None = None) -> int:
            while len(bin_data) % 4:
                bin_data.append(0)
            offset = len(bin_data)
            bin_data.extend(blob)
            view = {"buffer": 0, "byteOffset": offset, "byteLength": len(blob)}
            if target:
                view["target"] = target
            views.append(view)
            return len(views) - 1

        def pack(data, fmt: str) -> bytes:
            if hasattr(data, "tobytes"):  # numpy fast path (cad_dummies meshes)
                return data.astype("<f4" if fmt == "f" else "<u4").tobytes()
            flat = [c for v in data for c in v] if data and isinstance(data[0], (tuple, list)) else data
            return struct.pack(f"<{len(flat)}{fmt}", *flat)

        for prim in self.prims:
            pos = prim["pos"]
            idx = prim["idx"]
            streams = [("pos", pos, "f", 34962, 5126, len(pos), "VEC3"),
                       ("nrm", prim["nrm"], "f", 34962, 5126, len(prim["nrm"]), "VEC3")]
            if prim["uv"] is not None:
                streams.append(("uv", prim["uv"], "f", 34962, 5126, len(prim["uv"]), "VEC2"))
            streams.append(("idx", idx, "I", 34963, 5125, len(idx), "SCALAR"))
            entries = {}
            for kind, data, fmt, target, ctype, count, typ in streams:
                view = put(pack(data, fmt), target)
                acc = {"bufferView": view, "componentType": ctype, "count": count, "type": typ}
                if kind == "pos":
                    if hasattr(pos, "min"):
                        acc["min"], acc["max"] = [float(x) for x in pos.min(0)], [float(x) for x in pos.max(0)]
                    else:
                        acc["min"] = [min(v[k] for v in pos) for k in range(3)]
                        acc["max"] = [max(v[k] for v in pos) for k in range(3)]
                accessors.append(acc)
                entries[kind] = len(accessors) - 1
            pbr = {"baseColorFactor": [*prim["color"], 1.0], "metallicFactor": prim["metallic"],
                   "roughnessFactor": prim["rough"]}
            if prim["texture"] is not None:
                key = id(prim["texture"])
                if key not in image_index:
                    mime = "image/png" if prim["texture"][:4] == b"\x89PNG" else "image/jpeg"
                    images.append({"bufferView": put(prim["texture"]), "mimeType": mime})
                    textures.append({"sampler": 0, "source": len(images) - 1})
                    image_index[key] = len(textures) - 1
                pbr["baseColorTexture"] = {"index": image_index[key]}
            materials.append({"pbrMetallicRoughness": pbr, "doubleSided": True})
            attrs = {"POSITION": entries["pos"], "NORMAL": entries["nrm"]}
            if "uv" in entries:
                attrs["TEXCOORD_0"] = entries["uv"]
            meshes_prims.append({"attributes": attrs, "indices": entries["idx"], "material": len(materials) - 1})
        doc = {
            "asset": {"version": "2.0", "generator": "kamikaze-procedural", **({"extras": extras} if extras else {})},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": meshes_prims}], "materials": materials,
            "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": 0}],
        }
        if images:
            # 33071 = CLAMP_TO_EDGE vertically, 10497 = REPEAT around the seam
            doc.update(images=images, textures=textures,
                       samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 33071}])
        bin_data += b"\x00" * (-len(bin_data) % 4)
        doc["buffers"][0]["byteLength"] = len(bin_data)
        js = json.dumps(doc, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        total = 12 + 8 + len(js) + 8 + len(bin_data)
        return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
                + struct.pack("<II", len(bin_data), 0x004E4942) + bytes(bin_data))


def _rounded_outline(w: float, h: float, r: float, segs: int) -> list[tuple[float, float, float, float]]:
    """Counter-clockwise outline of a w x h rounded rectangle: [(x, y, nx, ny)] with outward normals."""
    r = max(0.0, min(r, w / 2, h / 2))
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0.0), (-w / 2 + r, h / 2 - r, 90.0), (-w / 2 + r, -h / 2 + r, 180.0),
                       (w / 2 - r, -h / 2 + r, 270.0)):
        for i in range(segs + 1):
            a = math.radians(a0 + 90.0 * i / segs)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a), math.cos(a), math.sin(a)))
    return pts


def rounded_prism(m, c: tuple, w: float, h: float, d: float, r: float, color: tuple, metallic: float, rough: float,
                  segs: int = 10) -> None:
    """Rounded-rectangle slab (w along X, h along Y, d along Z) centred on c."""
    cx, cy, cz = c
    outline = _rounded_outline(w, h, r, segs)
    pos, nrm, idx = [], [], []
    n = len(outline)
    for x, y, nx, ny in outline:
        for z in (-d / 2, d / 2):
            pos.append((cx + x, cy + y, cz + z))
            nrm.append((nx, ny, 0.0))
    for i in range(n):
        j = (i + 1) % n
        a, b, c2, e = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
        idx += [a, c2, b, b, c2, e]
    for z, nz in ((d / 2, 1.0), (-d / 2, -1.0)):
        base = len(pos)
        pos.append((cx, cy, cz + z))
        nrm.append((0.0, 0.0, nz))
        for x, y, _, _ in outline:
            pos.append((cx + x, cy + y, cz + z))
            nrm.append((0.0, 0.0, nz))
        for i in range(n):
            tri = [base, base + 1 + i, base + 1 + (i + 1) % n]
            idx += tri if nz > 0 else [tri[0], tri[2], tri[1]]
    m.add(pos, nrm, idx, color, metallic, rough)


def color_for(text: str) -> tuple | None:
    low = (text or "").lower()
    for word, rgb in NAMED_COLORS.items():
        if re.search(rf"\b{word}\b", low):
            return rgb
    return None


def _generation(name: str) -> int:
    m = re.search(r"(?:iphone|galaxy s|\bs)\s?(\d{1,2})", name.lower())
    return int(m.group(1)) if m else 0


def _layout_lenses(m, layout: dict, w: float, h: float, rear: float, body: tuple) -> int:
    """Explicit rear-camera layout in mm, measured on the BACK of the phone from its top-left corner (as seen when
    looking at the back): {"island": {"x", "y", "w", "h", "r", "bump"} | None, "lenses": [{"x", "y", "r", "h"}, ...]}
    (x, y = centres). Seen from behind, the model's +X is the viewer's left, so x_model = w/2 - x."""
    base = rear
    island = layout.get("island")
    if island:
        iw, ih = island["w"] / 1000.0, island["h"] / 1000.0
        bump = island.get("bump", 1.0) / 1000.0
        cx, cy = w / 2 - island["x"] / 1000.0, h / 2 - island["y"] / 1000.0
        rounded_prism(m, (cx, cy, rear - bump / 2), iw, ih, bump, island.get("r", 0.2 * min(island["w"], island["h"])) / 1000.0,
                      body, 0.6, 0.25)
        base = rear - bump
    lenses = layout.get("lenses") or []
    for lens in lenses:
        lr, lh = lens["r"] / 1000.0, lens.get("h", 1.0) / 1000.0
        x, y = w / 2 - lens["x"] / 1000.0, h / 2 - lens["y"] / 1000.0
        m.cylinder((x, y, base - lh / 2), lr, lh, LENS, axis="z", metallic=0.9, rough=0.1)
        m.cylinder((x, y, base - lh - 0.0001), lr * 0.62, 0.0002, GLASS, axis="z", metallic=0.2, rough=0.02)
    return len(lenses)


def validate_layout(layout: dict, width_mm: float, height_mm: float) -> None:
    """Raise ValueError unless every island / lens lies on the back of a width x height phone and a source is cited."""
    if not isinstance(layout, dict) or not isinstance(layout.get("lenses"), list) or not layout["lenses"]:
        raise ValueError("layout needs a non-empty 'lenses' list")
    if not str(layout.get("source") or "").strip():
        raise ValueError("layout needs a 'source' (where the lens coordinates were measured / published)")
    parts = list(layout["lenses"]) + ([layout["island"]] if layout.get("island") else [])
    for part in parts:
        for key in ("x", "y"):
            if not isinstance(part.get(key), (int, float)):
                raise ValueError(f"layout entry {part} is missing numeric {key!r}")
        if not (0 < part["x"] < width_mm and 0 < part["y"] < height_mm):
            raise ValueError(f"layout entry at ({part['x']}, {part['y']}) mm lies outside the {width_mm} x {height_mm} mm back")
    for lens in layout["lenses"]:
        if not isinstance(lens.get("r"), (int, float)) or not 0.5 <= lens["r"] <= width_mm / 4:
            raise ValueError(f"lens radius {lens.get('r')} mm is implausible")


def build_phone_spec(width_mm: float, height_mm: float, depth_mm: float, rear_cameras: int, maker: str,
                     name: str = "", color: tuple | None = None, extras: dict | None = None,
                     layout: dict | None = None, island_type: str | None = None) -> bytes:
    """Phone body at its published outer dimensions with the published number of rear camera modules.

    `island_type` (models/devices_db.json design family) overrides the generation heuristic: Apple "triple_square",
    "diagonal_dual", "vertical_pill"; Samsung "floating_rings" (no housing) or "rect_housing" / "contour_housing" /
    "contour_island" / "vertical_housing" (lenses on a raised housing).

    With `layout` (validate_layout: cited, mm from the top-left of the back) the island and lenses are placed exactly
    there. Without it the approximations flagged by spec_search apply: corner radius 11 % of width, 2 % bezel, lens /
    island size scaled to the width, island layout per maker generation (Apple square island; Samsung S23+ separate
    lens column, S20-S22 rectangular camera housing). The island is at the top-left of the back (+X, +Y, -Z).
    """
    if min(width_mm, height_mm, depth_mm) <= 0 or rear_cameras < 1:
        raise ValueError("phone dimensions and rear camera count must be positive")
    w, h, d = width_mm / 1000.0, height_mm / 1000.0, depth_mm / 1000.0
    body = color or NEUTRAL
    m = _Mesh()
    rounded_prism(m, (0, 0, 0), w, h, d, 0.11 * w, body, 0.8, 0.35)
    bezel = 0.02 * w
    rounded_prism(m, (0, 0, d / 2 + 0.0002), w - 2 * bezel, h - 2 * bezel, 0.0004, 0.11 * w - bezel, GLASS, 0.1, 0.05)
    rear = -d / 2
    gen = _generation(name)
    n = int(rear_cameras)
    meta = {}
    if layout:
        validate_layout(layout, width_mm, height_mm)
        n = _layout_lenses(m, layout, w, h, rear, body)
        meta = {"layout_source": layout["source"]}
    elif maker.lower() == "apple":
        side = (0.45 if n >= 3 else 0.42) * w
        bump = 0.0015
        ix, iy = w / 2 - 0.045 * w - side / 2, h / 2 - 0.045 * w - side / 2
        rounded_prism(m, (ix, iy, rear - bump / 2), side, side, bump, 0.22 * side, body, 0.6, 0.25)
        lr = 0.19 * side
        q = side * 0.24
        # (dx, dy) seen from the FRONT axes; from behind, -dx is the viewer's right
        if n >= 3 or island_type == "triple_square":
            spots = [(q, q), (q, -q), (-q, 0.0)]
        elif island_type == "vertical_pill" or (island_type is None and gen in (12, 16)):
            spots = [(q, q), (q, -q)]
        else:
            spots = [(q, q), (-q, -q)][:n]
        for dx, dy in spots:
            m.cylinder((ix + dx, iy + dy, rear - bump - 0.0011), lr, 0.0022, LENS, axis="z", metallic=0.9, rough=0.1)
            m.cylinder((ix + dx, iy + dy, rear - bump - 0.0023), lr * 0.62, 0.0002, GLASS, axis="z", metallic=0.2, rough=0.02)
    else:
        lr = 0.072 * w
        pitch = 2.75 * lr
        x = w / 2 - 0.06 * w - lr
        y0 = h / 2 - 0.06 * w - lr
        housed = island_type != "floating_rings" if island_type else bool(gen and gen < 23)
        if housed:
            hw, hh = 2 * lr + 0.012 * w, (n - 1) * pitch + 2 * lr + 0.024 * w
            rounded_prism(m, (x, y0 - (n - 1) * pitch / 2, rear - 0.0008), hw, hh, 0.0016, 0.3 * hw, body, 0.6, 0.3)
            base = rear - 0.0016
        else:
            base = rear
        for i in range(n):
            y = y0 - i * pitch
            m.cylinder((x, y, base - 0.0011), lr, 0.0022, LENS, axis="z", metallic=0.9, rough=0.1)
            m.cylinder((x, y, base - 0.0023), lr * 0.62, 0.0002, GLASS, axis="z", metallic=0.2, rough=0.02)
    return m.to_glb({"kind": SPEC_TAG, "category": "smartphone",
                     "dimensions_mm": {"width": width_mm, "height": height_mm, "depth": depth_mm},
                     "rear_cameras": n, **({"island_type": island_type} if island_type else {}), **meta, **(extras or {})})


def build_can_spec(diameter_mm: float, height_mm: float, neck_diameter_mm: float, neck_height_mm: float = 10.0,
                   color: tuple | None = None, extras: dict | None = None, label: bytes | None = None) -> bytes:
    """Necked aluminium beverage can at its published overall diameter and height (bottom at y = 0).

    `label` = JPEG of the full 360-degree unwrapped label (packaging.unwrap_front / label_stitch.LabelStitcher), its
    height spanning the whole can (bottom rim to lid rim); it is mapped onto the body and the neck taper.
    Approximations (flagged by spec_search): lid diameter from the standard end size, straight neck taper.
    """
    if min(diameter_mm, height_mm, neck_diameter_mm, neck_height_mm) <= 0 or neck_height_mm >= height_mm:
        raise ValueError("can dimensions must be positive and the neck shorter than the can")
    r, h, rn, hn = diameter_mm / 2000.0, height_mm / 1000.0, neck_diameter_mm / 2000.0, neck_height_mm / 1000.0
    body_h = h - hn
    shell = color or ALUMINIUM
    m = _Mesh()
    v_neck = 1.0 - body_h / h
    if label is not None:
        # printed ink on aluminium: lower metalness than bare metal so the artwork reads under a neutral environment
        m.cylinder((0, body_h / 2, 0), r, body_h, shell, segs=96, metallic=0.35, rough=0.35, caps=False,
                   texture=label, v_range=(v_neck, 1.0))
        m.cylinder((0, 0.0005, 0), r * 0.999, 0.001, ALUMINIUM, segs=96, metallic=0.85, rough=0.3)  # base disc
        m.cylinder((0, body_h + (hn - 0.002) / 2, 0), r, hn - 0.002, ALUMINIUM, segs=96, metallic=0.35, rough=0.35,
                   r_top=rn, caps=False, texture=label, v_range=(0.002 / h, v_neck))
    else:
        m.cylinder((0, body_h / 2, 0), r, body_h, shell, metallic=0.85, rough=0.3, segs=96)
        m.cylinder((0, body_h + (hn - 0.002) / 2, 0), r, hn - 0.002, ALUMINIUM, metallic=0.85, rough=0.3, segs=96, r_top=rn)
    m.cylinder((0, h - 0.001, 0), rn, 0.002, ALUMINIUM, metallic=0.9, rough=0.25, segs=96)
    m.cylinder((0, h + 0.0001, 0), rn * 0.9, 0.0002, (0.70, 0.71, 0.74), metallic=0.9, rough=0.35, segs=96)
    m.box((0, h + 0.0004, rn * 0.25), (rn * 0.5, 0.0006, rn * 0.55), (0.70, 0.71, 0.74), metallic=0.9, rough=0.35)
    return m.to_glb({"kind": SPEC_TAG, "category": "beverage_can",
                     "dimensions_mm": {"diameter": diameter_mm, "height": height_mm, "neck_diameter": neck_diameter_mm},
                     "textured_label": label is not None, **(extras or {})})


def build_from_spec(spec: dict, label_text: str = "", extras: dict | None = None, label: bytes | None = None,
                    layout: dict | None = None) -> bytes:
    """GLB for a verified spec_search result (+ optional can label texture / cited phone layout). Raises ValueError
    when the spec is not verified."""
    if not spec.get("verified"):
        raise ValueError(spec.get("reason") or "spec not verified")
    color = color_for(label_text)
    meta = {"sources": spec.get("sources", []), "approximations": spec.get("approximations", []), **(extras or {})}
    if spec["category"] == "smartphone":
        return build_phone_spec(spec["width_mm"], spec["height_mm"], spec["depth_mm"], spec["rear_cameras"],
                                spec.get("maker", ""), spec.get("name", ""), color, meta, layout, spec.get("island_type"))
    if spec["category"] == "beverage_can":
        return build_can_spec(spec["diameter_mm"], spec["height_mm"], spec["neck_diameter_mm"],
                              spec.get("neck_height_mm", 10.0), None if label else color, meta, label)
    raise ValueError(f"no spec builder for category {spec['category']!r}")
