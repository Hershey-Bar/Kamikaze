"""Canonical orientation for downloaded / uploaded GLB models, so the 3D stage and the physical pose agree.

Repository models arrive in arbitrary frames (Z-up exports, phones lying flat, cans on their side). The dashboard
drives <model-viewer> with yaw 0 = the object's front facing the camera and "up" = +Y, which is also the glTF
convention (+Y up, front toward +Z). This module fixes a model non-destructively by wrapping every scene root in one
extra node, "kamikaze_orient", that carries a rotation quaternion - meshes, materials and the binary buffer are left
untouched, and the fix can be replaced or removed at any time.

Auto mode looks at the world-space bounding box (POSITION accessor min/max through the node transforms) and picks the
smallest proper rotation among the 24 axis-aligned ones that
  * stands the longest dimension along +Y (a can's height, a phone's length), when it is clearly the longest, and
  * for a phone, puts the thinnest dimension (its thickness) along Z so the screen / back face the camera.
Which of +Z / -Z the screen faces cannot be told from a box; a manual 180 deg turn about Y fixes that
(rotate(path, "y", 180)). Every change is recorded in the attribution sidecar as a modification (CC-BY requires
indicating changes).
"""

from __future__ import annotations

import itertools
import json
import math
import os
import struct
from pathlib import Path

import numpy as np

WRAPPER = "kamikaze_orient"
GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = b"glTF", 0x4E4F534A, 0x004E4942
LONG_RATIO = float(os.environ.get("KAMIKAZE_ORIENT_LONG_RATIO", "1.15"))  # longest must beat the next by this factor
THIN_RATIO = float(os.environ.get("KAMIKAZE_ORIENT_THIN_RATIO", "0.6"))  # phone thickness <= this x its width


class GlbError(ValueError):
    """The file is not a GLB this module can rewrite."""


# --------------------------------------------------------------------------------------
# GLB container I/O
# --------------------------------------------------------------------------------------


def read_glb(path: Path) -> tuple[dict, list[tuple[int, bytes]]]:
    """(JSON document, [(chunk type, payload)] of the chunks after the JSON one)."""
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != GLB_MAGIC:
        raise GlbError("not a binary glTF (GLB)")
    version, total = struct.unpack("<II", data[4:12])
    if version != 2 or total > len(data):
        raise GlbError("unsupported or truncated GLB")
    off, chunks, doc = 12, [], None
    while off + 8 <= total:
        length, ctype = struct.unpack("<II", data[off:off + 8])
        payload = data[off + 8:off + 8 + length]
        if doc is None:
            if ctype != JSON_CHUNK:
                raise GlbError("first GLB chunk is not JSON")
            doc = json.loads(payload.decode("utf-8"))
        else:
            chunks.append((ctype, payload))
        off += 8 + length
    if not isinstance(doc, dict):
        raise GlbError("GLB has no JSON chunk")
    return doc, chunks


def write_glb(path: Path, doc: dict, chunks: list[tuple[int, bytes]]) -> None:
    """Atomically rewrite a GLB (JSON chunk space-padded to 4 bytes, other chunks copied byte for byte)."""
    js = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), JSON_CHUNK) + js
    for ctype, payload in chunks:
        payload = payload + b"\0" * (-len(payload) % 4)
        body += struct.pack("<II", len(payload), ctype) + payload
    path = Path(path)
    tmp = path.with_suffix(".orient.part")
    tmp.write_bytes(GLB_MAGIC + struct.pack("<II", 2, 12 + len(body)) + body)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# Math
# --------------------------------------------------------------------------------------


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m: np.ndarray) -> list[float]:
    """3x3 rotation -> glTF quaternion [x, y, z, w] (w >= 0)."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = [x, y, z, w]
    if w < 0:
        q = [-v for v in q]
    return [round(float(v), 9) + 0.0 for v in q]


def axis_rotation(axis: str, deg: float) -> np.ndarray:
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    axis = axis.lower()
    if axis == "x":
        m = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == "y":
        m = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    elif axis == "z":
        m = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    else:
        raise ValueError("axis must be x, y or z")
    return np.round(np.array(m, dtype=np.float64), 12)


def _node_local(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF matrices are column-major
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(node.get("rotation", [0, 0, 0, 1])) * np.array(node.get("scale", [1, 1, 1]))[None, :]
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def _scene_roots(doc: dict) -> list[int]:
    scenes = doc.get("scenes") or []
    if not scenes:
        return []
    return list(scenes[int(doc.get("scene", 0) or 0)].get("nodes") or [])


def world_bounds(doc: dict, skip_wrapper: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    """World-space AABB (min, max) of every mesh POSITION accessor reachable from the default scene."""
    nodes, accessors, meshes = doc.get("nodes") or [], doc.get("accessors") or [], doc.get("meshes") or []
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    stack = [(i, np.eye(4)) for i in _scene_roots(doc)]
    seen = set()
    while stack:
        idx, parent = stack.pop()
        if idx in seen or not 0 <= idx < len(nodes):
            continue
        seen.add(idx)
        node = nodes[idx]
        world = parent if (skip_wrapper and node.get("name") == WRAPPER) else parent @ _node_local(node)
        mesh = node.get("mesh")
        if isinstance(mesh, int) and 0 <= mesh < len(meshes):
            for prim in meshes[mesh].get("primitives") or []:
                ref = (prim.get("attributes") or {}).get("POSITION")
                acc = accessors[ref] if isinstance(ref, int) and 0 <= ref < len(accessors) else None
                if not acc or "min" not in acc or "max" not in acc:
                    continue
                a, b = np.array(acc["min"][:3], dtype=np.float64), np.array(acc["max"][:3], dtype=np.float64)
                corners = np.array([[x, y, z, 1.0] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])
                pts = (world @ corners.T).T[:, :3]
                lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
        stack += [(c, world) for c in node.get("children") or []]
    if not np.all(np.isfinite(lo)):
        return None
    return lo, hi


# --------------------------------------------------------------------------------------
# Orientation choice
# --------------------------------------------------------------------------------------


def _proper_signed_permutations() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sgn) in enumerate(zip(perm, signs)):
                m[row, col] = sgn
            if round(np.linalg.det(m)) == 1:
                out.append(m)
    return out


ROTATIONS = _proper_signed_permutations()  # the 24 axis-aligned proper rotations


def choose_rotation(extent, category: str | None) -> tuple[np.ndarray, str]:
    """Smallest axis-aligned rotation that stands the object up for its category. Returns (3x3, explanation)."""
    e = np.asarray(extent, dtype=np.float64)
    order = np.argsort(-e)  # source axes, longest first
    names = "XYZ"
    want: dict[int, int] = {}  # target axis -> source axis
    notes = []
    if e[order[0]] >= LONG_RATIO * e[order[1]]:
        want[1] = int(order[0])
        notes.append(f"longest {names[order[0]]} -> up (Y)")
    if category == "smartphone" and e[order[2]] <= THIN_RATIO * e[order[1]]:
        want[2] = int(order[2])
        notes.append(f"thinnest {names[order[2]]} -> depth (Z)")
    if not want:
        return np.eye(3), "no dominant axis - kept as authored"
    best = max((m for m in ROTATIONS if all(m[t, s] != 0 for t, s in want.items())), key=lambda m: np.trace(m))
    return best, ", ".join(notes)


def _wrapper_index(doc: dict) -> int | None:
    roots = _scene_roots(doc)
    nodes = doc.get("nodes") or []
    if len(roots) == 1 and 0 <= roots[0] < len(nodes) and nodes[roots[0]].get("name") == WRAPPER:
        return roots[0]
    return None


def current_rotation(doc: dict) -> np.ndarray:
    idx = _wrapper_index(doc)
    if idx is None:
        return np.eye(3)
    return quat_to_matrix(doc["nodes"][idx].get("rotation", [0, 0, 0, 1]))


def _set_rotation(doc: dict, m: np.ndarray) -> None:
    """Install (or update / remove when identity) the wrapper node carrying rotation m."""
    idx = _wrapper_index(doc)
    identity = np.allclose(m, np.eye(3), atol=1e-9)
    if idx is not None:
        if identity:
            scene = doc["scenes"][int(doc.get("scene", 0) or 0)]
            scene["nodes"] = list(doc["nodes"][idx].get("children") or [])
            doc["nodes"][idx] = {"name": WRAPPER + "_unused"}  # keep indices stable; an unreferenced node is valid
        else:
            doc["nodes"][idx]["rotation"] = matrix_to_quat(m)
        return
    if identity:
        return
    scenes = doc.setdefault("scenes", [{"nodes": []}])
    scene = scenes[int(doc.get("scene", 0) or 0)]
    nodes = doc.setdefault("nodes", [])
    roots = list(scene.get("nodes") or [])
    # reuse a previously emptied wrapper slot if present
    slot = next((i for i, n in enumerate(nodes) if n.get("name") == WRAPPER + "_unused"), None)
    wrapper = {"name": WRAPPER, "children": roots, "rotation": matrix_to_quat(m),
               "extras": {"kamikaze": "orientation fix; remove this node to restore the authored frame"}}
    if slot is None:
        nodes.append(wrapper)
        slot = len(nodes) - 1
    else:
        nodes[slot] = wrapper
    scene["nodes"] = [slot]


def describe(m: np.ndarray) -> str:
    if np.allclose(m, np.eye(3), atol=1e-9):
        return "authored frame"
    q = matrix_to_quat(m)
    angle = math.degrees(2 * math.acos(max(-1.0, min(1.0, q[3]))))
    s = math.sqrt(max(1e-12, 1 - q[3] * q[3]))
    axis = [q[0] / s, q[1] / s, q[2] / s]
    return f"rotated {angle:.0f} deg about ({axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f})"


def _info(doc: dict, note: str) -> dict:
    m = current_rotation(doc)
    b = world_bounds(doc, skip_wrapper=False)
    ext = (b[1] - b[0]) if b else None
    return {"rotation": matrix_to_quat(m), "description": describe(m), "note": note,
            "extent": [round(float(v), 6) for v in ext] if ext is not None else None,
            "identity": bool(np.allclose(m, np.eye(3), atol=1e-9))}


# --------------------------------------------------------------------------------------
# Public operations
# --------------------------------------------------------------------------------------


def auto_orient(path: Path, category: str | None) -> dict:
    """Replace any previous fix with the automatic one for `category` (idempotent)."""
    doc, chunks = read_glb(path)
    bounds = world_bounds(doc, skip_wrapper=True)
    if bounds is None:
        return {**_info(doc, "no POSITION bounds - kept as authored"), "changed": False}
    m, note = choose_rotation(bounds[1] - bounds[0], category)
    before = current_rotation(doc)
    changed = not np.allclose(before, m, atol=1e-9)
    if changed:
        _set_rotation(doc, m)
        write_glb(path, doc, chunks)
    return {**_info(doc, note), "changed": changed}


def rotate(path: Path, axis: str, deg: float) -> dict:
    """Turn the model further by `deg` degrees about world axis x / y / z (manual correction from the dashboard)."""
    if abs(float(deg)) > 360:
        raise ValueError("deg must be within [-360, 360]")
    doc, chunks = read_glb(path)
    m = axis_rotation(axis, deg) @ current_rotation(doc)
    m = np.round(m, 12)
    _set_rotation(doc, m)
    write_glb(path, doc, chunks)
    return {**_info(doc, f"manual {deg:+g} deg about {axis.upper()}"), "changed": True}


def reset(path: Path) -> dict:
    doc, chunks = read_glb(path)
    changed = _wrapper_index(doc) is not None
    if changed:
        _set_rotation(doc, np.eye(3))
        write_glb(path, doc, chunks)
    return {**_info(doc, "authored frame restored"), "changed": changed}


def status(path: Path) -> dict:
    doc, _ = read_glb(path)
    return {**_info(doc, ""), "changed": False}
