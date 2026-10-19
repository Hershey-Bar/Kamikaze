"""1:1 smartphone "dummy" CAD from 3D-printing repositories, verified against published dimensions.

Game-art sites rarely carry an exact iPhone / Galaxy chassis; phone-case makers do. Their 1:1 dummies (camera island,
lens rings, corner radii) are published on Printables (public GraphQL, no key) and Thingiverse (THINGIVERSE_TOKEN).

    search(product)            ranked prints whose title names the exact generation (no download)
    harvest_dummy(product, dest, spec)  download -> parse STL / 3MF / OBJ -> dimension gate -> GLB (metres, Y up,
                               screen +Z, camera island top / -Z) -> (Candidate-like dict, GLB bytes)

Nothing is accepted on the title alone. Every mesh must pass the dimension gate against the GSMArena spec of the exact
phone (spec_search.phone_specs): longest side within DIM_TOL of the published height, middle side within DIM_TOL of
the width and thickness between 0.9 x depth and depth + BUMP_MAX_MM (the camera bump). Phone cases (several mm larger)
and other generations fail it. Variants with identical bodies (iPhone 12 / 12 Pro) are told apart by the file name /
print title tiers (a "pro" file is never used for the base phone). Licences follow cad_harvester (CC0 / CC-BY /
CC-BY-SA / CC-BY-ND; NonCommercial only with KAMIKAZE_CAD_ALLOW_NC=1; Printables "Standard Digital File" rejected).
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

import netguard
import procedural_glb

log = logging.getLogger("kamikaze.dummies")

PRINTABLES_GQL = "https://api.printables.com/graphql/"
PRINTABLES_MEDIA = "https://media.printables.com/"
THINGIVERSE_API = "https://api.thingiverse.com"
USER_AGENT = "Kamikaze/1.0 (+https://github.com/Hershey-Bar/Kamikaze; edge CAD retrieval)"
TIMEOUT_S = float(os.environ.get("KAMIKAZE_CAD_SEARCH_TIMEOUT", "10"))
MAX_MESH_BYTES = int(float(os.environ.get("KAMIKAZE_DUMMY_MAX_MB", "40")) * 1024 * 1024)
MAX_UNZIPPED = 200 * 1024 * 1024
MAX_TRIS = int(os.environ.get("KAMIKAZE_DUMMY_MAX_TRIS", "120000"))
FILE_TRIES = 6
DIM_TOL = 0.025
BUMP_MAX_MM = 7.0
CREASE_DEG = 35.0
UNIT_SCALES = (1.0, 10.0, 25.4, 1000.0)  # mm, cm, inch, m files
MESH_EXT = (".stl", ".3mf", ".obj")
TIERS = {"pro", "max", "plus", "ultra", "mini", "fe", "edge", "air", "lite", "e"}
DUMMY_WORDS = {"dummy", "replica", "mockup", "mock", "model", "1:1", "template", "blank", "prop"}
ACCESSORY_WORDS = {"case", "cover", "holder", "stand", "mount", "clip", "dock", "charger", "bumper", "sleeve", "grip",
                   "wallet", "tripod", "rig", "cage", "lens", "protector", "box", "adapter", "bracket", "pouch", "skin"}
# accessory words that may appear in a dummy's title ("dummy for case design") but disqualify a single file name
FILE_REJECT = ACCESSORY_WORDS | {"screen", "button", "buttons", "half", "top", "bottom", "part", "insert", "support"}
SURFACE = (0.62, 0.64, 0.68)  # grey-silver PBR: dummies carry no colour


class DummyError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Text matching
# --------------------------------------------------------------------------------------


def tokens(text: str) -> list[str]:
    """Lower-case tokens with camelCase / letter-digit boundaries split: "iPhone15ProMax_v2" -> iphone 15 pro max v 2;
    "+" -> plus, "S24U" -> s 24 ultra."""
    text = re.sub(r"(?i)iphone", " iphone ", text or "")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(?<![A-Za-z])[sS](\d{2})\s*[uU](?![A-Za-z])", r"s\1 ultra", text)
    text = text.lower().replace("+", " plus ")
    text = re.sub(r"(\d)([a-z])", r"\1 \2", re.sub(r"([a-z])(\d)", r"\1 \2", text))
    return [t for t in re.split(r"[^a-z0-9:]+", text) if t]


def identity(product) -> dict:
    """{"brand", "family", "gen", "tiers"} of a CanonicalProduct smartphone ("Galaxy S24 Ultra" -> gen "24", ultra)."""
    toks = tokens(getattr(product, "model_or_variant", "") or getattr(product, "canonical_name", ""))
    gen = next((t for t in toks if t.isdigit()), "")
    tiers = {t for t in toks if t in TIERS}
    if getattr(product, "category", "") != "smartphone" or not gen:
        raise DummyError(f"{getattr(product, 'canonical_name', product)!r} is not a phone with a model number")
    brand = getattr(product, "brand", "")
    line = "fold" if "fold" in toks else "flip" if "flip" in toks else "iphone" if brand == "Apple" else "s"
    if line in ("fold", "flip"):
        gen = next((t for i, t in enumerate(toks) if t.isdigit() and i and toks[i - 1] == line), gen)
    return {"brand": brand, "gen": gen, "tiers": tiers, "line": line,
            "family": {"iphone"} if brand == "Apple" else {"galaxy", "samsung", "s"}}


def _names_generation(toks: list[str], ident: dict) -> bool:
    for i, t in enumerate(toks):
        if t != ident["gen"]:
            continue
        prev = toks[i - 1] if i else ""
        if ident.get("line") in ("fold", "flip"):
            if prev == ident["line"]:
                return True
            continue
        if ident["brand"] == "Apple" and prev == "iphone":
            return True
        if ident["brand"] == "Samsung" and prev == "s":
            return True
    return False


def title_relevance(title: str, ident: dict) -> float | None:
    """Relevance of a print title for the phone, None when it names neither the brand family nor the generation."""
    toks = tokens(title)
    if not _names_generation(toks, ident):
        return None
    score = 20.0
    words = set(toks)
    if words & DUMMY_WORDS or "1:1" in (title or ""):
        score += 15.0
    if words & ACCESSORY_WORDS and not words & DUMMY_WORDS:
        score -= 25.0
    if {t for t in toks if t in TIERS} == ident["tiers"]:
        score += 8.0
    return score


def file_matches(filename: str, title: str, ident: dict) -> tuple[bool, str]:
    """Whether a file of a print is the requested variant (not a case part, another generation or another tier)."""
    stem = re.sub(r"\.[a-z0-9]+$", "", filename.lower())
    toks = tokens(re.sub(r"\.[A-Za-z0-9]+$", "", filename))
    if not filename.lower().endswith(MESH_EXT):
        return False, "not a mesh file"
    if set(toks) & FILE_REJECT:
        return False, "accessory / partial file"
    other = {t for t in toks if t.isdigit() and 10 <= int(t) <= 30 and t != ident["gen"]}
    if ident.get("line") in ("fold", "flip"):
        other = {t for i, t in enumerate(toks) if t.isdigit() and i and toks[i - 1] == ident["line"] and t != ident["gen"]}
    if other and ident["gen"] not in toks:
        return False, f"different generation ({', '.join(sorted(other))})"
    ftiers = {t for t in toks if t in TIERS}
    if "e" in ftiers and not ident["gen"] in stem:
        ftiers.discard("e")
    if ftiers:
        return (ftiers == ident["tiers"], "tier tokens in file name" if ftiers == ident["tiers"] else
                f"file is the {' '.join(sorted(ftiers))} variant")
    if ident["gen"] in toks and not ident["tiers"]:
        return True, "base model named in file"
    named = set()
    for seg in re.split(r"[/&,+|]|\band\b|\bvs\b", (title or "").lower()):
        st = tokens(seg)
        if any(t in TIERS or (t.isdigit() and 10 <= int(t) <= 30) for t in st):
            named.add(frozenset(t for t in st if t in TIERS))
    if len(named) > 1:
        return False, "print covers several variants and the file name does not say which"
    ttiers = {t for t in tokens(title) if t in TIERS}
    if ttiers == ident["tiers"]:
        return True, "print title names exactly this variant"
    return False, "file name does not say which variant it is"


def license_ok(name: str) -> bool:
    import cad_harvester

    low = (name or "").lower()
    if "standard digital file" in low:
        return False
    return cad_harvester.license_allowed(low.replace("\u2014", " ").replace("creative commons", "cc"))


def license_link(name: str) -> str:
    """Deed URL for a Printables / Thingiverse licence name ("Creative Commons - Attribution - Share Alike")."""
    low = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    if "cc0" in low or "publicdomain" in low:
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    if "attribution" not in low and "ccby" not in low:
        return ""
    parts = ["by"] + (["nc"] if "noncommercial" in low else [])
    parts += ["sa"] if "sharealike" in low else ["nd"] if "noderiv" in low else []
    return f"https://creativecommons.org/licenses/{'-'.join(parts)}/4.0/"


def _no_derivatives(name: str) -> bool:
    low = (name or "").lower()
    return "noderiv" in low or "nd" in re.split(r"[^a-z]+", low)


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


def _request(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = TIMEOUT_S,
             limit: int = 8 * 1024 * 1024) -> bytes:
    netguard.check(url)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        size = int(resp.headers.get("Content-Length") or 0)
        if size > limit:
            raise DummyError(f"{size / 1e6:.1f} MB exceeds the {limit / 1e6:.0f} MB limit")
        body = resp.read(limit + 1)
    if len(body) > limit:
        raise DummyError(f"download exceeds the {limit / 1e6:.0f} MB limit")
    return body


def _graphql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    out = json.loads(_request(PRINTABLES_GQL, body, {"Content-Type": "application/json", "Accept": "application/json"}))
    if out.get("errors"):
        raise DummyError(f"Printables: {out['errors'][0].get('message', out['errors'])}")
    return out.get("data") or {}


# --------------------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------------------

_PRINTABLES_SEARCH = """query($q: String!, $n: Int!) { searchPrints2(query: $q, limit: $n) { items {
  id name slug likesCount downloadCount license { name abbreviation } user { publicUsername handle }
  image { filePath } } } }"""
_PRINTABLES_FILES = """query($id: ID!) { print(id: $id) { id stls { id name fileSize } } }"""
_PRINTABLES_LINK = """mutation($id: ID!, $print: ID!) { getDownloadLink(id: $id, printId: $print, fileType: stl,
  source: model_detail) { ok output { link } errors { field messages } } }"""


def search_queries(product) -> list[str]:
    model = getattr(product, "model_or_variant", "") or getattr(product, "canonical_name", "")
    brand = getattr(product, "brand", "")
    full = model if model.lower().startswith(brand.lower()) or brand == "Samsung" and "galaxy" in model.lower() \
        else f"{brand} {model}"
    out = [f"{model} dummy", f"{full} 1:1", model]
    return list(dict.fromkeys(q.strip() for q in out if q.strip()))


def popularity(likes: int, downloads: int) -> float:
    return min(25.0, 2.5 * math.log1p(max(0, likes)) + 1.0 * math.log1p(max(0, downloads)))


def search_printables(product, limit: int = 12) -> list[dict]:
    ident = identity(product)
    seen: dict[str, dict] = {}
    for q in search_queries(product):
        try:
            items = (_graphql(_PRINTABLES_SEARCH, {"q": q, "n": limit}).get("searchPrints2") or {}).get("items") or []
        except (urllib.error.URLError, OSError, ValueError, DummyError) as exc:
            log.warning("Printables search %r failed: %s", q, exc)
            continue
        for it in items:
            pid = str(it.get("id") or "")
            if not pid or pid in seen:
                continue
            rel = title_relevance(it.get("name", ""), ident)
            lic = (it.get("license") or {}).get("name") or (it.get("license") or {}).get("abbreviation") or ""
            if rel is None or not license_ok(lic):
                continue
            user = it.get("user") or {}
            likes, dls = int(it.get("likesCount") or 0), int(it.get("downloadCount") or 0)
            img = (it.get("image") or {}).get("filePath") or ""
            seen[pid] = {
                "provider": "printables", "uid": f"printables:{pid}", "id": pid, "title": it.get("name", ""),
                "author": user.get("publicUsername") or user.get("handle") or "",
                "author_url": f"https://www.printables.com/@{user['handle']}" if user.get("handle") else "",
                "license": lic, "source_url": f"https://www.printables.com/model/{pid}-{it.get('slug') or ''}".rstrip("-"),
                "thumbnail_url": PRINTABLES_MEDIA + img if img else "", "likes": likes, "downloads": dls,
                "score": rel + popularity(likes, dls) + _license_bonus(lic), "query": q,
            }
    return sorted(seen.values(), key=lambda c: -c["score"])


def _license_bonus(lic: str) -> float:
    import cad_harvester

    return cad_harvester.license_rank(lic.lower().replace("creative commons", "cc").replace("\u2014", " "))


def printables_files(cand: dict) -> list[dict]:
    pr = _graphql(_PRINTABLES_FILES, {"id": cand["id"]}).get("print") or {}
    return [{"id": str(f["id"]), "name": f.get("name", ""), "size": int(f.get("fileSize") or 0)}
            for f in pr.get("stls") or [] if f.get("id")]


def printables_download(cand: dict, f: dict) -> bytes:
    out = (_graphql(_PRINTABLES_LINK, {"id": f["id"], "print": cand["id"]}).get("getDownloadLink") or {})
    link = (out.get("output") or {}).get("link")
    if not out.get("ok") or not link:
        raise DummyError(f"Printables refused the download: {out.get('errors')}")
    return _request(link, timeout=TIMEOUT_S * 4, limit=MAX_MESH_BYTES)


def _thingiverse_token() -> str:
    return os.environ.get("THINGIVERSE_TOKEN", "").strip()


def _tv_json(path: str, params: dict | None = None):
    if not _thingiverse_token():
        raise DummyError("THINGIVERSE_TOKEN is not set (https://www.thingiverse.com/developers)")
    query = urllib.parse.urlencode({**(params or {}), "access_token": _thingiverse_token()})
    return json.loads(_request(f"{THINGIVERSE_API}{path}?{query}", headers={"Accept": "application/json"}))


def search_thingiverse(product, limit: int = 12) -> list[dict]:
    if not _thingiverse_token():
        return []
    ident = identity(product)
    seen: dict[str, dict] = {}
    for q in search_queries(product):
        try:
            data = _tv_json(f"/search/{urllib.parse.quote(q)}", {"type": "things", "per_page": limit, "sort": "relevant"})
        except (urllib.error.URLError, OSError, ValueError, DummyError) as exc:
            log.warning("Thingiverse search %r failed: %s", q, exc)
            continue
        hits = data.get("hits") if isinstance(data, dict) else data
        for it in hits or []:
            tid = str(it.get("id") or "")
            rel = title_relevance(it.get("name", ""), ident)
            if not tid or tid in seen or rel is None:
                continue
            creator = it.get("creator") or {}
            likes = int(it.get("like_count") or 0)
            seen[tid] = {
                "provider": "thingiverse", "uid": f"thingiverse:{tid}", "id": tid, "title": it.get("name", ""),
                "author": creator.get("name", ""), "author_url": creator.get("public_url", ""), "license": "",
                "source_url": it.get("public_url") or f"https://www.thingiverse.com/thing:{tid}",
                "thumbnail_url": it.get("thumbnail") or it.get("preview_image") or "", "likes": likes,
                "downloads": int(it.get("download_count") or it.get("collect_count") or 0),
                "score": rel + popularity(likes, int(it.get("collect_count") or 0)), "query": q,
            }
    return sorted(seen.values(), key=lambda c: -c["score"])


def thingiverse_files(cand: dict) -> list[dict]:
    if not cand.get("license"):
        thing = _tv_json(f"/things/{cand['id']}")
        cand["license"] = thing.get("license", "") if isinstance(thing, dict) else ""
    if not license_ok(cand["license"]):
        raise DummyError(f"licence {cand['license']!r} not allowed")
    return [{"id": str(f["id"]), "name": f.get("name", ""), "size": int(f.get("size") or 0),
             "url": f.get("download_url") or f.get("direct_url") or ""}
            for f in _tv_json(f"/things/{cand['id']}/files") or [] if f.get("id")]


def thingiverse_download(cand: dict, f: dict) -> bytes:
    if not f.get("url"):
        raise DummyError("no download URL")
    sep = "&" if "?" in f["url"] else "?"
    return _request(f"{f['url']}{sep}access_token={urllib.parse.quote(_thingiverse_token())}",
                    timeout=TIMEOUT_S * 4, limit=MAX_MESH_BYTES)


PROVIDERS = {
    "printables": (search_printables, printables_files, printables_download),
    "thingiverse": (search_thingiverse, thingiverse_files, thingiverse_download),
}


def search(product, providers: tuple[str, ...] = ("printables", "thingiverse")) -> list[dict]:
    out: list[dict] = []
    for name in providers:
        out += PROVIDERS[name][0](product)
    return sorted(out, key=lambda c: -c["score"])


# --------------------------------------------------------------------------------------
# Mesh loading (numpy; triangles as an (n, 3, 3) float64 array)
# --------------------------------------------------------------------------------------


def load_stl(data: bytes) -> np.ndarray:
    if len(data) >= 84:
        n = int.from_bytes(data[80:84], "little")
        if 84 + 50 * n == len(data):
            rec = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(data, dtype=rec, count=n, offset=84)["v"].astype(np.float64)
    text = data.decode("ascii", "replace")
    if "facet" not in text:
        raise DummyError("not an STL file")
    nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not nums or len(nums) % 3:
        raise DummyError("malformed ASCII STL")
    return np.asarray(nums, dtype=np.float64).reshape(-1, 3, 3)


def load_obj(data: bytes) -> np.ndarray:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in data.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "f" and len(parts) >= 4:
            idx = []
            for p in parts[1:]:
                i = int(p.split("/")[0])
                idx.append(i - 1 if i > 0 else len(verts) + i)
            faces += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    if not verts or not faces:
        raise DummyError("OBJ has no faces")
    return np.asarray(verts, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]


def _3mf_matrix(attr: str | None) -> np.ndarray:
    m = np.eye(4)
    if attr:
        v = [float(x) for x in attr.split()]
        if len(v) == 12:
            m[:3, :3] = np.asarray(v[:9]).reshape(3, 3)
            m[3, :3] = v[9:]
    return m


def load_3mf(data: bytes) -> np.ndarray:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DummyError("not a 3MF (zip) file") from exc
    models = [i for i in zf.infolist() if i.filename.lower().endswith(".model")]
    if not models or sum(i.file_size for i in models) > MAX_UNZIPPED:
        raise DummyError("3MF has no model part or is too large")
    docs = {}
    for info in models:
        raw = zf.read(info)
        if b"<!ENTITY" in raw:
            raise DummyError("3MF with XML entities refused")
        docs["/" + info.filename.lstrip("/")] = ET.fromstring(raw)
    objects: dict[tuple[str, str], ET.Element] = {}
    for path, root in docs.items():
        for el in root.iter():
            if el.tag.endswith("}object") or el.tag == "object":
                objects[(path, el.get("id", ""))] = el

    def local(el: ET.Element, name: str) -> list[ET.Element]:
        return [c for c in el if c.tag.split("}")[-1] == name]

    def expand(path: str, oid: str, mat: np.ndarray, depth: int = 0) -> list[np.ndarray]:
        obj = objects.get((path, oid))
        if obj is None or depth > 8:
            return []
        out = []
        for mesh in local(obj, "mesh"):
            vs = local(mesh, "vertices")
            ts = local(mesh, "triangles")
            if not vs or not ts:
                continue
            v = np.asarray([(float(e.get("x")), float(e.get("y")), float(e.get("z"))) for e in local(vs[0], "vertex")])
            t = np.asarray([(int(e.get("v1")), int(e.get("v2")), int(e.get("v3"))) for e in local(ts[0], "triangle")],
                           dtype=np.int64)
            if len(v) and len(t):
                v = np.c_[v, np.ones(len(v))] @ mat
                out.append(v[:, :3][t])
        for comps in local(obj, "components"):
            for comp in local(comps, "component"):
                cpath = next((v for k, v in comp.attrib.items() if k.split("}")[-1] == "path"), None) or path
                out += expand("/" + cpath.lstrip("/"), comp.get("objectid", ""),
                              _3mf_matrix(comp.get("transform")) @ mat, depth + 1)
        return out

    tris: list[np.ndarray] = []
    for path, root in docs.items():
        for build in (e for e in root.iter() if e.tag.split("}")[-1] == "build"):
            for item in local(build, "item"):
                tris += expand(path, item.get("objectid", ""), _3mf_matrix(item.get("transform")))
    if not tris:
        raise DummyError("3MF build has no mesh")
    return np.concatenate(tris)


def load_mesh(data: bytes, filename: str) -> np.ndarray:
    name = filename.lower()
    if name.endswith(".3mf") or data[:2] == b"PK":
        tris = load_3mf(data)
    elif name.endswith(".obj"):
        tris = load_obj(data)
    else:
        tris = load_stl(data)
    tris = tris[np.isfinite(tris).all(axis=(1, 2))]
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    tris = tris[area > 0]
    if len(tris) < 12:
        raise DummyError("mesh has fewer than 12 triangles")
    return tris


# --------------------------------------------------------------------------------------
# Geometry: orientation, dimension gate, decimation, GLB
# --------------------------------------------------------------------------------------


def _surface_moments(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact area-weighted mean and covariance of the surface (independent of how faces are triangulated)."""
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2
    total = max(float(area.sum()), 1e-12)
    s = tris.sum(axis=1)
    mu = (s * area[:, None]).sum(0) / (3 * total)
    second = np.einsum("t,tvi,tvj->ij", area, tris, tris) + np.einsum("t,ti,tj->ij", area, s, s)
    return mu, second / (12 * total) - np.outer(mu, mu)


def _frame_from(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _bbox_volume(pts: np.ndarray, frame: np.ndarray) -> float:
    return float(np.prod(np.ptp(pts @ frame, axis=0)))


def principal_frame(tris: np.ndarray) -> np.ndarray:
    """3x3 rotation whose columns are the target X (width), Y (height), Z (thickness) axes in source coordinates.
    Candidates: the authored axes (ordered by extent) and the surface-PCA axes; the tighter bounding box wins, with
    a 3 % preference for the authored axes because dummies are almost always modelled axis-aligned."""
    pts = tris.reshape(-1, 3)
    ext = np.ptp(pts, axis=0)
    order = np.argsort(-ext)
    axis = _frame_from(np.eye(3)[order[0]], np.eye(3)[order[1]])
    _, cov = _surface_moments(tris)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, np.argsort(-vals)]
    pca = _frame_from(vecs[:, 0], vecs[:, 1])
    pe = np.ptp(pts @ pca, axis=0)
    o = np.argsort(-pe)
    pca = _frame_from(pca[:, o[0]], pca[:, o[1]])
    return pca if _bbox_volume(pts, pca) < 0.97 * _bbox_volume(pts, axis) else axis


def dimension_gate(extent_mm: np.ndarray, spec: dict) -> tuple[bool, float, str]:
    """(ok, unit scale, reason) - extent = sorted (height, width, thickness) in file units."""
    H, W, D = float(spec["height_mm"]), float(spec["width_mm"]), float(spec["depth_mm"])
    e = np.sort(np.asarray(extent_mm, dtype=np.float64))[::-1]
    best = ""
    for s in UNIT_SCALES:
        h, w, d = e * s
        dh, dw = abs(h - H) / H, abs(w - W) / W
        if dh <= DIM_TOL and dw <= DIM_TOL and 0.9 * D <= d <= D + BUMP_MAX_MM:
            return True, s, f"{h:.1f} x {w:.1f} x {d:.1f} mm vs published {H} x {W} x {D} mm"
        if s == 1.0 or not best:
            best = (f"{h:.1f} x {w:.1f} x {d:.1f} mm (x{s:g}) vs published {H} x {W} x {D} mm "
                    f"(height {dh:+.1%}, width {dw:+.1%})")
    return False, 0.0, best


def orient_phone(tris: np.ndarray, spec: dict) -> tuple[np.ndarray, dict]:
    """Rotate into the dashboard frame (mm): height +Y, width X, screen +Z, camera bump on -Z at the top."""
    frame = principal_frame(tris)
    pts = tris.reshape(-1, 3) @ frame
    ext = pts.max(0) - pts.min(0)
    ok, scale, why = dimension_gate(ext[[1, 0, 2]], spec)
    if not ok:
        raise DummyError(f"dimension gate: {why}")
    pts = pts * scale
    pts -= (pts.max(0) + pts.min(0)) / 2
    notes = [f"unit scale x{scale:g}" if scale != 1 else "millimetres"]
    zmax, zmin = pts[:, 2].max(), pts[:, 2].min()
    band = 0.35

    def footprint(sel: np.ndarray) -> float:
        p = pts[sel]
        return float(np.ptp(p[:, 0]) * np.ptp(p[:, 1])) if len(p) > 2 else 0.0

    top, bot = pts[:, 2] > zmax - band, pts[:, 2] < zmin + band
    a_top, a_bot = footprint(top), footprint(bot)
    bump = None
    if a_top < 0.6 * a_bot:
        pts[:, [0, 2]] *= -1  # bump on +Z: rotate 180 deg about Y so the screen faces +Z
        bump = top
        notes.append("flipped: camera bump to the back (-Z)")
    elif a_bot < 0.6 * a_top:
        bump = bot
    else:
        notes.append("no camera bump detected - front/back kept as authored")
    if bump is not None:
        cy = float(pts[bump, 1].mean())
        if cy < 0:
            pts[:, [0, 1]] *= -1  # rotate 180 deg about Z: camera island to the top
            notes.append("rotated 180 deg: camera island to the top")
        cx = float(pts[bump, 0].mean())
        notes.append("camera island top-left seen from the back" if cx > 0 else "camera island centred / right")
    out = pts.reshape(-1, 3, 3)
    e = out.reshape(-1, 3)
    dims = np.ptp(e, axis=0)
    return out, {"dimension_check": why, "measured_mm": {"width": round(float(dims[0]), 2),
                 "height": round(float(dims[1]), 2), "thickness": round(float(dims[2]), 2)}, "notes": notes}


def decimate(tris: np.ndarray, target: int = MAX_TRIS) -> np.ndarray:
    """Vertex-clustering simplification to <= target triangles (only used for very dense scans)."""
    if len(tris) <= target:
        return tris
    v = tris.reshape(-1, 3)
    lo = v.min(0)
    cell = float(np.linalg.norm(np.ptp(v, axis=0))) / 1500.0
    for _ in range(20):
        q = np.floor((v - lo) / cell).astype(np.int64)
        key = (q[:, 0] * 1_000_003 + q[:, 1]) * 1_000_003 + q[:, 2]
        _, inv = np.unique(key, return_inverse=True)
        rep = np.zeros((inv.max() + 1, 3))
        np.add.at(rep, inv, v)
        rep /= np.bincount(inv)[:, None]
        f = inv.reshape(-1, 3)
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        if len(f) <= target:
            return rep[f]
        cell *= 1.35
    raise DummyError("mesh could not be simplified under the triangle budget")


def to_glb(tris_mm: np.ndarray, extras: dict) -> bytes:
    """Indexed GLB in metres with smooth normals except across creases sharper than CREASE_DEG."""
    v = tris_mm.reshape(-1, 3) / 1000.0
    eps = max(float(np.linalg.norm(np.ptp(v, axis=0))) * 1e-6, 1e-9)
    q = np.round(v / eps).astype(np.int64)
    q -= q.min(0)
    key = (q[:, 0] * 2_000_003 + q[:, 1]) * 2_000_003 + q[:, 2]
    _, wid = np.unique(key, return_inverse=True)
    fn = np.cross(tris_mm[:, 1] - tris_mm[:, 0], tris_mm[:, 2] - tris_mm[:, 0])
    fu = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = np.zeros((wid.max() + 1, 3))
    np.add.at(vn, wid, np.repeat(fn, 3, axis=0))
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    cf = np.repeat(fu, 3, axis=0)
    smooth = (cf * vn[wid]).sum(1) >= math.cos(math.radians(CREASE_DEG))
    nb = np.clip(np.round(cf * 8).astype(np.int64) + 8, 0, 16)
    bucket = np.where(smooth, 0, nb[:, 0] * 289 + nb[:, 1] * 17 + nb[:, 2] + 1)
    _, first, cidx = np.unique(wid * 5000 + bucket, return_index=True, return_inverse=True)
    nrm = np.zeros((len(first), 3))
    np.add.at(nrm, cidx, np.where(smooth[:, None], vn[wid], cf))
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    m = procedural_glb._Mesh()
    m.add(v[first].astype(np.float32), nrm.astype(np.float32), cidx.astype(np.uint32), SURFACE, 0.8, 0.35)
    return m.to_glb(extras)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def convert(data: bytes, filename: str, spec: dict, extras: dict | None = None,
            allow_simplify: bool = True) -> tuple[bytes, dict]:
    """Mesh file bytes -> (GLB bytes, report). Raises DummyError when the mesh is unreadable or fails the gate."""
    tris = load_mesh(data, filename)
    oriented, report = orient_phone(tris, spec)
    report["triangles_source"] = int(len(oriented))
    if len(oriented) > MAX_TRIS:
        if not allow_simplify:
            raise DummyError(f"{len(oriented)} triangles exceeds {MAX_TRIS} and the licence forbids simplification")
        oriented = decimate(oriented)
        report["notes"].append(f"simplified {report['triangles_source']} -> {len(oriented)} triangles")
    report["triangles"] = int(len(oriented))
    glb = to_glb(oriented, {"kind": "kamikaze-dummy-v1", "source_file": filename, **report, **(extras or {})})
    return glb, report


def harvest_dummy(product, spec: dict, deadline: float, providers: tuple[str, ...] = ("printables", "thingiverse"),
                  cands: list[dict] | None = None) -> tuple[dict, bytes] | None:
    """Best candidate whose file passes the variant check and the dimension gate -> (candidate + report, GLB)."""
    ident = identity(product)
    if not spec.get("verified"):
        raise DummyError(f"no verified dimensions to check dummies against: {spec.get('reason', '')}")
    tried = 0
    for cand in cands if cands is not None else search(product, providers):
        if time.monotonic() > deadline or tried >= FILE_TRIES:
            break
        _, list_files, download = PROVIDERS[cand["provider"]]
        try:
            files = list_files(cand)
        except (urllib.error.URLError, OSError, ValueError, DummyError) as exc:
            log.info("dummy %r: file list failed: %s", cand["title"], exc)
            continue
        picks = []
        for f in files:
            ok, why = file_matches(f["name"], cand["title"], ident)
            if ok and (not f.get("size") or f["size"] <= MAX_MESH_BYTES):
                picks.append(f)
        picks.sort(key=lambda f: (f["name"].lower().endswith(".stl") is False, -len(set(tokens(f["name"])) & ident["tiers"])))
        for f in picks[:2]:
            if time.monotonic() > deadline or tried >= FILE_TRIES:
                break
            tried += 1
            try:
                data = download(cand, f)
                glb, report = convert(data, f["name"], spec, {"source_url": cand["source_url"]},
                                      allow_simplify=not _no_derivatives(cand["license"]))
            except (urllib.error.URLError, OSError, ValueError, DummyError, zipfile.BadZipFile, ET.ParseError) as exc:
                log.info("dummy %r / %s rejected: %s", cand["title"], f["name"], exc)
                continue
            log.info("dummy %r / %s accepted: %s", cand["title"], f["name"], report["dimension_check"])
            return {**cand, "file": f["name"], "report": report}, glb
    return None
