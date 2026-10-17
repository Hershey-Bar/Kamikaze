#!/usr/bin/env python3
"""Live hardware + inference verification, run ON the VENTUNO Q.

  python verify_live.py [--url http://127.0.0.1:5000] [--images /tmp/kz] [--skip-leds]

Checks (each prints PASS/FAIL, exit code 1 if any FAIL):
  1. Raw MessagePack-RPC to /var/run/arduino-router.sock: green, amber, cyan (you confirm visually)
  2. Embedder latency on a dummy 224x224x3 tensor (<30 ms median budget)
  3. Few-shot enrollment + cosine similarity on synthetic embeddings (temporary bank file)
  4. Optional: real photos (--images DIR, files named <label>_<n>.jpg) enrolled and cross-scored
  5. Receiver HTTP: GET / (200 + HTML), /api/state, /api/asset, synthetic frame POST
  6. End-to-end: enroll a temporary class through the HTTP API, identify it, check LEDs and asset, clean up
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import msgpack
import numpy as np
from PIL import Image

import receiver

ROUTER_SOCK = "/var/run/arduino-router.sock"
LATENCY_BUDGET_MS = 30.0
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    return ok


def rpc(msg_id: int, params: list[int]) -> list:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        s.connect(ROUTER_SOCK)
        s.sendall(msgpack.packb([0, msg_id, "set_pixels", params]))
        unpacker = msgpack.Unpacker(raw=False)
        while True:
            chunk = s.recv(4096)
            if not chunk:
                raise ConnectionError("router closed the connection before replying")
            unpacker.feed(chunk)
            for msg in unpacker:
                return msg


def test_leds() -> None:
    print("\n== 1. Modulino Pixels via /var/run/arduino-router.sock ==")
    steps = [("green (IDENTIFIED)", [0, 255, 0, 50, 8]), ("amber (GUIDING)", [255, 120, 0, 40, 4]), ("cyan", [0, 255, 255, 40, 8])]
    for i, (name, params) in enumerate(steps, start=1):
        try:
            reply = rpc(100 + i, params)
            ok = isinstance(reply, list) and len(reply) == 4 and reply[0] == 1 and reply[1] == 100 + i and reply[2] is None
            check(f"RPC set_pixels{tuple(params)} {name}", ok, f"response={reply}")
        except (OSError, ConnectionError, TimeoutError) as exc:
            check(f"RPC set_pixels{tuple(params)} {name}", False, f"{type(exc).__name__}: {exc}")
        time.sleep(1.2)


def test_latency() -> receiver.FeatureExtractor | None:
    print("\n== 2. Embedder latency (dummy 224x224x3 tensor) ==")
    try:
        extractor = receiver.FeatureExtractor()
    except Exception as exc:
        check("embedder loads", False, f"{type(exc).__name__}: {exc}")
        return None
    check("embedder loads", True, f"backend={extractor.name}")
    rng = np.random.default_rng(0)
    arr = np.ascontiguousarray(rng.standard_normal((1, 3, 224, 224)).astype(np.float32))
    for _ in range(5):
        extractor.embed_array(arr)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        vec = extractor.embed_array(arr)
        times.append((time.perf_counter() - t0) * 1000)
    med, p95 = statistics.median(times), sorted(times)[int(0.95 * len(times)) - 1]
    check("embedding shape/norm", tuple(vec.shape) == (receiver.EMBED_DIM,) and abs(float(vec.norm()) - 1.0) < 1e-3, f"shape={tuple(vec.shape)}")
    check(f"embedding latency < {LATENCY_BUDGET_MS:.0f} ms", med < LATENCY_BUDGET_MS, f"median={med:.1f}ms p95={p95:.1f}ms over 50 runs")
    return extractor


def test_bank() -> None:
    print("\n== 3. Few-shot enrollment + similarity (synthetic) ==")
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bank.pt"
        bank = receiver.EmbeddingBank(path)
        g = torch.Generator().manual_seed(1)
        axes = {k: torch.nn.functional.normalize(torch.randn(receiver.EMBED_DIM, generator=g), dim=0) for k in ("alpha", "beta")}
        for label, axis in axes.items():
            vecs = [torch.nn.functional.normalize(axis + 0.01 * torch.randn(receiver.EMBED_DIM, generator=g), dim=0) for _ in range(12)]
            info = bank.add(label, vecs)
            check(f"enroll {label}", info["vectors"] == 12 and info["consistency"] > 0.9, str(info))
        ranked = bank.scores(axes["alpha"])
        check("similarity ranks the right class", ranked[0][0] == "alpha" and ranked[0][1] > 0.95 and ranked[1][1] < 0.3, f"scores={[(k, round(v, 3)) for k, v in ranked]}")
        reloaded = receiver.EmbeddingBank(path)
        check("bank persistence round-trip", reloaded.load() == 2 and sorted(reloaded.labels()) == ["alpha", "beta"], f"file={path.name}")
        check("unenroll", bank.remove("beta") and bank.labels() == ["alpha"])


def test_images(extractor, folder: Path) -> None:
    print(f"\n== 4. Real photos from {folder} ==")
    groups: dict[str, list] = {}
    for f in sorted(folder.glob("*.jpg")):
        label = f.stem.rsplit("_", 1)[0]
        groups.setdefault(label, []).append(extractor.embed_image(Image.open(f).convert("RGB")))
    if len(groups) < 2:
        check("photos found", False, f"need >=2 labels in {folder}, found {list(groups)}")
        return
    bank = receiver.EmbeddingBank(None)
    for label, vecs in groups.items():
        print(f"   enrolled {label}: {bank.add(label, vecs)}")
    correct = total = 0
    for label, vecs in groups.items():
        for v in vecs:
            total += 1
            correct += bank.scores(v)[0][0] == label
    check("self-retrieval on the enrollment photos", correct == total, f"{correct}/{total} top-1 correct")


def http(method: str, base: str, path: str, body: bytes | None = None, ctype: str = "application/json", timeout: float = 15.0):
    req = urllib.request.Request(base + path, data=body, method=method, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def synthetic_jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((640, 480), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def test_http(base: str) -> bool:
    print(f"\n== 5. Receiver HTTP ({base}) ==")
    try:
        code, body = http("GET", base, "/")
    except OSError as exc:
        check("GET / reachable", False, f"{exc} - start ./run_receiver.sh first")
        return False
    check("GET / -> 200 + HTML dashboard", code == 200 and b"<html" in body.lower() and b"model-viewer" in body, f"{code}, {len(body)} bytes")
    code, body = http("GET", base, "/api/state")
    state = json.loads(body) if code == 200 else {}
    check("GET /api/state", code == 200 and state.get("state") in ("SEARCHING", "GUIDING", "IDENTIFIED"), f"state={state.get('state')} backend={state.get('backend')}")
    code, body = http("GET", base, "/api/asset")
    asset = json.loads(body) if code == 200 else {}
    check("GET /api/asset schema", code == 200 and {"label", "confidence", "glb_url", "camera_orbit"} <= set(asset), str(asset))
    code, body = http("POST", base, "/api/frame", synthetic_jpeg(1), "image/jpeg")
    verdict = json.loads(body) if code == 200 else {}
    check("POST /api/frame synthetic JPEG", code == 200 and verdict.get("ok") is True, f"embed={verdict.get('embed_ms')}ms total={verdict.get('latency_ms')}ms")
    code, _ = http("POST", base, "/api/frame", b"not a jpeg", "image/jpeg")
    check("POST /api/frame rejects garbage with 400", code == 400, f"{code}")
    code, _ = http("GET", base, "/static/../receiver.py")
    check("static path traversal blocked", code == 404, f"{code}")
    code, body = http("GET", base, "/static/models/default.glb")
    check("GET /static/models/default.glb (GLB served)", code == 200 and body[:4] == b"glTF", f"{code}, {len(body)} bytes")
    return True


def solid_jpeg(value: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (value, value, value)).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def test_end_to_end(base: str) -> None:
    print("\n== 6. End-to-end: enroll -> identify -> LEDs -> asset -> cleanup ==")
    label = "verify_tmp_object"
    obj_a, obj_b = synthetic_jpeg(11), solid_jpeg(200)
    try:
        http("POST", base, "/api/unenroll", json.dumps({"label": label}).encode())
        code, body = http("POST", base, "/api/enroll", json.dumps({"label": label, "frames": 4}).encode())
        check("POST /api/enroll accepted", code == 202, body.decode()[:120])
        for _ in range(4):
            http("POST", base, "/api/frame", obj_a, "image/jpeg")
        _, body = http("GET", base, "/api/state")
        st = json.loads(body)
        check("enrollment completed from live frames", label in st["classes"] and st["enrollment"]["ok"], f"classes={st['classes']}")
        verdict = {}
        for _ in range(4):
            _, body = http("POST", base, "/api/frame", obj_a, "image/jpeg")
            verdict = json.loads(body)
        check("same object -> IDENTIFIED", verdict.get("state") == "IDENTIFIED" and verdict.get("label") == label, f"{verdict}")
        time.sleep(1.5)
        _, body = http("GET", base, "/api/state")
        led = json.loads(body)["led"]
        check("recognition LEDs commanded green (count>=1)", led["g"] == 255 and led["r"] == 0 and led["count"] >= 1 and led["ok"] is True, f"{led}")
        _, body = http("GET", base, "/api/asset")
        asset = json.loads(body)
        check("asset payload for identified object", asset["state"] == "IDENTIFIED" and asset["label"] == label and asset["glb_url"].startswith("/static/models/"), f"{asset}")
        for _ in range(6):
            _, body = http("POST", base, "/api/frame", obj_b, "image/jpeg")
        verdict = json.loads(body)
        check("different object leaves IDENTIFIED", verdict.get("state") != "IDENTIFIED", f"{verdict}")
    finally:
        code, _ = http("POST", base, "/api/unenroll", json.dumps({"label": label}).encode())
        check("cleanup (unenroll temporary class)", code in (200, 404), f"{code}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--images", type=Path, help="folder of <label>_<n>.jpg photos for the real-image test")
    parser.add_argument("--skip-leds", action="store_true")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    if not args.skip_leds:
        test_leds()
    extractor = test_latency()
    if extractor is not None:
        test_bank()
        if args.images:
            test_images(extractor, args.images)
    if test_http(base):
        test_end_to_end(base)

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
