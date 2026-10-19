#!/usr/bin/env python3
"""Local test server for Kamikaze — runs on any dev machine (Windows / Mac / Linux).

Mirrors the receiver.py HTTP ingest API so that tools/stream_cam.py works against
it unmodified.  No board, ADB tunnel, or Modulino hardware required.

What differs from the real board receiver.py
--------------------------------------------
* No MessagePack-RPC / arduino-router socket calls — LED output is printed to
  stdout as ANSI colour blocks so you can see what the board *would* light up.
* No QNN / Hexagon NPU — inference always uses the CPU PyTorch path (or an
  ExecuTorch portable backend if a valid .pte is found at MODEL_PATH).
* Runs on localhost; no adb forward needed — just point stream_cam.py at
  http://127.0.0.1:5000.

Quick start
-----------
  # From repo root:
  python -m venv venv
  # Windows:   .\\venv\\Scripts\\activate
  # Mac/Linux: source venv/bin/activate
  pip install -r requirements-local.txt

  # Terminal 1 — start the mock receiver:
  python tools/local_test_server.py

  # Terminal 2 — stream your webcam at it:
  python tools/stream_cam.py --url http://127.0.0.1:5000

  # Or post a single test image:
  # curl -s -X POST http://127.0.0.1:5000/api/ingest
  #      -H "Content-Type: image/jpeg" --data-binary @<some_image.jpg>

Environment variables (all optional, same names as the real receiver)
---------------------------------------------------------------------
  MODEL_PATH     path to a .pte file (default: models/model.pte)
  KAMIKAZE_PORT  listen port (default: 5000)
  KAMIKAZE_MODE  training | deployment  (default: training)
"""

import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour helpers — simulate what the Modulino Pixels would show on the board
# ---------------------------------------------------------------------------
_ANSI = {
    "red":   "\033[41m",
    "green": "\033[42m",
    "blue":  "\033[44m",
    "reset": "\033[0m",
}
_COLOR_NAMES = ["red", "green", "blue"]  # index 1-3 matching receiver.py dominant logic


def _show_leds(dominant: int, count: int, label: str) -> None:
    """Print an ASCII approximation of the 8 Modulino Pixels to stdout."""
    name = _COLOR_NAMES[dominant - 1] if 1 <= dominant <= 3 else "white"
    filled = _ANSI.get(name, "") + "  " + _ANSI["reset"]
    empty = "░░"
    bar = "".join(filled if i < count else empty for i in range(8))
    sys.stdout.write(f"\r[PIXELS] {bar}  ({name.upper()} x{count})  {label[:40]:<40}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Model bootstrap — try ExecuTorch portable backend, fall back to raw torch
# ---------------------------------------------------------------------------
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.pte"))

_et_prog = None
try:
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )
    if MODEL_PATH.exists():
        buf = MODEL_PATH.read_bytes()
        if buf[:4] == b"ET12":
            _et_prog = _load_for_executorch_from_buffer(buf)
            print(f"[MODEL] ExecuTorch portable backend active ({MODEL_PATH})")
        else:
            print(f"[MODEL] {MODEL_PATH} not a valid ET12 binary — using torch fallback")
    else:
        print(f"[MODEL] {MODEL_PATH} not found — using torch fallback")
except Exception as exc:
    print(f"[MODEL] ExecuTorch unavailable ({exc}) — using torch fallback")

try:
    import numpy as np
    import torch
    from PIL import Image

    class _FallbackModel(torch.nn.Module):
        """Tiny linear model matching the (1,4)->(1,2) shape used on the board."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)
            with torch.no_grad():
                self.fc.weight.copy_(
                    torch.tensor([[0.5, 0.2, 0.1, 0.2], [-0.3, 0.4, 0.4, -0.1]])
                )
                self.fc.bias.copy_(torch.tensor([0.1, -0.1]))

    _torch_model = _FallbackModel().eval()
    print("[MODEL] Torch CPU fallback ready")
except ImportError as exc:
    print(f"[FATAL] Missing dependency: {exc}")
    print("        Run: pip install -r requirements-local.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
_frame_count = 0
_TRAINING = os.environ.get("KAMIKAZE_MODE", "training") == "training"


def _infer(jpeg_bytes: bytes) -> dict:
    global _frame_count

    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img_np = np.array(image, dtype=np.float32) / 255.0
    mean_val = float(np.mean(img_np))
    r_val    = float(np.mean(img_np[:, :, 0]))
    g_val    = float(np.mean(img_np[:, :, 1]))
    b_val    = float(np.mean(img_np[:, :, 2]))

    tensor_in = torch.tensor(
        [[mean_val, r_val, g_val, b_val]], dtype=torch.float32
    ).contiguous()

    t0 = time.perf_counter()
    output_vals: list
    if _et_prog is not None:
        try:
            out = _et_prog.forward([tensor_in])
            output_vals = out[0][0].tolist()
        except Exception:
            with torch.no_grad():
                output_vals = _torch_model(tensor_in)[0].tolist()
    else:
        with torch.no_grad():
            output_vals = _torch_model(tensor_in)[0].tolist()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    dominant   = int(np.argmax([r_val, g_val, b_val]) + 1)
    num_pixels = int(np.clip(mean_val * 8, 1, 8))
    label      = _COLOR_NAMES[dominant - 1].upper()

    _frame_count += 1
    _show_leds(dominant, num_pixels, f"frame {_frame_count:05d} | {dt_ms:.1f} ms")

    return {
        "frame":      _frame_count,
        "dt_ms":      round(dt_ms, 2),
        "dominant":   label,
        "pixels":     num_pixels,
        "r":          round(r_val, 4),
        "g":          round(g_val, 4),
        "b":          round(b_val, 4),
        "output":     [round(v, 4) for v in output_vals],
        # Fields that stream_cam.py reads from the board verdict:
        "fps":        round(1000.0 / max(dt_ms, 0.01), 1),
        "state":      "LOCAL_TEST",
        "label":      label,
        "similarity": None,
    }


# ---------------------------------------------------------------------------
# HTTP server — compatible with stream_cam.py and receiver.py's real API
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path not in ("/api/ingest", "/upload"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        try:
            verdict = _infer(data)
            body = json.dumps(verdict).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode())

    def do_GET(self) -> None:
        if self.path == "/api/ingest":
            body = json.dumps({"stream": {"ingested": _frame_count}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            body = json.dumps({"state": "LOCAL_TEST", "fps": 0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        pass  # silence per-request log spam


def main() -> None:
    port = int(os.environ.get("KAMIKAZE_PORT", 5000))
    server = HTTPServer(("0.0.0.0", port), _Handler)
    mode = "TRAINING (preview frame kept in RAM)" if _TRAINING else "DEPLOYMENT (no frame stored)"
    print(f"\n[SERVER] Local Kamikaze test receiver → http://127.0.0.1:{port}")
    print(f"[SERVER] Mode: {mode}")
    print(f"[SERVER] Stream from another terminal:")
    print(f"         python tools/stream_cam.py --url http://127.0.0.1:{port}")
    print("[SERVER] ANSI LED simulation ↓\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] stopped")


if __name__ == "__main__":
    main()
