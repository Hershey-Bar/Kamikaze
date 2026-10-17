#!/usr/bin/env python3
"""Kamikaze edge inference + orchestration server for the Arduino VENTUNO Q.

Receives JPEG frames over HTTP POST (stream_cam.py -> adb tunnel -> 0.0.0.0:5000), classifies
them against the constrained target domain (Apple iPhone 12-16, Samsung Galaxy S20-S24,
Coca-Cola / Diet Coke / Coke Zero / Pepsi / Sprite cans) and drives the Modulino Pixels and the
3D-asset lookup through the tools in mcp_server.py.

Classification chain (per frame):
  1. Fast numpy color/shape classifier (milliseconds, no model runtime needed).
  2. Optional InternVL3 prompt evaluation running asynchronously in a worker thread, enabled by
     KAMIKAZE_VLM_CMD (a command template with {image} and {prompt} placeholders that prints the
     model's answer). Its latest fresh answer overrides the fast result when more confident, which
     is how phone *models* get identified (the fast path can only recognise "a smartphone").

Decision logic:
  - High-confidence label held for KAMIKAZE_CONFIRM consecutive frames
      -> LEDs solid green (0,255,0, 50, 8), search_3d_asset(label), emit asset payload.
  - Low confidence / ambiguous / nothing detected for KAMIKAZE_GUIDE_AFTER frames
      -> trigger_guidance (amber, 4 LEDs) + rotation instructions.

Frontend endpoints (CORS enabled): GET /asset (latest asset payload), GET /events?since=N,
GET /state. Asset payloads are also printed to stdout as "ASSET_READY_PAYLOAD {json}".

Optional header on POST for demo overrides: X-Kamikaze-Hint: <target label>.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, UnidentifiedImageError

from mcp_server import (
    CAN_LABELS,
    PHONE_LABELS,
    TARGET_LABELS,
    actuate_modulino,
    label_category,
    search_3d_asset,
    trigger_guidance,
)

log = logging.getLogger("kamikaze.receiver")

CONF_HIGH = float(os.environ.get("KAMIKAZE_CONF_HIGH", "0.75"))
CONFIRM_FRAMES = int(os.environ.get("KAMIKAZE_CONFIRM", "3"))
GUIDE_AFTER_FRAMES = int(os.environ.get("KAMIKAZE_GUIDE_AFTER", "4"))
GUIDE_INTERVAL = float(os.environ.get("KAMIKAZE_GUIDE_INTERVAL", "5.0"))
VLM_CMD = os.environ.get("KAMIKAZE_VLM_CMD", "")
VLM_TTL = float(os.environ.get("KAMIKAZE_VLM_TTL", "4.0"))
VLM_TIMEOUT = float(os.environ.get("KAMIKAZE_VLM_TIMEOUT", "20"))
VLM_ALWAYS = os.environ.get("KAMIKAZE_VLM_ALWAYS", "0") == "1"
MAX_FRAME_BYTES = 5 * 1024 * 1024

GUIDANCE_PROMPTS = {
    "phone": "Rotate 90° to show the camera island or branding",
    "can": "Rotate the can to show the logo and flavor label",
    None: "No target detected. Hold a phone or soda can in the center of the frame, 20-40 cm away",
}


@dataclass
class Detection:
    label: str | None
    category: str | None
    confidence: float
    source: str
    note: str = ""


NO_DETECTION = Detection(None, None, 0.0, "none", "nothing detected")


# --------------------------------------------------------------------------------------
# Label normalisation
# --------------------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", text.lower()).strip()


def _build_label_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for label in TARGET_LABELS:
        n = _norm(label)
        keys = {n, n.replace("apple ", "").replace("samsung ", ""), n.replace("samsung galaxy ", "")}
        for key in list(keys):
            keys.add(key.replace(" plus", "+"))
        for key in keys:
            index[key] = label
    index.update(
        {
            "coca cola": "Coca-Cola Classic",
            "coca cola classic": "Coca-Cola Classic",
            "coke": "Coca-Cola Classic",
            "coke classic": "Coca-Cola Classic",
            "diet coca cola": "Diet Coke",
            "coca cola zero": "Coke Zero",
            "pepsi cola": "Pepsi",
        }
    )
    return index


_LABEL_INDEX = _build_label_index()


def match_label(text: str) -> str | None:
    """Return the canonical target label named in text (longest match wins), else None."""
    n = f" {_norm(text)} "
    best: tuple[int, str] | None = None
    for key, label in _LABEL_INDEX.items():
        if f" {key} " in n and (best is None or len(key) > best[0]):
            best = (len(key), label)
    return best[1] if best else None


# --------------------------------------------------------------------------------------
# Fast numpy color / shape classifier
# --------------------------------------------------------------------------------------


def _extent(mask: np.ndarray) -> tuple[int, int, float] | None:
    """Robust bounding box (height, width, fill ratio) of a boolean mask."""
    h, w = mask.shape
    rows = np.flatnonzero(mask.sum(axis=1) > max(1, 0.02 * w))
    cols = np.flatnonzero(mask.sum(axis=0) > max(1, 0.02 * h))
    if rows.size == 0 or cols.size == 0:
        return None
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    box = mask[r0:r1, c0:c1]
    return int(r1 - r0), int(c1 - c0), float(box.mean())


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _can_score(mask: np.ndarray) -> float:
    cov = float(mask.mean())
    ext = _extent(mask)
    if ext is None or cov < 0.01:
        return 0.0
    bh, bw, fill = ext
    aspect = max(bh, bw) / max(1, min(bh, bw))
    aspect_score = 1.0 if 1.4 <= aspect <= 3.2 else 0.5 if 1.15 <= aspect < 1.4 else 0.2
    score = 0.55 * _clip01(cov / 0.15) + 0.25 * aspect_score + 0.20 * _clip01((fill - 0.35) / 0.35)
    if cov > 0.65:
        score *= 0.4
    return _clip01(score)


class ColorShapeClassifier:
    """Fast heuristic: a can is a dominant-color, upright, filled rounded rectangle."""

    def classify(self, image: Image.Image) -> Detection:
        hsv = np.asarray(image.convert("RGB").resize((160, 160)).convert("HSV"), dtype=np.float32)
        hue = hsv[..., 0] * (360.0 / 255.0)
        sat = hsv[..., 1] / 255.0
        val = hsv[..., 2] / 255.0

        red = ((hue < 12) | (hue > 345)) & (sat > 0.55) & (val > 0.30)
        blue = (hue > 195) & (hue < 250) & (sat > 0.50) & (val > 0.20)
        green = (hue > 85) & (hue < 160) & (sat > 0.45) & (val > 0.25)
        dark = val < 0.22
        silver = (sat < 0.14) & (val > 0.60)

        red_cov = float(red.mean())
        candidates: list[tuple[float, str]] = [
            (_can_score(red), "Coca-Cola Classic"),
            (_can_score(blue), "Pepsi"),
            (_can_score(green), "Sprite"),
        ]
        if 0.02 <= red_cov <= 0.25:
            candidates.append((min(0.80, _can_score(silver) * 0.9), "Diet Coke"))
            candidates.append((min(0.80, _can_score(dark) * 0.9), "Coke Zero"))

        score, label = max(candidates, key=lambda c: c[0])
        if score >= 0.35:
            return Detection(label, "can", round(score, 3), "color-shape", "can-like dominant color blob")

        if red_cov < 0.02:
            ext = _extent(dark)
            cov = float(dark.mean())
            if ext and 0.12 <= cov <= 0.75:
                bh, bw, fill = ext
                aspect = max(bh, bw) / max(1, min(bh, bw))
                if 1.6 <= aspect <= 2.4 and fill >= 0.75:
                    return Detection(None, "phone", 0.5, "color-shape", "smartphone-like slab; model not identified")
        return NO_DETECTION


# --------------------------------------------------------------------------------------
# Optional InternVL3 prompt evaluation (async)
# --------------------------------------------------------------------------------------

VLM_PROMPT = (
    "Identify the single object held in front of the camera. It must be exactly one of: "
    + "; ".join(TARGET_LABELS)
    + '. Answer with JSON only: {"label": "<one label from the list or none>", '
    '"confidence": <0.0-1.0>}. If the exact model or flavor cannot be determined from what is '
    "visible, answer none with a low confidence."
)


def parse_vlm_output(text: str) -> Detection:
    for blob in re.findall(r"\{[^{}]*\}", text, flags=re.S):
        try:
            data = json.loads(blob)
        except ValueError:
            continue
        label = match_label(str(data.get("label", "")))
        try:
            conf = _clip01(float(data.get("confidence", 0.8)))
        except (TypeError, ValueError):
            conf = 0.8
        if label:
            return Detection(label, label_category(label), conf, "internvl3")
        return Detection(None, None, 0.0, "internvl3", "model answered none")
    label = match_label(text)
    if label:
        return Detection(label, label_category(label), 0.8, "internvl3", "free-text answer")
    return Detection(None, None, 0.0, "internvl3", "unparseable answer")


class VlmWorker:
    """Runs the VLM command off the request path so the camera never blocks on it."""

    def __init__(self, template: str) -> None:
        self.argv = shlex.split(template) if template else []
        self.enabled = bool(self.argv)
        self._busy = threading.Event()
        self._lock = threading.Lock()
        self._result: tuple[Detection, float] | None = None

    def submit(self, image: Image.Image) -> None:
        if not self.enabled or self._busy.is_set():
            return
        self._busy.set()
        threading.Thread(target=self._run, args=(image.copy(),), daemon=True).start()

    def _run(self, image: Image.Image) -> None:
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                path = tmp.name
            image.save(path, "JPEG", quality=90)
            argv = [a.replace("{image}", path).replace("{prompt}", VLM_PROMPT) for a in self.argv]
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=VLM_TIMEOUT, check=False)
            det = parse_vlm_output(proc.stdout)
            log.info("VLM -> %s (%.2f)", det.label, det.confidence)
            with self._lock:
                self._result = (det, time.monotonic())
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("VLM run failed: %s", exc)
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self._busy.clear()

    def fresh_result(self) -> Detection | None:
        with self._lock:
            if self._result and time.monotonic() - self._result[1] <= VLM_TTL:
                return self._result[0]
        return None


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------


class Orchestrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._led_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="led")
        self._asset_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asset")
        self._events: collections.deque[dict] = collections.deque(maxlen=100)
        self._seq = 0
        self._epoch = 0
        self.state = "idle"
        self.locked_label: str | None = None
        self._streak_label: str | None = None
        self._streak = 0
        self._low = 0
        self._last_guidance = 0.0
        self._last_category: str | None = None
        self.frames = 0
        self.latest_asset: dict | None = None
        self.last_detection: Detection = NO_DETECTION

    def _emit(self, event: dict) -> dict:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "timestamp": time.time(), **event}
            self._events.append(event)
            if event["event"] == "asset_ready":
                self.latest_asset = event
        print(f"{event['event'].upper()}_PAYLOAD {json.dumps(event, ensure_ascii=False)}", flush=True)
        return event

    def events_since(self, seq: int) -> list[dict]:
        with self._lock:
            return [e for e in self._events if e["seq"] > seq]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "locked_label": self.locked_label,
                "frames": self.frames,
                "last_detection": self.last_detection.__dict__,
                "seq": self._seq,
            }

    def _do_lock(self, det: Detection, epoch: int) -> None:
        self._led_pool.submit(actuate_modulino, 0, 255, 0, 50, 8)
        try:
            asset = search_3d_asset(det.label)
        except Exception:
            log.exception("search_3d_asset failed for %s", det.label)
            return
        with self._lock:
            stale = epoch != self._epoch
        if stale:
            log.info("Dropping stale asset result for %s", det.label)
            return
        self._emit(
            {
                "event": "asset_ready",
                "label": det.label,
                "category": det.category,
                "confidence": det.confidence,
                "source": det.source,
                "asset": asset,
            }
        )

    def _do_guidance(self, reason: str, message: str, category: str | None) -> None:
        def run() -> None:
            try:
                result = trigger_guidance(reason, message)
            except Exception:
                log.exception("trigger_guidance failed")
                return
            self._emit({"event": "guidance", "reason": reason, "category": category, **result})

        self._led_pool.submit(run)

    def process(self, det: Detection) -> dict:
        now = time.monotonic()
        with self._lock:
            self.frames += 1
            self.last_detection = det
            if det.category:
                self._last_category = det.category
            confident = det.label is not None and det.confidence >= CONF_HIGH
            action: tuple | None = None

            if confident:
                self._low = 0
                if det.label == self._streak_label:
                    self._streak += 1
                else:
                    self._streak_label, self._streak = det.label, 1
                if self.locked_label == det.label:
                    self.state = "locked"
                elif self._streak >= CONFIRM_FRAMES:
                    self.locked_label = det.label
                    self.state = "locked"
                    self._epoch += 1
                    action = ("lock", det, self._epoch)
                else:
                    self.state = "confirming"
            else:
                self._streak_label, self._streak = None, 0
                self._low += 1
                if self._low >= GUIDE_AFTER_FRAMES:
                    self.state = "guidance"
                    if now - self._last_guidance >= GUIDE_INTERVAL:
                        self._last_guidance = now
                        self.locked_label = None
                        self._epoch += 1
                        category = det.category or self._last_category
                        reason = "no_target" if det.category is None else "low_confidence"
                        action = ("guide", reason, GUIDANCE_PROMPTS.get(category, GUIDANCE_PROMPTS[None]), category)

            state, locked = self.state, self.locked_label

        if action and action[0] == "lock":
            self._asset_pool.submit(self._do_lock, action[1], action[2])
        elif action:
            self._do_guidance(action[1], action[2], action[3])
        return {
            "state": state,
            "label": det.label,
            "category": det.category,
            "confidence": det.confidence,
            "source": det.source,
            "locked_label": locked,
        }


# --------------------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------------------

fast_classifier = ColorShapeClassifier()
vlm = VlmWorker(VLM_CMD)
orchestrator = Orchestrator()


def classify_frame(image: Image.Image, hint: str | None) -> Detection:
    if hint:
        label = match_label(hint)
        if label:
            return Detection(label, label_category(label), 1.0, "hint")
    det = fast_classifier.classify(image)
    if vlm.enabled:
        if VLM_ALWAYS or det.category is not None:
            vlm.submit(image)
        vlm_det = vlm.fresh_result()
        if vlm_det and vlm_det.label and vlm_det.confidence >= det.confidence:
            if det.category is None or det.category == vlm_det.category:
                return vlm_det
    return det


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, code: int, body: object) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Kamikaze-Hint")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/asset":
            asset = orchestrator.latest_asset
            self._send_json(200 if asset else 404, asset or {"error": "no asset yet"})
        elif url.path == "/events":
            try:
                since = int(parse_qs(url.query).get("since", ["0"])[0])
            except ValueError:
                since = 0
            self._send_json(200, orchestrator.events_since(since))
        elif url.path in ("/state", "/"):
            self._send_json(200, orchestrator.snapshot())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_FRAME_BYTES:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        data = self.rfile.read(length)
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            self._send_json(400, {"error": f"bad image: {exc}"})
            return

        t0 = time.perf_counter()
        det = classify_frame(image, self.headers.get("X-Kamikaze-Hint"))
        result = orchestrator.process(det)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        log.info(
            "[Frame %05d] %s | %s %.2f (%s) | %.1fms",
            orchestrator.frames, result["state"], det.label or "-", det.confidence, det.source, result["latency_ms"],
        )
        self._send_json(200, result)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("KAMIKAZE_PORT", "5000")))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log.info(
        "Kamikaze receiver on %s:%d | VLM %s | targets: %d phones, %d cans",
        args.host, args.port, "enabled" if vlm.enabled else "disabled (color/shape only)",
        len(PHONE_LABELS), len(CAN_LABELS),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
