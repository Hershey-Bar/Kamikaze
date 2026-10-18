#!/usr/bin/env python3
"""Kamikaze edge receiver: few-shot visual identification, guidance state machine, 3D dashboard.

One process, one port (0.0.0.0:5000):
  POST /api/ingest | /ingest | / | /upload | /api/upload | /frame
                                     live camera frames: return at once with the latest verdict; a background NPU worker
                                     always processes the newest frame (older unprocessed frames are dropped). Accepts raw
                                     image/jpeg, multipart/form-data, JSON/form base64 or data URLs, and PNG/BMP/WebP/GIF
  POST /api/frame (or any ingest route with ?sync=1)   same payloads -> synchronous JSON verdict (tests, replay tools)
  GET  /api/ingest                   ingest routes, accepted formats and live stream stats (source client/route/format)
  GET  /                             interactive dashboard + <model-viewer> (static/index.html)
  GET  /api/state                    live telemetry (state, top-1, confidence, guidance, LEDs, ...)
  GET  /api/asset                    active 3D asset for the identified object
  GET  /api/stream                   MJPEG live view (multipart/x-mixed-replace, each new frame, <= KAMIKAZE_STREAM_FPS)
  GET  /api/frame.jpg                latest camera frame (404 once the stream has been idle for KAMIKAZE_STALE_S)
  GET  /api/ring | /api/ring/frame.jpg?ago=N   opt-in RAM frame ring (KAMIKAZE_RING_GB, training mode only)
  GET  /api/classes                  enrolled classes: metadata, view count, created_at, CAD status/credit
  DELETE /api/classes/<id>[?delete_cad=0]   remove one class + its cached GLB/attribution (keep the GLB with delete_cad=0)
  POST /api/classes/reset            {"delete_cad": true} drop every enrolled class (and their cached GLBs)
  POST /api/classes/purge            {"delete_cad": true} hard reset: drop every class AND delete enrolled_classes.pt
  POST /api/classes/<id>/fetch_cad   202: re-harvest that class's CAD model in the background and show it in the viewer
  POST /api/resolve                  {"text": "coke zero can 330ml"} -> canonical product (class_id, brand, tier, CAD terms)
  GET  /api/telemetry                SoC identity, thermals, per-core freq/load, RAM, uptime, router IPC, bridge latency,
                                     Wire1/Modulino status, NPU (cDSP/NSP), detector + embedder median/p95, cosine/margin, FPS
                                     (SSE: /api/telemetry/stream)
  GET  /api/credits                  author/licence attribution of every cached 3D model
  POST /api/enroll                   legacy single-shot {"label": "coke_can", "frames": 12, "append": false, "wait": false}
  POST /api/unenroll                 {"label": "coke_can"}
  POST /api/enroll/start             multi-view guided enrollment: {"raw_name", "category", "brand", "variant",
                                     "package_type", "container_volume", "frames_per_view": 10}
  POST /api/enroll/capture           {"view": "front|back|side|top", "wait": false} capture the current/named view
  POST /api/enroll/finish | /cancel  finish early with the captured views / abort the session
  GET  /api/enroll/status            wizard state (current view, per-view progress + stability)
  GET  /api/pose                     smoothed physical pose {"pitch", "yaw", "roll"}
  GET  /api/events[?telemetry=1]     Server-Sent Events stream of the live state (10 Hz, change-driven); with telemetry=1 the
                                     same connection also carries `event: telemetry` every KAMIKAZE_TELEMETRY_PERIOD
  POST /api/db/save | GET /api/db/export   persist / download the enrolled class database
  POST /api/cad/fetch                {"raw_name": "...", "force": true} resolve + (re)harvest the CAD model, with attribution
  GET  /static/...                   static files, incl. cached GLBs in static/models/

Pipeline per frame: Ultralytics YOLO26n object detection on the Hexagon NPU (npu.py; phones, bottles/cans) ->
square crop around the target inside the central 60% box (temporally smoothed; the fixed center-60% crop when nothing
is detected) -> 224x224 -> 1280-d MobileNetV3-Large embedding on the NPU (QNN TFLite delegate; ExecuTorch/XNNPACK or
torch on the Arm CPU as fallbacks) -> L2-normalised vector -> cosine similarity against the multi-view prototypes
(front/back/side/top centroids + global centroid) -> 5-frame temporal state machine -> Modulino Pixels via the Arduino
Router socket (STM32 firmware untouched), background 3D CAD retrieval and physical pose estimation (pose.py).

Environment (all optional): KAMIKAZE_HOST/PORT, KAMIKAZE_BANK, KAMIKAZE_EMBED_BACKEND (auto|npu|executorch|torch),
KAMIKAZE_DETECT_CROP (1|0), KAMIKAZE_DETECTOR (yolo26n|yolo11n|lite2|lite0), KAMIKAZE_DETECT_SCORE, KAMIKAZE_DETECTOR_CPU (0|1), KAMIKAZE_NPU (1|0),
KAMIKAZE_EMBED_PRECISION (fp16|fp32), KAMIKAZE_TORCH_THREADS, KAMIKAZE_CAL_FLOOR / C_IDENT / C_SEARCH / MARGIN
(calibrated confidence), KAMIKAZE_SCENE_GATE (1|0) / LAPLACIAN_MIN / EDGE_MIN (empty-reticle gate), KAMIKAZE_PHONE_MARGIN,
KAMIKAZE_LOG_FRAMES (1|0), KAMIKAZE_MODE (training|deployment), KAMIKAZE_PREVIEW (1|0), KAMIKAZE_LED_HEARTBEAT (s),
KAMIKAZE_STALE_S, KAMIKAZE_STREAM_FPS, KAMIKAZE_ASSET_CACHE_MB (RAM cache for GLBs/JS), KAMIKAZE_RING_GB (default 0).

Privacy: no raw image is ever written to disk; only 1280-d vectors reach enrolled_classes.pt. The dashboard's live
view keeps ONE most-recent JPEG in RAM (overwritten every frame, dropped when the stream goes idle, disabled in
deployment mode). KAMIKAZE_RING_GB > 0 opts in to a RAM ring of recent frames (training mode only, reported in
/api/telemetry privacy + memory_pools).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import collections
import datetime
import io
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # the state machine / HTTP layer stay importable without torch (e.g. on a laptop)
    torch = None

try:
    import cv2
except ImportError:  # numpy fallback for the empty-scene gate
    cv2 = None

import cad_harvester
import device_taxonomy
import entity_resolver
import npu as npu_mod
import pose as pose_mod
import taxonomy
import telemetry
from mcp_server import (
    STATIC_DIR,
    is_valid_glb,
    local_glb_path,
    set_leds,
    slugify,
)

log = logging.getLogger("kamikaze.receiver")

BASE_DIR = Path(__file__).resolve().parent
HOST = os.environ.get("KAMIKAZE_HOST", "0.0.0.0")
PORT = int(os.environ.get("KAMIKAZE_PORT", "5000"))
BANK_PATH = Path(os.environ.get("KAMIKAZE_BANK", BASE_DIR / "enrolled_classes.pt"))
PTE_DIR = Path(os.environ.get("KAMIKAZE_PTE_DIR", BASE_DIR / "models"))
EMBED_BACKEND = os.environ.get("KAMIKAZE_EMBED_BACKEND", "auto")  # auto | npu | executorch | torch
DETECT_CROP = os.environ.get("KAMIKAZE_DETECT_CROP", "1") != "0"
EMBED_PRECISION = os.environ.get("KAMIKAZE_EMBED_PRECISION", "fp16")
TORCH_THREADS = int(os.environ.get("KAMIKAZE_TORCH_THREADS", "1"))
LOG_FRAMES = os.environ.get("KAMIKAZE_LOG_FRAMES", "1") != "0"
LED_HEARTBEAT_S = float(os.environ.get("KAMIKAZE_LED_HEARTBEAT", "5"))
MODE = os.environ.get("KAMIKAZE_MODE", "training").strip().lower()  # training | deployment
PREVIEW_FRAME = os.environ.get("KAMIKAZE_PREVIEW", "1" if MODE == "training" else "0") != "0"

INPUT_SIZE = 224
CROP_FRACTION = 0.60
EMBED_DIM = 1280
EMBEDDER_ID = "mobilenet_v3_large-fc1280"  # torchvision weights (ExecuTorch / eager); the NPU embedder has its own id
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Calibrated confidence: C = 0 for raw cosine <= CAL_FLOOR, else (S - CAL_FLOOR) / (1 - CAL_FLOOR). High-dimensional
# embeddings sit at a 0.35-0.50 cosine baseline for arbitrary scenes, so raw cosine is not a usable confidence.
CAL_FLOOR = float(os.environ.get("KAMIKAZE_CAL_FLOOR", "0.52"))
C_IDENT = float(os.environ.get("KAMIKAZE_C_IDENT", "0.70"))
C_SEARCH = float(os.environ.get("KAMIKAZE_C_SEARCH", "0.25"))
MIN_MARGIN = float(os.environ.get("KAMIKAZE_MARGIN", "0.15"))  # calibrated top-1 minus top-2
T_IDENT = CAL_FLOOR + C_IDENT * (1.0 - CAL_FLOOR)  # raw cosine equivalents (0.856 / 0.640)
T_SEARCH = CAL_FLOOR + C_SEARCH * (1.0 - CAL_FLOOR)
WINDOW = 5
IDENT_FRAMES = 3
GUIDE_FRAMES = 3
SEARCH_FRAMES = 3
# Empty-scene gate on the 60% center crop (resampled to SALIENCY_SIZE px so thresholds do not depend on resolution)
SCENE_GATE = os.environ.get("KAMIKAZE_SCENE_GATE", "1") != "0"
LAPLACIAN_MIN = float(os.environ.get("KAMIKAZE_LAPLACIAN_MIN", "18"))
EDGE_MIN = float(os.environ.get("KAMIKAZE_EDGE_MIN", "0.002"))
SALIENCY_SIZE = 256
ENROLL_FRAMES = 12
ENROLL_STALE_S = 15.0

INITIAL_CLASSES = ("coke_can", "pepsi_can", "s25_ultra")
DEFAULT_GLB_URL = "/static/models/default.glb"
CAMERA_ORBIT = "45deg 75deg 1.5m"
MODEL_VIEWER_URL = "https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"
MAX_FRAME_BYTES = 8 * 1024 * 1024

SEARCHING, GUIDING, IDENTIFIED = "SEARCHING", "GUIDING", "IDENTIFIED"
MSG_SEARCHING = "Place target object inside the central alignment box."
MSG_GUIDE_CAN = "Rotate can 90° slowly to align the front branding with the camera."
MSG_GUIDE_PHONE = "Flip phone over to reveal the rear camera island and lenses."
MSG_GUIDE_GENERAL = "Center the item within the reticle and adjust distance for optimal lighting."
MSG_GUIDE_BOTTLE = "Rotate bottle slowly to show the front label to the camera."
MSG_EMPTY = "Awaiting object inside center reticle..."
MSG_PHONE_FLIP = taxonomy.MSG_PHONE_FLIP
MSG_PHONE_TILT = taxonomy.MSG_PHONE_TILT
MSG_VERIFY = device_taxonomy.MSG_VERIFY_GEOMETRY
PHONE_AMBIG_MARGIN = float(os.environ.get("KAMIKAZE_PHONE_MARGIN", "0.15"))  # calibrated
FRAMES_PER_VIEW = 10
MIN_VIEW_FRAMES = 4
MV_IDLE_EXPIRE_S = 900.0
SSE_PERIOD_S = 0.1
SSE_KEEPALIVE_S = 1.0
TELEMETRY_PERIOD_S = float(os.environ.get("KAMIKAZE_TELEMETRY_PERIOD", "1.0"))
STALE_S = float(os.environ.get("KAMIKAZE_STALE_S", "3.0"))  # no camera frame for this long -> stream idle, preview dropped
STREAM_MAX_FPS = float(os.environ.get("KAMIKAZE_STREAM_FPS", "30"))  # MJPEG /api/stream rate cap
RING_GB = float(os.environ.get("KAMIKAZE_RING_GB", "0")) if MODE == "training" else 0.0
ASSET_CACHE_MB = float(os.environ.get("KAMIKAZE_ASSET_CACHE_MB", "1024"))
JPEG_SOI = b"\xff\xd8\xff"


# --------------------------------------------------------------------------------------
# Preprocessing + embedding backbone
# --------------------------------------------------------------------------------------


def center_crop_box(width: int, height: int) -> tuple[int, int, int]:
    """Center square covering CROP_FRACTION of the short side: (size, left, top)."""
    size = max(1, int(CROP_FRACTION * min(width, height)))
    return size, (width - size) // 2, (height - size) // 2


def preprocess(image: Image.Image, box: tuple[int, int, int] | None = None) -> np.ndarray:
    """Square crop (detector box, else the center 60%) -> 224x224 -> ImageNet normalisation. float32 (1,3,224,224)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    size, left, top = box or center_crop_box(*image.size)
    crop = image.crop((left, top, left + size, top + size)).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = (np.asarray(crop, dtype=np.float32) * (1.0 / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def calibrate(sim: float) -> float:
    """Raw cosine -> calibrated confidence in [0, 1] (0 at or below CAL_FLOOR)."""
    if sim <= CAL_FLOOR:
        return 0.0
    return min(1.0, (sim - CAL_FLOOR) / (1.0 - CAL_FLOOR))


def scene_saliency(image: Image.Image) -> dict:
    """Laplacian variance + Canny edge density of the 60% center crop. empty=True for blank walls, bare desks, a
    covered lens or a dark room, so the detector and embedder never score background noise."""
    size, left, top = center_crop_box(*image.size)
    crop = image.crop((left, top, left + size, top + size)).convert("L")
    if size != SALIENCY_SIZE:
        crop = crop.resize((SALIENCY_SIZE, SALIENCY_SIZE), Image.BILINEAR)
    gray = np.asarray(crop, dtype=np.uint8)
    if cv2 is not None:
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = float(np.count_nonzero(cv2.Canny(gray, 50, 150))) / gray.size
    else:
        g = gray.astype(np.float64)
        lap = float((g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4.0 * g[1:-1, 1:-1]).var())
        gx = g[1:-1, 2:] - g[1:-1, :-2]
        gy = g[2:, 1:-1] - g[:-2, 1:-1]
        edges = float(np.count_nonzero(np.hypot(gx, gy) > 100.0)) / gray.size
    return {"laplacian_var": round(lap, 2), "edge_density": round(edges, 5),
            "empty": lap < LAPLACIAN_MIN or edges < EDGE_MIN}


def build_eager_model():
    import torchvision

    model = torchvision.models.mobilenet_v3_large(weights="DEFAULT").eval()
    model.classifier[-1] = torch.nn.Identity()  # drop the 1000-way head: output is the 1280-d embedding
    return model


def build_pte(path: Path, precision: str) -> None:
    """Export MobileNetV3-Large to an XNNPACK-delegated ExecuTorch program."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = build_eager_model().to(dtype)
    example = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=dtype)
    program = to_edge_transform_and_lower(
        torch.export.export(model, (example,)), partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(program.buffer)
    os.replace(tmp, path)
    log.info("Exported %s (%d KB)", path, path.stat().st_size // 1024)


class FeatureExtractor:
    """Image -> L2-normalised 1280-d embedding.

    Backends, in "auto" order: npu (MediaPipe MobileNetV3-Large embedder on the Hexagon HTP through the QNN TFLite
    delegate; the same .tflite on XNNPACK if the HTP is unavailable), executorch (torchvision MobileNetV3-Large,
    XNNPACK fp16 .pte), torch (eager fp32). npu and executorch/torch use different weights, so each has its own
    embedder_id and enrolled classes do not carry across backends.
    """

    def __init__(self, backend: str = EMBED_BACKEND, precision: str = EMBED_PRECISION, threads: int = TORCH_THREADS):
        if torch is None:
            raise RuntimeError("PyTorch is required for the embedding backbone")
        self._lock = threading.Lock()
        self._runtime = None
        self._model = None
        self._npu = None
        self.dtype = torch.float32
        self.name = ""
        self.embedder_id = EMBEDDER_ID
        self.device = "cpu"
        if backend in ("auto", "npu"):
            try:
                self._npu = npu_mod.NpuEmbedder()
                self.embedder_id, self.device = self._npu.embedder_id, self._npu.device
                self.name = "npu-qnn-htp-mediapipe-mnv3" if self._npu.device == "npu" else "tflite-xnnpack-cpu-mediapipe-mnv3"
            except Exception as exc:
                log.warning("NPU embedder unavailable (%s: %s)", type(exc).__name__, exc)
                if backend == "npu":
                    raise
        if self._npu is None and backend in ("auto", "executorch"):
            try:
                self._init_executorch(precision)
            except Exception as exc:  # export/runtime problems must degrade, not kill the demo
                log.warning("ExecuTorch backend unavailable (%s: %s)", type(exc).__name__, exc)
                if backend == "executorch":
                    raise
        if self._npu is None and self._runtime is None:
            torch.set_num_threads(threads)
            self._model = build_eager_model()
            self.name = "torch-eager-fp32"
        self.embed_tensor(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE) + 0.1)

    def info(self) -> dict:
        if self._npu is not None:
            return self._npu.info()
        return {"model": "torchvision MobileNetV3-Large (ImageNet)", "device": "cpu", "runtime": self.name, "dim": EMBED_DIM}

    def _init_executorch(self, precision: str) -> None:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch

        path = PTE_DIR / f"mnv3_embed_{precision}.pte"
        if not path.is_file():
            log.info("No %s yet - exporting (one-time, ~1 min)...", path.name)
            build_pte(path, precision)
        dtype = torch.float16 if precision == "fp16" else torch.float32
        runtime = _load_for_executorch(str(path))
        out = runtime.forward((torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=dtype),))[0]
        if tuple(out.shape) != (1, EMBED_DIM) or not bool(torch.isfinite(out.float()).all()):
            raise RuntimeError(f"unexpected output {tuple(out.shape)} from {path.name}")
        self._runtime, self.dtype, self.name = runtime, dtype, f"executorch-xnnpack-{precision}"

    def embed_tensor(self, x) -> "torch.Tensor":
        """x: float (1,3,224,224), already normalised. Returns a unit-norm fp32 vector of shape (D,)."""
        if self._npu is not None:
            nhwc = x.float().numpy()[0].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
            return torch.from_numpy(self._npu.embed_rgb01(np.clip(nhwc, 0.0, 1.0)[None]))
        x = x.to(self.dtype).contiguous()
        with self._lock:
            if self._runtime is not None:
                out = self._runtime.forward((x,))[0]
            else:
                with torch.inference_mode():
                    out = self._model(x)
        vec = out.reshape(-1).float()
        norm = vec.norm()
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise ValueError("degenerate embedding")
        return vec / norm

    def embed_array(self, arr: np.ndarray) -> "torch.Tensor":
        return self.embed_tensor(torch.from_numpy(arr))

    def embed_image(self, image: Image.Image) -> "torch.Tensor":
        return self.embed_array(preprocess(image))


# --------------------------------------------------------------------------------------
# Few-shot prototype bank
# --------------------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,39}$")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def normalize_label(raw: object) -> str:
    label = re.sub(r"[\s\-]+", "_", str(raw or "").strip().lower())
    if not _LABEL_RE.match(label):
        raise ValueError("label must be 1-40 chars of a-z, 0-9, _ (e.g. coke_can)")
    return label


class EmbeddingBank:
    """label -> exemplar embeddings and prototypes.

    Each label has a global L2-normalised centroid and, for multi-view enrollments, angle-tagged view centroids
    (front/back/side/top) plus metadata and the resolved CAD asset. A label's score for a live vector is the
    best cosine over its global and view centroids.
    """

    def __init__(self, path: Path | None = None, embedder_id: str = EMBEDDER_ID):
        self.path = Path(path) if path else None
        self.embedder_id = embedder_id
        self._lock = threading.RLock()
        self.exemplars: dict[str, list] = {}
        self.centroids: dict[str, "torch.Tensor"] = {}
        self.views: dict[str, dict[str, "torch.Tensor"]] = {}
        self.view_counts: dict[str, dict[str, int]] = {}
        self.meta: dict[str, dict] = {}
        self.cad: dict[str, dict | None] = {}
        self.created: dict[str, str] = {}
        self._rows: list[str] = []
        self._row_views: list[str | None] = []
        self._labels: list[str] = []
        self._matrix = None

    def _rebuild(self) -> None:
        self._labels = list(self.centroids)
        rows, row_views, vecs = [], [], []
        for label in self._labels:
            rows.append(label)
            row_views.append(None)
            vecs.append(self.centroids[label])
            for view_name, view_vec in self.views.get(label, {}).items():
                rows.append(label)
                row_views.append(view_name)
                vecs.append(view_vec)
        self._rows = rows
        self._row_views = row_views
        self._matrix = torch.stack(vecs) if vecs else None

    def labels(self) -> list[str]:
        with self._lock:
            return list(self._labels)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self.exemplars.items()}

    @staticmethod
    def _unit_mean(vectors: list) -> "torch.Tensor":
        total = torch.stack(vectors).sum(dim=0)
        return total / total.norm().clamp_min(1e-12)

    def add(self, label: str, vectors: list, append: bool = False) -> dict:
        """Legacy single-shot enrollment: one global centroid, no view tags."""
        with self._lock:
            kept = list(self.exemplars.get(label, [])) if append else []
            stack = torch.stack(kept + [v.detach().float().clone() for v in vectors])
            centroid = self._unit_mean(list(stack.unbind(0)))
            self.exemplars[label] = list(stack.unbind(0))
            self.centroids[label] = centroid
            self.views.pop(label, None)
            self.view_counts.pop(label, None)
            self.meta.setdefault(label, {})
            self.created.setdefault(label, _now_iso())
            self._rebuild()
            self.save()
            return {"label": label, "vectors": int(stack.shape[0]), "consistency": round(float((stack @ centroid).mean()), 4)}

    def add_multiview(self, label: str, view_vectors: dict[str, list], metadata: dict) -> dict:
        """Store angle-tagged centroids; the global centroid is the normalised mean of the view centroids."""
        with self._lock:
            views, counts, all_vecs = {}, {}, []
            for view, vecs in view_vectors.items():
                if not vecs:
                    continue
                clean = [v.detach().float().clone() for v in vecs]
                views[view] = self._unit_mean(clean)
                counts[view] = len(clean)
                all_vecs += clean
            if not views:
                raise ValueError("no frames captured")
            global_centroid = self._unit_mean(list(views.values()))
            self.exemplars[label] = all_vecs
            self.centroids[label] = global_centroid
            self.views[label] = views
            self.view_counts[label] = counts
            self.meta[label] = dict(metadata)
            self.cad.setdefault(label, None)
            self.created[label] = _now_iso()
            self._rebuild()
            self.save()
            stack = torch.stack(all_vecs)
            return {
                "label": label, "vectors": len(all_vecs), "views": counts,
                "consistency": round(float((stack @ global_centroid).mean()), 4), "metadata": dict(metadata),
            }

    def remove(self, label: str) -> bool:
        with self._lock:
            if label not in self.centroids:
                return False
            del self.centroids[label]
            for table in (self.exemplars, self.views, self.view_counts, self.meta, self.cad, self.created):
                table.pop(label, None)
            self._rebuild()
            self.save()
            return True

    def reset(self) -> int:
        """Drop every enrolled class and persist the empty database. Returns how many were removed."""
        with self._lock:
            n = len(self.centroids)
            for table in (self.exemplars, self.centroids, self.views, self.view_counts, self.meta, self.cad, self.created):
                table.clear()
            self._rebuild()
            self.save()
            return n

    def purge(self) -> int:
        """Drop every enrolled class and delete the database file (plus old-embedder backups) from disk."""
        with self._lock:
            n = len(self.centroids)
            for table in (self.exemplars, self.centroids, self.views, self.view_counts, self.meta, self.cad, self.created):
                table.clear()
            self._rebuild()
            if self.path is not None:
                victims = [self.path, self.path.with_suffix(".tmp"),
                           *self.path.parent.glob(f"{self.path.stem}.*.bak{self.path.suffix}")]
                for victim in victims:
                    try:
                        victim.unlink()
                    except FileNotFoundError:
                        pass
            return n

    def rank(self, vec, domain: tuple[str, ...] | None = None) -> list[tuple[str, float, str | None]]:
        """(label, score, matched view) best first. Smartphone classes with view centroids use rear-weighted view
        reliability (device_taxonomy); other classes take the best cosine over their global + view centroids.
        `domain` restricts the search to those categories (classes without a category always stay eligible)."""
        with self._lock:
            if self._matrix is None:
                return []
            sims = (self._matrix @ vec).tolist()
            per: dict[str, dict] = {}
            for label, view, sim in zip(self._rows, self._row_views, sims):
                per.setdefault(label, {})["global" if view is None else view] = sim
            out = []
            for label, table in per.items():
                category = (self.meta.get(label) or {}).get("category")
                if domain and category and category not in domain:
                    continue
                views = {k: v for k, v in table.items() if k != "global"}
                if category == "smartphone" and views:
                    score, view = device_taxonomy.weighted_phone_score(views)
                else:
                    view = max(table, key=table.get)
                    score = table[view]
                    view = None if view == "global" else view
                out.append((label, score, view))
            return sorted(out, key=lambda item: -item[1])

    def scores(self, vec, domain: tuple[str, ...] | None = None) -> list[tuple[str, float]]:
        """Per-label match score (see rank), best first."""
        return [(label, score) for label, score, _ in self.rank(vec, domain)]

    def view_sims(self, label: str, vec) -> dict[str, float]:
        with self._lock:
            return {view: float(c @ vec) for view, c in self.views.get(label, {}).items()}

    def metadata(self, label: str) -> dict | None:
        with self._lock:
            meta = self.meta.get(label)
            return dict(meta) if meta else None

    def category(self, label: str | None) -> str | None:
        meta = self.metadata(label) if label else None
        return meta.get("category") if meta else None

    def set_cad(self, label: str, info: dict | None) -> None:
        with self._lock:
            if label in self.centroids:
                self.cad[label] = dict(info) if info else None
                self.save()

    def info(self) -> list[dict]:
        with self._lock:
            return [self._entry_info(k) for k in self._labels]

    def _entry_info(self, k: str) -> dict:
        meta = dict(self.meta.get(k) or {})
        cad = self.cad.get(k)
        counts = dict(self.view_counts.get(k, {}))
        glb = cad_harvester.glb_path(k)
        try:
            cad_path = str(glb.relative_to(BASE_DIR)).replace("\\", "/") if glb.is_file() else None
        except ValueError:
            cad_path = str(glb)
        return {
            "label": k, "class_id": k, "canonical_name": taxonomy.display_name(meta) if meta else k.replace("_", " ").title(),
            "vectors": len(self.exemplars.get(k, [])), "views": counts, "view_count": len(counts),
            "metadata": meta, "category": meta.get("category"), "created_at": self.created.get(k),
            "phone_tier": device_taxonomy.tier_for(meta), "cad_path": cad_path,
            "cad_asset": cad, "cad_model_url": (cad or {}).get("glb_url") or None,
            "cad_status": "ready" if cad and cad.get("glb_url") else "missing",
            "cad_source": (cad or {}).get("source"), "cad_credit": (cad or {}).get("credit"),
            "cad_match": (cad or {}).get("match") or ((cad or {}).get("attribution") or {}).get("match"),
        }

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            prototypes = {}
            for label in self.centroids:
                entry = self._entry_info(label)
                prototypes[label] = {
                    "class_id": label,
                    "canonical_name": entry["canonical_name"],
                    "metadata": dict(self.meta.get(label) or {}),
                    "cad_model_url": entry["cad_model_url"],
                    "views": dict(self.views.get(label, {})),
                    "view_counts": dict(self.view_counts.get(label, {})),
                    "global_centroid": self.centroids[label],
                    "created_at": self.created.get(label),
                    "cad_asset": self.cad.get(label),
                }
            payload = {
                "version": 2,
                "embedder": self.embedder_id,
                "exemplars": {k: torch.stack(v) for k, v in self.exemplars.items()},
                "centroids": dict(self.centroids),
                "prototypes": prototypes,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, self.path)

    def load(self) -> int:
        if self.path is None or not self.path.is_file():
            return 0
        try:
            data = torch.load(self.path, map_location="cpu", weights_only=True)
            if data.get("embedder") != self.embedder_id:
                backup = self.path.with_name(f"{self.path.stem}.{slugify(str(data.get('embedder')))}.bak{self.path.suffix}")
                if not backup.exists():
                    shutil.copyfile(self.path, backup)
                log.warning("Ignoring %s: it was enrolled with embedder %r but the active embedder is %r - re-enroll your "
                            "classes (old database kept as %s)", self.path, data.get("embedder"), self.embedder_id, backup.name)
                return 0
            with self._lock:
                self.exemplars = {k: list(v.unbind(0)) for k, v in data["exemplars"].items()}
                self.centroids = {k: v.float() for k, v in data["centroids"].items()}
                protos = data.get("prototypes") or {}
                self.views = {k: {taxonomy.canonical_view(vn): vv.float() for vn, vv in (p.get("views") or {}).items()}
                              for k, p in protos.items()}
                self.view_counts = {k: {taxonomy.canonical_view(vn): int(n) for vn, n in (p.get("view_counts") or {}).items()}
                                    for k, p in protos.items()}
                self.meta = {k: dict(p.get("metadata") or {}) for k, p in protos.items()}
                self.cad = {k: p.get("cad_asset") for k, p in protos.items()}
                self.created = {k: p.get("created_at") or _now_iso() for k, p in protos.items()}
                for k in self.centroids:
                    self.created.setdefault(k, _now_iso())
                self._rebuild()
            return len(self.centroids)
        except Exception as exc:
            log.warning("Could not load %s: %s", self.path, exc)
            return 0

    def export_bytes(self) -> bytes | None:
        """Current database as .pt bytes (for download); None when nothing is persisted."""
        with self._lock:
            self.save()
            return self.path.read_bytes() if self.path and self.path.is_file() else None


class EnrollmentBusy(Exception):
    pass


class Enroller:
    """Collects the next N consecutive frame embeddings and stores their centroid in the bank."""

    def __init__(self, bank: EmbeddingBank):
        self.bank = bank
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._active = False
        self._label = ""
        self._target = ENROLL_FRAMES
        self._append = False
        self._vectors: list = []
        self._last_activity = 0.0
        self._result: dict | None = None
        self._error: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self, label: str, frames: int = ENROLL_FRAMES, append: bool = False) -> dict:
        with self._lock:
            if self._active and time.monotonic() - self._last_activity < ENROLL_STALE_S:
                raise EnrollmentBusy(f"enrollment of {self._label!r} is still collecting frames")
            self._active, self._label, self._target, self._append = True, label, frames, append
            self._vectors, self._result, self._error = [], None, None
            self._last_activity = time.monotonic()
            self._done.clear()
        log.info("Enrollment started: %s (%d frames)", label, frames)
        return self.status()

    def feed(self, vec) -> bool:
        """Offer a frame embedding; returns True when this frame completed a successful enrollment."""
        with self._lock:
            if not self._active:
                return False
            self._vectors.append(vec)
            self._last_activity = time.monotonic()
            if len(self._vectors) < self._target:
                return False
            try:
                self._result = self.bank.add(self._label, self._vectors, append=self._append)
                log.info("Enrolled %s: %s", self._label, self._result)
            except Exception as exc:
                self._error = str(exc)
                log.exception("Enrollment failed")
            self._active = False
            self._done.set()
            return self._error is None

    def cancel(self, reason: str) -> None:
        with self._lock:
            if self._active:
                self._active, self._error = False, reason
                self._done.set()

    def wait(self, timeout: float) -> dict:
        if not self._done.wait(timeout):
            with self._lock:
                collected = len(self._vectors)
                target = self._target
            self.cancel(f"timed out after {timeout:.0f}s with {collected}/{target} frames - is stream_cam.py running?")
        return self.status()

    def status(self) -> dict:
        with self._lock:
            done = not self._active and (self._result is not None or self._error is not None)
            return {
                "active": self._active,
                "label": self._label,
                "collected": len(self._vectors),
                "total": self._target,
                "done": done,
                "ok": done and self._error is None,
                "error": self._error,
                "result": self._result,
            }


# --------------------------------------------------------------------------------------
# Multi-view guided enrollment
# --------------------------------------------------------------------------------------


class MultiViewEnroller:
    """Guided 4-angle enrollment wizard: front -> back -> side -> top, N frames per view.

    Phases: idle -> ready (waiting for the user to present the current view) -> capturing -> ready (next view)
    ... -> done. A view can be re-captured, and the session can be finished early once a view exists.
    """

    def __init__(self, bank: EmbeddingBank, on_complete: Callable[[str], None] | None = None):
        self.bank = bank
        self.on_complete = on_complete
        self._cond = threading.Condition(threading.RLock())
        self._reset()

    def _reset(self) -> None:
        self.phase = "idle"
        self.meta: dict | None = None
        self.label = ""
        self.per_view = FRAMES_PER_VIEW
        self.current: str | None = None
        self.captured: dict[str, list] = {}
        self.stability: dict[str, float] = {}
        self.warnings: dict[str, str] = {}
        self._buf: list = []
        self.result: dict | None = None
        self.error: str | None = None
        self._touched = time.monotonic()

    @property
    def capturing(self) -> bool:
        return self.phase == "capturing"

    @property
    def active(self) -> bool:
        return self.phase in ("ready", "capturing")

    def start(self, meta: dict, frames_per_view: int = FRAMES_PER_VIEW) -> dict:
        product = entity_resolver.resolve_metadata(meta)
        meta = {**meta, **product.metadata()}
        label = normalize_label(product.class_id)
        if not 3 <= frames_per_view <= 60:
            raise ValueError("frames_per_view must be 3-60")
        with self._cond:
            if self.capturing and time.monotonic() - self._touched < ENROLL_STALE_S:
                raise EnrollmentBusy(f"a view of {self.label!r} is still being captured")
            self._reset()
            self.meta, self.label, self.per_view = dict(meta), label, frames_per_view
            self.phase, self.current = "ready", taxonomy.VIEWS[0]
            self._cond.notify_all()
        log.info("Multi-view enrollment started: %s %s (%d frames/view)", label, meta, frames_per_view)
        return self.status()

    def capture(self, view: str | None = None) -> dict:
        with self._cond:
            if self.phase == "capturing" and time.monotonic() - self._touched < ENROLL_STALE_S:
                raise EnrollmentBusy(f"view {self.current!r} is already being captured")
            if self.phase not in ("ready", "capturing"):
                raise EnrollmentBusy("no multi-view enrollment in progress - call /api/enroll/start first")
            if view is not None:
                if view not in taxonomy.VIEWS:
                    raise ValueError(f"view must be one of {', '.join(taxonomy.VIEWS)}")
                self.current = view
            self._buf, self.error, self.phase = [], None, "capturing"
            self._touched = time.monotonic()
            self._cond.notify_all()
        return self.status()

    def feed(self, vec) -> str | None:
        """Offer a frame embedding. Returns "view_done", "complete" or None."""
        with self._cond:
            if self.phase != "capturing":
                return None
            self._buf.append(vec)
            self._touched = time.monotonic()
            if len(self._buf) < self.per_view:
                return None
            view, buf = self.current, self._buf
            centroid = EmbeddingBank._unit_mean(buf)
            self.captured[view] = buf
            self.stability[view] = round(float((torch.stack(buf) @ centroid).mean()), 3)
            front = self.captured.get("front")
            self.warnings.pop(view, None)
            if view != "front" and front:
                if float(EmbeddingBank._unit_mean(front) @ centroid) > 0.97:
                    self.warnings[view] = "looks identical to the front view - rotate the object further"
            if self.stability[view] < 0.6:
                self.warnings[view] = "unstable capture - hold the object steadier"
            self._buf = []
            pending = [v for v in taxonomy.VIEWS if v not in self.captured]
            if pending:
                self.phase, self.current = "ready", pending[0]
                self._cond.notify_all()
                return "view_done"
            self._finalize()
            return "complete" if self.error is None else None

    def _finalize(self) -> None:
        try:
            self.result = self.bank.add_multiview(self.label, self.captured, self.meta or {})
            self.phase = "done"
            log.info("Multi-view enrolled %s: %s", self.label, {k: v for k, v in self.result.items() if k != "metadata"})
            if self.on_complete:
                self.on_complete(self.label)
        except Exception as exc:
            self.phase, self.error = "error", str(exc)
            log.exception("Multi-view enrollment failed")
        self._cond.notify_all()

    def finish(self) -> dict:
        with self._cond:
            if self.phase != "ready":
                raise EnrollmentBusy("finish is only possible between view captures")
            if not self.captured:
                raise ValueError("capture at least one view first")
            self._finalize()
        return self.status()

    def cancel(self, reason: str = "cancelled") -> dict:
        with self._cond:
            if self.active:
                self.phase, self.error, self._buf = "error", reason, []
                self._cond.notify_all()
        return self.status()

    def wait_idle(self, timeout: float) -> dict:
        """Block until the current view capture finished (or the session ended); abort the capture on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self.phase != "capturing", timeout):
                self._buf, self.phase = [], "ready"
                self.error = f"timed out after {timeout:.0f}s - is stream_cam.py running?"
        return self.status()

    def status(self) -> dict:
        with self._cond:
            category = (self.meta or {}).get("category", "beverage_can")
            views = []
            for v in taxonomy.VIEWS:
                title, instruction = taxonomy.view_step(category, v)
                done = v in self.captured
                is_current = v == self.current and self.active
                views.append({
                    "view": v, "title": title, "instruction": instruction,
                    "state": "done" if done and not (is_current and self.capturing) else ("current" if is_current else "pending"),
                    "captured": len(self.captured[v]) if done else (len(self._buf) if is_current else 0),
                    "total": self.per_view, "stability": self.stability.get(v), "warning": self.warnings.get(v),
                })
            current = next((x for x in views if x["view"] == self.current), None) if self.active else None
            return {
                "active": self.active, "phase": self.phase, "label": self.label, "metadata": self.meta,
                "category": category if self.meta else None, "frames_per_view": self.per_view,
                "current_view": self.current if self.active else None,
                "current_title": current["title"] if current else None,
                "current_instruction": current["instruction"] if current else None,
                "collected": len(self._buf), "views": views,
                "done": self.phase == "done", "ok": self.phase == "done", "error": self.error, "result": self.result,
            }


# --------------------------------------------------------------------------------------
# Guidance state machine
# --------------------------------------------------------------------------------------


@dataclass
class FrameScore:
    label: str | None
    s1: float  # raw cosine of the best class
    s2: float = 0.0  # raw cosine of the runner-up
    top2: str | None = None
    hint: str | None = None  # "flip" | "tilt" | "verify": the match needs another look before it can be trusted
    cap: float | None = None  # calibrated-confidence ceiling (front-screen penalty)
    prompt: str | None = None  # model-specific guidance text for the hint
    empty: bool = False  # the empty-scene gate fired: nothing inside the reticle

    @property
    def c1(self) -> float:
        if self.empty or self.label is None:
            return 0.0
        c = calibrate(self.s1)
        return min(c, self.cap) if self.cap is not None else c

    @property
    def c2(self) -> float:
        return calibrate(self.s2) if self.top2 else 0.0

    @property
    def margin(self) -> float:
        """Calibrated top-1 minus top-2 (0 when there is no candidate)."""
        if self.empty or self.label is None:
            return 0.0
        return max(0.0, calibrate(self.s1) - self.c2)


class StateMachine:
    """5-frame sliding window over calibrated confidence with hysteresis between SEARCHING / GUIDING / IDENTIFIED."""

    def __init__(self) -> None:
        self.window: collections.deque = collections.deque(maxlen=WINDOW)
        self.state = SEARCHING
        self.label: str | None = None
        self.hint: str | None = None
        self.prompt: str | None = None
        self.empty = False

    def reset(self) -> None:
        self.window.clear()
        self.state, self.label, self.hint, self.prompt, self.empty = SEARCHING, None, None, None, False

    @staticmethod
    def frame_kind(frame: FrameScore) -> str:
        if frame.empty or (frame.label is None and not frame.hint):
            return "search"
        if frame.hint:
            return "guide"
        if frame.c1 < C_SEARCH:
            return "search"
        if frame.c1 >= C_IDENT and frame.margin >= MIN_MARGIN:
            return "ident"
        return "guide"

    def _streak(self, pred: Callable[[tuple], bool]) -> int:
        n = 0
        for item in reversed(self.window):
            if not pred(item):
                break
            n += 1
        return n

    def update(self, frame: FrameScore) -> dict:
        kind = self.frame_kind(frame)
        self.hint, self.prompt, self.empty = frame.hint, frame.prompt, frame.empty
        self.window.append((kind, frame.label, frame.c1, frame.margin))
        if kind == "ident" and self._streak(lambda w: w[0] == "ident" and w[1] == frame.label) >= IDENT_FRAMES:
            self.state, self.label = IDENTIFIED, frame.label
        elif kind == "guide" and self._streak(lambda w: w[0] == "guide") >= GUIDE_FRAMES:
            self.state, self.label = GUIDING, frame.label
        elif kind == "search" and self._streak(lambda w: w[0] == "search") >= SEARCH_FRAMES:
            self.state, self.label = SEARCHING, None
        elif self.state == GUIDING and kind != "search":
            self.label = frame.label

        if self.state == SEARCHING and kind == "search":
            return {"state": self.state, "label": None, "score": 0.0 if frame.empty else frame.c1,
                    "margin": 0.0 if frame.empty else frame.margin}
        subject = self.label if self.state != SEARCHING else frame.label
        # Average the current run only, so frames from a previous pose (e.g. a capped front view) don't drag the score.
        picks = []
        for w in reversed(self.window):
            if w[1] != subject or w[0] != kind:
                break
            picks.append(w)
        picks = picks or [w for w in self.window if w[1] == subject] or [self.window[-1]]
        score = sum(w[2] for w in picks) / len(picks)
        if frame.cap is not None:
            score = min(score, frame.cap)
        return {
            "state": self.state,
            "label": self.label,
            "score": score,
            "margin": sum(w[3] for w in picks) / len(picks),
        }


def guidance_message(state: str, label: str | None, score: float, hint: str | None = None,
                     category: str | None = None, prompt: str | None = None, empty: bool = False) -> str:
    if state == IDENTIFIED:
        return f"Object Verified: {label} ({max(0, round(score * 100))}%)"
    if state == GUIDING:
        if hint == "verify":
            return MSG_VERIFY
        if hint == "flip":
            return MSG_PHONE_FLIP
        if hint == "tilt":
            return prompt or MSG_PHONE_TILT
        if category == "smartphone":
            return MSG_GUIDE_PHONE
        if category == "beverage_can":
            return MSG_GUIDE_CAN
        if category == "beverage_bottle":
            return MSG_GUIDE_BOTTLE
        name = (label or "").lower()
        if "can" in re.split(r"[^a-z0-9]+", name) or any(k in name for k in ("coke", "pepsi", "sprite", "soda", "cola")):
            return MSG_GUIDE_CAN
        if any(k in name for k in ("s25", "phone", "galaxy", "iphone", "pixel")):
            return MSG_GUIDE_PHONE
        return MSG_GUIDE_GENERAL
    return MSG_EMPTY if empty else MSG_SEARCHING


def phone_hint(category: str | None, best_view: str | None, margin: float, front: bool | None = None) -> str | None:
    """Smartphones: the front screen is nearly identical across models -> "flip"; a rear match too close to a
    sibling model (calibrated margin) -> "tilt" to expose the camera island / lens count."""
    if category != "smartphone":
        return None
    if front if front is not None else best_view == "front":
        return "flip"
    if margin < PHONE_AMBIG_MARGIN:
        return "tilt"
    return None


def led_params(state: str, score: float) -> tuple[int, int, int, int, int]:
    if state == IDENTIFIED:
        return (0, 255, 0, 50, max(1, min(8, int(score * 8))))
    if state == GUIDING:
        return (255, 120, 0, 40, 4)
    return (0, 100, 150, 15, 1)


def capture_led_params(collected: int, total: int) -> tuple[int, int, int, int, int]:
    """Blue progress bar while an enrollment view is being captured."""
    return (0, 80, 255, 30, max(1, min(8, math.ceil(8 * collected / max(1, total)))))


# --------------------------------------------------------------------------------------
# Modulino LED dispatcher (latest-wins, off the frame path)
# --------------------------------------------------------------------------------------


class LedDriver:
    """Sends set_pixels over the router socket on a worker thread; tracks bridge RPC latency / success.

    When idle it re-sends the last successful params every KAMIKAZE_LED_HEARTBEAT seconds (same colour, so the LEDs
    do not change) which keeps the bridge latency / health figures live. 0 disables the heartbeat.
    """

    def __init__(self, setter: Callable[..., bool] = set_leds, heartbeat_s: float | None = None) -> None:
        self._setter = setter
        self._cv = threading.Condition()
        self._target: tuple | None = None
        self._sent: tuple | None = None
        self.ok: bool | None = None
        self.latency_ms = 0.0
        self.heartbeat_s = LED_HEARTBEAT_S if heartbeat_s is None else heartbeat_s
        self._lat: collections.deque = collections.deque(maxlen=60)
        self.n_ok = 0
        self.n_fail = 0
        self.last_ok_at: float | None = None
        threading.Thread(target=self._run, name="led", daemon=True).start()

    def set(self, params: tuple) -> None:
        with self._cv:
            self._target = tuple(params)
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._target is None or self._target == self._sent:
                    woke = self._cv.wait(self.heartbeat_s if self.heartbeat_s > 0 else None)
                    if not woke and self._target is not None and self.heartbeat_s > 0:
                        break  # idle heartbeat: re-send the unchanged target
                params = self._target
            t0 = time.perf_counter()
            try:
                ok = bool(self._setter(*params))
            except Exception:
                log.exception("LED setter crashed")
                ok = False
            with self._cv:
                self.ok = ok
                self.latency_ms = (time.perf_counter() - t0) * 1000.0
                if ok:
                    first = self._target != self._sent or self._sent is None
                    self._sent = params
                    self.n_ok += 1
                    self.last_ok_at = time.time()
                    self._lat.append(self.latency_ms)
                    if first:
                        log.info("LEDs set_pixels%s -> ok (%.1f ms)", params, self.latency_ms)
                else:
                    self.n_fail += 1
            if not ok:
                time.sleep(1.0)

    def status(self) -> dict:
        with self._cv:
            r, g, b, brightness, count = self._target or (0, 0, 0, 0, 0)
            return {"r": r, "g": g, "b": b, "brightness": brightness, "count": count, "ok": self.ok,
                    "latency_ms": round(self.latency_ms, 1),
                    "latency_avg_ms": round(sum(self._lat) / len(self._lat), 2) if self._lat else None,
                    "latency_p95_ms": telemetry.percentile(self._lat, 95), "rpc_ok": self.n_ok, "rpc_fail": self.n_fail,
                    "last_ok_age_s": round(time.time() - self.last_ok_at, 1) if self.last_ok_at else None}


# --------------------------------------------------------------------------------------
# 3D asset resolution (local cache first, then Sketchfab / Poly Pizza / public fallbacks)
# --------------------------------------------------------------------------------------


class AssetManager:
    RETRY_AFTER_S = 60.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._info: dict[str, dict] = {}
        self._inflight: set[str] = set()
        self._failed_at: dict[str, float] = {}
        self._gen: dict[str, int] = {}
        self.meta_lookup: Callable[[str], dict | None] | None = None
        self.on_resolved: Callable[[str, dict | None], None] | None = None

    def forget(self, label: str, delete_files: bool = True) -> bool:
        """Drop all state for a deleted class; an in-flight background harvest is discarded when it lands."""
        with self._lock:
            self._gen[label] = self._gen.get(label, 0) + 1
            self._info.pop(label, None)
            self._failed_at.pop(label, None)
        return cad_harvester.delete_cached(label) if delete_files else False

    def peek(self, label: str) -> dict:
        slug = slugify(label)
        with self._lock:
            info = dict(self._info.get(label, {}))
            busy = label in self._inflight
        glb = local_glb_path(slug)
        if is_valid_glb(glb):
            attr = cad_harvester.read_attribution(label) or {}
            try:
                version = glb.stat().st_mtime_ns
            except OSError:
                version = 0
            return {"glb_url": f"/static/models/{slug}.glb?v={version}", "status": "ready",
                    "title": attr.get("title") or info.get("title", label), "attribution": attr or None,
                    "credit": cad_harvester.attribution_line(attr) if attr else ""}
        return {"glb_url": DEFAULT_GLB_URL, "status": "resolving" if busy else "pending", "title": info.get("title", label),
                "attribution": None, "credit": ""}

    def refetch(self, label: str) -> bool:
        """Force a fresh CAD harvest for `label` in the background (replaces static/models/<label>.glb).
        Returns False when a harvest for that class is already running."""
        with self._lock:
            if label in self._inflight:
                return False
            self._inflight.add(label)
            self._failed_at.pop(label, None)
        threading.Thread(target=self._resolve, args=(label, True), name=f"refetch-{label}", daemon=True).start()
        return True

    def get(self, label: str) -> dict:
        entry = self.peek(label)
        if entry["status"] == "ready":
            return entry
        with self._lock:
            recently_failed = time.monotonic() - self._failed_at.get(label, -1e9) < self.RETRY_AFTER_S
            if label in self._inflight or recently_failed:
                return entry
            self._inflight.add(label)
        threading.Thread(target=self._resolve, args=(label,), name=f"asset-{label}", daemon=True).start()
        return {**entry, "status": "resolving"}

    def _resolve(self, label: str, force: bool = False) -> None:
        with self._lock:
            gen = self._gen.get(label, 0)
        try:
            meta = self.meta_lookup(label) if self.meta_lookup else None
            result = cad_harvester.retrieve(label, meta, force=force)
            with self._lock:
                if self._gen.get(label, 0) != gen:
                    log.info("Discarding CAD result for %s: the class was deleted while harvesting", label)
                    cad_harvester.delete_cached(label)
                elif result.get("glb_url"):
                    self._info[label] = {"title": result.get("title", label), "source": result.get("source") or result.get("status")}
                    self._failed_at.pop(label, None)
                    if self.on_resolved:
                        self.on_resolved(label, {k: result.get(k) for k in
                                                 ("title", "glb_url", "source", "status", "query", "attribution", "credit", "match")})
                else:
                    self._failed_at[label] = time.monotonic()
        except Exception:
            log.exception("Asset resolution failed for %s", label)
            with self._lock:
                self._failed_at[label] = time.monotonic()
        finally:
            with self._lock:
                self._inflight.discard(label)

    def prepare_static(self, labels: tuple[str, ...] = INITIAL_CLASSES) -> None:
        """Background: default model, vendored <model-viewer>, and GLBs for the initial classes."""

        def work() -> None:
            if not is_valid_glb(local_glb_path("default")):
                base = cad_harvester.ensure_base_models().get("smartphone")
                if base:
                    shutil.copyfile(base, local_glb_path("default"))
            vendor = STATIC_DIR / "vendor" / "model-viewer.min.js"
            if not vendor.is_file() or vendor.stat().st_size < 50_000:
                try:
                    vendor.parent.mkdir(parents=True, exist_ok=True)
                    req = urllib.request.Request(MODEL_VIEWER_URL, headers={"User-Agent": "kamikaze-edge/1.0"})
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        vendor.write_bytes(resp.read())
                    log.info("Vendored model-viewer (%d KB)", vendor.stat().st_size // 1024)
                except (urllib.error.URLError, OSError, TimeoutError) as exc:
                    log.warning("Could not vendor model-viewer (dashboard will use the CDN): %s", exc)
            for label in labels:
                self.get(label)

        threading.Thread(target=work, name="asset-prep", daemon=True).start()


class SystemStats:
    """CPU / RAM utilisation from /proc (None on non-Linux hosts), cached for 1 s."""

    def __init__(self) -> None:
        self._prev: tuple[int, int] | None = None
        self._cache: dict = {}
        self._at = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _cpu_times() -> tuple[int, int] | None:
        try:
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            vals = [int(x) for x in fields]
            return sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0)
        except (OSError, ValueError, IndexError):
            return None

    def read(self) -> dict:
        with self._lock:
            now = time.monotonic()
            if now - self._at < 1.0 and self._cache:
                return self._cache
            cpu = ram = temp = None
            times = self._cpu_times()
            if times and self._prev and times[0] > self._prev[0]:
                cpu = round(100.0 * (1 - (times[1] - self._prev[1]) / (times[0] - self._prev[0])), 1)
            self._prev = times or self._prev
            try:
                mem = {k: int(v.split()[0]) for k, v in (ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines())}
                ram = round(100.0 * (1 - mem["MemAvailable"] / mem["MemTotal"]), 1)
            except (OSError, ValueError, KeyError):
                pass
            try:
                temp = round(max(int(p.read_text()) for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")) / 1000.0, 1)
            except (OSError, ValueError):
                pass
            self._cache = {"cpu_pct": cpu if cpu is not None else self._cache.get("cpu_pct"), "ram_pct": ram, "temp_c": temp}
            self._at = now
            return self._cache


# --------------------------------------------------------------------------------------
# RAM pools: static/GLB cache and the opt-in camera frame ring
# --------------------------------------------------------------------------------------


def _phys_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


class StaticCache:
    """LRU RAM cache for static files (GLB models, model-viewer JS): served from memory, revalidated by mtime/size."""

    PRELOAD = ("models/*.glb", "vendor/*.js")

    def __init__(self, budget_bytes: int) -> None:
        self.budget = max(0, int(budget_bytes))
        self._lock = threading.Lock()
        self._items: collections.OrderedDict[str, tuple[int, int, bytes]] = collections.OrderedDict()
        self._bytes = 0
        self.hits = self.misses = 0

    def read(self, path: Path) -> bytes:
        st = path.stat()
        key, sig = str(path), (st.st_mtime_ns, st.st_size)
        with self._lock:
            item = self._items.get(key)
            if item is not None and item[:2] == sig:
                self._items.move_to_end(key)
                self.hits += 1
                return item[2]
        data = path.read_bytes()
        with self._lock:
            self.misses += 1
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= len(old[2])
            if len(data) <= self.budget:
                self._items[key] = (*sig, data)
                self._bytes += len(data)
                while self._bytes > self.budget and self._items:
                    self._bytes -= len(self._items.popitem(last=False)[1][2])
        return data

    def preload(self, root: Path) -> int:
        n = 0
        for pattern in self.PRELOAD:
            for path in sorted(root.glob(pattern)):
                try:
                    self.read(path)
                    n += 1
                except OSError:
                    pass
        return n

    def prune(self) -> None:
        with self._lock:
            for key in [k for k in self._items if not os.path.exists(k)]:
                self._bytes -= len(self._items.pop(key)[2])

    def info(self) -> dict:
        self.prune()
        with self._lock:
            return {"files": len(self._items), "mb": round(self._bytes / 2**20, 1), "budget_mb": round(self.budget / 2**20, 1),
                    "hits": self.hits, "misses": self.misses}


class FrameRing:
    """Opt-in RAM ring of recent decoded RGB camera frames (KAMIKAZE_RING_GB > 0, training mode only). Pages are
    committed as frames arrive, never written to disk, and dropped on clear() or when the frame size changes."""

    def __init__(self, budget_bytes: int) -> None:
        phys = _phys_ram_bytes()
        cap = int(phys * 0.6) if phys else None
        self.budget = max(0, min(int(budget_bytes), cap) if cap else int(budget_bytes))
        self._lock = threading.Lock()
        self._buf: np.ndarray | None = None
        self._ts: np.ndarray | None = None
        self._head = self._count = 0

    @property
    def enabled(self) -> bool:
        return self.budget > 0

    def push(self, image: Image.Image) -> None:
        if not self.budget:
            return
        arr = np.asarray(image, dtype=np.uint8)
        with self._lock:
            if self._buf is None or self._buf.shape[1:] != arr.shape:
                self._buf = None
                capacity = max(1, self.budget // max(1, arr.nbytes))
                self._buf = np.empty((capacity, *arr.shape), np.uint8)
                self._ts = np.zeros(capacity, np.float64)
                self._head = self._count = 0
            self._buf[self._head] = arr
            self._ts[self._head] = time.time()
            self._head = (self._head + 1) % len(self._buf)
            self._count = min(self._count + 1, len(self._buf))

    def get(self, ago: int = 0) -> np.ndarray | None:
        with self._lock:
            if self._buf is None or not 0 <= ago < self._count:
                return None
            return self._buf[(self._head - 1 - ago) % len(self._buf)].copy()

    def __len__(self) -> int:
        return self._count

    def clear(self) -> None:
        with self._lock:
            self._buf = self._ts = None
            self._head = self._count = 0

    def info(self) -> dict:
        with self._lock:
            out = {"enabled": self.enabled, "budget_mb": round(self.budget / 2**20, 1), "frames": self._count,
                   "capacity": 0, "used_mb": 0.0, "shape": None, "span_s": 0.0}
            if self._buf is not None and self._count:
                cap, frame_bytes = len(self._buf), self._buf[0].nbytes
                newest, oldest = self._ts[(self._head - 1) % cap], self._ts[(self._head - self._count) % cap]
                out.update(capacity=cap, used_mb=round(self._count * frame_bytes / 2**20, 1),
                           shape=list(self._buf.shape[1:]), span_s=round(float(newest - oldest), 2))
            return out


# --------------------------------------------------------------------------------------
# Pipeline: frame -> embedding -> scores -> state -> actuation / assets
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Frame payload normalisation (any streamer -> JPEG bytes)
# --------------------------------------------------------------------------------------

INGEST_ROUTES = ("/api/ingest", "/ingest", "/", "/upload", "/api/upload", "/frame")  # fast async path
SYNC_FRAME_ROUTES = ("/api/frame",)  # synchronous: waits for inference and returns the fresh verdict
_IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"BM", b"RIFF", b"GIF8")
_FRAME_KEYS = ("image", "frame", "jpeg", "jpg", "img", "data", "file", "b64", "image_base64", "base64")
_B64_RE = re.compile(rb"[A-Za-z0-9+/=_\-\r\n]+")


def _b64_image(text: str | bytes) -> bytes:
    if isinstance(text, bytes):
        text = text.decode("latin-1")
    text = text.strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    text = "".join(text.split())
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), altchars=b"-_" if ("-" in text or "_" in text) else None)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 image: {exc}") from None


def _to_jpeg(data: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=85)
            return buf.getvalue()
    except Exception as exc:
        raise ValueError(f"corrupt or unsupported image: {exc}") from None


def _from_multipart(body: bytes, content_type: str, depth: int) -> tuple[bytes, str]:
    m = re.search(r'boundary="?([^";]+)"?', content_type or "", re.I)
    if m:
        boundary = m.group(1).encode("latin-1")
    else:
        first = body.split(b"\r\n", 1)[0]
        if not first.startswith(b"--") or len(first) < 3:
            raise ValueError("multipart body without boundary")
        boundary = first[2:]
    for chunk in body.split(b"--" + boundary)[1:]:
        if chunk.startswith(b"--"):
            break
        head, sep, content = chunk.lstrip(b"\r\n").partition(b"\r\n\r\n")
        if not sep:
            continue
        if content.endswith(b"\r\n"):
            content = content[:-2]
        ct = re.search(rb"(?im)^content-type:\s*([^\r\n]+)", head)
        try:
            frame, fmt = extract_frame(content, ct.group(1).decode("latin-1") if ct else "", depth + 1)
            return frame, "multipart/" + fmt
        except ValueError:
            continue
    raise ValueError("multipart body has no image part")


def extract_frame(body: bytes, content_type: str = "", depth: int = 0) -> tuple[bytes, str]:
    """Normalise whatever a camera streamer POSTs into JPEG bytes: raw JPEG (stream_cam.py), multipart/form-data
    (requests files=...), JSON or form fields holding base64 / data-URL images, bare base64, or PNG/BMP/WebP/GIF
    (transcoded to JPEG in memory). Returns (jpeg, format) or raises ValueError with an actionable message."""
    if not body or len(body) < 16:
        raise ValueError("empty frame payload")
    if body.startswith(JPEG_SOI):
        return body, "jpeg"
    if depth > 2:
        raise ValueError("nested frame payload too deep")
    ctype = (content_type or "").lower()
    stripped = body.lstrip()
    if "multipart/" in ctype or body.startswith(b"--"):
        return _from_multipart(body, content_type, depth)
    if stripped.startswith(b"{"):
        try:
            obj = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ValueError("JSON frame body is not valid JSON") from None
        if isinstance(obj, dict):
            for key in _FRAME_KEYS:
                if isinstance(obj.get(key), str) and len(obj[key]) > 16:
                    frame, fmt = extract_frame(_b64_image(obj[key]), "", depth + 1)
                    return frame, "json-base64/" + fmt
        raise ValueError(f"JSON frame body has no base64 image under any of: {', '.join(_FRAME_KEYS)}")
    if "x-www-form-urlencoded" in ctype:
        form = urllib.parse.parse_qs(body.decode("latin-1"))
        for key in _FRAME_KEYS:
            if form.get(key):
                frame, fmt = extract_frame(_b64_image(form[key][0]), "", depth + 1)
                return frame, "form-base64/" + fmt
        raise ValueError(f"form body has no base64 image under any of: {', '.join(_FRAME_KEYS)}")
    if body.startswith(_IMAGE_MAGIC):
        return _to_jpeg(body), "transcoded"
    if stripped.startswith(b"data:") or _B64_RE.fullmatch(stripped[:512]):
        frame, fmt = extract_frame(_b64_image(stripped), "", depth + 1)
        return frame, "base64/" + fmt
    raise ValueError("not a JPEG frame: send raw image/jpeg bytes, multipart/form-data, or JSON {\"image\": <base64>}")


class Pipeline:
    def __init__(self, extractor, bank: EmbeddingBank, leds: LedDriver, assets: AssetManager, detector=None) -> None:
        self.extractor, self.bank, self.leds, self.assets = extractor, bank, leds, assets
        self.detector = detector
        self.crop_tracker = npu_mod.CropTracker()
        self._detect_lat: collections.deque = collections.deque(maxlen=240)
        self._detection: dict = {"target": None, "objects": [], "crop": None, "crop_source": "center", "ms": 0.0}
        self.enroller = Enroller(bank)
        self.mv = MultiViewEnroller(bank, on_complete=self._mv_complete)
        self.pose_est = pose_mod.PoseEstimator()
        self.stats = SystemStats()
        self._pose: dict = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "valid": False, "view": None, "ms": 0.0}
        self._best_view: str | None = None
        self._category: str | None = None
        self._version = 0
        self._ver_cv = threading.Condition()
        if hasattr(assets, "meta_lookup"):
            assets.meta_lookup = bank.metadata
            assets.on_resolved = self._cad_resolved
        self.sm = StateMachine()
        self._infer_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._snap_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None  # single most-recent preview frame, in RAM only, overwritten each frame
        self._frame_cv = threading.Condition(self._frame_lock)
        self._frame_seq = 0
        self._ingested = 0
        self._ingest_fps = 0.0
        self._last_ingest_mono = 0.0
        self._pending: bytes | None = None
        self._pending_cv = threading.Condition()
        self._dropped = 0
        self._decode_errors = 0
        self._rejected = 0
        self._source: dict | None = None
        self._idle_logged = True
        self._worker: threading.Thread | None = None
        self.ring = FrameRing(int(RING_GB * 2**30))
        self.static_cache = StaticCache(int(ASSET_CACHE_MB * 2**20))
        self._embed_lat: collections.deque = collections.deque(maxlen=240)
        self._total_lat: collections.deque = collections.deque(maxlen=240)
        self._started = time.time()
        self._frames = 0
        self._fps = 0.0
        self._last_frame_mono = 0.0
        self._viewer_label: str | None = None
        self._last = {
            "label": None, "similarity": 0.0, "confidence": 0.0, "margin": 0.0, "top2": None, "frame_confidence": 0.0,
            "embed_ms": 0.0, "detect_ms": 0.0, "total_ms": 0.0, "state": SEARCHING, "guidance": MSG_SEARCHING,
            "hint": None, "empty": False, "saliency": None, "domain": None, "conflict": False, "front_dominant": False,
            "raw_top1": None,
        }

    @property
    def latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            if self._latest_jpeg is not None and time.monotonic() - self._last_ingest_mono > STALE_S:
                self._latest_jpeg = None  # stream went idle: never keep showing (or holding) an old frame
            return self._latest_jpeg

    def _publish_frame(self, data: bytes) -> None:
        now = time.monotonic()
        with self._frame_cv:
            if PREVIEW_FRAME:
                self._latest_jpeg = data
            gap = now - self._last_ingest_mono if self._last_ingest_mono else None
            if gap is None or gap > STALE_S:
                self._ingest_fps = 0.0
            else:
                inst = 1.0 / max(1e-3, gap)
                self._ingest_fps = inst if self._ingest_fps == 0 else 0.85 * self._ingest_fps + 0.15 * inst
            self._last_ingest_mono = now
            self._ingested += 1
            self._frame_seq += 1
            self._frame_cv.notify_all()

    def wait_frame(self, seq: int, timeout: float) -> tuple[bytes | None, int]:
        """Block until a frame newer than `seq` arrives (MJPEG stream). (None, seq) on timeout."""
        with self._frame_cv:
            self._frame_cv.wait_for(lambda: self._frame_seq != seq, timeout)
            if self._frame_seq == seq or time.monotonic() - self._last_ingest_mono > STALE_S:
                return None, self._frame_seq
            return self._latest_jpeg, self._frame_seq

    def ingest(self, data: bytes) -> dict:
        """Fast live-stream path: publish the frame for the preview and hand it to the NPU worker (latest frame wins,
        older unprocessed frames are dropped). Returns the most recent verdict without waiting for inference."""
        if not data.startswith(JPEG_SOI):
            raise ValueError("not a JPEG frame")
        self._publish_frame(data)
        with self._pending_cv:
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="npu-worker", daemon=True)
                self._worker.start()
            if self._pending is not None:
                self._dropped += 1
            self._pending = data
            self._pending_cv.notify()
        with self._snap_lock:
            last = dict(self._last)
        return {"ok": True, "queued": True, "seq": self._frame_seq, "state": last["state"], "label": last["label"],
                "similarity": round(last["similarity"], 3), "confidence": round(last["confidence"], 3),
                "fps": round(self._fps, 1), "ingest_fps": round(self._ingest_fps, 1), "latency_ms": round(last["total_ms"], 1)}

    def note_source(self, source: dict) -> None:
        """Record who is streaming; log when a stream (re)starts so a route/format mismatch is obvious in the log."""
        with self._frame_lock:
            idle = not self._last_ingest_mono or time.monotonic() - self._last_ingest_mono > STALE_S
            changed = self._source is None or any(self._source.get(k) != source.get(k) for k in ("client", "path", "format"))
            self._source, self._idle_logged = {**source, "since": self._source.get("since") if self._source and not idle and not changed else time.time()}, False
        if idle or changed:
            log.info("Live stream started: %s POST %s (%s, %s, %d bytes/frame)", source.get("client"), source.get("path"),
                     source.get("format"), source.get("content_type") or "no content-type", source.get("bytes", 0))

    def note_rejected(self, source: dict, size: int, head: bytes, error: Exception) -> None:
        self._rejected += 1
        if self._rejected <= 5 or self._rejected % 100 == 0:
            log.warning("Rejected frame #%d from %s POST %s (content-type %r, %d bytes, starts %r): %s", self._rejected,
                        source.get("client"), source.get("path"), source.get("content_type"), size, head, error)

    def _check_idle(self) -> None:
        with self._frame_lock:
            if self._idle_logged or not self._last_ingest_mono:
                return
            age = time.monotonic() - self._last_ingest_mono
            if age <= STALE_S:
                return
            self._idle_logged = True
        log.info("Live stream idle: no frame for %.0f s (preview dropped, waiting for stream_cam.py)", age)

    def _work(self) -> None:
        while True:
            with self._pending_cv:
                if not self._pending_cv.wait_for(lambda: self._pending is not None, 1.0):
                    data = None
                else:
                    data, self._pending = self._pending, None
            if data is None:
                self._check_idle()
                continue
            try:
                self.process_jpeg(data, publish=False)
            except ValueError:
                self._decode_errors += 1
            except Exception:
                log.exception("Frame processing failed")

    def stream_stats(self) -> dict:
        age = time.monotonic() - self._last_ingest_mono if self._last_ingest_mono else None
        live = age is not None and age <= STALE_S
        return {"live": live, "frame_age_s": round(age, 2) if age is not None else None,
                "ingest_fps": round(self._ingest_fps, 1) if live else 0.0, "infer_fps": round(self._fps, 1) if live else 0.0,
                "ingested": self._ingested, "processed": self._frames, "dropped": self._dropped,
                "decode_errors": self._decode_errors, "rejected": self._rejected, "source": self._source,
                "async_worker": self._worker is not None}

    def _bump(self) -> None:
        with self._ver_cv:
            self._version += 1
            self._ver_cv.notify_all()

    def wait_version(self, last: int, timeout: float) -> int:
        with self._ver_cv:
            self._ver_cv.wait_for(lambda: self._version != last, timeout)
            return self._version

    def _cad_resolved(self, label: str, info: dict | None) -> None:
        self.bank.set_cad(label, info)
        self._bump()  # push the new GLB URL to the 3D viewer even when no frames are streaming

    def _mv_complete(self, label: str) -> None:
        self.sm.reset()
        self.pose_est.reset()
        self._viewer_label = label
        self.assets.get(label)
        self._bump()

    def process_jpeg(self, data: bytes, publish: bool = True) -> dict:
        t0 = time.perf_counter()
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:
            raise ValueError(f"undecodable image: {exc}") from exc
        image = image.convert("RGB")
        if publish:
            self._publish_frame(data)
        del data
        self.ring.push(image)
        with self._infer_lock:
            enrolling = self.mv.capturing or self.enroller.active
            saliency = scene_saliency(image) if SCENE_GATE else {"laplacian_var": None, "edge_density": None, "empty": False}
            empty = bool(saliency["empty"]) and not enrolling
            capturing = self.mv.capturing
            if empty:
                # Nothing inside the reticle: skip the detector + embedder entirely (no background noise scoring).
                t1 = t2 = time.perf_counter()
                box, detect_ms = None, 0.0
                self.crop_tracker.reset()
                self._detection = {"target": None, "objects": [], "crop": None, "crop_source": "center", "ms": 0.0,
                                   "skipped": "empty_scene"}
                frame, views, best_view, category = FrameScore(None, 0.0, empty=True), {}, None, None
                arb = {"domain": None, "conflict": False, "front_dominant": False, "raw_top1": None}
            else:
                box, detect_ms = self._locate(image)
                arr = preprocess(image, box)
                t1 = time.perf_counter()
                vec = self.extractor.embed_array(arr)
                t2 = time.perf_counter()
                if self.enroller.feed(vec):
                    self.sm.reset()
                if self.mv.feed(vec) == "complete":
                    capturing = False
                frame, views, best_view, category, arb = self._arbitrate(vec)
            prev = (self.sm.state, self.sm.label)
            snap = self.sm.update(frame)
            state, label, score = snap["state"], snap["label"], snap["score"]
            hint = frame.hint if state == GUIDING else None
            message = guidance_message(state, label, score, hint, category, frame.prompt, frame.empty)
            if self.mv.active:
                mv = self.mv.status()
                self.leds.set(capture_led_params(mv["collected"], mv["frames_per_view"]) if capturing else led_params(SEARCHING, 0.0))
            else:
                self.leds.set(led_params(state, score))
            if state == IDENTIFIED:
                self._viewer_label = label
                self.assets.get(label)
            pose = self._pose
            if frame.label and state != SEARCHING:
                try:
                    pose = self.pose_est.update(image, box or center_crop_box(*image.size), views, frame.label, category)
                except Exception:
                    log.exception("pose estimation failed")
            self._pose, self._best_view, self._category = pose, best_view, category
            self._frames += 1
            now = time.monotonic()
            if self._last_frame_mono and now - self._last_frame_mono <= STALE_S:
                inst = 1.0 / max(1e-3, now - self._last_frame_mono)
                self._fps = inst if self._fps == 0 else 0.85 * self._fps + 0.15 * inst
            else:
                self._fps = 0.0
            self._last_frame_mono = now
            total_ms = (time.perf_counter() - t0) * 1000.0
            if not empty:
                self._embed_lat.append((t2 - t1) * 1000.0)
            self._total_lat.append(total_ms)
            with self._snap_lock:
                self._last = {
                    "label": frame.label, "similarity": frame.s1, "confidence": score, "margin": snap["margin"],
                    "frame_confidence": frame.c1,
                    "top2": {"label": frame.top2, "score": frame.s2, "confidence": frame.c2} if frame.top2 else None,
                    "embed_ms": (t2 - t1) * 1000.0, "detect_ms": detect_ms, "total_ms": total_ms, "state": state,
                    "guidance": message, "hint": hint, "empty": frame.empty, "saliency": saliency, **arb,
                }
            frames_seen = self._frames
        if LOG_FRAMES:
            log.info(
                "[Frame %05d] %s | %s s1=%.3f C=%.2f margin=%.2f lap=%s edge=%s | detect %.1fms embed %.1fms total %.1fms",
                frames_seen, state, frame.label or "-", frame.s1, frame.c1, frame.margin, saliency["laplacian_var"],
                saliency["edge_density"], detect_ms, (t2 - t1) * 1000, total_ms,
            )
        if (state, label) != prev:
            self._announce(state, label, score, snap["margin"], message)
        self._bump()
        return {
            "ok": True, "state": state, "label": frame.label, "confidence": round(score, 3),
            "similarity": round(frame.s1, 3), "margin": round(snap["margin"], 3), "hint": hint, "empty": frame.empty,
            "embed_ms": round((t2 - t1) * 1000, 1), "detect_ms": round(detect_ms, 1), "latency_ms": round(total_ms, 1),
            "detection": self._detection["target"], "saliency": saliency, "domain": arb["domain"],
        }

    def _arbitrate(self, vec) -> tuple[FrameScore, dict, str | None, str | None, dict]:
        """Rank enrolled classes inside the detector's domain gate and attach phone / conflict guidance.
        Returns (frame score, view sims of top-1, best view, top-1 category, diagnostics)."""
        target = self._detection.get("target")
        domain = device_taxonomy.domain_for(target)
        ranked_all = self.bank.rank(vec)
        ranked = self.bank.rank(vec, domain) if domain else ranked_all
        info = {"domain": list(domain) if domain else None, "conflict": False, "front_dominant": False,
                "raw_top1": {"label": ranked_all[0][0], "score": round(ranked_all[0][1], 4)} if ranked_all else None}
        lead = ranked_all[0] if ranked_all else None
        lead_cat = self.bank.category(lead[0]) if lead else None
        conflict = bool(domain and lead and lead_cat and lead_cat not in domain and calibrate(lead[1]) >= C_SEARCH)
        if not ranked:
            info["conflict"] = conflict
            return FrameScore(None, 0.0, hint="verify" if conflict else None), {}, None, None, info
        label, s1, best_view = ranked[0]
        top2, s2 = (ranked[1][0], ranked[1][1]) if len(ranked) > 1 else (None, 0.0)
        category = self.bank.category(label)
        views = self.bank.view_sims(label, vec)
        if best_view is None and views:
            best_view = max(views, key=views.get)
        frame = FrameScore(label, s1, s2, top2)
        if conflict:
            info["conflict"] = True
            frame.hint = "verify"
        elif category == "smartphone":
            front = device_taxonomy.front_dominates(views)
            front_live = front and (frame.c1 >= C_SEARCH or domain == device_taxonomy.PHONE_DOMAIN
                                    or calibrate(views.get("front", 0.0)) >= C_SEARCH)
            info["front_dominant"] = bool(front_live)
            if front_live:
                frame.hint, frame.cap = "flip", device_taxonomy.FRONT_CONFIDENCE_CAP
            elif frame.c1 >= C_SEARCH and top2 and self.bank.category(top2) == "smartphone" \
                    and phone_hint(category, best_view, frame.margin, front=False) == "tilt":
                frame.hint = "tilt"
                frame.prompt = device_taxonomy.disambiguation_prompt(self.bank.metadata(label), self.bank.metadata(top2))
        return frame, views, best_view, category, info

    def _locate(self, image: Image.Image) -> tuple[tuple[int, int, int] | None, float]:
        """NPU object detection -> smoothed square crop around the target inside the reticle (None = center crop)."""
        if self.detector is None or not DETECT_CROP:
            return None, 0.0
        width, height = image.size
        t0 = time.perf_counter()
        try:
            dets = self.detector.detect(image)
        except Exception:
            log.exception("object detection failed")
            dets = []
        ms = (time.perf_counter() - t0) * 1000.0
        self._detect_lat.append(ms)
        target = npu_mod.pick_target(dets, width, height, CROP_FRACTION)
        box = self.crop_tracker.update(npu_mod.square_crop(target, width, height) if target else None, width, height)
        crop = box or center_crop_box(width, height)
        self._detection = {
            "target": target.to_dict() if target else None,
            "objects": [d.to_dict() for d in dets[:5]],
            "crop": [round(crop[1] / width, 4), round(crop[2] / height, 4), round(crop[0] / width, 4), round(crop[0] / height, 4)],
            "crop_source": "detector" if box else "center",
            "ms": round(ms, 2),
        }
        return box, ms

    def _announce(self, state: str, label: str | None, score: float, margin: float, message: str) -> None:
        if state == IDENTIFIED:
            payload = {"event": "asset_ready", "label": label, "confidence": round(score, 3),
                       "margin": round(margin, 3), "asset": self.assets.peek(label), "timestamp": time.time()}
            log.info("IDENTIFIED %s (%.0f%%) -> %s", label, score * 100, payload["asset"]["glb_url"])
            print(f"ASSET_READY_PAYLOAD {json.dumps(payload, ensure_ascii=False)}", flush=True)
        elif state == GUIDING:
            payload = {"event": "guidance", "label": label, "action": "reorient", "message": message, "timestamp": time.time()}
            log.info("GUIDING %s: %s", label, message)
            print(f"GUIDANCE_PAYLOAD {json.dumps(payload, ensure_ascii=False)}", flush=True)
        else:
            log.info("SEARCHING: %s", message)

    def _clear_verdict(self) -> None:
        self.sm.reset()
        self.pose_est.reset()
        with self._snap_lock:
            self._last.update(label=None, similarity=0.0, confidence=0.0, margin=0.0, top2=None, frame_confidence=0.0,
                              state=SEARCHING, guidance=MSG_SEARCHING, hint=None, conflict=False, front_dominant=False,
                              raw_top1=None)
        self.leds.set(led_params(SEARCHING, 0.0))

    def unenroll(self, label: str) -> bool:
        with self._infer_lock:
            removed = self.bank.remove(label)
            if removed:
                if self._viewer_label == label:
                    self._viewer_label = None
                self._clear_verdict()
            return removed

    def reset_bank(self) -> int:
        with self._infer_lock:
            n = self.bank.reset()
            self._viewer_label = None
            self._clear_verdict()
        self._bump()
        return n

    def purge(self, delete_cad: bool = True) -> dict:
        """Hard reset of the training database: every class, the .pt file on disk and (optionally) cached CAD."""
        with self._infer_lock:
            labels = list(self.bank.centroids)
            self.enroller.cancel("database purged")
            self.mv.cancel("database purged")
            n = self.bank.purge()
            self._viewer_label = None
            self._clear_verdict()
        cad_removed = 0
        if delete_cad:
            for label in labels:
                if hasattr(self.assets, "forget") and self.assets.forget(label):
                    cad_removed += 1
        self._bump()
        path = self.bank.path
        log.warning("Training database purged: %d classes removed, %d CAD caches deleted, %s %s", n, cad_removed,
                    path, "deleted" if path is None or not path.exists() else "STILL PRESENT")
        return {"ok": True, "removed": n, "labels": labels, "cad_removed": cad_removed,
                "db_file": str(path) if path else None, "db_file_exists": bool(path and path.exists())}

    def refetch_cad(self, label: str) -> dict:
        """Re-harvest the CAD model of an enrolled class in the background and point the 3D viewer at it."""
        if label not in self.bank.centroids:
            raise KeyError(label)
        started = self.assets.refetch(label) if hasattr(self.assets, "refetch") else False
        self._viewer_label = label
        self._bump()
        return {"ok": True, "label": label, "started": started, "status": "resolving" if started else "busy",
                "asset": self.assets.peek(label)}

    def snapshot(self) -> dict:
        with self._snap_lock:
            last = dict(self._last)
        stream = self.stream_stats()
        age = stream["frame_age_s"]
        viewer = (
            self.assets.peek(self._viewer_label)
            if self._viewer_label
            else {"glb_url": DEFAULT_GLB_URL, "status": "default", "title": ""}
        )
        state = last["state"]
        return {
            "ok": True,
            "state": state,
            "label": last["label"],
            "confirmed_label": self.sm.label if state == IDENTIFIED else None,
            "viewer_label": self._viewer_label,
            "confidence": round(last["confidence"], 3),
            "similarity": round(last["similarity"], 3),
            "margin": round(last["margin"], 3),
            "top2": last["top2"],
            "guidance": last["guidance"],
            "action": {IDENTIFIED: "none", GUIDING: "reorient"}.get(state, "present_object"),
            "glb_url": viewer["glb_url"],
            "asset_status": viewer["status"],
            "asset_title": viewer.get("title", ""),
            "asset_credit": viewer.get("credit", ""),
            "asset_attribution": viewer.get("attribution"),
            "mode": MODE,
            "privacy": self.privacy(),
            "latency_ms": {"embed": round(last["embed_ms"], 1), "detect": round(last["detect_ms"], 1), "total": round(last["total_ms"], 1)},
            "detection": self._detection,
            "accelerator": self.accelerator(),
            "fps": stream["infer_fps"],
            "ingest_fps": stream["ingest_fps"],
            "stream": stream,
            "frames": self._frames,
            "frame_age_s": age,
            "led": self.leds.status(),
            "enrollment": self.enroller.status(),
            "multiview": self.mv.status(),
            "pose": self._pose,
            "category": self._category,
            "best_view": self._best_view,
            "hint": last.get("hint"),
            "empty_scene": last.get("empty", False),
            "saliency": last.get("saliency"),
            "frame_confidence": round(last.get("frame_confidence", 0.0), 3),
            "domain": last.get("domain"),
            "domain_conflict": last.get("conflict", False),
            "front_dominant": last.get("front_dominant", False),
            "raw_top1": last.get("raw_top1"),
            "crop_fraction": CROP_FRACTION,
            "system": self.stats.read(),
            "bank": self.bank.info(),
            "classes": self.bank.counts(),
            "backend": self.extractor.name,
            "thresholds": {"calibration_floor": CAL_FLOOR, "identified": C_IDENT, "searching": C_SEARCH,
                           "margin": MIN_MARGIN, "identified_raw_cosine": round(T_IDENT, 4),
                           "searching_raw_cosine": round(T_SEARCH, 4), "phone_margin": PHONE_AMBIG_MARGIN,
                           "front_cap": device_taxonomy.FRONT_CONFIDENCE_CAP, "scene_gate": SCENE_GATE,
                           "laplacian_min": LAPLACIAN_MIN, "edge_min": EDGE_MIN,
                           "detector_gate": device_taxonomy.DETECTOR_GATE_MIN},
        }

    def privacy(self) -> dict:
        held = self.latest_jpeg is not None
        return {"raw_frames_persisted": False, "images_written_to_disk": False, "stored_in_class_db": "1280-d vectors only",
                "preview_frame_in_ram": bool(PREVIEW_FRAME and held), "preview_enabled": PREVIEW_FRAME, "mode": MODE,
                "ram_ring_enabled": self.ring.enabled, "ram_ring_frames": len(self.ring)}

    def accelerator(self) -> str:
        on_npu = [name for name, part in (("embedder", self.extractor), ("detector", self.detector))
                  if part is not None and getattr(part, "device", "cpu") == "npu"]
        if on_npu:
            arch = npu_mod.htp_arch()
            return f"Hexagon NPU ({f'HTP {arch}' if arch else 'HTP'}, QNN TFLite delegate): {' + '.join(on_npu)}"
        return "Arm CPU (XNNPACK); Hexagon NPU unavailable"

    def telemetry_snapshot(self) -> dict:
        """Hardware (/proc, /sys), bridge IPC, and inference telemetry for GET /api/telemetry."""
        hw = telemetry.hardware()
        led = self.leds.status()
        with self._snap_lock:
            last = dict(self._last)
        embed = list(self._embed_lat)
        total = list(self._total_lat)
        detect = list(self._detect_lat)
        router = hw["router"]
        led_ok = led["ok"]
        return {
            "ok": True, "time": time.time(), "mode": MODE, **hw,
            "ipc": {"router_socket": router, "serial_bridge": hw["serial"],
                    "bridge_rpc": {"ok": led_ok, "last_ms": led["latency_ms"], "avg_ms": led["latency_avg_ms"],
                                   "p95_ms": led["latency_p95_ms"], "calls_ok": led["rpc_ok"], "calls_failed": led["rpc_fail"],
                                   "last_ok_age_s": led["last_ok_age_s"]}},
            "modulino": {
                "bus": "Wire1 (I2C4) on the STM32H5F5 (reached through arduino-router; not visible to Linux i2c-dev)",
                "device": "Modulino Pixels", "status": "online" if led_ok else ("unreachable" if led_ok is False else "unknown"),
                "color": [led["r"], led["g"], led["b"]], "brightness": led["brightness"], "leds_on": led["count"],
                "leds_total": 8, "bus_latency_ms": led["latency_ms"],
            },
            "inference": {
                "backend": self.extractor.name, "embedder": getattr(self.extractor, "embedder_id", EMBEDDER_ID),
                "embed_dim": EMBED_DIM,
                "embed_ms_last": round(embed[-1], 2) if embed else None,
                "embed_ms_median": telemetry.percentile(embed, 50), "embed_ms_p95": telemetry.percentile(embed, 95),
                "detect_ms_median": telemetry.percentile(detect, 50), "detect_ms_p95": telemetry.percentile(detect, 95),
                "total_ms_median": telemetry.percentile(total, 50), "total_ms_p95": telemetry.percentile(total, 95),
                "samples": len(embed), "fps": self.stream_stats()["infer_fps"], "frames": self._frames,
                "cosine_similarity": round(last["similarity"], 4), "margin": round(last["margin"], 4),
                "confidence": round(last["confidence"], 4), "label": last["label"],
                "similarity_pct": round(last["confidence"] * 100, 1), "margin_pct": round(last["margin"] * 100, 1),
                "state": last["state"], "empty_scene": last.get("empty", False), "saliency": last.get("saliency"),
                "domain": last.get("domain"),
                "accelerator": self.accelerator(),
                "models": {
                    "embedder": self.extractor.info() if hasattr(self.extractor, "info") else {"runtime": self.extractor.name},
                    "detector": self.detector.info() if self.detector is not None else None,
                },
                "detection": self._detection,
            },
            "privacy": self.privacy(),
            "stream": self.stream_stats(),
            "memory_pools": {"static_cache": self.static_cache.info(), "frame_ring": self.ring.info()},
            "uptime_service_s": round(time.time() - self._started, 1),
        }

    def asset_payload(self) -> dict:
        snap = self.snapshot()
        return {
            "label": snap["confirmed_label"] or (self._viewer_label if snap["state"] != SEARCHING else None),
            "confidence": snap["confidence"],
            "margin": snap["margin"],
            "state": snap["state"],
            "glb_url": snap["glb_url"],
            "camera_orbit": CAMERA_ORBIT,
            "guidance": snap["guidance"],
            "asset_status": snap["asset_status"],
            "title": snap["asset_title"],
            "credit": snap["asset_credit"],
            "attribution": snap["asset_attribution"],
        }


# --------------------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------------------

_MIME = {".glb": "model/gltf-binary", ".js": "text/javascript", ".mjs": "text/javascript", ".html": "text/html; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Kamikaze/2.0"
    pipeline: Pipeline

    def log_message(self, fmt, *args) -> None:  # request logging is done by the pipeline
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json")

    def _read_body(self) -> bytes | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_FRAME_BYTES:
            self._json(400, {"ok": False, "error": "missing or oversized body"})
            self.close_connection = True
            return None
        return self.rfile.read(length)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        p = self.pipeline
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path in ("/api/state", "/state"):
            return self._json(200, p.snapshot())
        if path in ("/api/asset", "/asset"):
            return self._json(200, p.asset_payload())
        if path == "/api/classes":
            return self._json(200, {"ok": True, "count": len(p.bank.labels()), "classes": p.bank.counts(),
                                    "prototypes": p.bank.info(), "db_path": str(p.bank.path) if p.bank.path else None})
        if path == "/api/telemetry":
            return self._json(200, p.telemetry_snapshot())
        if path == "/api/telemetry/stream":
            return self._telemetry_sse()
        if path == "/api/credits":
            return self._json(200, {"ok": True, "credits": cad_harvester.credits()})
        if path == "/api/pose":
            snap = p.snapshot()
            return self._json(200, {"ok": True, "pose": snap["pose"], "label": snap["label"], "state": snap["state"],
                                    "best_view": snap["best_view"], "category": snap["category"]})
        if path == "/api/enroll/status":
            return self._json(200, {"ok": True, **p.mv.status()})
        if path == "/api/events":
            return self._sse("telemetry=1" in urllib.parse.urlsplit(self.path).query)
        if path in INGEST_ROUTES[:2] or path in SYNC_FRAME_ROUTES:
            return self._json(200, {"ok": True, "hint": "POST camera frames here (raw image/jpeg, multipart/form-data, or "
                                    "JSON {\"image\": <base64>})", "async_routes": list(INGEST_ROUTES),
                                    "sync_routes": list(SYNC_FRAME_ROUTES), "stream": p.stream_stats()})
        if path == "/api/db/export":
            blob = p.bank.export_bytes()
            if blob is None:
                return self._json(404, {"ok": False, "error": "nothing enrolled yet"})
            return self._send(200, blob, "application/octet-stream")
        if path == "/api/frame.jpg":
            frame = p.latest_jpeg
            if frame is None:
                return self._json(404, {"ok": False, "error": "no live frame (stream idle)" if PREVIEW_FRAME else "preview disabled (deployment mode)"})
            return self._send(200, frame, "image/jpeg")
        if path == "/api/stream":
            if not PREVIEW_FRAME:
                return self._json(404, {"ok": False, "error": "preview disabled (deployment mode)"})
            return self._mjpeg()
        if path == "/api/ring":
            return self._json(200, {"ok": True, **p.ring.info()})
        if path == "/api/ring/frame.jpg":
            try:
                ago = int(urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("ago", ["0"])[0])
            except ValueError:
                return self._json(400, {"ok": False, "error": "ago must be an integer"})
            arr = p.ring.get(ago)
            if arr is None:
                return self._json(404, {"ok": False, "error": "frame not in RAM ring (KAMIKAZE_RING_GB=0 or out of range)"})
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "JPEG", quality=85)
            return self._send(200, buf.getvalue(), "image/jpeg")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        self._json(404, {"ok": False, "error": "not found"})

    def _mjpeg(self) -> None:
        """multipart/x-mixed-replace live view: pushes each new camera frame as it arrives (capped at STREAM_MAX_FPS)."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=kzframe")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        p, seq, last = self.pipeline, -1, 0.0
        min_dt = 1.0 / max(1.0, STREAM_MAX_FPS)
        try:
            while True:
                data, seq = p.wait_frame(seq, SSE_KEEPALIVE_S * 5)
                if data is None:
                    continue
                wait = min_dt - (time.monotonic() - last)
                if wait > 0:
                    time.sleep(wait)
                    data = p.latest_jpeg or data
                last = time.monotonic()
                self.wfile.write(b"--kzframe\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()
        except (OSError, ValueError):
            return

    def _telemetry_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                self.wfile.write(b"data: " + json.dumps(self.pipeline.telemetry_snapshot(), ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(TELEMETRY_PERIOD_S)
        except (OSError, ValueError):
            return

    def do_DELETE(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        parts = [urllib.parse.unquote(s) for s in url.path.split("/") if s]
        if len(parts) != 3 or parts[:2] != ["api", "classes"]:
            return self._json(404, {"ok": False, "error": "not found"})
        query = urllib.parse.parse_qs(url.query)
        delete_cad = query.get("delete_cad", ["1"])[0].lower() not in ("0", "false", "no")
        try:
            label = normalize_label(parts[2])
        except ValueError as exc:
            return self._json(400, {"ok": False, "error": str(exc)})
        removed = self.pipeline.unenroll(label)
        cad_removed = False
        if removed and delete_cad:
            forget = getattr(self.pipeline.assets, "forget", None) or cad_harvester.delete_cached
            cad_removed = forget(label)
        self._json(200 if removed else 404, {"ok": removed, "class_id": label, "cad_deleted": cad_removed,
                                             "classes": self.pipeline.bank.counts()})

    def _sse(self, with_telemetry: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        p, version, last_sent, last_tel = self.pipeline, -1, 0.0, 0.0
        try:
            while True:
                if with_telemetry and time.monotonic() - last_tel >= TELEMETRY_PERIOD_S:
                    last_tel = time.monotonic()  # one connection carries state + telemetry (browsers cap ~6 per host)
                    self.wfile.write(b"event: telemetry\ndata: " + json.dumps(p.telemetry_snapshot(), ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                timeout = min(SSE_KEEPALIVE_S, TELEMETRY_PERIOD_S) if with_telemetry else SSE_KEEPALIVE_S
                new = p.wait_version(version, timeout)
                now = time.monotonic()
                if new != version:
                    if now - last_sent < SSE_PERIOD_S:
                        time.sleep(SSE_PERIOD_S - (now - last_sent))
                    version, last_sent = new, time.monotonic()
                    self.wfile.write(b"data: " + json.dumps(p.snapshot(), ensure_ascii=False).encode() + b"\n\n")
                else:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (OSError, ValueError):
            return

    def _json_body(self, body: bytes) -> dict:
        req = json.loads(body.decode("utf-8") or "{}")
        if not isinstance(req, dict):
            raise ValueError("JSON object expected")
        return req

    def _post_api(self, path: str, body: bytes) -> bool:
        """Multi-view enrollment, db and CAD endpoints. Returns False when the path is not one of them."""
        p = self.pipeline
        parts = [urllib.parse.unquote(s) for s in path.split("/") if s]
        if len(parts) == 4 and parts[:2] == ["api", "classes"] and parts[3] in ("fetch_cad", "refetch_cad"):
            try:
                label = normalize_label(parts[2])
                self._json(202, p.refetch_cad(label))
            except KeyError:
                self._json(404, {"ok": False, "error": f"class {parts[2]!r} is not enrolled"})
            except ValueError as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return True
        if path not in ("/api/enroll/start", "/api/enroll/capture", "/api/enroll/finish", "/api/enroll/cancel",
                        "/api/db/save", "/api/cad/fetch", "/api/classes/reset", "/api/classes/purge", "/api/resolve"):
            return False
        try:
            req = self._json_body(body)
            if path == "/api/classes/purge":
                result = p.purge(delete_cad=bool(req.get("delete_cad", True)))
                self._json(200, {**result, "classes": p.bank.counts()})
            elif path == "/api/enroll/start":
                meta = taxonomy.parse_metadata(req)
                frames = int(req.get("frames_per_view", FRAMES_PER_VIEW))
                with p._infer_lock:
                    status = p.mv.start(meta, frames)
                    p.sm.reset()
                p._bump()
                self._json(200, {**status, "ok": True})
            elif path == "/api/enroll/capture":
                status = p.mv.capture(req.get("view"))
                p._bump()
                if req.get("wait"):
                    status = p.mv.wait_idle(float(req.get("timeout", 30)))
                    p._bump()
                self._json(200 if not status["error"] else 504, {"ok": not status["error"], **status})
            elif path == "/api/enroll/finish":
                with p._infer_lock:
                    status = p.mv.finish()
                p._bump()
                self._json(200 if status["ok"] else 500, {"ok": status["ok"], **status})
            elif path == "/api/enroll/cancel":
                status = p.mv.cancel()
                p._bump()
                self._json(200, {**status, "ok": True})
            elif path == "/api/db/save":
                p.bank.save()
                self._json(200, {"ok": True, "path": str(p.bank.path), "classes": p.bank.counts()})
            elif path == "/api/classes/reset":
                labels = p.bank.labels()
                p.reset_bank()
                drop = getattr(p.assets, "forget", None) or cad_harvester.delete_cached
                cad_removed = [x for x in labels if req.get("delete_cad", True) and drop(x)]
                self._json(200, {"ok": True, "removed": labels, "cad_deleted": cad_removed})
            elif path == "/api/resolve":
                product = entity_resolver.resolve(str(req.get("text") or req.get("raw_name") or ""), str(req.get("category", "")),
                                                  str(req.get("brand", "")), str(req.get("variant", "")),
                                                  str(req.get("package_type", "")), str(req.get("container_volume", "")))
                self._json(200, {"ok": True, **product.to_dict()})
            else:
                if req.get("label"):
                    label = normalize_label(req.get("label"))
                    meta = p.bank.metadata(label) or (taxonomy.parse_metadata(req) if req.get("raw_name") else None)
                    result = cad_harvester.retrieve(label, meta, force=bool(req.get("force")))
                else:
                    product = entity_resolver.resolve(str(req.get("raw_name") or req.get("text") or ""), str(req.get("category", "")),
                                                      str(req.get("brand", "")), str(req.get("variant", "")))
                    label = normalize_label(product.class_id)
                    result = cad_harvester.harvest(product, force=bool(req.get("force")))
                keys = ("title", "glb_url", "source", "status", "query", "attribution", "credit", "match")
                p.bank.set_cad(label, {k: result.get(k) for k in keys})
                if label == p._viewer_label:
                    p._bump()
                self._json(200, {"ok": True, "label": label, **result})
        except EnrollmentBusy as exc:
            self._json(409, {"ok": False, "error": str(exc)})
        except (TypeError, ValueError) as exc:  # includes JSON and MetadataError
            self._json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            log.exception("API call %s failed", path)
            self._json(500, {"ok": False, "error": str(exc)})
        return True

    def _static(self, rel: str) -> None:
        root = STATIC_DIR.resolve()
        try:
            target = (root / urllib.request.url2pathname(rel).lstrip("/\\")).resolve()
        except (OSError, ValueError):
            return self._json(404, {"ok": False, "error": "not found"})
        if (root != target and root not in target.parents) or not target.is_file():
            return self._json(404, {"ok": False, "error": "not found"})
        ctype = _MIME.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        cache = "no-cache" if target.suffix == ".html" else "public, max-age=3600"
        try:
            body = self.pipeline.static_cache.read(target)
        except OSError:
            return self._json(404, {"ok": False, "error": "not found"})
        self._send(200, body, ctype, cache)

    def do_POST(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        path = url.path.rstrip("/") or "/"
        body = self._read_body()
        if body is None:
            return
        if path in INGEST_ROUTES or path in SYNC_FRAME_ROUTES:
            return self._frame_post(path, body, path in SYNC_FRAME_ROUTES or "sync=1" in url.query)
        if self._post_api(path, body):
            return
        if path in ("/api/enroll", "/api/unenroll"):
            try:
                req = json.loads(body.decode("utf-8") or "{}")
                if not isinstance(req, dict):
                    raise ValueError("JSON object expected")
                label = normalize_label(req.get("label"))
            except ValueError as exc:
                return self._json(400, {"ok": False, "error": str(exc)})
            if path == "/api/unenroll":
                removed = self.pipeline.unenroll(label)
                return self._json(200 if removed else 404, {"ok": removed, "label": label})
            return self._enroll(label, req)
        self._json(404, {"ok": False, "error": "not found"})

    def _frame_post(self, path: str, body: bytes, sync: bool) -> None:
        """Camera frame on any ingest alias, in any common encoding. Async routes answer in ~1 ms; /api/frame waits."""
        p = self.pipeline
        ctype = self.headers.get("Content-Type", "")
        source = {"client": self.client_address[0], "path": path, "content_type": ctype.split(";")[0].strip() or None}
        try:
            frame, fmt = extract_frame(body, ctype)
        except ValueError as exc:
            p.note_rejected(source, len(body), body[:16], exc)
            return self._json(400, {"ok": False, "error": str(exc)})
        p.note_source({**source, "format": fmt, "bytes": len(frame)})
        try:
            return self._json(200, p.process_jpeg(frame) if sync else p.ingest(frame))
        except ValueError as exc:
            p.note_rejected(source, len(body), body[:16], exc)
            return self._json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            log.exception("Frame processing failed")
            return self._json(500, {"ok": False, "error": str(exc)})

    def _enroll(self, label: str, req: dict) -> None:
        try:
            frames = int(req.get("frames", ENROLL_FRAMES))
            if not 1 <= frames <= 120:
                raise ValueError("frames must be 1-120")
            timeout = float(req.get("timeout", 30))
            status = self.pipeline.enroller.start(label, frames, bool(req.get("append", False)))
        except EnrollmentBusy as exc:
            return self._json(409, {"ok": False, "error": str(exc)})
        except (TypeError, ValueError) as exc:
            return self._json(400, {"ok": False, "error": str(exc)})
        if not req.get("wait"):
            return self._json(202, {**status, "ok": True, "status": "collecting"})
        status = self.pipeline.enroller.wait(timeout)
        if status["ok"]:
            return self._json(200, {"ok": True, "label": label, "frames": status["total"], **(status["result"] or {})})
        self._json(504, {"ok": False, "label": label, "error": status["error"], "collected": status["collected"]})


def make_server(pipeline: Pipeline, host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"pipeline": pipeline})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def build_detector():
    """YOLO (KAMIKAZE_DETECTOR, default yolo26n) or EfficientDet-Lite object detector on the Hexagon NPU; None (fixed
    center crop) if the NPU is unavailable, unless KAMIKAZE_DETECTOR_CPU=1 accepts the slower CPU detector."""
    ok, why = npu_mod.npu_available()
    if not ok and os.environ.get("KAMIKAZE_DETECTOR_CPU", "0") != "1":
        log.warning("Object detector off (NPU unavailable: %s); using the fixed center crop", why)
        return None
    try:
        return npu_mod.make_detector(npu_mod.DETECTOR_VARIANT, prefer_npu=ok)
    except Exception as exc:
        log.warning("Object detector unavailable, using the fixed center crop (%s: %s)", type(exc).__name__, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--build-model", action="store_true", help="export the ExecuTorch embedder and exit")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("KAMIKAZE_LOG", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.build_model:
        build_pte(PTE_DIR / f"mnv3_embed_{EMBED_PRECISION}.pte", EMBED_PRECISION)
        return

    t0 = time.perf_counter()
    extractor = FeatureExtractor()
    detector = build_detector() if DETECT_CROP else None
    bank = EmbeddingBank(BANK_PATH, extractor.embedder_id)
    loaded = bank.load()
    leds = LedDriver()
    assets = AssetManager()
    pipeline = Pipeline(extractor, bank, leds, assets, detector)
    assets.prepare_static()
    threading.Thread(target=lambda: log.info("RAM asset cache: %d static files preloaded (%s)", pipeline.static_cache.preload(STATIC_DIR),
                                             pipeline.static_cache.info()), name="asset-preload", daemon=True).start()
    if pipeline.ring.enabled:
        log.info("RAM frame ring enabled: %.1f GB budget (training mode, never written to disk)", pipeline.ring.budget / 2**30)
    leds.set(led_params(SEARCHING, 0.0))
    server = make_server(pipeline, args.host, args.port)
    log.info(
        "Kamikaze receiver on %s:%d | embedder %s | detector %s | %s | ready in %.1fs | %d enrolled classes %s | "
        "dashboard http://localhost:%d/",
        args.host, args.port, extractor.name, detector.info()["runtime"] if detector else "off", pipeline.accelerator(),
        time.perf_counter() - t0, loaded, bank.labels(), args.port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        set_leds(0, 0, 0, 0, 0)


if __name__ == "__main__":
    main()
