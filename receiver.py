#!/usr/bin/env python3
"""Kamikaze edge receiver: few-shot visual identification, guidance state machine, 3D dashboard.

One process, one port (0.0.0.0:5000):
  POST /upload | /api/frame | /      raw JPEG frame from stream_cam.py -> JSON verdict
  GET  /                             interactive dashboard + <model-viewer> (static/index.html)
  GET  /api/state                    live telemetry (state, top-1, confidence, guidance, LEDs, ...)
  GET  /api/asset                    active 3D asset for the identified object
  GET  /api/frame.jpg                latest camera frame (dashboard live view)
  GET  /api/classes                  enrolled classes (+ multi-view prototype details)
  POST /api/enroll                   legacy single-shot {"label": "coke_can", "frames": 12, "append": false, "wait": false}
  POST /api/unenroll                 {"label": "coke_can"}
  POST /api/enroll/start             multi-view guided enrollment: {"raw_name", "category", "brand", "variant",
                                     "package_type", "container_volume", "frames_per_view": 8}
  POST /api/enroll/capture           {"view": "front|back|side|detail", "wait": false} capture the current/named view
  POST /api/enroll/finish | /cancel  finish early with the captured views / abort the session
  GET  /api/enroll/status            wizard state (current view, per-view progress + stability)
  GET  /api/pose                     smoothed physical pose {"pitch", "yaw", "roll"}
  GET  /api/events                   Server-Sent Events stream of the live state (10 Hz, change-driven)
  POST /api/db/save | GET /api/db/export   persist / download the enrolled class database
  POST /api/cad/fetch                {"label": "...", "force": true} (re)retrieve the CAD model
  GET  /static/...                   static files, incl. cached GLBs in static/models/

Pipeline per frame: center-60% crop -> 224x224 ImageNet normalisation -> MobileNetV3-Large embedding
(ExecuTorch/XNNPACK fp16 on the Arm CPU, torch eager as fallback) -> L2-normalised vector -> cosine
similarity against the multi-view prototypes (front/back/side/detail centroids + global centroid) -> 5-frame
temporal state machine -> Modulino Pixels via the Arduino Router socket (STM32 firmware untouched), 3D CAD
retrieval and physical pose estimation (pose.py).

Environment (all optional): KAMIKAZE_HOST/PORT, KAMIKAZE_BANK, KAMIKAZE_EMBED_BACKEND (auto|executorch|torch),
KAMIKAZE_EMBED_PRECISION (fp16|fp32), KAMIKAZE_TORCH_THREADS, KAMIKAZE_T_IDENT/T_SEARCH/MARGIN,
KAMIKAZE_LOG_FRAMES (1|0).
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import logging
import math
import mimetypes
import os
import re
import threading
import time
import urllib.error
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

import cad_retriever
import pose as pose_mod
import taxonomy
from mcp_server import (
    FALLBACK_MODELS,
    STATIC_DIR,
    download_glb,
    is_valid_glb,
    local_glb_path,
    search_3d_asset,
    set_leds,
    slugify,
)

log = logging.getLogger("kamikaze.receiver")

BASE_DIR = Path(__file__).resolve().parent
HOST = os.environ.get("KAMIKAZE_HOST", "0.0.0.0")
PORT = int(os.environ.get("KAMIKAZE_PORT", "5000"))
BANK_PATH = Path(os.environ.get("KAMIKAZE_BANK", BASE_DIR / "enrolled_classes.pt"))
PTE_DIR = Path(os.environ.get("KAMIKAZE_PTE_DIR", BASE_DIR / "models"))
EMBED_BACKEND = os.environ.get("KAMIKAZE_EMBED_BACKEND", "auto")
EMBED_PRECISION = os.environ.get("KAMIKAZE_EMBED_PRECISION", "fp16")
TORCH_THREADS = int(os.environ.get("KAMIKAZE_TORCH_THREADS", "1"))
LOG_FRAMES = os.environ.get("KAMIKAZE_LOG_FRAMES", "1") != "0"

INPUT_SIZE = 224
CROP_FRACTION = 0.60
EMBED_DIM = 1280
EMBEDDER_ID = "mobilenet_v3_large-fc1280"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

T_IDENT = float(os.environ.get("KAMIKAZE_T_IDENT", "0.75"))
T_SEARCH = float(os.environ.get("KAMIKAZE_T_SEARCH", "0.40"))
MIN_MARGIN = float(os.environ.get("KAMIKAZE_MARGIN", "0.12"))
WINDOW = 5
IDENT_FRAMES = 3
GUIDE_FRAMES = 4
SEARCH_FRAMES = 3
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
MSG_PHONE_FLIP = taxonomy.MSG_PHONE_FLIP
MSG_PHONE_TILT = taxonomy.MSG_PHONE_TILT
PHONE_AMBIG_MARGIN = float(os.environ.get("KAMIKAZE_PHONE_MARGIN", "0.20"))
FRAMES_PER_VIEW = 8
MIN_VIEW_FRAMES = 4
MV_IDLE_EXPIRE_S = 900.0
SSE_PERIOD_S = 0.1
SSE_KEEPALIVE_S = 1.0


# --------------------------------------------------------------------------------------
# Preprocessing + embedding backbone
# --------------------------------------------------------------------------------------


def center_crop_box(width: int, height: int) -> tuple[int, int, int]:
    """Center square covering CROP_FRACTION of the short side: (size, left, top)."""
    size = max(1, int(CROP_FRACTION * min(width, height)))
    return size, (width - size) // 2, (height - size) // 2


def preprocess(image: Image.Image) -> np.ndarray:
    """Center-60% crop -> 224x224 -> ImageNet normalisation. Returns contiguous float32 (1,3,224,224)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    size, left, top = center_crop_box(*image.size)
    crop = image.crop((left, top, left + size, top + size)).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = (np.asarray(crop, dtype=np.float32) * (1.0 / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


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
    """Image -> L2-normalised 1280-d embedding."""

    def __init__(self, backend: str = EMBED_BACKEND, precision: str = EMBED_PRECISION, threads: int = TORCH_THREADS):
        if torch is None:
            raise RuntimeError("PyTorch is required for the embedding backbone")
        self._lock = threading.Lock()
        self._runtime = None
        self._model = None
        self.dtype = torch.float32
        self.name = ""
        if backend in ("auto", "executorch"):
            try:
                self._init_executorch(precision)
            except Exception as exc:  # export/runtime problems must degrade, not kill the demo
                log.warning("ExecuTorch backend unavailable (%s: %s)", type(exc).__name__, exc)
                if backend == "executorch":
                    raise
        if self._runtime is None:
            torch.set_num_threads(threads)
            self._model = build_eager_model()
            self.name = "torch-eager-fp32"
        self.embed_tensor(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE) + 0.1)

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


def normalize_label(raw: object) -> str:
    label = re.sub(r"[\s\-]+", "_", str(raw or "").strip().lower())
    if not _LABEL_RE.match(label):
        raise ValueError("label must be 1-40 chars of a-z, 0-9, _ (e.g. coke_can)")
    return label


class EmbeddingBank:
    """label -> exemplar embeddings and prototypes.

    Each label has a global L2-normalised centroid and, for multi-view enrollments, angle-tagged view centroids
    (front/back/side/detail) plus metadata and the resolved CAD asset. A label's score for a live vector is the
    best cosine over its global and view centroids.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self.exemplars: dict[str, list] = {}
        self.centroids: dict[str, "torch.Tensor"] = {}
        self.views: dict[str, dict[str, "torch.Tensor"]] = {}
        self.view_counts: dict[str, dict[str, int]] = {}
        self.meta: dict[str, dict] = {}
        self.cad: dict[str, dict | None] = {}
        self._labels: list[str] = []
        self._rows: list[str] = []
        self._matrix = None

    def _rebuild(self) -> None:
        self._labels = list(self.centroids)
        rows, vecs = [], []
        for label in self._labels:
            rows.append(label)
            vecs.append(self.centroids[label])
            for view_vec in self.views.get(label, {}).values():
                rows.append(label)
                vecs.append(view_vec)
        self._rows = rows
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
            for table in (self.exemplars, self.views, self.view_counts, self.meta, self.cad):
                table.pop(label, None)
            self._rebuild()
            self.save()
            return True

    def scores(self, vec) -> list[tuple[str, float]]:
        """Best cosine similarity per label over its global + view centroids, best first."""
        with self._lock:
            if self._matrix is None:
                return []
            sims = (self._matrix @ vec).tolist()
            best: dict[str, float] = {}
            for label, sim in zip(self._rows, sims):
                if sim > best.get(label, -2.0):
                    best[label] = sim
            return sorted(best.items(), key=lambda kv: -kv[1])

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
            return [
                {"label": k, "vectors": len(self.exemplars.get(k, [])), "views": dict(self.view_counts.get(k, {})),
                 "metadata": dict(self.meta.get(k) or {}), "cad_asset": self.cad.get(k)}
                for k in self._labels
            ]

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            prototypes = {}
            for label in self.centroids:
                prototypes[label] = {
                    "metadata": dict(self.meta.get(label) or {}),
                    "views": dict(self.views.get(label, {})),
                    "view_counts": dict(self.view_counts.get(label, {})),
                    "global_centroid": self.centroids[label],
                    "cad_asset": self.cad.get(label),
                }
            payload = {
                "version": 2,
                "embedder": EMBEDDER_ID,
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
            if data.get("embedder") != EMBEDDER_ID:
                log.warning("Ignoring %s: embedder %r != %r", self.path, data.get("embedder"), EMBEDDER_ID)
                return 0
            with self._lock:
                self.exemplars = {k: list(v.unbind(0)) for k, v in data["exemplars"].items()}
                self.centroids = {k: v.float() for k, v in data["centroids"].items()}
                protos = data.get("prototypes") or {}
                self.views = {k: {vn: vv.float() for vn, vv in (p.get("views") or {}).items()} for k, p in protos.items()}
                self.view_counts = {k: dict(p.get("view_counts") or {}) for k, p in protos.items()}
                self.meta = {k: dict(p.get("metadata") or {}) for k, p in protos.items()}
                self.cad = {k: p.get("cad_asset") for k, p in protos.items()}
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
    """Guided 4-angle enrollment wizard: front -> back -> side -> detail, N frames per view.

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
        label = normalize_label(taxonomy.label_from_metadata(meta))
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
    s1: float
    s2: float = 0.0
    top2: str | None = None
    hint: str | None = None  # "flip" | "tilt": the match is ambiguous and needs another look at the phone

    @property
    def margin(self) -> float:
        return self.s1 - self.s2


class StateMachine:
    """5-frame sliding window with hysteresis between SEARCHING / GUIDING / IDENTIFIED."""

    def __init__(self) -> None:
        self.window: collections.deque = collections.deque(maxlen=WINDOW)
        self.state = SEARCHING
        self.label: str | None = None
        self.hint: str | None = None

    def reset(self) -> None:
        self.window.clear()
        self.state, self.label, self.hint = SEARCHING, None, None

    @staticmethod
    def frame_kind(frame: FrameScore) -> str:
        if frame.label is None or frame.s1 < T_SEARCH:
            return "search"
        if frame.s1 >= T_IDENT and frame.margin >= MIN_MARGIN and not frame.hint:
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
        self.hint = frame.hint
        self.window.append((kind, frame.label, frame.s1, frame.margin))
        if kind == "ident" and self._streak(lambda w: w[0] == "ident" and w[1] == frame.label) >= IDENT_FRAMES:
            self.state, self.label = IDENTIFIED, frame.label
        elif kind == "guide" and self._streak(lambda w: w[0] == "guide") >= GUIDE_FRAMES:
            self.state, self.label = GUIDING, frame.label
        elif kind == "search" and self._streak(lambda w: w[0] == "search") >= SEARCH_FRAMES:
            self.state, self.label = SEARCHING, None
        elif self.state == GUIDING and kind != "search":
            self.label = frame.label

        subject = self.label if self.state != SEARCHING else frame.label
        picks = [w for w in self.window if w[1] == subject] or [self.window[-1]]
        return {
            "state": self.state,
            "label": self.label,
            "score": sum(w[2] for w in picks) / len(picks),
            "margin": sum(w[3] for w in picks) / len(picks),
        }


def guidance_message(state: str, label: str | None, score: float, hint: str | None = None,
                     category: str | None = None) -> str:
    if state == IDENTIFIED:
        return f"Object Verified: {label} ({max(0, round(score * 100))}%)"
    if state == GUIDING:
        if hint == "tilt":
            return MSG_PHONE_TILT
        if hint == "flip":
            return MSG_PHONE_FLIP
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
    return MSG_SEARCHING


def phone_hint(category: str | None, best_view: str | None, margin: float) -> str | None:
    """Smartphone matches that are too close to a sibling model need the rear island / lens count to be shown."""
    if category != "smartphone" or margin >= PHONE_AMBIG_MARGIN:
        return None
    return "flip" if best_view in (None, "front") else "tilt"


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
    def __init__(self, setter: Callable[..., bool] = set_leds) -> None:
        self._setter = setter
        self._cv = threading.Condition()
        self._target: tuple | None = None
        self._sent: tuple | None = None
        self.ok: bool | None = None
        self.latency_ms = 0.0
        threading.Thread(target=self._run, name="led", daemon=True).start()

    def set(self, params: tuple) -> None:
        with self._cv:
            self._target = tuple(params)
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._target is None or self._target == self._sent:
                    self._cv.wait()
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
                    self._sent = params
                    log.info("LEDs set_pixels%s -> ok (%.1f ms)", params, self.latency_ms)
            if not ok:
                time.sleep(1.0)

    def status(self) -> dict:
        with self._cv:
            r, g, b, brightness, count = self._target or (0, 0, 0, 0, 0)
            return {"r": r, "g": g, "b": b, "brightness": brightness, "count": count, "ok": self.ok,
                    "latency_ms": round(self.latency_ms, 1)}


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
        self.meta_lookup: Callable[[str], dict | None] | None = None
        self.on_resolved: Callable[[str, dict | None], None] | None = None

    def peek(self, label: str) -> dict:
        slug = slugify(label)
        with self._lock:
            info = dict(self._info.get(label, {}))
            busy = label in self._inflight
        if is_valid_glb(local_glb_path(slug)):
            return {"glb_url": f"/static/models/{slug}.glb", "status": "ready", "title": info.get("title", label)}
        return {"glb_url": DEFAULT_GLB_URL, "status": "resolving" if busy else "pending", "title": info.get("title", label)}

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

    def _resolve(self, label: str) -> None:
        try:
            meta = self.meta_lookup(label) if self.meta_lookup else None
            result = cad_retriever.retrieve(label, meta) if meta else search_3d_asset(label)
            with self._lock:
                if result.get("glb_url"):
                    self._info[label] = {"title": result.get("title", label), "source": result.get("source") or result.get("status")}
                    self._failed_at.pop(label, None)
                    if self.on_resolved:
                        self.on_resolved(label, {k: result.get(k) for k in ("title", "glb_url", "source", "status", "query")})
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
                download_glb(FALLBACK_MODELS["_generic_phone"]["model_url"], "default")
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
# Pipeline: frame -> embedding -> scores -> state -> actuation / assets
# --------------------------------------------------------------------------------------


class Pipeline:
    def __init__(self, extractor, bank: EmbeddingBank, leds: LedDriver, assets: AssetManager) -> None:
        self.extractor, self.bank, self.leds, self.assets = extractor, bank, leds, assets
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
            assets.on_resolved = bank.set_cad
        self.sm = StateMachine()
        self._infer_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._snap_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._frames = 0
        self._fps = 0.0
        self._last_frame_mono = 0.0
        self._viewer_label: str | None = None
        self._last = {
            "label": None, "similarity": 0.0, "confidence": 0.0, "margin": 0.0, "top2": None,
            "embed_ms": 0.0, "total_ms": 0.0, "state": SEARCHING, "guidance": MSG_SEARCHING,
        }

    @property
    def latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def _bump(self) -> None:
        with self._ver_cv:
            self._version += 1
            self._ver_cv.notify_all()

    def wait_version(self, last: int, timeout: float) -> int:
        with self._ver_cv:
            self._ver_cv.wait_for(lambda: self._version != last, timeout)
            return self._version

    def _mv_complete(self, label: str) -> None:
        self.sm.reset()
        self.pose_est.reset()
        self._viewer_label = label
        self.assets.get(label)
        self._bump()

    def process_jpeg(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:
            raise ValueError(f"undecodable image: {exc}") from exc
        image = image.convert("RGB")
        with self._frame_lock:
            self._latest_jpeg = data

        with self._infer_lock:
            arr = preprocess(image)
            t1 = time.perf_counter()
            vec = self.extractor.embed_array(arr)
            t2 = time.perf_counter()
            if self.enroller.feed(vec):
                self.sm.reset()
            capturing = self.mv.capturing
            mv_event = self.mv.feed(vec)
            if mv_event == "complete":
                capturing = False
            ranked = self.bank.scores(vec)
            top1 = ranked[0] if ranked else (None, 0.0)
            top2 = ranked[1] if len(ranked) > 1 else (None, 0.0)
            category = self.bank.category(top1[0])
            views = self.bank.view_sims(top1[0], vec) if top1[0] else {}
            best_view = max(views, key=views.get) if views else None
            hint = phone_hint(category, best_view, top1[1] - top2[1]) if top1[1] >= T_SEARCH else None
            frame = FrameScore(top1[0], top1[1], top2[1], top2[0], hint)
            prev = (self.sm.state, self.sm.label)
            snap = self.sm.update(frame)
            state, label, score = snap["state"], snap["label"], snap["score"]
            message = guidance_message(state, label, score, hint if state == GUIDING else None, category)
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
                    size, left, top = center_crop_box(*image.size)
                    pose = self.pose_est.update(image, (size, left, top), views, frame.label, category)
                except Exception:
                    log.exception("pose estimation failed")
            self._pose, self._best_view, self._category = pose, best_view, category
            self._frames += 1
            now = time.monotonic()
            if self._last_frame_mono:
                inst = 1.0 / max(1e-3, now - self._last_frame_mono)
                self._fps = inst if self._fps == 0 else 0.8 * self._fps + 0.2 * inst
            self._last_frame_mono = now
            total_ms = (time.perf_counter() - t0) * 1000.0
            with self._snap_lock:
                self._last = {
                    "label": frame.label, "similarity": frame.s1, "confidence": score, "margin": snap["margin"],
                    "top2": {"label": top2[0], "score": top2[1]} if top2[0] else None,
                    "embed_ms": (t2 - t1) * 1000.0, "total_ms": total_ms, "state": state, "guidance": message,
                }
            frames_seen = self._frames
        if LOG_FRAMES:
            log.info(
                "[Frame %05d] %s | %s s1=%.3f margin=%.3f | preprocess %.1fms embed %.1fms total %.1fms",
                frames_seen, state, frame.label or "-", frame.s1, frame.margin,
                (t1 - t0) * 1000, (t2 - t1) * 1000, total_ms,
            )
        if (state, label) != prev:
            self._announce(state, label, score, snap["margin"], message)
        self._bump()
        return {
            "ok": True, "state": state, "label": frame.label, "confidence": round(score, 3),
            "similarity": round(frame.s1, 3), "margin": round(snap["margin"], 3),
            "embed_ms": round((t2 - t1) * 1000, 1), "latency_ms": round(total_ms, 1),
        }

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

    def unenroll(self, label: str) -> bool:
        with self._infer_lock:
            removed = self.bank.remove(label)
            if removed:
                self.sm.reset()
                self.pose_est.reset()
                if self._viewer_label == label:
                    self._viewer_label = None
                with self._snap_lock:
                    self._last.update(label=None, similarity=0.0, confidence=0.0, margin=0.0, top2=None,
                                      state=SEARCHING, guidance=MSG_SEARCHING)
                self.leds.set(led_params(SEARCHING, 0.0))
            return removed

    def snapshot(self) -> dict:
        with self._snap_lock:
            last = dict(self._last)
        age = round(time.monotonic() - self._last_frame_mono, 2) if self._last_frame_mono else None
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
            "confidence": round(last["confidence"], 3),
            "similarity": round(last["similarity"], 3),
            "margin": round(last["margin"], 3),
            "top2": last["top2"],
            "guidance": last["guidance"],
            "action": {IDENTIFIED: "none", GUIDING: "reorient"}.get(state, "present_object"),
            "glb_url": viewer["glb_url"],
            "asset_status": viewer["status"],
            "latency_ms": {"embed": round(last["embed_ms"], 1), "total": round(last["total_ms"], 1)},
            "fps": round(self._fps, 1),
            "frames": self._frames,
            "frame_age_s": age,
            "led": self.leds.status(),
            "enrollment": self.enroller.status(),
            "multiview": self.mv.status(),
            "pose": self._pose,
            "category": self._category,
            "best_view": self._best_view,
            "hint": "flip" if last["guidance"] == MSG_PHONE_FLIP else ("tilt" if last["guidance"] == MSG_PHONE_TILT else None),
            "crop_fraction": CROP_FRACTION,
            "system": self.stats.read(),
            "bank": self.bank.info(),
            "classes": self.bank.counts(),
            "backend": self.extractor.name,
            "thresholds": {"identified": T_IDENT, "searching": T_SEARCH, "margin": MIN_MARGIN},
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
            return self._json(200, {"ok": True, "classes": p.bank.counts(), "prototypes": p.bank.info()})
        if path == "/api/pose":
            snap = p.snapshot()
            return self._json(200, {"ok": True, "pose": snap["pose"], "label": snap["label"], "state": snap["state"],
                                    "best_view": snap["best_view"], "category": snap["category"]})
        if path == "/api/enroll/status":
            return self._json(200, {"ok": True, **p.mv.status()})
        if path == "/api/events":
            return self._sse()
        if path == "/api/db/export":
            blob = p.bank.export_bytes()
            if blob is None:
                return self._json(404, {"ok": False, "error": "nothing enrolled yet"})
            return self._send(200, blob, "application/octet-stream")
        if path == "/api/frame.jpg":
            frame = p.latest_jpeg
            if frame is None:
                return self._json(404, {"ok": False, "error": "no frame received yet"})
            return self._send(200, frame, "image/jpeg")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        self._json(404, {"ok": False, "error": "not found"})

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        p, version, last_sent = self.pipeline, -1, 0.0
        try:
            while True:
                new = p.wait_version(version, SSE_KEEPALIVE_S)
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
        if path not in ("/api/enroll/start", "/api/enroll/capture", "/api/enroll/finish", "/api/enroll/cancel",
                        "/api/db/save", "/api/cad/fetch"):
            return False
        try:
            req = self._json_body(body)
            if path == "/api/enroll/start":
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
            else:
                label = normalize_label(req.get("label"))
                meta = p.bank.metadata(label) or (taxonomy.parse_metadata(req) if req.get("raw_name") else None)
                result = cad_retriever.retrieve(label, meta, force=bool(req.get("force")))
                p.bank.set_cad(label, {k: result.get(k) for k in ("title", "glb_url", "source", "status", "query")})
                self._json(200, {"ok": True, **result})
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
        self._send(200, target.read_bytes(), ctype, cache)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        body = self._read_body()
        if body is None:
            return
        if path in ("/", "/upload", "/api/frame"):
            try:
                return self._json(200, self.pipeline.process_jpeg(body))
            except ValueError as exc:
                return self._json(400, {"ok": False, "error": str(exc)})
            except Exception as exc:
                log.exception("Frame processing failed")
                return self._json(500, {"ok": False, "error": str(exc)})
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
    bank = EmbeddingBank(BANK_PATH)
    loaded = bank.load()
    leds = LedDriver()
    assets = AssetManager()
    pipeline = Pipeline(extractor, bank, leds, assets)
    assets.prepare_static()
    leds.set(led_params(SEARCHING, 0.0))
    server = make_server(pipeline, args.host, args.port)
    log.info(
        "Kamikaze receiver on %s:%d | embedder %s (%.1fs) | %d enrolled classes %s | dashboard http://localhost:%d/",
        args.host, args.port, extractor.name, time.perf_counter() - t0, loaded, bank.labels(), args.port,
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
