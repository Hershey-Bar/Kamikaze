"""Hexagon NPU acceleration for the RapidOCR (PP-OCRv4) pipeline.

The PP-OCRv4 text detector (DBNet, the dominant OCR cost) is exported to a fixed-shape TFLite graph (export_ppocr.sh)
and swapped into RapidOCR's detector in place of its ONNX Runtime session; it runs on the HTP through the same QNN
TFLite delegate as the embedder / detector / DINOv2 (16 ms vs 110 ms on the Kryo CPU at 320 px). The recogniser stays on
ONNX Runtime CPU. Any NPU failure falls back to the original CPU session for that call, so OCR never stops.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard dependency of rapidocr
    cv2 = None

log = logging.getLogger("kamikaze.ocr")

DET_SIDE = int(os.environ.get("KAMIKAZE_OCR_SIDE", "320"))
ENABLED = os.environ.get("KAMIKAZE_OCR_NPU", "1") != "0"
MAX_ERRORS = 5


def det_model_path(side: int = DET_SIDE) -> Path:
    import npu

    return npu.MODELS_DIR / f"ppocrv4_det_{side}_float32.tflite"


class HtpDetSession:
    """Drop-in for rapidocr's OrtInferSession on the detector: __call__(x NCHW) -> [prob map NCHW]."""

    def __init__(self, runtime, cpu_session, side: int) -> None:
        self.rt, self.cpu, self.side = runtime, cpu_session, int(side)
        self._out = runtime.outputs()[0]["index"]
        self._lat: collections.deque = collections.deque(maxlen=120)
        self._lock = threading.Lock()
        self.npu_runs = 0
        self.cpu_runs = 0
        self.errors = 0
        self.last_error: str | None = None

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        if self.errors >= MAX_ERRORS or x.ndim != 4 or x.shape[0] != 1:
            self.cpu_runs += 1
            return self.cpu(x)
        t0 = time.perf_counter()
        try:
            h, w = int(x.shape[2]), int(x.shape[3])
            nhwc = x[0].transpose(1, 2, 0)
            if (h, w) != (self.side, self.side):
                nhwc = cv2.resize(nhwc, (self.side, self.side), interpolation=cv2.INTER_LINEAR)
            prob = self.rt.run(np.ascontiguousarray(nhwc[None], dtype=np.float32), [self._out])[0]
            prob = prob.reshape(self.side, self.side).astype(np.float32)
            if (h, w) != (self.side, self.side):
                prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception as exc:  # delegate hiccup: this frame on the CPU, give up on the NPU after MAX_ERRORS
            self.errors += 1
            self.last_error = str(exc)
            log.warning("PP-OCRv4 det on the NPU failed (%d/%d): %s", self.errors, MAX_ERRORS, exc)
            self.cpu_runs += 1
            return self.cpu(x)
        with self._lock:
            self._lat.append((time.perf_counter() - t0) * 1000.0)
        self.npu_runs += 1
        return [prob[None, None]]

    def info(self) -> dict:
        with self._lock:
            lat = sorted(self._lat)
        return {"device": self.rt.device if self.errors < MAX_ERRORS else "cpu", "runtime": self.rt.note,
                "model": det_model_path(self.side).name, "input": [self.side, self.side],
                "median_ms": round(lat[len(lat) // 2], 2) if lat else None, "npu_runs": self.npu_runs,
                "cpu_runs": self.cpu_runs, "errors": self.errors, "last_error": self.last_error}


def accelerate(engine, side: int = DET_SIDE) -> HtpDetSession | None:
    """Swap RapidOCR's PP-OCRv4 detector session for the HTP graph. Returns the session, or None (engine unchanged)
    when disabled, the exported model is missing, or the delegate cannot load it on the NPU."""
    if not ENABLED:
        log.info("OCR NPU detector disabled (KAMIKAZE_OCR_NPU=0)")
        return None
    det = getattr(engine, "text_det", None)
    if det is None or not hasattr(det, "infer") or cv2 is None:
        return None
    path = det_model_path(side)
    if not path.is_file():
        log.warning("OCR detector stays on the CPU: %s missing (run export_ppocr.sh on the board)", path)
        return None
    try:
        import npu

        rt = npu.Runtime(path, True)
    except Exception as exc:
        log.warning("OCR detector stays on the CPU: %s", exc)
        return None
    if rt.device != "npu":
        log.warning("OCR detector stays on ONNX Runtime CPU: %s", rt.note)
        return None
    session = HtpDetSession(rt, det.infer, side)
    det.infer = session
    log.info("OCR detector PP-OCRv4 on the Hexagon NPU (%s, %d px, init %.0f ms)", rt.note, side, rt.init_ms)
    return session
