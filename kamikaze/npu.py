"""Hexagon NPU acceleration for Kamikaze (Qualcomm QCS8275 HTP v75 via the QNN TFLite delegate).

    NpuEmbedder  - MediaPipe Image Embedder (MobileNetV3-Large, Apache-2.0): 224x224 RGB -> L2-normalised 1280-d vector.
                   ~0.7 ms/frame on the HTP vs ~24 ms on the Arm CPU; HTP/CPU output cosine 0.99998.
    YoloDetector - Ultralytics YOLO26n / YOLO11n (COCO-80, AGPL-3.0) exported to FP32 TFLite (export_yolo.sh) and run
                   in FP16 on the HTP: ~6 ms/frame at 640x640 vs ~170-200 ms on the Arm CPU. Letterboxed input,
                   raw (1, 4+nc, 8400) head decoded + NMS here. Without the .tflite it falls back to Ultralytics
                   PyTorch on the CPU (UltralyticsDetector).
    NpuDetector  - MediaPipe EfficientDet-Lite0/2 int8 (COCO-90, Apache-2.0), kept as an alternative
                   (KAMIKAZE_DETECTOR=lite2). Raw SSD outputs decoded with anchors matching TFLite_Detection_PostProcess.
    CropTracker  - smooths the detector box over frames into the square crop that feeds the embedder.
    DinoPoseEmbedder - Meta DINOv2 ViT-S/14 (Apache-2.0) exported by export_dinov2.py: 224x224 crop -> [CLS | mean patch]
                   viewpoint descriptor for continuous yaw estimation, ~20 ms on the HTP vs ~240 ms on the CPU.

Runtime: ai-edge-litert + libQnnTFLiteDelegate.so from Qualcomm's qairt-libs / qairt-dsp-binaries packages
(ppa:ubuntu-qcom-iot/qcom-ppa). Each model is compiled for the HTP on first load (~0.7 s embedder, ~2-4 s detector);
if the delegate, /dev/fastrpc-cdsp or the model is unavailable the same .tflite runs on the CPU (XNNPACK) instead.

Models live in models/npu/ (override: KAMIKAZE_NPU_MODELS): MediaPipe models are downloaded on first use, YOLO models
are produced on the board by ./export_yolo.sh. Environment: KAMIKAZE_NPU (1|0), KAMIKAZE_QNN_DELEGATE,
KAMIKAZE_DETECTOR (yolo26n|yolo11n|<custom stem>|lite2|lite0), KAMIKAZE_DETECT_SCORE, KAMIKAZE_NPU_VERBOSE (0|1).
"""

from __future__ import annotations

import contextlib
import functools
import glob
import json
import logging
import math
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("kamikaze.npu")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("KAMIKAZE_NPU_MODELS", ROOT / "models" / "npu"))
NPU_ENABLED = os.environ.get("KAMIKAZE_NPU", "1") != "0"
QNN_DELEGATE = os.environ.get("KAMIKAZE_QNN_DELEGATE", "libQnnTFLiteDelegate.so")
QNN_OPTIONS = {"backend_type": "htp", "htp_performance_mode": "2", "htp_precision": "1", "log_level": "0"}
FASTRPC_DEVICE = "/dev/fastrpc-cdsp"
DSP_LIB_DIRS = ("/usr/lib/dsp/cdsp", "/usr/lib/rfsa/adsp", "/usr/share/qcom")
DETECTOR_VARIANT = os.environ.get("KAMIKAZE_DETECTOR", "yolov8s-worldv2-kz")
DETECT_SCORE = float(os.environ.get("KAMIKAZE_DETECT_SCORE", "0.30"))
VERBOSE = os.environ.get("KAMIKAZE_NPU_VERBOSE", "0") == "1"
CPU_THREADS = 4
MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class ModelSpec:
    file: str
    url: str
    title: str
    license: str = "Apache-2.0"


EMBEDDER_SPEC = ModelSpec(
    "mobilenet_v3_large_embedder.tflite",
    f"{MEDIAPIPE}/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite",
    "MediaPipe Image Embedder - MobileNetV3-Large (float32)",
)
DETECTOR_SPECS = {
    "lite0": ModelSpec("efficientdet_lite0_int8.tflite",
                       f"{MEDIAPIPE}/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite",
                       "MediaPipe Object Detector - EfficientDet-Lite0 (int8, COCO)"),
    "lite2": ModelSpec("efficientdet_lite2_int8.tflite",
                       f"{MEDIAPIPE}/object_detector/efficientdet_lite2/int8/latest/efficientdet_lite2.tflite",
                       "MediaPipe Object Detector - EfficientDet-Lite2 (int8, COCO)"),
}
YOLO_SPECS = {
    "yolov8s-worldv2-kz": ModelSpec("yolov8s-worldv2-kz_float32.tflite", "",
                                    "Ultralytics YOLO-World v2 S, open vocabulary {beverage can, soda can, smartphone, "
                                    "bottle} (TFLite FP32 -> HTP FP16)", "AGPL-3.0"),
    "yolo26n": ModelSpec("yolo26n_float32.tflite", "", "Ultralytics YOLO26n (COCO-80, TFLite FP32 -> HTP FP16)", "AGPL-3.0"),
    "yolo11n": ModelSpec("yolo11n_float32.tflite", "", "Ultralytics YOLO11n (COCO-80, TFLite FP32 -> HTP FP16)", "AGPL-3.0"),
}
DINO_SPEC = ModelSpec("dinov2_vits14_pose_float32.tflite", "",
                      "Meta DINOv2 ViT-S/14 (224 px, [CLS | mean patch] 2x384-d, TFLite FP32 -> HTP FP16)")
DINO_ID = "dinov2_vits14-cls+meanpatch-768"
WORLD_VARIANT = "yolov8s-worldv2-kz"  # export_yolo.sh world  (YOLO-World with the domain vocabulary baked in)
WORLD_CLASSES = ("beverage can", "soda can", "smartphone", "bottle")
FALLBACK_DETECTOR = "yolo26n"
EMBEDDER_ID = "mediapipe-mobilenet_v3_large-embedder-1280"

COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "???", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "???", "backpack", "umbrella", "???", "???", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "???", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "???", "dining table", "???", "???", "toilet", "???", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "???", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)
COCO80_LABELS = tuple(name for name in COCO_LABELS if name != "???")  # Ultralytics/YOLO class order
# COCO has no "can": cans come out as bottle/cup; phones as cell phone (sometimes remote when edge-on).
# The extra names cover a custom-trained YOLO (export_yolo.sh path/to/best.pt) with its own phone/can classes.
TARGET_CLASSES = {"cell phone": "smartphone", "remote": "smartphone", "bottle": "beverage", "cup": "beverage",
                  "smartphone": "smartphone", "phone": "smartphone", "can": "beverage", "soda can": "beverage",
                  "beverage can": "beverage"}
TARGET_IDS = np.array([COCO_LABELS.index(name) for name in TARGET_CLASSES if name in COCO_LABELS], dtype=np.int64)


# --------------------------------------------------------------------------------------
# Runtime: QNN HTP delegate with CPU fallback
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def htp_arch() -> str | None:
    """Hexagon HTP generation from the installed DSP skeleton libraries, e.g. "v75"."""
    for base in DSP_LIB_DIRS:
        for path in sorted(glob.glob(os.path.join(base, "**", "libQnnHtpV*Skel.so"), recursive=True)):
            name = os.path.basename(path)
            return "v" + name[len("libQnnHtpV"):-len("Skel.so")]
    return None


def npu_available() -> tuple[bool, str]:
    if not NPU_ENABLED:
        return False, "disabled (KAMIKAZE_NPU=0)"
    if not os.path.exists(FASTRPC_DEVICE):
        return False, f"{FASTRPC_DEVICE} not present"
    if not os.access(FASTRPC_DEVICE, os.R_OK | os.W_OK):
        return False, f"no read/write access to {FASTRPC_DEVICE} (add the user to the fastrpc group)"
    return True, "ok"


@contextlib.contextmanager
def _quiet_native():
    """Silence QNN's native stdout/stderr chatter ("Starting stage...", rpcmem) during graph preparation."""
    if VERBOSE:
        yield
        return
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    saved = []
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    try:
        for fd in (1, 2):
            with contextlib.suppress(OSError):
                saved.append((fd, os.dup(fd)))
                os.dup2(devnull, fd)
        yield
    finally:
        for fd, copy in saved:
            with contextlib.suppress(OSError):
                os.dup2(copy, fd)
                os.close(copy)
        os.close(devnull)


def ensure_model(spec: ModelSpec, models_dir: Path | None = None, timeout: float = 120.0) -> Path:
    """Path to a local copy of `spec`, downloading it once (atomic) if needed."""
    directory = Path(models_dir or MODELS_DIR)
    path = directory / spec.file
    if _valid_tflite(path):
        return path
    if not spec.url:
        raise FileNotFoundError(f"{path} not found: run ./export_yolo.sh {path.name.split('_float32')[0]} on the board")
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log.info("Downloading %s ...", spec.url)
    req = urllib.request.Request(spec.url, headers={"User-Agent": "kamikaze-edge/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if not _valid_tflite(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {spec.file} is not a TFLite flatbuffer")
    os.replace(tmp, path)
    log.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def _valid_tflite(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
        return path.stat().st_size > 100_000 and head[4:8] == b"TFL3"
    except OSError:
        return False


@dataclass
class Runtime:
    """One TFLite interpreter bound to the HTP (or the CPU fallback)."""

    path: Path
    prefer_npu: bool = True
    device: str = field(init=False, default="cpu")
    note: str = field(init=False, default="")
    init_ms: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        from ai_edge_litert.interpreter import Interpreter, load_delegate

        t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.interpreter = None
        ok, why = npu_available() if self.prefer_npu else (False, "CPU requested")
        if ok:
            try:
                with _quiet_native():
                    delegate = load_delegate(QNN_DELEGATE, dict(QNN_OPTIONS))
                    it = Interpreter(model_path=str(self.path), experimental_delegates=[delegate], num_threads=CPU_THREADS)
                    it.allocate_tensors()
                    self._warmup(it)
                self.interpreter, self.device = it, "npu"
                self.note = f"QNN HTP {htp_arch() or ''}".strip()
            except (ValueError, RuntimeError, OSError) as exc:
                why = f"QNN delegate failed: {exc}"
        if self.interpreter is None:
            it = Interpreter(model_path=str(self.path), num_threads=CPU_THREADS)
            it.allocate_tensors()
            self._warmup(it)
            self.interpreter, self.device, self.note = it, "cpu", f"XNNPACK CPU ({why})"
            log.warning("%s runs on the CPU: %s", self.path.name, why)
        self.init_ms = (time.perf_counter() - t0) * 1000.0
        log.info("%s ready on %s in %.0f ms", self.path.name, self.note, self.init_ms)

    @staticmethod
    def _warmup(it) -> None:
        inp = it.get_input_details()[0]
        it.set_tensor(inp["index"], np.zeros(inp["shape"], dtype=inp["dtype"]))
        it.invoke()

    @property
    def input(self) -> dict:
        return self.interpreter.get_input_details()[0]

    def outputs(self) -> list[dict]:
        return self.interpreter.get_output_details()

    def run(self, x: np.ndarray, output_indices: list[int]) -> list[np.ndarray]:
        with self.lock:
            try:
                self.interpreter.set_tensor(self.input["index"], x)
                self.interpreter.invoke()
            except RuntimeError as exc:
                # A failed QNN graph (e.g. HTP error 1003) never recovers and its error text grows on every
                # invoke; rebuild once on the NPU, then fall back to the CPU so the pipeline keeps running.
                self._recover(str(exc)[:160])
                self.interpreter.set_tensor(self.input["index"], x)
                self.interpreter.invoke()
            return [_dequantize(self.interpreter.get_tensor(i), d) for i, d in
                    ((i, next(o for o in self.outputs() if o["index"] == i)) for i in output_indices)]

    def _recover(self, why: str) -> None:
        self.failures = getattr(self, "failures", 0) + 1
        old_device = self.device
        log.error("%s invoke failed on %s (%s); rebuilding (%s)", self.path.name, old_device, why,
                  "NPU" if self.failures == 1 and self.prefer_npu else "CPU")
        self.interpreter = None
        if self.failures > 1:
            self.prefer_npu = False
        lock = self.lock
        self.__post_init__()
        self.lock = lock


def _dequantize(arr: np.ndarray, detail: dict) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.floating):
        return arr
    scale, zero = detail.get("quantization", (0.0, 0))
    return (arr.astype(np.float32) - zero) * scale if scale else arr.astype(np.float32)


# --------------------------------------------------------------------------------------
# Embedder
# --------------------------------------------------------------------------------------


class NpuEmbedder:
    """RGB crop -> unit-norm 1280-d embedding (MediaPipe MobileNetV3-Large image embedder)."""

    embedder_id = EMBEDDER_ID

    def __init__(self, path: Path | None = None, prefer_npu: bool = True):
        self.spec = EMBEDDER_SPEC
        self.rt = Runtime(Path(path) if path else ensure_model(EMBEDDER_SPEC), prefer_npu)
        shape = self.rt.input["shape"]
        self.size = (int(shape[2]), int(shape[1]))
        out = max(self.rt.outputs(), key=lambda o: int(o["shape"][-1]))
        self._out = out["index"]
        self.dim = int(out["shape"][-1])

    @property
    def device(self) -> str:
        return self.rt.device

    def embed_rgb01(self, nhwc: np.ndarray) -> np.ndarray:
        """nhwc: float32 (1,224,224,3) RGB in [0,1]. Returns float32 (dim,) with unit L2 norm."""
        vec = self.rt.run(np.ascontiguousarray(nhwc, dtype=np.float32), [self._out])[0].reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("degenerate embedding")
        return vec / norm

    def embed_image(self, image) -> np.ndarray:
        """PIL RGB image (already cropped) -> embedding."""
        from PIL import Image

        crop = image.convert("RGB").resize(self.size, Image.BILINEAR)
        return self.embed_rgb01(np.asarray(crop, dtype=np.float32)[None] * (1.0 / 255.0))

    def info(self) -> dict:
        return {"model": self.spec.title, "file": self.rt.path.name, "license": self.spec.license, "device": self.rt.device,
                "runtime": self.rt.note, "init_ms": round(self.rt.init_ms, 1), "input": list(self.size), "dim": self.dim}


class DinoPoseEmbedder:
    """RGB crop -> unit-norm DINOv2 ViT-S/14 viewpoint descriptor (export_dinov2.py).

    The graph takes 224x224 RGB in [0,1] (ImageNet normalisation is baked in) and returns [CLS | mean patch token]
    (2x384). Each half is L2-normalised and the halves are averaged, so a cosine between two descriptors is the mean of
    the CLS and patch-token cosines. ~20 ms on the HTP, ~240 ms on the CPU."""

    embedder_id = DINO_ID

    def __init__(self, path: Path | None = None, prefer_npu: bool = True):
        self.spec = DINO_SPEC
        path = Path(path) if path else MODELS_DIR / DINO_SPEC.file
        if not _valid_tflite(path):
            raise FileNotFoundError(f"{path} missing - run export_dinov2.py on the board")
        self.rt = Runtime(path, prefer_npu)
        shape = self.rt.input["shape"]
        self.size = (int(shape[2]), int(shape[1]))
        out = self.rt.outputs()[0]
        self._out = out["index"]
        self.dim = int(out["shape"][-1])
        self.half = self.dim // 2

    @property
    def device(self) -> str:
        return self.rt.device

    def embed_rgb01(self, nhwc: np.ndarray) -> np.ndarray:
        raw = self.rt.run(np.ascontiguousarray(nhwc, dtype=np.float32), [self._out])[0].reshape(-1).astype(np.float32)
        parts = [raw[:self.half], raw[self.half:]]
        norms = [float(np.linalg.norm(p)) for p in parts]
        if not all(math.isfinite(n) and n > 0 for n in norms):
            raise ValueError("degenerate DINOv2 descriptor")
        return np.concatenate([p / n for p, n in zip(parts, norms)]) * np.float32(1.0 / math.sqrt(2.0))

    def info(self) -> dict:
        return {"model": self.spec.title, "file": self.rt.path.name, "license": self.spec.license, "device": self.rt.device,
                "runtime": self.rt.note, "init_ms": round(self.rt.init_ms, 1), "input": list(self.size), "dim": self.dim}


def make_pose_embedder(prefer_npu: bool = True) -> DinoPoseEmbedder | None:
    """DINOv2 pose descriptor on the HTP; None when the exported model is missing (pose falls back to MobileNet views)."""
    try:
        return DinoPoseEmbedder(prefer_npu=prefer_npu)
    except Exception as exc:  # missing model, missing runtime or delegate failure
        log.warning("DINOv2 pose embedder unavailable (%s) - pose uses the MobileNet view centroids", exc)
        return None


# --------------------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------------------


def generate_anchors(size: int, min_level: int = 3, max_level: int = 7, num_scales: int = 3,
                     anchor_scale: float = 3.0) -> np.ndarray:
    """EfficientDet anchors as (N, 4) [yc, xc, h, w], normalised; ordering matches TFLite_Detection_PostProcess."""
    ratios = ((1.0, 1.0), (1 / math.sqrt(2), math.sqrt(2)), (math.sqrt(2), 1 / math.sqrt(2)))
    rows = []
    for level in range(min_level, max_level + 1):
        feat = math.ceil(size / 2 ** level)
        stride = size / feat
        centers = (np.arange(feat, dtype=np.float64) + 0.5) / feat
        cy, cx = np.meshgrid(centers, centers, indexing="ij")
        per_cell = []
        for octave in range(num_scales):
            base = anchor_scale * stride * 2 ** (octave / num_scales) / size
            per_cell += [(base * rh, base * rw) for rh, rw in ratios]
        hw = np.array(per_cell)
        level_rows = np.empty((feat, feat, len(per_cell), 4))
        level_rows[..., 0] = cy[..., None]
        level_rows[..., 1] = cx[..., None]
        level_rows[..., 2] = hw[:, 0]
        level_rows[..., 3] = hw[:, 1]
        rows.append(level_rows.reshape(-1, 4))
    return np.concatenate(rows).astype(np.float32)


def decode_boxes(encodings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """SSD box decode (scale 1,1,1,1): (N,4) [ty,tx,th,tw] -> (N,4) [ymin,xmin,ymax,xmax], normalised."""
    yc = encodings[:, 0] * anchors[:, 2] + anchors[:, 0]
    xc = encodings[:, 1] * anchors[:, 3] + anchors[:, 1]
    h = np.exp(encodings[:, 2]) * anchors[:, 2]
    w = np.exp(encodings[:, 3]) * anchors[:, 3]
    return np.stack([yc - h / 2, xc - w / 2, yc + h / 2, xc + w / 2], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5, limit: int = 20) -> list[int]:
    """Greedy non-maximum suppression over [ymin,xmin,ymax,xmax] boxes; returns kept indices, best first."""
    order = np.argsort(-scores)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    keep: list[int] = []
    while order.size and len(keep) < limit:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        yy1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        xx1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        yy2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        xx2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(yy2 - yy1, 0, None) * np.clip(xx2 - xx1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou < iou_threshold]
    return keep


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (normalised to the frame)

    @property
    def category(self) -> str | None:
        return TARGET_CLASSES.get(self.label)

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    def kind(self, width: int | None = None, height: int | None = None) -> str:
        """Domain label shown to the user: COCO has no "can", so cup -> "beverage can" and can-shaped bottles too."""
        return domain_label(self.label, self.box, width, height)

    def to_dict(self, width: int | None = None, height: int | None = None) -> dict:
        return {"label": self.kind(width, height), "raw_label": self.label, "score": round(self.score, 3),
                "category": self.category, "box": [round(v, 4) for v in self.box]}


CAN_ASPECT = (1.4, 2.6)  # pixel H/W of an upright 330-500 ml can (incl. sleek cans) seen from the side


def domain_label(label: str, box=None, width: int | None = None, height: int | None = None) -> str:
    """Raw detector class -> domain vocabulary ("beverage can", "bottle", "smartphone"). A "bottle" box is renamed
    to "beverage can" only when its pixel aspect ratio is can-like; other classes pass through unchanged."""
    name = str(label).lower()
    if name in ("cup", "can", "soda can", "beverage can"):
        return "beverage can"
    if name in ("cell phone", "phone", "smartphone"):
        return "smartphone"
    if name == "bottle" and box is not None and width and height:
        bw = (box[2] - box[0]) * width
        bh = (box[3] - box[1]) * height
        if bw > 0 and CAN_ASPECT[0] <= bh / bw <= CAN_ASPECT[1]:
            return "beverage can"
    return label


class NpuDetector:
    """Full frame -> target-object detections (EfficientDet-Lite int8, COCO)."""

    def __init__(self, variant: str = DETECTOR_VARIANT, path: Path | None = None, prefer_npu: bool = True,
                 score_threshold: float = 0.30, iou_threshold: float = 0.5, all_classes: bool = False):
        self.spec = DETECTOR_SPECS[variant]
        self.variant = variant
        self.rt = Runtime(Path(path) if path else ensure_model(self.spec), prefer_npu)
        inp = self.rt.input
        self.size = int(inp["shape"][1])
        self.dtype = inp["dtype"]
        outs = {int(o["shape"][-1]): o["index"] for o in self.rt.outputs()}
        self._scores, self._boxes = outs[len(COCO_LABELS)], outs[4]
        self.anchors = generate_anchors(self.size)
        n = int(next(o for o in self.rt.outputs() if o["index"] == self._boxes)["shape"][1])
        if n != len(self.anchors):
            raise RuntimeError(f"{self.spec.file}: {n} boxes but {len(self.anchors)} anchors")
        self.score_threshold, self.iou_threshold = score_threshold, iou_threshold
        self.class_ids = np.arange(len(COCO_LABELS)) if all_classes else TARGET_IDS

    @property
    def device(self) -> str:
        return self.rt.device

    def detect_array(self, rgb: np.ndarray) -> list[Detection]:
        """rgb: uint8 (S,S,3) frame already resized to the model input."""
        x = rgb[None].astype(self.dtype, copy=False)
        scores, enc = self.rt.run(x, [self._scores, self._boxes])
        scores = scores[0][:, self.class_ids]
        if scores.max(initial=0.0) > 1.0 or scores.min(initial=0.0) < 0.0:
            scores = 1.0 / (1.0 + np.exp(-scores))
        best = scores.max(axis=1)
        idx = np.nonzero(best >= self.score_threshold)[0]
        if idx.size == 0:
            return []
        cls = self.class_ids[scores[idx].argmax(axis=1)]
        boxes = decode_boxes(enc[0][idx], self.anchors[idx])
        out = []
        for k in nms(boxes, best[idx], self.iou_threshold):
            ymin, xmin, ymax, xmax = (float(np.clip(v, 0.0, 1.0)) for v in boxes[k])
            if xmax - xmin > 1e-3 and ymax - ymin > 1e-3:
                out.append(Detection(COCO_LABELS[int(cls[k])], float(best[idx][k]), (xmin, ymin, xmax, ymax)))
        return out

    def detect(self, image) -> list[Detection]:
        """PIL image (any size) -> detections normalised to that image."""
        from PIL import Image

        frame = image if image.mode == "RGB" else image.convert("RGB")
        frame = frame.resize((self.size, self.size), Image.BILINEAR, reducing_gap=2.0)
        return self.detect_array(np.asarray(frame, dtype=np.uint8))

    def info(self) -> dict:
        return {"model": self.spec.title, "file": self.rt.path.name, "license": self.spec.license, "device": self.rt.device,
                "runtime": self.rt.note, "init_ms": round(self.rt.init_ms, 1), "input": [self.size, self.size],
                "anchors": int(len(self.anchors)), "classes": sorted(TARGET_CLASSES)}


# --------------------------------------------------------------------------------------
# YOLO detector (Ultralytics, exported to TFLite by export_yolo.sh)
# --------------------------------------------------------------------------------------


def yolo_spec(variant: str) -> ModelSpec:
    """Known YOLO variants, or any custom stem exported by export_yolo.sh (models/npu/<stem>_float32.tflite)."""
    return YOLO_SPECS.get(variant) or ModelSpec(f"{variant}_float32.tflite", "",
                                                f"Custom YOLO {variant} (TFLite FP32 -> HTP FP16)", "AGPL-3.0")


def detector_model_path(variant: str = DETECTOR_VARIANT, models_dir: Path | None = None) -> Path:
    spec = DETECTOR_SPECS.get(variant) or yolo_spec(variant)
    return Path(models_dir or MODELS_DIR) / spec.file


def load_yolo_names(stem: str, models_dir: Path | None = None) -> tuple[str, ...]:
    """Class names written by export_yolo.sh (<stem>.names.json); COCO-80 order otherwise."""
    try:
        names = json.loads((Path(models_dir or MODELS_DIR) / f"{stem}.names.json").read_text(encoding="utf-8"))["names"]
        if names:
            return tuple(str(n) for n in names)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return COCO80_LABELS


def letterbox(image, size: int, fill: int = 114) -> tuple[np.ndarray, float, int, int]:
    """PIL image -> uint8 (size, size, 3) letterboxed RGB plus the scale and left/top padding (Ultralytics style)."""
    from PIL import Image

    frame = image if image.mode == "RGB" else image.convert("RGB")
    w, h = frame.size
    r = min(size / w, size / h)
    nw, nh = max(1, min(size, round(w * r))), max(1, min(size, round(h * r)))
    if (nw, nh) != (w, h):
        frame = frame.resize((nw, nh), Image.BILINEAR, reducing_gap=2.0)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = np.asarray(frame, dtype=np.uint8)
    return canvas, r, left, top


def decode_yolo(raw: np.ndarray, class_ids: np.ndarray, score_threshold: float, iou_threshold: float = 0.5,
                size: int | None = None, max_candidates: int = 300, limit: int = 20) -> list[tuple[int, float, np.ndarray]]:
    """Raw YOLO head (1, 4+nc, N) or (1, N, 4+nc) with cx,cy,w,h boxes (pixels or normalised) and per-class scores
    -> [(class index, score, [xmin, ymin, xmax, ymax] normalised to the model input)], best first, after NMS."""
    out = raw[0] if raw.ndim == 3 else raw
    if out.shape[0] < out.shape[1]:
        out = out.T
    scores = out[:, 4:][:, class_ids]
    best = scores.max(axis=1)
    idx = np.nonzero(best >= score_threshold)[0]
    if idx.size == 0:
        return []
    if idx.size > max_candidates:
        idx = idx[np.argsort(-best[idx])[:max_candidates]]
    cls = class_ids[scores[idx].argmax(axis=1)]
    conf = best[idx]
    b = out[idx, :4].astype(np.float32)
    if size and float(np.abs(b).max()) > 2.0:
        b = b / float(size)
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
    return [(int(cls[k]), float(conf[k]), xyxy[k]) for k in nms(xyxy, conf, iou_threshold, limit)]


def unletterbox(box, size: int, scale: float, left: int, top: int, width: int, height: int) -> tuple[float, ...]:
    """[xmin, ymin, xmax, ymax] normalised to the letterboxed input -> normalised to the original frame."""
    x0, y0, x1, y1 = box
    return (float(np.clip((x0 * size - left) / (scale * width), 0.0, 1.0)),
            float(np.clip((y0 * size - top) / (scale * height), 0.0, 1.0)),
            float(np.clip((x1 * size - left) / (scale * width), 0.0, 1.0)),
            float(np.clip((y1 * size - top) / (scale * height), 0.0, 1.0)))


def _target_ids(names, all_classes: bool) -> np.ndarray:
    if all_classes:
        return np.arange(len(names), dtype=np.int64)
    return np.array([i for i, name in enumerate(names) if name in TARGET_CLASSES], dtype=np.int64)


class YoloDetector:
    """Full frame -> target-object detections with Ultralytics YOLO (TFLite FP32, run in FP16 on the HTP)."""

    def __init__(self, variant: str = DETECTOR_VARIANT, path: Path | None = None, prefer_npu: bool = True,
                 score_threshold: float = DETECT_SCORE, iou_threshold: float = 0.5, all_classes: bool = False):
        self.variant = variant
        self.spec = yolo_spec(variant)
        self.rt = Runtime(Path(path) if path else ensure_model(self.spec), prefer_npu)
        inp = self.rt.input
        self.size = int(inp["shape"][1])
        self.dtype = inp["dtype"]
        out = self.rt.outputs()[0]
        self._out = out["index"]
        shape = [int(v) for v in out["shape"]]
        num_classes = min(shape[1], shape[2]) - 4
        names = load_yolo_names(self.rt.path.name.split("_float32")[0], self.rt.path.parent)
        if len(names) != num_classes:
            names = COCO80_LABELS if num_classes == len(COCO80_LABELS) else tuple(f"class_{i}" for i in range(num_classes))
        self.names = names
        self.class_ids = _target_ids(names, all_classes)
        if self.class_ids.size == 0:
            raise RuntimeError(f"{self.rt.path.name}: no class maps to a phone or beverage ({names[:8]}...)")
        self.score_threshold, self.iou_threshold = score_threshold, iou_threshold

    @property
    def device(self) -> str:
        return self.rt.device

    def detect_array(self, rgb: np.ndarray) -> list[Detection]:
        """rgb: uint8 (S,S,3) letterboxed frame at the model input size. Boxes normalised to that input."""
        x = rgb[None].astype(np.float32) * (1.0 / 255.0) if np.issubdtype(self.dtype, np.floating) else rgb[None].astype(self.dtype)
        raw = self.rt.run(x, [self._out])[0]
        return [Detection(self.names[c], s, tuple(float(np.clip(v, 0.0, 1.0)) for v in box))
                for c, s, box in decode_yolo(raw, self.class_ids, self.score_threshold, self.iou_threshold, self.size)]

    def detect(self, image) -> list[Detection]:
        """PIL image (any size) -> detections normalised to that image."""
        width, height = image.size
        canvas, scale, left, top = letterbox(image, self.size)
        out = []
        for det in self.detect_array(canvas):
            box = unletterbox(det.box, self.size, scale, left, top, width, height)
            if box[2] - box[0] > 1e-3 and box[3] - box[1] > 1e-3:
                out.append(Detection(det.label, det.score, box))
        return out

    def info(self) -> dict:
        return {"model": self.spec.title, "file": self.rt.path.name, "license": self.spec.license, "device": self.rt.device,
                "runtime": self.rt.note, "init_ms": round(self.rt.init_ms, 1), "input": [self.size, self.size],
                "num_classes": len(self.names), "classes": [self.names[i] for i in self.class_ids]}


class UltralyticsDetector:
    """Ultralytics PyTorch YOLO on the Arm CPU: fallback when the exported .tflite is missing (~150-250 ms/frame)."""

    device = "cpu"

    def __init__(self, variant: str = DETECTOR_VARIANT, score_threshold: float = DETECT_SCORE, iou_threshold: float = 0.5,
                 size: int = 640, all_classes: bool = False):
        from ultralytics import YOLO

        t0 = time.perf_counter()
        self.variant, self.size = variant, size
        weights = Path(variant if variant.endswith(".pt") else MODELS_DIR / f"{variant}.pt")
        weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights))
        names = self.model.names
        self.names = tuple(names[k] for k in sorted(names)) if isinstance(names, dict) else tuple(names)
        self.class_ids = _target_ids(self.names, all_classes)
        if self.class_ids.size == 0:
            raise RuntimeError(f"{weights.name}: no class maps to a phone or beverage")
        self.score_threshold, self.iou_threshold = score_threshold, iou_threshold
        self.lock = threading.Lock()
        self.detect_array(np.zeros((size, size, 3), dtype=np.uint8))
        self.init_ms = (time.perf_counter() - t0) * 1000.0
        log.info("%s ready on the CPU (Ultralytics PyTorch) in %.0f ms", weights.name, self.init_ms)

    def _predict(self, source) -> list[Detection]:
        with self.lock:
            result = self.model.predict(source, imgsz=self.size, conf=self.score_threshold, iou=self.iou_threshold,
                                        classes=[int(i) for i in self.class_ids], device="cpu", verbose=False)[0]
        boxes = result.boxes
        return [Detection(self.names[int(c)], float(s), tuple(float(v) for v in b))
                for b, s, c in zip(boxes.xyxyn.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]

    def detect_array(self, rgb: np.ndarray) -> list[Detection]:
        return self._predict(np.ascontiguousarray(rgb[..., ::-1]))  # Ultralytics treats arrays as BGR

    def detect(self, image) -> list[Detection]:
        return self._predict(image if image.mode == "RGB" else image.convert("RGB"))

    def info(self) -> dict:
        return {"model": f"Ultralytics {self.variant} (PyTorch, COCO-80)", "file": f"{self.variant}.pt", "license": "AGPL-3.0",
                "device": "cpu", "runtime": "Ultralytics PyTorch CPU (run ./export_yolo.sh for the NPU)",
                "init_ms": round(self.init_ms, 1), "input": [self.size, self.size], "num_classes": len(self.names),
                "classes": [self.names[i] for i in self.class_ids]}


def make_detector(variant: str = DETECTOR_VARIANT, prefer_npu: bool = True):
    """YOLO (default: YOLO-World with the domain vocabulary) or EfficientDet-Lite. YOLO runs from
    models/npu/<variant>_float32.tflite on the HTP. A missing YOLO-World export falls back to YOLO26n (COCO, with the
    cup/bottle -> "beverage can" remap); any other missing export falls back to Ultralytics PyTorch on the CPU."""
    if variant in DETECTOR_SPECS:
        return NpuDetector(variant, prefer_npu=prefer_npu)
    if _valid_tflite(detector_model_path(variant)):
        return YoloDetector(variant, prefer_npu=prefer_npu)
    if variant == WORLD_VARIANT:
        log.warning("%s missing (run ./export_yolo.sh world); falling back to %s",
                    detector_model_path(variant).name, FALLBACK_DETECTOR)
        return make_detector(FALLBACK_DETECTOR, prefer_npu)
    log.warning("%s missing (run ./export_yolo.sh %s); using Ultralytics PyTorch on the CPU",
                detector_model_path(variant).name, variant)
    return UltralyticsDetector(variant)


# --------------------------------------------------------------------------------------
# Detector-guided crop
# --------------------------------------------------------------------------------------


def pick_target(detections: list[Detection], width: int, height: int, roi_fraction: float = 0.6,
                min_score: float = 0.35) -> Detection | None:
    """Best target-class detection whose center lies inside the central alignment box (roi_fraction of the short side)."""
    side = roi_fraction * min(width, height)
    x0, y0 = (width - side) / 2, (height - side) / 2
    best, best_key = None, -1.0
    for det in detections:
        if det.category is None or det.score < min_score:
            continue
        cx, cy = det.center[0] * width, det.center[1] * height
        if not (x0 <= cx <= x0 + side and y0 <= cy <= y0 + side):
            continue
        dist = math.hypot(cx - width / 2, cy - height / 2) / max(1.0, side)
        key = det.score - 0.25 * dist
        if key > best_key:
            best, best_key = det, key
    return best


def square_crop(det: Detection, width: int, height: int, margin: float = 0.15,
                min_fraction: float = 0.25) -> tuple[int, int, int]:
    """Square (size, left, top) around a detection with `margin` on each side, kept inside the frame."""
    bw = (det.box[2] - det.box[0]) * width
    bh = (det.box[3] - det.box[1]) * height
    short = min(width, height)
    size = int(min(short, max(min_fraction * short, max(bw, bh) * (1.0 + 2.0 * margin))))
    cx, cy = det.center[0] * width, det.center[1] * height
    left = int(round(min(max(cx - size / 2, 0), width - size)))
    top = int(round(min(max(cy - size / 2, 0), height - size)))
    return max(1, size), left, top


class CropTracker:
    """Temporal smoothing for the detector crop: EMA while tracked, short hold through misses, then release."""

    def __init__(self, alpha: float = 0.5, hold_frames: int = 4, jump: float = 0.5):
        self.alpha, self.hold_frames, self.jump = alpha, hold_frames, jump
        self._state: tuple[float, float, float] | None = None
        self._misses = 0

    def reset(self) -> None:
        self._state, self._misses = None, 0

    def update(self, box: tuple[int, int, int] | None, width: int, height: int) -> tuple[int, int, int] | None:
        if box is None:
            self._misses += 1
            if self._misses > self.hold_frames:
                self._state = None
        else:
            size, left, top = box
            new = (left + size / 2, top + size / 2, float(size))
            if self._state is None or math.hypot(new[0] - self._state[0], new[1] - self._state[1]) > self.jump * self._state[2]:
                self._state = new
            else:
                a = self.alpha
                self._state = tuple(a * n + (1 - a) * o for n, o in zip(new, self._state))
            self._misses = 0
        if self._state is None:
            return None
        cx, cy, size = self._state
        size = int(min(size, width, height))
        left = int(round(min(max(cx - size / 2, 0), width - size)))
        top = int(round(min(max(cy - size / 2, 0), height - size)))
        return size, left, top
