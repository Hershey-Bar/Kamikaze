"""On-device OCR (RapidOCR / ONNX Runtime, CPU) fused with the visual embedding score.

The OCR engine is far slower than the NPU path (~160-175 ms per 320 px crop on the QCS8275's Kryo cores), so it runs in
its own latest-frame-wins worker thread at OCR_HZ and never blocks a video frame. Recognised words from the last
WINDOW_S seconds are matched against each candidate class's brand / flavour / model words:

    S_fused = 0.65 * S_vision + 0.35 * S_text

  S_text = 1.0  brand and detail word both read  (e.g. "coca" + "zero" for a Coke Zero class)
         = 0.6  brand only (class has a detail word that was not read) / 1.0 when the class has no detail words
         = 0.4  detail word only (no brand word read)
         = 0.0  a *different* known brand was read and none of this class's brand words were (conflict)
  When the recent text says nothing about a class (no brand, detail or conflict word) the vision score is kept as is,
  so unrelated print (nutrition facts, barcodes) never moves the verdict.

No frame or crop is ever written to disk: the worker holds one RGB array in RAM and drops it after recognition.
"""

from __future__ import annotations

import collections
import logging
import os
import re
import threading
import time

import numpy as np

log = logging.getLogger("kamikaze.ocr")

W_VISION, W_TEXT = 0.65, 0.35
OCR_HZ = float(os.environ.get("KAMIKAZE_OCR_HZ", "3"))
WINDOW_S = float(os.environ.get("KAMIKAZE_OCR_WINDOW", "1.5"))
MIN_SCORE = float(os.environ.get("KAMIKAZE_OCR_MIN_SCORE", "0.5"))
OCR_SIDE = int(os.environ.get("KAMIKAZE_OCR_SIDE", "320"))
OCR_THREADS = int(os.environ.get("KAMIKAZE_OCR_THREADS", "4"))

# brand -> words that identify it in print (lower case, alphanumerics only)
BRAND_WORDS: dict[str, set[str]] = {
    "coca-cola": {"coca", "cola", "cocacola", "coke"},
    "pepsi": {"pepsi"},
    "sprite": {"sprite"},
    "fanta": {"fanta"},
    "red bull": {"redbull", "red", "bull"},
    "monster": {"monster"},
    "dr pepper": {"pepper", "drpepper"},
    "mountain dew": {"dew", "mtn"},
    "samsung": {"samsung", "galaxy"},
    "apple": {"iphone", "apple"},
    "google": {"pixel", "google"},
    "kirkland": {"kirkland"},
    "lacroix": {"lacroix", "croix"},
}
# words too generic to count as a brand conflict on their own
_WEAK = {"red", "cola", "apple", "google", "pepper", "galaxy"}
_STOP = {"can", "soda", "drink", "beverage", "bottle", "phone", "smartphone", "original", "classic", "regular",
         "standard", "the", "and", "with", "ml", "oz", "fl", "black", "white", "blue", "silver", "gold", "green"}


def tokens_of(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    out = set(words)
    out.update(a + b for a, b in zip(words, words[1:]))  # "coca cola" -> "cocacola", "red bull" -> "redbull"
    return out


def _brand_key(brand: str) -> str | None:
    b = (brand or "").lower().replace("-", " ").strip()
    for key in BRAND_WORDS:
        if key.replace("-", " ") == b or b.replace(" ", "") == key.replace("-", "").replace(" ", ""):
            return key
    return None


def class_words(meta: dict | None, label: str = "") -> dict:
    """{"brand": set, "detail": set} - printable words that identify a class (from its enrollment metadata)."""
    meta = meta or {}
    key = _brand_key(str(meta.get("brand", "")))
    brand = set(BRAND_WORDS.get(key, set())) if key else set(tokens_of(str(meta.get("brand", "")))) - _STOP
    detail: set[str] = set()
    for field in ("variant", "flavor", "color"):
        detail |= {w for w in re.findall(r"[a-z0-9]+", str(meta.get(field, "")).lower()) if len(w) >= 3}
    raw = re.findall(r"[a-z0-9]+", str(meta.get("raw_name") or label.replace("_", " ")).lower())
    detail |= {w for w in raw if len(w) >= 3 and not w.isdigit()}
    detail -= brand | _STOP | {"original", "regular", "standard", "plus"}
    detail = {w for w in detail if not re.fullmatch(r"\d+(ml|l|oz|cl)", w)}  # sizes are printed on every brand
    return {"brand": brand, "detail": detail, "brand_key": key}


def text_score(words: dict, seen: set[str]) -> float | None:
    """S_text for one class given the recently read tokens; None when the text is uninformative for this class."""
    if not seen:
        return None
    brand_hit = bool(words["brand"] & seen)
    detail_hit = bool(words["detail"] & seen)
    if brand_hit:
        return 1.0 if detail_hit or not words["detail"] else 0.6
    other = set()
    for key, ws in BRAND_WORDS.items():
        if key != words.get("brand_key"):
            other |= (ws - _WEAK) & seen
    if other:
        return 0.0
    if detail_hit:
        return 0.4
    return None


def fuse(s_vision: float, s_text: float | None) -> float:
    return s_vision if s_text is None else W_VISION * s_vision + W_TEXT * s_text


class OcrWorker:
    """Latest-crop-wins background OCR. submit() never blocks; recent() returns the words read in the last WINDOW_S."""

    def __init__(self, engine=None, hz: float = OCR_HZ) -> None:
        self.engine = engine
        self.period = 1.0 / max(0.1, hz)
        self._cv = threading.Condition()
        self._pending: np.ndarray | None = None
        self._reads: collections.deque = collections.deque(maxlen=64)  # (mono_time, text, score)
        self._lat: collections.deque = collections.deque(maxlen=120)
        self._runs = 0
        self._last_run = 0.0
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        if engine is not None:
            self._thread = threading.Thread(target=self._loop, name="ocr-worker", daemon=True)
            self._thread.start()

    @classmethod
    def create(cls) -> "OcrWorker":
        """Load RapidOCR (ONNX Runtime CPU). Returns a disabled worker when the package is missing or OCR is off."""
        if os.environ.get("KAMIKAZE_OCR", "1") == "0":
            worker = cls(None)
            worker.error = "disabled (KAMIKAZE_OCR=0)"
            return worker
        try:
            from rapidocr_onnxruntime import RapidOCR

            engine = RapidOCR(det_limit_side_len=OCR_SIDE, intra_op_num_threads=OCR_THREADS,
                              inter_op_num_threads=1)
            log.info("OCR ready: RapidOCR (ONNX Runtime CPU, %d threads, %d px, %.1f Hz)", OCR_THREADS, OCR_SIDE, OCR_HZ)
            return cls(engine)
        except Exception as exc:  # optional dependency: the pipeline runs vision-only without it
            worker = cls(None)
            worker.error = f"unavailable: {exc}"
            log.warning("OCR %s", worker.error)
            return worker

    @property
    def enabled(self) -> bool:
        return self.engine is not None

    def submit(self, crop_rgb: np.ndarray) -> None:
        if self.engine is None:
            return
        with self._cv:
            self._pending = crop_rgb
            self._cv.notify()

    def _loop(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._pending is not None)
                wait = self.period - (time.monotonic() - self._last_run)
            if wait > 0:
                time.sleep(wait)
            with self._cv:
                crop, self._pending = self._pending, None
            if crop is None:
                continue
            self._last_run = time.monotonic()
            try:
                self._recognise(crop)
            except Exception as exc:
                self.error = str(exc)
                log.exception("OCR failed")
            finally:
                del crop

    def _recognise(self, crop: np.ndarray) -> None:
        h, w = crop.shape[:2]
        if max(h, w) > OCR_SIDE:
            k = OCR_SIDE / max(h, w)
            ys = (np.arange(int(h * k)) / k).astype(int)
            xs = (np.arange(int(w * k)) / k).astype(int)
            crop = crop[ys][:, xs]
        bgr = np.ascontiguousarray(crop[:, :, ::-1])
        t0 = time.perf_counter()
        result, _ = self.engine(bgr, use_cls=False)
        ms = (time.perf_counter() - t0) * 1000.0
        now = time.monotonic()
        with self._cv:
            self._lat.append(ms)
            self._runs += 1
            for item in result or []:
                text, score = str(item[1]), float(item[2])
                if score >= MIN_SCORE and text.strip():
                    self._reads.append((now, text.strip(), score))

    def recent(self, window_s: float = WINDOW_S) -> dict:
        now = time.monotonic()
        with self._cv:
            reads = [(t, s, c) for t, s, c in self._reads if now - t <= window_s]
            lat = sorted(self._lat)
            runs = self._runs
        texts: list[str] = []
        for _, text, _ in sorted(reads, key=lambda r: -r[2]):
            if text not in texts:
                texts.append(text)
        seen: set[str] = set()
        for text in texts:
            seen |= tokens_of(text)
        return {
            "enabled": self.enabled, "texts": texts[:8], "tokens": sorted(seen), "runs": runs,
            "ms": round(lat[len(lat) // 2], 1) if lat else None, "hz": round(1.0 / self.period, 1),
            "error": self.error, "engine": "RapidOCR / ONNX Runtime (CPU)" if self.enabled else None,
        }
