"""On-board V4L2 capture for the VENTUNO Q (Logitech Brio on the USB 3.0 Type-A port).

The camera is opened with OpenCV's V4L2 backend in MJPG mode (native 1920x1080 @ 30 fps). Colour conversion is disabled
so each read returns the camera's own JPEG bitstream: it is handed to the receiver unchanged (no decode / re-encode on
the capture thread), the receiver decodes it once for inference and serves the same bytes on the MJPEG preview. Frames
live only in RAM - nothing is written to disk. If the backend ignores the passthrough request, decoded BGR frames are
re-encoded to JPEG in memory instead.

Device selection: "auto" picks the first USB UVC capture node (/sys/class/video4linux/videoN with index 0 and a USB
parent); "none" / "off" disables on-board capture; "0" means /dev/video0; anything else is used as a path.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

import v4l2_ctrl
try:
    import cv2
except ImportError:  # pragma: no cover - the board venv ships opencv-python-headless
    cv2 = None

log = logging.getLogger("kamikaze.camera")

JPEG_SOI = b"\xff\xd8"
SYSFS_V4L = Path("/sys/class/video4linux")
REOPEN_AFTER_FAILS = 30  # consecutive failed reads (~1 s) before the device is reopened
BACKOFF_MAX_S = 5.0


def find_camera(sysfs: Path = SYSFS_V4L) -> str | None:
    """First USB UVC capture node (index 0; index 1 is the UVC metadata node). Platform ISP nodes are skipped."""
    if not sysfs.is_dir():
        return None
    nodes = sorted(sysfs.glob("video*"), key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else 1 << 30)
    for node in nodes:
        try:
            index = (node / "index").read_text().strip()
            parent = str((node / "device").resolve())
        except OSError:
            continue
        if index == "0" and "/usb" in parent:
            return f"/dev/{node.name}"
    return None


def device_name(device: str, sysfs: Path = SYSFS_V4L) -> str | None:
    try:
        return (sysfs / Path(device).name / "name").read_text().strip() or None
    except OSError:
        return None


def resolve_device(spec: str | None, sysfs: Path = SYSFS_V4L) -> str | None:
    spec = (spec or "").strip()
    if spec.lower() in ("", "none", "off", "false", "http"):
        return None
    if spec.lower() == "auto":
        return find_camera(sysfs)
    if spec.isdigit():
        return f"/dev/video{spec}"
    return spec


def _default_opener(device: str):
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed (pip install opencv-python-headless)")
    return cv2.VideoCapture(device, cv2.CAP_V4L2)


class LocalBrioCapture:
    """Non-blocking capture thread. `on_frame(jpeg_bytes)` is called for every frame (latest-wins downstream)."""

    def __init__(self, device_path: str = "/dev/video0", width: int = 1920, height: int = 1080, fps: int = 30,
                 on_frame: Callable[[bytes], object] | None = None, opener: Callable[[str], object] | None = None,
                 jpeg_quality: int = 85) -> None:
        self.device = device_path
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.on_frame = on_frame
        self._opener = opener or _default_opener
        self._quality = int(jpeg_quality)
        self.cap = None
        self.actual_w = self.actual_h = 0
        self.actual_fps = 0.0
        self.fourcc = ""
        self.passthrough: bool | None = None
        self.name = device_name(device_path)
        self.status = "stopped"
        self.error: str | None = None
        self.frames = 0
        self.read_errors = 0
        self.callback_errors = 0
        self.reconnects = 0
        self.capture_fps = 0.0
        self.last_bytes = 0
        self._last_mono = 0.0
        self._latest: bytes | None = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.focus = v4l2_ctrl.FocusControl(v4l2_ctrl.V4L2Controls(device_path))
        self.focus_env = os.environ.get("KAMIKAZE_FOCUS", "")

    # -- device ---------------------------------------------------------------------------------------------------
    def open(self) -> bool:
        self.release()
        try:
            cap = self._opener(self.device)
        except Exception as exc:
            self.status, self.error = "unavailable", f"{type(exc).__name__}: {exc}"
            return False
        if cap is None or not cap.isOpened():
            self.status, self.error = "unavailable", f"cannot open {self.device}"
            if cap is not None:
                cap.release()
            return False
        if cv2 is not None:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            # No CAP_PROP_BUFFERSIZE=1: a single V4L2 buffer makes the Brio skip every other frame (15 fps measured).
            # The capture thread drains the default queue continuously and downstream is latest-wins, so no lag builds.
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # raw MJPG bitstream passthrough
            self.actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.fourcc = code.to_bytes(4, "little").decode("ascii", "replace").strip("\x00") if code > 0 else ""
        else:
            self.actual_w, self.actual_h, self.actual_fps, self.fourcc = self.width, self.height, float(self.fps), "MJPG"
        self.cap = cap
        self.status, self.error = "open", None
        log.info("[ONBOARD-CAM] Opened %s (%s): %dx%d %s @ %.0f FPS", self.device, self.name or "?", self.actual_w,
                 self.actual_h, self.fourcc or "?", self.actual_fps)
        self._apply_focus_env()
        return True

    def _apply_focus_env(self) -> None:
        """KAMIKAZE_FOCUS=auto|<0-255>: pin the focus on every (re)open; fixed-focus cameras just log it."""
        try:
            auto, value = v4l2_ctrl.parse_focus_env(self.focus_env)
            if auto is None:
                return
            st = self.focus.set(auto=auto, value=value)
            log.info("[ONBOARD-CAM] Focus: %s", "continuous AF" if st["auto"] else f"manual {st['value']}")
        except (LookupError, ValueError, OSError) as exc:
            log.warning("[ONBOARD-CAM] KAMIKAZE_FOCUS=%s not applied: %s", self.focus_env, exc)

    def release(self) -> None:
        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    # -- frames ---------------------------------------------------------------------------------------------------
    def _to_jpeg(self, frame) -> bytes | None:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.dtype == np.uint8 and (arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] == 1)):
            data = arr.tobytes()
            if data.startswith(JPEG_SOI):
                self.passthrough = True
                return data
            return None
        if arr.ndim == 3 and cv2 is not None:
            self.passthrough = False
            ok, enc = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            return enc.tobytes() if ok else None
        return None

    def _publish(self, data: bytes) -> None:
        now = time.monotonic()
        with self.lock:
            gap = now - self._last_mono if self._last_mono else None
            if gap is not None and gap < 2.0:
                inst = 1.0 / max(1e-3, gap)
                self.capture_fps = inst if self.capture_fps == 0 else 0.9 * self.capture_fps + 0.1 * inst
            else:
                self.capture_fps = 0.0
            self._last_mono = now
            self._latest = data
            self.frames += 1
            self.last_bytes = len(data)
            self.status = "streaming"
        if self.on_frame is not None:
            try:
                self.on_frame(data)
            except Exception as exc:
                self.callback_errors += 1
                if self.callback_errors <= 3 or self.callback_errors % 500 == 0:
                    log.warning("[ONBOARD-CAM] frame handler failed (%d): %s", self.callback_errors, exc)

    def _capture_loop(self) -> None:
        fails, backoff = 0, 0.5
        while self.running:
            if self.cap is None:
                if not self.open():
                    log.warning("[ONBOARD-CAM] %s; retrying in %.1fs", self.error, backoff)
                    self._sleep(backoff)
                    backoff = min(BACKOFF_MAX_S, backoff * 2)
                    continue
                backoff, fails = 0.5, 0
            try:
                ok, frame = self.cap.read()
            except Exception as exc:
                ok, frame = False, None
                self.error = f"{type(exc).__name__}: {exc}"
            data = self._to_jpeg(frame) if ok else None
            if data is None:
                self.read_errors += 1
                fails += 1
                if fails >= REOPEN_AFTER_FAILS:
                    self.status, self.error = "reconnecting", self.error or f"{fails} failed reads from {self.device}"
                    log.warning("[ONBOARD-CAM] %s - reopening", self.error)
                    self.release()
                    self.reconnects += 1
                    fails = 0
                else:
                    time.sleep(0.005)
                continue
            fails = 0
            self._publish(data)
        self.release()
        self.status = "stopped"

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while self.running and time.monotonic() < end:
            time.sleep(0.05)

    # -- control --------------------------------------------------------------------------------------------------
    def start(self) -> "LocalBrioCapture":
        if self.running:
            return self
        if self.cap is None:
            self.open()  # first open is synchronous so the startup log shows the negotiated mode
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, name="onboard-camera", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.release()
        self.status = "stopped"

    def latest_jpeg(self) -> bytes | None:
        with self.lock:
            return self._latest

    def read_latest(self):
        """Decoded copy (BGR ndarray) of the newest frame, or None."""
        data = self.latest_jpeg()
        if data is None or cv2 is None:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    @property
    def live(self) -> bool:
        return self.running and self._last_mono > 0 and time.monotonic() - self._last_mono < 2.0

    def info(self) -> dict:
        age = time.monotonic() - self._last_mono if self._last_mono else None
        return {
            "source": "onboard-v4l2", "device": self.device, "name": self.name, "status": self.status, "error": self.error,
            "live": self.live, "requested": {"width": self.width, "height": self.height, "fps": self.fps, "fourcc": "MJPG"},
            "width": self.actual_w, "height": self.actual_h, "negotiated_fps": round(self.actual_fps, 1),
            "fourcc": self.fourcc, "mjpeg_passthrough": self.passthrough,
            "capture_fps": round(self.capture_fps, 1) if self.live else 0.0, "frames": self.frames,
            "read_errors": self.read_errors, "reconnects": self.reconnects, "frame_bytes": self.last_bytes,
            "frame_age_s": round(age, 2) if age is not None else None, "persisted_to_disk": False,
        }
