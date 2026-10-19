"""Minimal V4L2 control access (VIDIOC_QUERYCTRL / G_CTRL / S_CTRL) over ioctl - no v4l-utils dependency.

Used for the on-board camera's focus: continuous autofocus hunting blurs the reticle crop, so the dashboard can pin a
manual focal distance. Cameras without focus controls (the Logitech Brio 101 on the VENTUNO Q is fixed-focus) report
supported=False with the reason instead of pretending.
"""

from __future__ import annotations

import os
import struct
import threading
from dataclasses import dataclass
from typing import Callable

try:
    import fcntl
except ImportError:  # Windows dev hosts: no V4L2
    fcntl = None

QUERYCTRL_FMT = "II32siiiiI2I"  # struct v4l2_queryctrl (68 bytes)
CONTROL_FMT = "Ii"  # struct v4l2_control
CTRL_FLAG_DISABLED = 0x0001
CTRL_FLAG_INACTIVE = 0x0010
CTRL_FLAG_NEXT_CTRL = 0x80000000
CTRL_TYPE_INTEGER, CTRL_TYPE_BOOLEAN, CTRL_TYPE_MENU = 1, 2, 3

CID_FOCUS_ABSOLUTE = 0x009A090A
CID_FOCUS_AUTO = 0x009A090C  # "focus_automatic_continuous" (older drivers: "focus_auto")


def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


VIDIOC_QUERYCTRL = _iowr(36, struct.calcsize(QUERYCTRL_FMT))
VIDIOC_G_CTRL = _iowr(27, struct.calcsize(CONTROL_FMT))
VIDIOC_S_CTRL = _iowr(28, struct.calcsize(CONTROL_FMT))


@dataclass(frozen=True)
class Control:
    cid: int
    type: int
    name: str
    minimum: int
    maximum: int
    step: int
    default: int
    flags: int

    def clamp(self, value: int) -> int:
        value = max(self.minimum, min(self.maximum, int(value)))
        step = max(1, self.step)
        return self.minimum + round((value - self.minimum) / step) * step if step > 1 else value


class V4L2Controls:
    """Controls of one V4L2 device node. `ioctl` / `opener` are injectable so the logic is testable without hardware."""

    def __init__(self, device: str, ioctl: Callable | None = None, opener: Callable[[str], int] | None = None,
                 closer: Callable[[int], None] | None = None) -> None:
        self.device = device
        self._ioctl = ioctl or (fcntl.ioctl if fcntl is not None else None)
        self._open = opener or (lambda path: os.open(path, os.O_RDWR | os.O_NONBLOCK))
        self._close = closer or os.close
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._ioctl is not None

    def _with_fd(self, fn):
        if self._ioctl is None:
            raise OSError("V4L2 ioctls are unavailable on this host")
        with self._lock:
            fd = self._open(self.device)
            try:
                return fn(fd)
            finally:
                self._close(fd)

    def _query(self, fd: int, cid: int) -> Control | None:
        buf = bytearray(struct.pack(QUERYCTRL_FMT, cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
        try:
            self._ioctl(fd, VIDIOC_QUERYCTRL, buf)
        except OSError:
            return None
        i, t, name, mn, mx, st, df, fl, _r0, _r1 = struct.unpack(QUERYCTRL_FMT, buf)
        return Control(i, t, name.rstrip(b"\0").decode("ascii", "replace"), mn, mx, st, df, fl)

    def query(self, cid: int) -> Control | None:
        ctrl = self._with_fd(lambda fd: self._query(fd, cid))
        return None if ctrl is None or ctrl.flags & CTRL_FLAG_DISABLED else ctrl

    def list(self) -> list[Control]:
        def walk(fd: int) -> list[Control]:
            out, cid = [], CTRL_FLAG_NEXT_CTRL
            for _ in range(512):
                ctrl = self._query(fd, cid)
                if ctrl is None:
                    break
                if ctrl.type in (CTRL_TYPE_INTEGER, CTRL_TYPE_BOOLEAN, CTRL_TYPE_MENU) and not ctrl.flags & CTRL_FLAG_DISABLED:
                    out.append(ctrl)
                cid = ctrl.cid | CTRL_FLAG_NEXT_CTRL
            return out

        return self._with_fd(walk)

    def get(self, cid: int) -> int:
        def run(fd: int) -> int:
            buf = bytearray(struct.pack(CONTROL_FMT, cid, 0))
            self._ioctl(fd, VIDIOC_G_CTRL, buf)
            return struct.unpack(CONTROL_FMT, buf)[1]

        return self._with_fd(run)

    def set(self, cid: int, value: int) -> int:
        def run(fd: int) -> int:
            self._ioctl(fd, VIDIOC_S_CTRL, bytearray(struct.pack(CONTROL_FMT, cid, int(value))))
            buf = bytearray(struct.pack(CONTROL_FMT, cid, 0))
            self._ioctl(fd, VIDIOC_G_CTRL, buf)
            return struct.unpack(CONTROL_FMT, buf)[1]

        return self._with_fd(run)


class FocusControl:
    """Auto / manual focus of a UVC camera (continuous AF off + focus_absolute)."""

    def __init__(self, controls: V4L2Controls) -> None:
        self.ctl = controls
        self.error: str | None = None

    def _controls(self) -> tuple[Control | None, Control | None]:
        return self.ctl.query(CID_FOCUS_AUTO), self.ctl.query(CID_FOCUS_ABSOLUTE)

    def status(self) -> dict:
        out = {"device": self.ctl.device, "supported": False, "auto_supported": False, "auto": None, "value": None,
               "min": None, "max": None, "step": None, "default": None, "reason": None}
        if not self.ctl.available:
            return {**out, "reason": "V4L2 controls are unavailable on this host"}
        try:
            auto, absolute = self._controls()
            if auto is not None:
                out.update(auto_supported=True, auto=bool(self.ctl.get(CID_FOCUS_AUTO)))
            if absolute is not None:
                out.update(supported=True, value=self.ctl.get(CID_FOCUS_ABSOLUTE), min=absolute.minimum,
                           max=absolute.maximum, step=absolute.step, default=absolute.default)
            if absolute is None and auto is None:
                out["reason"] = "camera exposes no V4L2 focus controls (fixed-focus lens)"
        except OSError as exc:
            out["reason"] = f"cannot read {self.ctl.device}: {exc}"
        if self.error:
            out["error"] = self.error
        return out

    def set(self, auto: bool | None = None, value: int | None = None) -> dict:
        """auto=True: continuous autofocus. auto=False (+ value): manual focus pinned at `value` (clamped/snapped)."""
        auto_ctl, absolute = self._controls()
        if absolute is None and auto_ctl is None:
            raise LookupError("camera exposes no V4L2 focus controls (fixed-focus lens)")
        if value is not None and absolute is None:
            raise LookupError("camera has no manual focus (focus_absolute) control")
        if auto is not None and auto_ctl is None and auto:
            raise LookupError("camera has no autofocus control")
        if auto_ctl is not None and (auto is not None or value is not None):
            self.ctl.set(CID_FOCUS_AUTO, 1 if auto else 0)  # a manual value implies AF off
        if value is not None and not auto:
            self.ctl.set(CID_FOCUS_ABSOLUTE, absolute.clamp(value))
        self.error = None
        return self.status()


def parse_focus_env(value: str | None) -> tuple[bool | None, int | None]:
    """KAMIKAZE_FOCUS: "" (leave the camera alone) | "auto" | "<int>" manual focal distance."""
    value = (value or "").strip().lower()
    if not value:
        return None, None
    if value in ("auto", "af", "continuous"):
        return True, None
    try:
        return False, int(value)
    except ValueError:
        raise ValueError(f"KAMIKAZE_FOCUS must be 'auto' or an integer, got {value!r}") from None
