"""Hardware telemetry from /proc and /sys (no third-party deps). Every reader degrades to None/empty off-board."""

from __future__ import annotations

import glob
import os
import platform
import socket
import threading
import time
from pathlib import Path

ROUTER_SOCK = os.environ.get("ARDUINO_ROUTER_SOCK", "/var/run/arduino-router.sock")
TOTAL_RAM_GB_NOMINAL = 16
_lock = threading.Lock()
_prev_stat: dict[str, tuple[int, int]] = {}
_router_cache: tuple[float, dict] = (0.0, {})


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip().replace("\x00", "")
    except OSError:
        return ""


def _int(path: str) -> int | None:
    text = _read(path)
    try:
        return int(text)
    except ValueError:
        return None


def soc_identity() -> dict:
    cpuinfo = _read("/proc/cpuinfo")
    parts = sorted({line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.lower().startswith("cpu part")})
    model_name = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.lower().startswith("model name")), "")
    os_release = {}
    for line in _read("/etc/os-release").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os_release[k] = v.strip('"')
    board = _read("/proc/device-tree/model") or _read("/sys/devices/virtual/dmi/id/product_name") or platform.node()
    machine = _read("/sys/devices/soc0/machine")
    tag = f"{board} {machine}".lower()
    processor = "Qualcomm Dragonwing QCS8275" if ("8275" in tag or "ventuno" in tag) else (machine or platform.processor())
    return {
        "processor": processor,
        "board_model": board,
        "soc_family": _read("/sys/devices/soc0/family"),
        "soc_machine": _read("/sys/devices/soc0/machine"),
        "soc_id": _read("/sys/devices/soc0/soc_id"),
        "arch": platform.machine(),
        "cpu_model": model_name,
        "cpu_parts": parts,
        "cores": os.cpu_count() or 0,
        "kernel": platform.release(),
        "os": os_release.get("PRETTY_NAME", platform.system()),
        "python": platform.python_version(),
    }


def thermal() -> list[dict]:
    out = []
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*"), key=lambda p: int(p.rsplit("zone", 1)[1] or 0)):
        raw = _int(f"{zone}/temp")
        if raw is None:
            continue
        out.append({"zone": os.path.basename(zone), "type": _read(f"{zone}/type"), "temp_c": round(raw / 1000.0, 1)})
    return out


def cpu_freqs() -> list[dict]:
    out = []
    for cpu in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"), key=lambda p: int(p.rsplit("cpu", 1)[1])):
        cur = _int(f"{cpu}/cpufreq/scaling_cur_freq")
        out.append({
            "cpu": os.path.basename(cpu),
            "mhz": round(cur / 1000.0) if cur is not None else None,
            "max_mhz": round((_int(f"{cpu}/cpufreq/cpuinfo_max_freq") or 0) / 1000.0) or None,
            "governor": _read(f"{cpu}/cpufreq/scaling_governor") or None,
        })
    return out


def cpu_load() -> dict:
    """Per-core and total busy % since the previous call (first call reports since boot)."""
    cores: dict[str, float] = {}
    total = 0.0
    with _lock:
        for line in _read("/proc/stat").splitlines():
            if not line.startswith("cpu"):
                continue
            name, *vals = line.split()
            nums = [int(v) for v in vals[:8]]
            idle = nums[3] + nums[4]
            tot = sum(nums)
            p_idle, p_tot = _prev_stat.get(name, (0, 0))
            d_tot = tot - p_tot
            pct = round(100.0 * (1.0 - (idle - p_idle) / d_tot), 1) if d_tot > 0 else 0.0
            _prev_stat[name] = (idle, tot)
            if name == "cpu":
                total = pct
            else:
                cores[name] = pct
    load1 = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    return {"total_pct": total, "per_core_pct": cores, "loadavg_1m": round(load1, 2) if load1 is not None else None}


def memory() -> dict:
    info: dict[str, int] = {}
    for line in _read("/proc/meminfo").splitlines():
        key, _, rest = line.partition(":")
        try:
            info[key] = int(rest.split()[0])
        except (ValueError, IndexError):
            pass
    total = info.get("MemTotal", 0) / 1024.0
    avail = info.get("MemAvailable", info.get("MemFree", 0)) / 1024.0
    used = max(0.0, total - avail)
    return {
        "total_mb": round(total), "available_mb": round(avail), "used_mb": round(used),
        "used_pct": round(100.0 * used / total, 1) if total else None,
        "nominal_gb": TOTAL_RAM_GB_NOMINAL,
        "swap_used_mb": round(max(0, info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / 1024.0),
    }


def uptime_s() -> float | None:
    text = _read("/proc/uptime")
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return None


def router_socket(max_age_s: float = 2.0) -> dict:
    """State of the arduino-router IPC socket: exists / connectable / connect latency (cached for max_age_s)."""
    global _router_cache
    now = time.monotonic()
    with _lock:
        if now - _router_cache[0] < max_age_s and _router_cache[1]:
            return dict(_router_cache[1])
    state = {"path": ROUTER_SOCK, "exists": os.path.exists(ROUTER_SOCK), "connectable": False, "connect_ms": None}
    if state["exists"] and hasattr(socket, "AF_UNIX"):
        t0 = time.perf_counter()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(ROUTER_SOCK)
            state["connectable"] = True
            state["connect_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        except OSError as exc:
            state["error"] = str(exc)
    with _lock:
        _router_cache = (time.monotonic(), state)
    return dict(state)


def serial_bridge() -> dict:
    return {"device": "/dev/ttyACM0", "present": os.path.exists("/dev/ttyACM0")}


def i2c_buses() -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob("/dev/i2c-*"))


def npu_status(zones: list[dict] | None = None) -> dict:
    """Hexagon NPU (cDSP) state: FastRPC device node, remoteproc state and NSP junction temperatures."""
    cdsp = None
    for rp in sorted(glob.glob("/sys/class/remoteproc/remoteproc*")):
        ident = " ".join((_read(f"{rp}/name"), _read(f"{rp}/firmware"), _read(f"{rp}/device/of_node/compatible")))
        if "cdsp" in ident.lower():
            cdsp = {"remoteproc": os.path.basename(rp), "name": _read(f"{rp}/name"),
                    "firmware": _read(f"{rp}/firmware"), "state": _read(f"{rp}/state")}
            break
    nsp = [z for z in (zones if zones is not None else thermal()) if z["type"].lower().startswith("nsp")]
    return {
        "fastrpc_cdsp": os.path.exists("/dev/fastrpc-cdsp"),
        "cdsp": cdsp,
        "nsp_temp_c": max((z["temp_c"] for z in nsp), default=None),
        "nsp_zones": len(nsp),
    }


def hardware() -> dict:
    """Everything that does not depend on the pipeline (SoC, thermals, CPU, RAM, IPC, NPU)."""
    zones = thermal()
    return {
        "soc": soc_identity(), "thermal": zones, "cpu_freq": cpu_freqs(), "cpu_load": cpu_load(),
        "memory": memory(), "uptime_s": uptime_s(), "router": router_socket(), "serial": serial_bridge(),
        "linux_i2c_buses": i2c_buses(), "npu": npu_status(zones),
    }


def percentile(values, pct: float) -> float | None:
    data = sorted(values)
    if not data:
        return None
    idx = min(len(data) - 1, max(0, int(round(pct / 100.0 * (len(data) - 1)))))
    return round(float(data[idx]), 2)
