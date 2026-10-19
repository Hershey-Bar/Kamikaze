"""Process-wide outbound network gate for the dashboard's strict Offline mode.

Every external HTTP call in the CAD path (Sketchfab / Poly Pizza search + download, spec crawlers, model-viewer
vendoring) calls check(url) first. While offline, any non-loopback destination raises NetworkBlocked - a URLError
subclass, so the callers' existing network-failure handling applies unchanged - and the attempt is counted, which
makes "zero outbound requests" verifiable from /api/settings/mode and the tests.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import time
import urllib.error
import urllib.parse

log = logging.getLogger("kamikaze.netguard")

_lock = threading.Lock()
_offline = False
_blocked = 0
_allowed = 0
_last_blocked: dict | None = None


class NetworkBlocked(urllib.error.URLError):
    def __init__(self, url: str, host: str) -> None:
        super().__init__(f"offline mode: outbound request to {host or url} blocked")
        self.url = url
        self.host = host


def set_offline(offline: bool) -> None:
    global _offline
    with _lock:
        changed = _offline != bool(offline)
        _offline = bool(offline)
    if changed:
        log.info("Network gate: %s", "OFFLINE (external requests blocked)" if offline else "online")


def is_offline() -> bool:
    return _offline


def is_local(url: str) -> bool:
    """True for loopback destinations (localhost, 127.0.0.0/8, ::1) - the receiver's own API stays reachable."""
    host = (urllib.parse.urlsplit(url).hostname or "").strip("[]").lower()
    if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check(url: str) -> None:
    """Raise NetworkBlocked when offline and `url` leaves the device."""
    global _blocked, _allowed, _last_blocked
    local = is_local(url)
    with _lock:
        if _offline and not local:
            _blocked += 1
            host = urllib.parse.urlsplit(url).hostname or ""
            _last_blocked = {"host": host, "at": time.time()}
        else:
            _allowed += 0 if local else 1
            return
    log.warning("Offline mode blocked an outbound request to %s", host)
    raise NetworkBlocked(url, host)


def stats() -> dict:
    with _lock:
        return {"offline": _offline, "blocked": _blocked, "external_allowed": _allowed,
                "last_blocked": dict(_last_blocked) if _last_blocked else None}
