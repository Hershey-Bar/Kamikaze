"""SMART kiosk mode: a 5-second autonomous scan that turns a stream of per-frame verdicts into one locked identity.

    idle --(object detected, detector score >= DETECT_MIN)--> scanning --(WINDOW_S elapsed)--> locked | inconclusive
    scanning --(object gone for LOST_S)--> idle          locked --(object gone for RELEASE_S)--> idle
    inconclusive --(HOLD_S)--> idle (re-arms on the next detection)

While scanning, every frame votes for its top label with its fused calibrated confidence (vision + OCR + colour); OCR
tokens (cans) and the sharpest Laplacian per viewpoint (phones) are accumulated as numbers / words only - no frame or
crop is kept. At the end of the window the label with the largest vote share locks if its share >= MIN_SHARE and its mean
confidence >= LOCK_CONF; otherwise the scan is inconclusive and the guidance asks for a better view.

The receiver maps the state to the Modulino Pixels: amber progress 1..8 while scanning, solid green x8 when locked.
"""

from __future__ import annotations

import collections
import math
import os
import time
from typing import Callable, Iterable

WINDOW_S = float(os.environ.get("KAMIKAZE_SMART_WINDOW", "5.0"))
DETECT_MIN = float(os.environ.get("KAMIKAZE_SMART_DETECT", "0.65"))
LOCK_CONF = float(os.environ.get("KAMIKAZE_SMART_LOCK", "0.50"))
MIN_SHARE = float(os.environ.get("KAMIKAZE_SMART_SHARE", "0.60"))
LOST_S = float(os.environ.get("KAMIKAZE_SMART_LOST", "1.0"))
RELEASE_S = float(os.environ.get("KAMIKAZE_SMART_RELEASE", "2.0"))
HOLD_S = float(os.environ.get("KAMIKAZE_SMART_HOLD", "2.0"))
MIN_FRAMES = 3
SWAP_FRAMES = int(os.environ.get("KAMIKAZE_SMART_SWAP", "15"))

IDLE, SCANNING, LOCKED, INCONCLUSIVE = "idle", "scanning", "locked", "inconclusive"


class SmartScan:
    def __init__(self, window_s: float = WINDOW_S, detect_min: float = DETECT_MIN, lock_conf: float = LOCK_CONF,
                 min_share: float = MIN_SHARE, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s, self.detect_min, self.lock_conf, self.min_share = window_s, detect_min, lock_conf, min_share
        self.clock = clock
        self.scans = 0
        self._reset(IDLE)

    def _reset(self, status: str, message: str = "") -> None:
        self.status = status
        self.started = 0.0
        self.last_seen = 0.0
        self.changed = self.clock()
        self.votes: dict[str, float] = collections.defaultdict(float)
        self.counts: dict[str, int] = collections.defaultdict(int)
        self.categories: dict[str, str] = {}
        self.tokens: collections.Counter = collections.Counter()
        self.views: dict[str, float] = {}
        self.frames = 0
        self.label: str | None = None
        self.confidence = 0.0
        self.share = 0.0
        self.message = message
        self._other = 0

    def reset(self) -> None:
        self._reset(IDLE)

    @property
    def active(self) -> bool:
        return self.status in (SCANNING, LOCKED)

    def progress(self, now: float | None = None) -> float:
        if self.status == LOCKED:
            return 1.0
        if self.status != SCANNING:
            return 0.0
        now = self.clock() if now is None else now
        return max(0.0, min(1.0, (now - self.started) / max(1e-3, self.window_s)))

    def leader(self) -> tuple[str | None, float, float]:
        """(label with the largest vote, its share of all votes, its mean confidence)."""
        total = sum(self.votes.values())
        if not total:
            return None, 0.0, 0.0
        lab = max(self.votes, key=self.votes.get)
        return lab, self.votes[lab] / total, self.votes[lab] / max(1, self.counts[lab])

    def update(self, *, det_score: float | None, label: str | None, confidence: float, category: str | None = None,
               tokens: Iterable[str] = (), sharpness: float | None = None, view: str | None = None,
               present_min: float = 0.35) -> str | None:
        """Feed one frame. det_score: detector score of the target inside the reticle (None = nothing detected).
        Returns an event: "started" | "locked" | "inconclusive" | "lost" | "released" | None."""
        now = self.clock()
        strong = det_score is not None and det_score >= self.detect_min
        present = (det_score is not None and det_score >= present_min) or (label is not None and confidence >= 0.25)
        if present:
            self.last_seen = now
        if self.status == IDLE:
            if strong:
                self._reset(SCANNING)
                self.started = self.last_seen = now
                self.scans += 1
                self._accumulate(label, confidence, category, tokens, sharpness, view)
                return "started"
            return None
        if self.status == INCONCLUSIVE:
            if now - self.changed >= HOLD_S:
                self._reset(IDLE)
            return None
        if self.status == LOCKED:
            # a different object held up without leaving the reticle: N confident frames of another label re-arm
            self._other = self._other + 1 if label and label != self.label and confidence >= self.detect_min else 0
            if now - self.last_seen >= RELEASE_S or self._other >= SWAP_FRAMES:
                self._reset(IDLE)
                return "released"
            return None
        # scanning
        if now - self.last_seen >= LOST_S:
            self._reset(IDLE, "Object left the reticle - scan cancelled")
            return "lost"
        if present:
            self._accumulate(label, confidence, category, tokens, sharpness, view)
        self.label, self.share, self.confidence = self.leader()
        if now - self.started < self.window_s:
            return None
        return self._finalize(now)

    def _accumulate(self, label, confidence, category, tokens, sharpness, view) -> None:
        self.frames += 1
        if label:
            self.votes[label] += max(0.0, float(confidence))
            self.counts[label] += 1
            if category:
                self.categories[label] = category
        self.tokens.update(set(tokens or ()))
        if view and sharpness is not None and math.isfinite(sharpness):
            self.views[view] = max(self.views.get(view, 0.0), float(sharpness))

    def _finalize(self, now: float) -> str:
        lab, share, conf = self.leader()
        votes, counts, tokens, views, frames = dict(self.votes), dict(self.counts), self.tokens, self.views, self.frames
        categories = dict(self.categories)
        ok = lab is not None and counts.get(lab, 0) >= MIN_FRAMES and share >= self.min_share and conf >= self.lock_conf
        status = LOCKED if ok else INCONCLUSIVE
        msg = (f"Locked: {lab} ({conf:.0%} mean confidence, {share:.0%} of votes)" if ok else
               "Scan inconclusive - hold the object steady and show its branding / camera island" if lab else
               "Scan inconclusive - show the brand / model name and rotate for more evidence")
        self._reset(status, msg)
        self.votes.update(votes)
        self.counts.update(counts)
        self.categories, self.tokens, self.views, self.frames = categories, tokens, views, frames
        self.label, self.share, self.confidence = lab, share, conf
        self.started, self.last_seen, self.changed = now - self.window_s, now, now
        return "locked" if ok else "inconclusive"

    def led(self) -> tuple[int, int, int, int, int] | None:
        """Modulino Pixels for the current state (None: let the regular state machine drive the LEDs)."""
        if self.status == SCANNING:
            return (255, 120, 0, 40, max(1, min(8, math.ceil(8 * self.progress()))))
        if self.status == LOCKED:
            return (0, 255, 0, 50, 8)
        if self.status == INCONCLUSIVE:
            return (255, 120, 0, 40, 4)
        return None

    def snapshot(self) -> dict:
        now = self.clock()
        top = sorted(self.votes.items(), key=lambda kv: -kv[1])[:3]
        total = sum(self.votes.values()) or 1.0
        return {
            "status": self.status, "progress": round(self.progress(now), 3), "window_s": self.window_s,
            "elapsed": round(now - self.started, 2) if self.status == SCANNING else None,
            "label": self.label, "category": self.categories.get(self.label) if self.label else None,
            "confidence": round(self.confidence, 3), "share": round(self.share, 3), "frames": self.frames,
            "votes": [{"label": k, "share": round(v / total, 3)} for k, v in top],
            "tokens": [w for w, _ in self.tokens.most_common(12)],
            "views": {k: round(v, 1) for k, v in sorted(self.views.items(), key=lambda kv: -kv[1])},
            "message": self.message, "scans": self.scans,
            "thresholds": {"detect": self.detect_min, "lock_confidence": self.lock_conf, "vote_share": self.min_share},
        }
