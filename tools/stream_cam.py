#!/usr/bin/env python3
"""Stream the laptop webcam to the Kamikaze receiver on the VENTUNO Q at the camera's native frame rate.

A capture thread always holds the newest frame; the main loop JPEG-encodes it and a small pool of keep-alive HTTP
senders POSTs it as raw image/jpeg to /api/ingest, which answers immediately (inference runs on the board's NPU worker
thread). Several connections are needed because each round trip through the ADB tunnel costs ~45 ms, which would
otherwise cap a single sequential sender at ~21 fps.

Run `adb forward tcp:5000 tcp:5000` first. Press 'q' in the preview window (or Ctrl+C) to stop.
Every problem is printed: receiver unreachable, HTTP status of the first responses and of every error, and a camera
that opens but delivers no frames (the capture then falls back to other backends / pixel formats automatically).
"""

import argparse
import statistics
import threading
import time
import urllib.parse

import cv2
import requests

CROP_FRACTION = 0.60  # must match receiver.CROP_FRACTION (the receiver crops/resizes itself)
NO_FRAME_TIMEOUT_S = 4.0


def _backend_configs() -> list[tuple[str, int, bool]]:
    """(name, OpenCV API, force MJPG) in the order tried. MJPG is what lets most webcams reach 30 fps; some refuse it."""
    return [("DirectShow+MJPG", cv2.CAP_DSHOW, True), ("DirectShow", cv2.CAP_DSHOW, False),
            ("MediaFoundation", cv2.CAP_MSMF, False), ("default", cv2.CAP_ANY, False)]


class Camera:
    """Background reader: keeps only the most recent frame. If the device opens but yields no frames for
    NO_FRAME_TIMEOUT_S, it is reopened with the next backend/pixel format so the stream never fails silently."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.read_fps = 0.0
        self.backend = "-"
        self.size = (0, 0)
        self._cv = threading.Condition()
        self._frame = None
        self._seq = 0
        self._running = True
        self._configs = _backend_configs()
        self._cfg = 0
        self.cap = self._open_next()
        if self.cap is None:
            raise SystemExit(f"[CAMERA] could not open webcam index {index} with any backend. "
                             "Close Teams/Zoom/the Camera app (only one program can use the webcam) or try --camera 1.")
        threading.Thread(target=self._run, name="camera", daemon=True).start()

    def _open_next(self):
        while self._cfg < len(self._configs):
            name, api, mjpg = self._configs[self._cfg]
            self._cfg += 1
            cap = cv2.VideoCapture(self.index, api)
            if not cap.isOpened():
                cap.release()
                print(f"[CAMERA] {name}: cannot open index {self.index}")
                continue
            if mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.backend = name
            self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print(f"[CAMERA] opened index {self.index} via {name} at {self.size[0]}x{self.size[1]}")
            return cap
        return None

    def _run(self) -> None:
        last = None
        opened = time.perf_counter()
        while self._running:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                if now - max(opened, last or 0.0) > NO_FRAME_TIMEOUT_S:
                    print(f"[CAMERA] {self.backend}: no frames for {NO_FRAME_TIMEOUT_S:.0f} s (camera busy or format unsupported)")
                    self.cap.release()
                    cap = self._open_next()
                    if cap is None:
                        self._cfg = 0  # cycle through the backends again (e.g. another app released the camera)
                        time.sleep(1.0)
                        cap = self._open_next()
                    if cap is None:
                        continue
                    self.cap, opened, last, self.read_fps = cap, time.perf_counter(), None, 0.0
                else:
                    time.sleep(0.005)
                continue
            if last is not None:
                inst = 1.0 / max(1e-3, now - last)
                self.read_fps = inst if not self.read_fps else 0.9 * self.read_fps + 0.1 * inst
            last = now
            if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
                self.size = (frame.shape[1], frame.shape[0])
            with self._cv:
                self._frame, self._seq = frame, self._seq + 1
                self._cv.notify_all()

    def next(self, seq: int, timeout: float = 1.0):
        with self._cv:
            self._cv.wait_for(lambda: self._seq != seq or not self._running, timeout)
            return self._frame, self._seq

    def close(self) -> None:
        self._running = False
        self.cap.release()


class Uplink:
    """Pool of keep-alive senders; every newly encoded frame is claimed by exactly one idle sender (stale ones are skipped)."""

    def __init__(self, url: str, senders: int) -> None:
        self.url = url
        self._cv = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self._claimed = 0
        self._running = True
        self.sent, self.fails, self.rtts, self.verdict = 0, 0, [], {}
        self.responses, self.errors, self._last_err_print = 0, 0, 0.0
        for i in range(max(1, senders)):
            threading.Thread(target=self._run, name=f"sender-{i}", daemon=True).start()

    def offer(self, jpeg: bytes) -> None:
        with self._cv:
            self._jpeg, self._seq = jpeg, self._seq + 1
            self._cv.notify()

    def stats(self) -> tuple[int, list, dict]:
        with self._cv:
            out = (self.sent, self.rtts[:], self.verdict)
            self.sent, self.rtts = 0, []
            return out

    def _log_response(self, url: str, resp: requests.Response) -> None:
        with self._cv:
            self.responses += 1
            n = self.responses
            if resp.status_code != 200:
                self.errors += 1
            now = time.monotonic()
            noisy = resp.status_code != 200 and now - self._last_err_print < 1.0
            if resp.status_code != 200 and not noisy:
                self._last_err_print = now
        if n <= 5 or (resp.status_code != 200 and not noisy):
            text = " ".join(resp.text[:100].split())
            print(f"[STREAM] POST {url} -> HTTP {resp.status_code}: {text}")

    def _run(self) -> None:
        session = requests.Session()
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._seq > self._claimed or not self._running)
                if not self._running:
                    return
                self._claimed, data, url = self._seq, self._jpeg, self.url
            t0 = time.perf_counter()
            try:
                resp = session.post(url, data=data, headers={"Content-Type": "image/jpeg"}, timeout=3.0)
                self._log_response(url, resp)
                if resp.status_code in (404, 405) and url.endswith("/api/ingest"):
                    with self._cv:
                        if self.url == url:
                            self.url = url[: -len("/api/ingest")] + "/upload"
                            print(f"[STREAM] receiver has no /api/ingest (older build); falling back to {self.url}")
                    continue
                verdict = resp.json() if resp.ok else None
                with self._cv:
                    self.rtts.append((time.perf_counter() - t0) * 1000.0)
                    self.sent += 1
                    if verdict is not None:
                        self.verdict = verdict
            except (requests.exceptions.RequestException, ValueError) as exc:
                with self._cv:
                    self.fails += 1
                    fails = self.fails
                if fails % 30 == 1:
                    print(f"[STREAM] POST {url} failed ({fails}x): {exc}\n"
                          "         -> is `adb forward tcp:5000 tcp:5000` active and receiver.py running on the board?")
                time.sleep(0.2)

    def close(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()


def normalize_url(url: str) -> str:
    """Accept a bare host/base URL (http://127.0.0.1:5000) and point it at the async ingest route."""
    if "://" not in url:
        url = "http://" + url
    parts = urllib.parse.urlsplit(url)
    if parts.path in ("", "/"):
        parts = parts._replace(path="/api/ingest")
    return urllib.parse.urlunsplit(parts)


def probe(url: str) -> None:
    base = urllib.parse.urlunsplit(urllib.parse.urlsplit(url)._replace(path="", query=""))
    try:
        info = requests.get(base + "/api/ingest", timeout=3.0)
        if info.ok:
            st = info.json().get("stream", {})
            print(f"[STREAM] receiver reachable at {base} (frames received so far: {st.get('ingested', 0)})")
        else:
            state = requests.get(base + "/api/state", timeout=3.0)
            print(f"[STREAM] receiver reachable at {base} (older build: /api/ingest GET -> {info.status_code}, "
                  f"/api/state -> {state.status_code})")
    except requests.exceptions.RequestException as exc:
        print(f"[STREAM] receiver NOT reachable at {base}: {exc}\n"
              "         -> run `adb forward tcp:5000 tcp:5000` and start ./run_receiver.sh on the board; streaming will retry.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:5000/api/ingest", help="receiver ingest endpoint (or base URL)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality")
    ap.add_argument("--senders", type=int, default=3, help="parallel keep-alive connections (hides ADB tunnel latency)")
    ap.add_argument("--no-preview", action="store_true", help="skip the local OpenCV preview window")
    args = ap.parse_args()

    url = normalize_url(args.url)
    probe(url)
    cam = Camera(args.camera, args.width, args.height, args.fps)
    uplink = Uplink(url, args.senders)
    print(f"[STREAM] streaming webcam frames to {url} with {args.senders} senders (q / Ctrl+C to stop)")

    seq = 0
    t_report = time.perf_counter()
    try:
        while True:
            frame, new = cam.next(seq)
            if frame is not None and new != seq:
                seq = new
                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                if ok:
                    uplink.offer(enc.tobytes())
                if not args.no_preview:
                    h, w = frame.shape[:2]
                    side = int(CROP_FRACTION * min(w, h))
                    x0, y0 = (w - side) // 2, (h - side) // 2
                    preview = frame.copy()
                    cv2.rectangle(preview, (x0, y0), (x0 + side, y0 + side), (246, 130, 59), 2)
                    cv2.imshow("Webcam Streamer Preview", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            now = time.perf_counter()
            if now - t_report >= 1.0:
                sent, rtts, verdict = uplink.stats()
                post_ms = f"{statistics.median(rtts):.1f}" if rtts else "-"
                warn = "  <- camera delivering no frames" if frame is None or cam.read_fps == 0 else ""
                warn = warn or ("  <- frames not reaching the receiver (see errors above)" if not sent else "")
                print(f"camera {cam.read_fps:4.1f} fps ({cam.backend}) | sent {sent / (now - t_report):4.1f} fps | "
                      f"POST {post_ms} ms | board NPU {verdict.get('fps', '-')} fps | {verdict.get('state', '?')} "
                      f"{verdict.get('label') or '-'} sim={verdict.get('similarity')}{warn}")
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        uplink.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
