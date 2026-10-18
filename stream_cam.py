#!/usr/bin/env python3
"""Stream the laptop webcam to the Kamikaze receiver on the VENTUNO Q at the camera's native frame rate.

A capture thread always holds the newest frame; the main loop JPEG-encodes it and a small pool of keep-alive HTTP
senders POSTs it to /api/ingest, which answers immediately (inference runs on the board's NPU worker thread).
Several connections are needed because each round trip through the ADB tunnel costs ~45 ms, which would otherwise
cap a single sequential sender at ~21 fps.
Run `adb forward tcp:5000 tcp:5000` first. Press 'q' in the preview window (or Ctrl+C) to stop.
"""

import argparse
import statistics
import threading
import time

import cv2
import requests

CROP_FRACTION = 0.60  # must match receiver.CROP_FRACTION (the receiver crops/resizes itself)


class Camera:
    """Background reader: keeps only the most recent frame so the sender never works on a stale one."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DirectShow avoids MSMF sample-reader locks on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise SystemExit("Error: could not open the webcam. Close Teams/Zoom/Camera app and retry.")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # most webcams only reach 30 fps in MJPG mode
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.read_fps = 0.0
        self._cv = threading.Condition()
        self._frame = None
        self._seq = 0
        self._running = True
        threading.Thread(target=self._run, name="camera", daemon=True).start()

    def _run(self) -> None:
        last = None
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            if last is not None:
                inst = 1.0 / max(1e-3, now - last)
                self.read_fps = inst if not self.read_fps else 0.9 * self.read_fps + 0.1 * inst
            last = now
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
                if resp.status_code == 404 and url.endswith("/api/ingest"):
                    with self._cv:
                        if self.url == url:
                            self.url = url[: -len("/api/ingest")] + "/api/frame"
                            print(f"Receiver has no /api/ingest (older build); falling back to synchronous {self.url}")
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
                    print(f"Transfer failed ({fails}x): {exc}  -- is `adb forward tcp:5000 tcp:5000` active and receiver.py running?")
                time.sleep(0.2)

    def close(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:5000/api/ingest", help="receiver ingest endpoint")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality")
    ap.add_argument("--senders", type=int, default=3, help="parallel keep-alive connections (hides ADB tunnel latency)")
    ap.add_argument("--no-preview", action="store_true", help="skip the local OpenCV preview window")
    args = ap.parse_args()

    cam = Camera(args.camera, args.width, args.height, args.fps)
    uplink = Uplink(args.url, args.senders)
    print(f"Streaming {cam.size[0]}x{cam.size[1]} webcam frames to the VENTUNO Q via {args.url} "
          f"with {args.senders} senders (q / Ctrl+C to stop)")

    seq = 0
    t_report = time.perf_counter()
    try:
        while True:
            frame, new = cam.next(seq)
            if frame is None or new == seq:
                continue
            seq = new
            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            if not ok:
                continue
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
                print(f"camera {cam.read_fps:4.1f} fps | sent {sent / (now - t_report):4.1f} fps | POST {post_ms} ms | "
                      f"board NPU {verdict.get('fps', '-')} fps | {verdict.get('state', '?')} {verdict.get('label') or '-'} "
                      f"sim={verdict.get('similarity')}")
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        uplink.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
