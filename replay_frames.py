#!/usr/bin/env python3
"""Replay still images through the live receiver like stream_cam.py does (full-resolution JPEG POSTs).

Usage: python replay_frames.py [--url http://127.0.0.1:5000] [--repeat 4] img1.jpg img2.jpg ...
Prints the receiver's JSON verdict after the last repeat of each image.
"""

import argparse
import io
import json
import urllib.request

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+")
parser.add_argument("--url", default="http://127.0.0.1:5000")
parser.add_argument("--repeat", type=int, default=4)
args = parser.parse_args()

for path in args.images:
    buf = io.BytesIO()
    img = Image.open(path).convert("RGB")
    img.thumbnail((640, 480))
    img.save(buf, "JPEG", quality=80)
    for _ in range(args.repeat):
        req = urllib.request.Request(args.url.rstrip("/") + "/api/frame", data=buf.getvalue(), headers={"Content-Type": "image/jpeg"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            r = json.load(resp)
    print(f"{path:28s} state={r['state']:10s} label={r['label']} sim={r['similarity']} margin={r['margin']} "
          f"embed={r['embed_ms']}ms total={r['latency_ms']}ms")
