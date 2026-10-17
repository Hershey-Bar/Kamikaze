#!/usr/bin/env python3
"""Replay still images through the live receiver like stream_cam.py does (224x224 JPEG POSTs).

Usage: python replay_frames.py [--url http://127.0.0.1:5000] [--repeat 4] img1.jpg img2.jpg ...
Prints the receiver's JSON verdict per frame.
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
    Image.open(path).convert("RGB").resize((224, 224)).save(buf, "JPEG", quality=75)
    for i in range(args.repeat):
        req = urllib.request.Request(args.url, data=buf.getvalue(), headers={"Content-Type": "image/jpeg"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            r = json.load(resp)
    print(f"{path:28s} state={r['state']:10s} label={r['label']} cat={r['category']} conf={r['confidence']} {r['latency_ms']}ms")
