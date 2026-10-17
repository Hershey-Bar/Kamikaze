"""Benchmark CLIP inference latency on the board for different thread counts / quantization."""
import sys
import time

import torch
from PIL import Image

import receiver

img = Image.open(sys.argv[1]).convert("RGB")
for threads in (4, 6, 8):
    clf = receiver.ClipClassifier(receiver.CLIP_MODEL_ID, threads)
    clf.probabilities(img)
    t = time.perf_counter()
    for _ in range(5):
        clf.probabilities(img)
    print(f"fp32 threads={threads}: {(time.perf_counter() - t) / 5 * 1000:.0f} ms", flush=True)

clf.model = torch.ao.quantization.quantize_dynamic(clf.model, {torch.nn.Linear}, dtype=torch.qint8)
clf.probabilities(img)
t = time.perf_counter()
for _ in range(5):
    clf.probabilities(img)
print(f"int8-dynamic threads=8: {(time.perf_counter() - t) / 5 * 1000:.0f} ms", flush=True)
