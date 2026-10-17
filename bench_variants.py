"""Compare CLIP precision/quantization variants on the board: latency and predictions."""
import copy
import glob
import os
import sys
import time

import torch
from PIL import Image

os.environ["KAMIKAZE_CLIP_QUANT"] = "0"
import receiver

paths = sorted(glob.glob(sys.argv[1] + "/*.jpg"))
imgs = [(os.path.basename(p), Image.open(p).convert("RGB")) for p in paths]
base = receiver.ClipClassifier(receiver.CLIP_MODEL_ID, 8)
Q = torch.ao.quantization.quantize_dynamic


def variant(name):
    clf = copy.copy(base)
    clf.model = copy.deepcopy(base.model)
    vm = clf.model.vision_model
    if name == "mlp_int8":
        for layer in vm.encoder.layers:
            layer.mlp = Q(layer.mlp, {torch.nn.Linear}, dtype=torch.qint8)
    elif name == "all_int8":
        clf.model.vision_model = Q(vm, {torch.nn.Linear}, dtype=torch.qint8)
    elif name == "bf16":
        clf.model.vision_model = vm.to(torch.bfloat16)
        orig = clf._preprocess
        clf._preprocess = lambda im: orig(im).to(torch.bfloat16)
        clf.model.visual_projection = clf.model.visual_projection.to(torch.bfloat16)
        pf = clf.probabilities

        def probs(image, _c=clf):
            with torch.inference_mode():
                feats = _c.model.get_image_features(pixel_values=_c._preprocess(image))
                feats = feats.float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return torch.softmax(_c.scale * feats @ _c.text.T, dim=-1)[0].numpy()

        clf.probabilities = probs
    return clf


for name in ("fp32", "mlp_int8", "all_int8", "bf16"):
    try:
        clf = base if name == "fp32" else variant(name)
        clf.probabilities(imgs[0][1])
        t = time.perf_counter()
        res = [(n, clf.classify(im)) for n, im in imgs]
        ms = (time.perf_counter() - t) / len(imgs) * 1000
        print(f"== {name}: {ms:.0f} ms/img", flush=True)
        for n, d in res:
            print(f"   {n:22s} {str(d.label):28s} {d.confidence}", flush=True)
    except Exception as exc:
        print(f"== {name}: failed {exc}", flush=True)
