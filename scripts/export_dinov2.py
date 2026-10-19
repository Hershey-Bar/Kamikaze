"""Export Meta DINOv2 ViT-S/14 as an HTP-friendly TFLite viewpoint descriptor for pose.py / receiver.py.

Run on the board inside the export venv used by export_yolo.sh (torch, onnx, onnxslim, onnx2tf, tensorflow):

    TORCH_HOME=models/dinov2 ~/yolo_export/venv/bin/python export_dinov2.py

Graph: 224x224 RGB in [0,1] -> ImageNet normalisation -> ViT-S/14 (position embedding pre-interpolated to 16x16
patches, GELU in its tanh form so no Flex Erf op is emitted) -> [CLS | mean patch token] (1, 768).
Output: models/npu/dinov2_vits14_pose_float32.tflite (~87 MB), validated against the torch hub model (cosine >= 0.999).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models" / "npu" / "dinov2_vits14_pose_float32.tflite"
SIZE = 224


def build(torch, nn):
    hub = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, skip_validation=True).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def reference(x):
        f = hub.forward_features((x - mean) / std)
        return torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1)

    class Pose(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            tok = m.patch_embed(torch.zeros(1, 3, SIZE, SIZE))
            tok = torch.cat((m.cls_token.expand(1, -1, -1), tok), 1)
            self.register_buffer("pos", m.interpolate_pos_encoding(tok, SIZE, SIZE).detach().clone())
            self.register_buffer("mean", mean.clone())
            self.register_buffer("std", std.clone())

        def forward(self, x):
            m = self.m
            t = m.patch_embed((x - self.mean) / self.std)
            t = torch.cat((m.cls_token.expand(t.shape[0], -1, -1), t), 1) + self.pos
            for blk in m.blocks:
                t = blk(t)
            t = m.norm(t)
            return torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)

    return hub, reference, Pose


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import onnx
    import onnxslim
    import torch
    import torch.nn as nn

    torch.set_grad_enabled(False)
    hub, reference, Pose = build(torch, nn)
    rng = np.random.default_rng(0)
    probe = torch.from_numpy(rng.random((2, 3, SIZE, SIZE), dtype=np.float32))
    want = reference(probe).numpy()
    for mod in list(hub.modules()):
        for name, child in mod.named_children():
            if isinstance(child, nn.GELU):
                setattr(mod, name, nn.GELU(approximate="tanh"))
    model = Pose(hub).eval()
    got = model(probe).numpy()
    print("tanh-GELU wrapper vs hub cosine:", [round(cosine(g, w), 5) for g, w in zip(got, want)])

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        onnx_path = work / "dinov2_vits14_pose.onnx"
        torch.onnx.export(model, probe[:1], str(onnx_path), input_names=["image"], output_names=["feat"],
                          opset_version=17, dynamo=False)
        onnx.save(onnxslim.slim(onnx.load(str(onnx_path))), str(onnx_path))

        import onnx2tf.onnx2tf as o2t
        # onnx2tf downloads a calibration image set that no longer loads with allow_pickle=False; float export only
        # needs a shape-compatible sample.
        o2t.download_test_image_data = lambda: rng.random((20, SIZE, SIZE, 3), dtype=np.float32)
        o2t.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(work / "tf"), non_verbose=True)
        tfl = work / "tf" / "dinov2_vits14_pose_float32.tflite"

        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
        it = Interpreter(model_path=str(tfl))
        it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        for i in range(len(want)):
            it.set_tensor(inp["index"], np.ascontiguousarray(probe[i:i + 1].numpy().transpose(0, 2, 3, 1)))
            it.invoke()
            c = cosine(it.get_tensor(out["index"]).reshape(-1), want[i])
            print(f"tflite vs hub cosine [{i}]: {c:.5f}")
            if c < 0.999:
                raise SystemExit("export does not match the torch model")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tfl, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
