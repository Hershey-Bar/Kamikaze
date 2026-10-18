# Attribution

Kamikaze trains nothing from scratch. Every model below is a published pretrained checkpoint, run on-device
on the Arduino VENTUNO Q (Qualcomm Dragonwing QCS8275: Hexagon NPU via the QNN TFLite delegate, Arm CPU fallback).
Enrollment stores only normalised embedding vectors and OCR keyword lexicons (`enrolled_classes.pt`). It never stores images.

## Perception models

| Stage | Model | Authors / repository | Licence | How it runs here |
|---|---|---|---|---|
| Object detection (domain gate) | YOLO-World v2 (`yolov8s-worldv2`), vocabulary baked in by `export_yolo.sh` | Ultralytics – <https://github.com/ultralytics/ultralytics>; YOLO-World by Tencent AI Lab – <https://github.com/AILab-CVC/YOLO-World> | AGPL-3.0 (Ultralytics) / GPL-3.0 (YOLO-World) | TFLite FP32 → Hexagon NPU (QNN HTP, FP16) |
| Identity embedding | MobileNetV3-Large image embedder (1280-d) | Google MediaPipe – <https://github.com/google-ai-edge/mediapipe> | Apache-2.0 | TFLite → Hexagon NPU |
| Identity embedding (fallback) | MobileNetV3-Large ImageNet weights | PyTorch torchvision – <https://github.com/pytorch/vision> | BSD-3-Clause | ExecuTorch / eager on the Arm CPU |
| Smartphone identity + 360° pose | DINOv2 ViT-S/14 (`dinov2_vits14`), exported by `export_dinov2.py` | Meta AI Research – <https://github.com/facebookresearch/dinov2> | Apache-2.0 | TFLite → Hexagon NPU |
| Text spotting (cans) | PP-OCRv4 mobile detection + recognition, packaged by RapidOCR | PaddlePaddle – <https://github.com/PaddlePaddle/PaddleOCR>; RapidAI – <https://github.com/RapidAI/RapidOCR> | Apache-2.0 | det: TFLite → Hexagon NPU (`export_ppocr.sh`); rec: ONNX Runtime CPU |
| ONNX execution | ONNX Runtime | Microsoft – <https://github.com/microsoft/onnxruntime> | MIT | CPU execution provider. The QNN EP plugin (onnxruntime-qnn 2.7.0) is incompatible with this board's BSP, so NPU work goes through the QNN TFLite delegate. |
| NPU runtime | Qualcomm AI Engine Direct (QNN) TFLite delegate, `qairt-libs` | Qualcomm – <https://github.com/qualcomm-ai-hub/models> (AI Hub reference models for Dragonwing IQ-8275) | Qualcomm AI Hub / QAIRT licence terms | `libQnnTFLiteDelegate.so` + `libQnnHtp.so` |
| TFLite interpreter | LiteRT (`ai-edge-litert`) | Google – <https://github.com/google-ai-edge/LiteRT> | Apache-2.0 | Host for the TFLite graphs |

## Dashboard

- `<model-viewer>` – Google – <https://github.com/google/model-viewer> – Apache-2.0.

## 3D assets

- **Kirkland Signature Sparkling Water – Lemon, Lime, Grapefruit**: 3D hand scans by
  [Hershey-Bar](https://github.com/Hershey-Bar). They are owner-provided, all rights reserved by the author, and kept
  only in the device's local model library (`static/models/library/`). They are never committed to this repository.
- **Online CAD** (Sketchfab, Poly Pizza): each downloaded model keeps its own title, author, licence and source link
  in a sidecar `.json`. The dashboard credit line and `GET /api/credits` show them for whichever model is on stage.
  No model is shown without its attribution. No geometry is invented for products whose dimensions are not verified.
- **1:1 phone dummies** (Printables <https://www.printables.com>, Thingiverse <https://www.thingiverse.com>): each
  converted dummy keeps the print title, author, licence (with deed link) and print URL, and is marked as modified
  (converted to GLB, re-oriented, scaled, possibly simplified; never simplified under ND licences). Only licences
  allowing redistribution are used; NC only with `KAMIKAZE_CAD_ALLOW_NC=1`.
- **Packaging photos**: Open Food Facts contributors, <https://world.openfoodfacts.org>, images under
  CC BY-SA 3.0 (<https://creativecommons.org/licenses/by-sa/3.0/>), product page linked per model; the photo is
  segmented and unwrapped onto the can (modified).
- **Dimensions**: smartphones from GSMArena spec pages (linked per model), cans from Crown's standard can size table.
- **Live label scans** are made by the operator's own camera and stay on the device.

## Hardware

- Arduino VENTUNO Q, Arduino Modulino Pixels (`Arduino_Modulino`, MPL-2.0), and Arduino RouterBridge / `arduino-router`.
