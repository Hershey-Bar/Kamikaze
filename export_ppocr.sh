#!/usr/bin/env bash
# Export the PP-OCRv4 text DETECTOR bundled with RapidOCR to a fixed-shape TFLite graph for the Hexagon NPU.
#
#   ./export_ppocr.sh            # -> models/npu/ppocrv4_det_320_float32.tflite (QNN TFLite delegate, HTP FP16)
#   SIDE=480 ./export_ppocr.sh   # other square input size (must match KAMIKAZE_OCR_SIDE)
#
# Measured on the VENTUNO Q (QCS8275, HTP v75): 16 ms on the NPU vs 110 ms on ONNX Runtime CPU at 320 px, probability
# map IoU 0.999 vs the ONNX reference. The PP-OCRv4 recogniser (SVTR-LCNet) stays on ONNX Runtime CPU: onnx2tf cannot
# convert its LayerNorm / attention reshapes. onnxruntime-qnn 2.7.0 (QNN plugin EP) was also evaluated and fails to
# create the HTP device on this BSP (QNN_DEVICE_ERROR_INVALID_CONFIG), so the TFLite delegate is the NPU path.
#
# Uses the same isolated export venv as export_yolo.sh (TensorFlow 2.19 + onnx2tf) - run export_yolo.sh once first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="${OCR_EXPORT_DIR:-$HOME/ocr_export}"
VENV="${YOLO_EXPORT_DIR:-$HOME/yolo_export}/venv"
OUT="${KAMIKAZE_NPU_MODELS:-$ROOT/models/npu}"
SIDE="${SIDE:-320}"
SRC="$("$ROOT/.venv/bin/python" -c 'import os, rapidocr_onnxruntime as r; print(os.path.join(os.path.dirname(r.__file__), "models"))')"

[ -x "$VENV/bin/onnx2tf" ] || { echo "missing $VENV/bin/onnx2tf - run ./export_yolo.sh first" >&2; exit 1; }
mkdir -p "$WORK" "$OUT"
cd "$WORK"
cp "$SRC/ch_PP-OCRv4_det_infer.onnx" det.onnx
# onnx2tf needs fully static shapes (dynamic H/W leaves a broken TRANSPOSE_CONV output shape in the graph)
"$VENV/bin/python" -m onnxruntime.tools.make_dynamic_shape_fixed --input_name x --input_shape "1,3,$SIDE,$SIDE" \
  det.onnx det_s.onnx
CAL=calibration_image_sample_data_20x128x128x3_float32.npy
[ -f "$CAL" ] || cp "${YOLO_EXPORT_DIR:-$HOME/yolo_export}/$CAL" . 2>/dev/null || true
rm -rf det_tf
"$VENV/bin/onnx2tf" -i det_s.onnx -o det_tf -b 1 -n
cp det_tf/det_s_float32.tflite "$OUT/ppocrv4_det_${SIDE}_float32.tflite"
ls -la "$OUT/ppocrv4_det_${SIDE}_float32.tflite"
