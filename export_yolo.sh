#!/usr/bin/env bash
# Export Ultralytics YOLO detectors to TFLite for the Hexagon NPU (QNN TFLite delegate) on the VENTUNO Q.
#
#   ./export_yolo.sh                 # yolo26n + yolo11n at 640 px
#   ./export_yolo.sh yolo26s         # any Ultralytics detect model (or a custom best.pt path)
#   IMGSZ=480 ./export_yolo.sh yolo26n
#
# The export toolchain (TensorFlow 2.19 + onnx2tf) pins numpy<2.2, so it lives in its own venv (~/yolo_export/venv)
# and never touches the receiver's .venv. litert-torch (Ultralytics' format=litert) has no aarch64 wheel, so the
# SavedModel route is used; it writes <model>_float32.tflite, which the QNN delegate runs on the HTP in FP16.
# Output: models/npu/<model>_float32.tflite + models/npu/<model>.names.json (class-index -> name).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="${YOLO_EXPORT_DIR:-$HOME/yolo_export}"
OUT="${KAMIKAZE_NPU_MODELS:-$ROOT/models/npu}"
IMGSZ="${IMGSZ:-640}"
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(yolo26n yolo11n)

mkdir -p "$WORK" "$OUT"
cd "$WORK"
if [ ! -x venv/bin/python ]; then
  if ! python3 -m venv venv 2>/dev/null; then
    rm -rf venv
    python3 -m venv --without-pip venv   # python3-venv (ensurepip) is not installed on the board image
    "$ROOT/.venv/bin/python" -m pip --python "$WORK/venv/bin/python" install -q pip
  fi
fi
venv/bin/python -m pip install -q "ultralytics>=8.4" "tensorflow==2.19.1" "tf_keras==2.19.0" \
  "onnx2tf>=1.26.3,<1.29.0" "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26" "ai-edge-litert>=1.2.0" \
  onnxruntime "onnx>=1.12" onnxslim "protobuf>=5"

for model in "${MODELS[@]}"; do
  weights="$model"
  [[ "$weights" == *.pt ]] || weights="$model.pt"
  stem="$(basename "${weights%.pt}")"
  echo "== exporting $weights (imgsz $IMGSZ)"
  venv/bin/yolo export model="$weights" format=saved_model imgsz="$IMGSZ"
  src_dir="$(dirname "$weights")/${stem}_saved_model"
  [ -d "$src_dir" ] || src_dir="$WORK/${stem}_saved_model"
  cp "$src_dir/${stem}_float32.tflite" "$OUT/${stem}_float32.tflite"
  venv/bin/python - "$src_dir/metadata.yaml" "$OUT/${stem}.names.json" <<'PY'
import json, sys, yaml
meta = yaml.safe_load(open(sys.argv[1]))
names = meta["names"]
names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
json.dump({"names": names, "imgsz": meta.get("imgsz"), "description": meta.get("description"),
           "license": meta.get("license"), "version": meta.get("version")}, open(sys.argv[2], "w"), indent=1)
print(f"{sys.argv[2]}: {len(names)} classes")
PY
  ls -la "$OUT/${stem}_float32.tflite"
done
