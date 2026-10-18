# Kamikaze

Real-time edge AI: a webcam feed is classified on the **Arduino VENTUNO Q** (Qualcomm Dragonwing QCS8275, Hexagon NPU), the
matching 3D model is fetched and shown in a pose-synchronised `<model-viewer>`, and the **Modulino Pixels** on the STM32H5F5
co-processor show the recognition state.

```
Windows webcam (stream_cam.py) --HTTP POST /upload--> adb forward tcp:5000 --> receiver.py (VENTUNO Q, 0.0.0.0:5000)
   receiver.py: YOLO26n (Ultralytics, Hexagon NPU)       -> crop the phone/can inside the 60% reticle
                MobileNetV3-Large embedder (Hexagon NPU) -> 1280-d unit vector -> few-shot multi-view prototypes
                pose.py (OpenCV/NumPy + view matching)   -> pitch/yaw/roll for the 3D viewer
                cad_harvester.py (Sketchfab, Poly Pizza, Khronos) -> static/models/<class>.glb + attribution
                /var/run/arduino-router.sock (MessagePack-RPC set_pixels) -> STM32 -> Wire1 (I2C4) -> Modulino Pixels
```

## Board setup (once)

```bash
cd ~/hackathon_iot
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
# Hexagon NPU runtime (QNN TFLite delegate + cDSP skeletons)
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-ppa && sudo apt install qairt-libs qairt-dsp-binaries
cp .env.example .env   # add SKETCHFAB_TOKEN and POLY_PIZZA_API_KEY
./export_yolo.sh       # YOLO26n + YOLO11n -> models/npu/*_float32.tflite (~10 min the first time)
```

The embedder (MediaPipe, Apache-2.0) is downloaded to `models/npu/` on first start. The YOLO detectors (Ultralytics,
AGPL-3.0) are exported on the board by `export_yolo.sh`: Ultralytics' `litert` route needs `litert-converter`, which has
no aarch64 wheel, so the script uses the TensorFlow SavedModel route in its own venv (`~/yolo_export/venv`, TensorFlow
2.19 pins numpy<2.2) and copies the FP32 `.tflite` that the QNN delegate runs in FP16. Without the exported file the
receiver falls back to Ultralytics PyTorch on the CPU. `./export_yolo.sh path/to/best.pt` exports a custom-trained
YOLO; set `KAMIKAZE_DETECTOR=best`. The STM32 firmware in
`modulino_bridge/` is already flashed; it does not need to be reflashed and `arduino-router.service` must stay running.

## Run

```bash
# board
./run_receiver.sh                        # dashboard: http://<board>:5000 (or http://localhost:5000 through adb)
# Windows
adb forward tcp:5000 tcp:5000
python stream_cam.py
```

The startup log prints the accelerator, e.g. `Hexagon NPU (HTP v75, QNN TFLite delegate): embedder + detector`.
`KAMIKAZE_NPU=0` forces CPU (XNNPACK); `KAMIKAZE_DETECT_CROP=0` falls back to the fixed center crop.

| Stage (QCS8275, measured)  | Hexagon NPU | Arm CPU (XNNPACK, 4 threads) |
|----------------------------|-------------|------------------------------|
| MobileNetV3-L embedder     | ~0.9 ms     | ~20 ms                       |
| YOLO26n 640 (default)      | ~6–8.5 ms   | ~200 ms                      |
| YOLO11n 640                | ~6.7 ms     | ~166 ms                      |
| EfficientDet-Lite2 int8    | ~3–6 ms     | ~84 ms                       |

End to end per frame (JPEG decode, letterbox, detect, crop, embed) is ~20–30 ms, dominated by image resizing on the CPU.
COCO has no "can" class, so cans crop as `bottle`/`cup` and phones as `cell phone`/`remote`; when nothing is found the center crop is used.
For real can/phone classes, train a custom YOLO and run `./export_yolo.sh best.pt` with `KAMIKAZE_DETECTOR=best`.

## Training (enrollment)

1. Type a plain-text name in the dashboard (e.g. `coke zero can 330ml`, `s25 ultra black`, `iphone 15 pro max`). It is
   resolved to a canonical class (`coca_cola_zero_330ml_can`, `galaxy_s25_ultra`, ...); colour is kept as metadata only.
2. For each view, hold the object inside the reticle and click **Capture [view] & Next**. The server samples the next
   10 frames, keeps only the embedding vectors (no images are written to disk) and advances:
   - **Front (0°)**: upright, label or screen facing the camera
   - **Back (180°)**: turned around (phone camera island, can nutrition panel)
   - **Side (90°)**: turned a quarter (phone button rail, can profile)
   - **Top (tilt 30–45°)**: tilted toward the camera (can pull-tab, lens cluster)
3. Click **Done** to save. CAD retrieval runs in the background and never blocks enrollment.

Tips: plain background, even light, object filling about 60–70% of the reticle. Use the red **Delete** button (or
**Reset all classes**) to remove a bad run; its vectors, `enrolled_classes.pt` entry and cached CAD model are removed at once.

Classes enrolled with another embedder (for example the earlier torchvision backbone) are not mixed: the old database is
kept as `enrolled_classes.<embedder>.bak.pt` and the objects must be re-enrolled.

## 3D models and attribution

`cad_harvester.build_search_queries()` turns plain text into a cascade of queries, most specific first
(`s25 ultra` -> `Samsung Galaxy S25 Ultra`, `Samsung S25 Ultra`, `Galaxy S25 Ultra`, `Galaxy S24 Ultra`, `Smartphone`).
Sketchfab (downloadable, non-NC licences) and Poly Pizza (`https://api.poly.pizza/v1.1/search/{query}`, `X-Auth-Token`)
are tried at each level, then Khronos sample assets, then a procedural model. Every downloaded model gets a JSON sidecar
with title, author, author URL, licence and source; credits are shown in the dashboard and written to
`static/models/CREDITS.md`.

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload`, `/api/frame` | JPEG frame in, classification out |
| GET | `/api/state`, `/api/asset`, `/api/pose` | recognition state, current 3D asset, pose |
| GET | `/api/telemetry`, `/api/telemetry/stream` | SoC, thermals, RAM, CPU, NPU, LED bridge, inference (JSON / SSE) |
| POST | `/api/enroll/start` / `capture` / `finish` / `cancel`, GET `/api/enroll/status` | guided enrollment |
| GET | `/api/classes` | enrolled classes |
| DELETE | `/api/classes/<id>` | delete a class and its CAD cache (`?delete_cad=0` keeps the model) |
| POST | `/api/classes/reset` | delete all classes and their CAD cache |
| POST | `/api/resolve`, `/api/cad/fetch` | plain-text resolution, CAD retrieval |
| GET | `/api/credits`, `/api/db/export`; POST `/api/db/save` | attribution, database export/save |

`mcp_server.py` exposes the same capabilities as FastMCP tools over stdio (`search_3d_asset`, `actuate_modulino`,
`trigger_guidance`, `enroll_multiview`, `fetch_cad_asset`, `resolve_product`, `get_telemetry`, `get_pose_telemetry`, ...).

## Tests

```bash
.venv/bin/python -W ignore -m unittest test_kamikaze test_comprehensive test_full_suite
```

Hardware tests (LED RPC, Hexagon NPU, online CAD) skip themselves when the router socket, the NPU or credentials are absent.