# Kamikaze

Real-time edge AI: a webcam feed is classified on the **Arduino VENTUNO Q** (Qualcomm Dragonwing QCS8275, Hexagon NPU), the
matching 3D model is fetched and shown in a pose-synchronised `<model-viewer>`, and the **Modulino Pixels** on the STM32H5F5
co-processor show the recognition state.

```
Windows webcam (stream_cam.py) --HTTP POST /api/ingest, 30 fps--> adb forward tcp:5000 --> receiver.py (VENTUNO Q, 0.0.0.0:5000)
   receiver.py: ingest returns at once; an NPU worker thread always processes the newest frame (older ones are dropped)
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

`stream_cam.py` captures in MJPG mode at 30 fps and POSTs over 3 parallel keep-alive connections (`--senders`), because
one round trip through the ADB tunnel costs ~45 ms and a single sequential sender tops out at ~21 fps. The dashboard
reads the MJPEG stream (`/api/stream`) with `fetch()` and draws each frame on the canvas; if no frame has arrived for
`KAMIKAZE_STALE_S` (3 s) the preview is dropped and it shows *Awaiting stream from stream_cam.py on port 5000...*, so a
stale or test image is never displayed. State and telemetry share one SSE connection (`/api/events?telemetry=1`), so
several open tabs do not exhaust the browser's per-host connection limit.
Measured live: camera 30 fps, 29.8 fps sent, NPU worker 29-30 fps (detect 8.7 ms, embed 2.3 ms, 17 ms per frame), MJPEG 29.6 fps, 0 dropped.

### Frame ingest

Every route below accepts a frame; the payload format is detected from the body, not only from `Content-Type`:

- Asynchronous (reply at once, latest frame wins): `/api/ingest`, `/ingest`, `/`, `/upload`, `/api/upload`, `/frame`
- Synchronous (reply contains the classification): `/api/frame`, or any route with `?sync=1`
- Formats: raw `image/jpeg` bytes; `multipart/form-data` (any field name); JSON `{"image": "<base64 or data URL>"}`
  (keys such as `image`, `frame`, `jpeg`, `img`, `data`, `file`, `b64`); form-urlencoded `frame=<base64>`; bare base64;
  PNG/BMP/WebP/GIF (transcoded to JPEG). Anything else is rejected with HTTP 400 and counted in telemetry (`stream.rejected`).

`GET /api/ingest` returns the accepted routes and the live stream counters. The receiver logs
`Live stream started: <client> POST <path> (<format>, ...)` when frames start and `Live stream idle` when they stop.

### Troubleshooting "Awaiting stream"

1. `stream_cam.py` prints `[STREAM] receiver reachable at ...` at startup, then `[STREAM] POST ... -> HTTP 200` for the
   first 5 frames and every error. *not reachable* means the ADB forward is missing: run `adb forward tcp:5000 tcp:5000`.
2. `[CAMERA]` lines show which backend opened the webcam (DirectShow+MJPG, DirectShow, MediaFoundation, default); a
   camera that delivers no frames for 4 s is reopened with the next backend. Use `--camera 1` for another device.
3. The once-per-second stats line warns when the camera delivers no frames or frames do not reach the receiver.
4. On the board, `curl -s localhost:5000/api/ingest` shows `ingested`, `rejected` and the last `source`;
   `/tmp/receiver.log` has the `Live stream ...` lines.

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

### Memory

Models, the embedding bank and all `static/` files (GLB models, JS) are kept in RAM (`KAMIKAZE_ASSET_CACHE_MB`,
default 1024), so `<model-viewer>` loads models without touching the disk. The pipeline needs ~1.5 GB in total; the
remaining RAM is deliberately left free. An optional rewind ring of decoded frames can be enabled in training mode with
`KAMIKAZE_RING_GB=8` (capped at 60% of physical RAM, filled as frames arrive, ~27 MB/s at 640x480 30 fps, browsable via
`/api/ring` and `/api/ring/frame.jpg?ago=N`). It is off by default because it keeps camera images in memory and does not
make inference faster; it is always off in deployment mode.

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

**PURGE DATABASE** (dashboard, or `POST /api/classes/purge`) deletes every class, removes `enrolled_classes.pt` (and its
`.tmp` / `.bak.pt` siblings) from disk and clears the cached CAD models: a clean slate before a fresh training run.

## Recognition logic

Every frame passes these stages in order (`receiver.py` `Pipeline.process_jpeg` / `_arbitrate`):

1. **Empty-scene gate.** The 60% reticle crop is scored for structure (Laplacian variance and Canny edge density). Below
   `KAMIKAZE_LAPLACIAN_MIN` (default 18) or `KAMIKAZE_EDGE_MIN` (0.002) the frame is a blank wall / dark scene: detector and
   embedder are skipped, the state is **SEARCHING** with confidence 0% / margin 0%, the banner reads
   *Awaiting object inside center reticle...* and one cyan LED is lit `(0,100,150,15,1)`. Disable with `KAMIKAZE_SCENE_GATE=0`.
2. **Detector = domain gate.** YOLO26n on the NPU only restricts *which classes may match*: `cell phone` ≥ 0.50 → smartphones
   only, `cup` / `bottle` ≥ 0.50 → beverage cans/bottles only. If the unrestricted top-1 is from the other domain the frame is
   **GUIDING** with *Verifying object geometry... Hold steady in center.*
3. **Calibrated confidence.** Raw cosine `S` maps to `C = max(0, (S − 0.52) / 0.48)` (`KAMIKAZE_CAL_FLOOR`), so the empty
   background (raw 0.38–0.45) reads 0%. **IDENTIFIED** needs `C ≥ 0.70` (raw ≈ 0.856) *and* margin `Δ ≥ 0.15` for 3 consecutive
   frames; `0.25 ≤ C < 0.70` or `Δ < 0.15` is **GUIDING**; `C < 0.25` is **SEARCHING**.
4. **Smartphones.** Views are weighted back 0.65 / side 0.15 / top 0.10 / front 0.10 (`device_taxonomy.py`). Because the camera
   sees one face at a time, the weights act as per-view reliability (back × 1.00, side × 0.81, top/front × 0.79): only the rear
   shell can push a phone over the IDENTIFIED floor. If the live frame matches a phone's screen better than its back,
   confidence is capped at 35%, LEDs go amber and the banner says *Front screen detected. Flip phone 180° to expose camera
   island and sensors.* When two enrolled phones are within Δ 0.15, the tier taxonomy picks the cue that separates them
   (Pro LiDAR aperture, Ultra boxy corners + periscope, iPhone lens layout, Apple vs Samsung island).

## Dashboard

Row 1: live camera with the 60% corner-bracket reticle, status pill and guidance banner (45%) | interactive 3D CAD stage
(55%, `<model-viewer>`, Auto Rotate / Sync Live Pose / Re-Fetch CAD for Active, author + licence badge).
Row 2: four telemetry cards — Host QCS8275 (junction °C, per-core GHz, load), Memory (16 GB LPDDR5 used/total), STM32H5F5
bridge (`set_pixels` RPC ms, LEDs, router socket) and Edge-AI pipeline (NPU runtime, embed/detect ms, fps, calibrated
similarity %, margin %, scene-gate values, active detector domain).
Row 3: training console (name + category, Capture Front 0° → Back 180° → Side 90° → Top) | enrolled classes table
(Name · Category · Views · CAD · Re-Fetch CAD · Delete) and **PURGE DATABASE**.

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
| POST | `/api/ingest`, `/ingest`, `/`, `/upload`, `/api/upload`, `/frame` | frame in, returns at once (async NPU worker, latest-frame-wins) |
| POST | `/api/frame` (or any ingest route with `?sync=1`) | frame in, synchronous classification out |
| GET | `/api/ingest` | accepted ingest routes/formats and live stream counters |
| GET | `/api/stream`, `/api/frame.jpg` | live MJPEG preview / latest frame (404 when the stream is idle; training mode only) |
| GET | `/api/ring`, `/api/ring/frame.jpg?ago=N` | optional RAM frame ring status / rewind (`KAMIKAZE_RING_GB`) |
| GET | `/api/state`, `/api/asset`, `/api/pose` | recognition state, current 3D asset, pose |
| GET | `/api/events` (`?telemetry=1` adds `event: telemetry`) | state SSE (the dashboard's single event connection) |
| GET | `/api/telemetry`, `/api/telemetry/stream` | SoC, thermals, RAM, CPU, NPU, LED bridge, inference (JSON / SSE) |
| POST | `/api/enroll/start` / `capture` / `finish` / `cancel`, GET `/api/enroll/status` | guided enrollment |
| GET | `/api/classes` | enrolled classes (views, phone tier, CAD status and `cad_path`) |
| DELETE | `/api/classes/<id>` | delete a class and its CAD cache (`?delete_cad=0` keeps the model) |
| POST | `/api/classes/<id>/fetch_cad` | force a background CAD re-harvest for one class (202; 404 if not enrolled) |
| POST | `/api/classes/purge` | wipe all classes, delete `enrolled_classes.pt` and the CAD cache (`{"delete_cad": false}` keeps models) |
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