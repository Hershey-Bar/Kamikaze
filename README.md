# Kamikaze

Real-time edge AI: a webcam feed is classified on the **Arduino VENTUNO Q** (Qualcomm Dragonwing QCS8275, Hexagon NPU), the
matching 3D model is fetched and shown in a pose-synchronised `<model-viewer>`, and the **Modulino Pixels** on the STM32H5F5
co-processor show the recognition state.

```
Logitech Brio (USB 3.0 on the board, /dev/video0, V4L2 MJPG 1920x1080 @ 30 fps) --> onboard_camera.py capture thread
   (raw camera JPEG passthrough, no re-encode) --> receiver.py (VENTUNO Q, 0.0.0.0:5000) <-- adb forward tcp:5000 (dashboard)
   receiver.py: capture returns at once; an NPU worker thread always processes the newest frame (older ones are dropped)
   receiver.py: YOLO-World v2-S (Hexagon NPU, vocabulary "beverage can" / "soda can" / "smartphone" / "bottle")
                                                         -> crop the phone/can inside the 75% reticle
                MobileNetV3-Large embedder (Hexagon NPU) -> 1280-d unit vector -> few-shot multi-view prototypes
                DINOv2 ViT-S/14 pose worker (Hexagon NPU, own thread, 10 Hz) -> per-view centroids -> continuous yaw
                pose.py (OpenCV/NumPy + view matching)   -> pitch/yaw/roll for the 3D viewer
                ocr_fusion.py (RapidOCR worker, CPU, ~3 Hz) -> brand/flavour words -> 0.65 vision + 0.35 text
                cad_harvester.py (local library -> phones: 1:1 CAD dummies (Printables / Thingiverse, dimension-gated)
                                  -> Sketchfab + Poly Pizza exact matches -> cans: Open Food Facts packaging photo;
                                  else NOT_FOUND -> browse / upload / 360° label scan / generate from verified specs)
                                                         -> static/models/<class>.glb + attribution
                /var/run/arduino-router.sock (MessagePack-RPC set_pixels) -> STM32 -> Wire1 (I2C4) -> Modulino Pixels
```

## Board setup (once)

```bash
cd ~/hackathon_iot
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
# Hexagon NPU runtime (QNN TFLite delegate + cDSP skeletons)
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-ppa && sudo apt install qairt-libs qairt-dsp-binaries
cp .env.example .env   # add SKETCHFAB_TOKEN, POLY_PIZZA_API_KEY and (optional) THINGIVERSE_TOKEN
./export_yolo.sh       # YOLO-World (domain vocabulary) + YOLO26n + YOLO11n -> models/npu/*_float32.tflite
TORCH_HOME=models/dinov2 ~/yolo_export/venv/bin/python export_dinov2.py  # DINOv2 ViT-S/14 -> models/npu/dinov2_vits14_pose_float32.tflite (pose worker)
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
# board (Brio plugged into the USB 3.0 Type-A port)
./run_receiver.sh                        # = receiver.py --camera auto (first USB UVC camera, e.g. /dev/video0)
./run_receiver.sh --camera /dev/video0   # explicit device; --cam-width/--cam-height/--cam-fps (default 1920/1080/30)
# Windows: only the dashboard
adb forward tcp:5000 tcp:5000            # then open http://localhost:5000
```

Video is captured **on the board**. `onboard_camera.py` opens the camera with OpenCV's V4L2 backend in MJPG mode
(1920x1080 @ 30 fps) with colour conversion disabled, so each read is the camera's own JPEG: the capture thread hands it
unchanged to the pipeline (no decode/re-encode), the NPU worker decodes it once, and the same bytes are served on the
MJPEG preview (native 16:9, full 1080p). Frames stay in RAM only. The startup log prints
`[ONBOARD-CAM] Opened /dev/video0 (Brio 101): 1920x1080 MJPG @ 30 FPS`; if the camera is unplugged or stops delivering
frames it is reopened with back-off. The V4L2 queue keeps its default depth: `CAP_PROP_BUFFERSIZE=1` makes the Brio drop
every other frame (15 fps measured), while the capture thread drains the queue continuously so no lag builds up.

For inference the 1080p JPEG is decoded at half scale (`KAMIKAZE_DECODE_MAX=960`, JPEG DCT scaling, ~6.5 ms instead of
~16 ms), and the 75% center square (810x810 px on 1080p, `[555,135,1365,945]`; `KAMIKAZE_CROP_FRACTION`) is resized to 224x224. Measured on the board with the Brio: capture 29.9 fps,
inference 30.8 fps, 0 dropped frames, 26 ms median per frame (detector + embedder + decode), CPU ~27%. With
`KAMIKAZE_DECODE_MAX=0` (full-res decode) inference falls to ~17 fps (48 ms) and ~40% of frames are skipped.

The dashboard reads the MJPEG stream (`/api/stream`) with `fetch()` and draws each frame on the canvas at the camera's
aspect ratio; if no frame has arrived for `KAMIKAZE_STALE_S` (3 s) the preview is dropped and it shows
*Awaiting on-board camera ...* with the camera status, so a stale or test image is never displayed. State and telemetry
share one SSE connection (`/api/events?telemetry=1`), so several open tabs do not exhaust the browser's per-host
connection limit. Camera device, negotiated mode, capture fps, reconnects and passthrough are in `stream.camera` of
`/api/state` and `/api/telemetry`.

**Legacy Windows streaming (decommissioned).** `stream_cam.py` + the HTTP ingest routes remain only as a fallback for a
board without a USB camera: start `./run_receiver.sh --camera none` (or `auto` with no camera attached), then
`python stream_cam.py` on Windows. While the on-board camera is live every ingest route answers **409** so two sources
never interleave.

### Frame ingest (legacy HTTP fallback, `--camera none`)

Every route below accepts a frame; the payload format is detected from the body, not only from `Content-Type`:

- Asynchronous (reply at once, latest frame wins): `/api/ingest`, `/ingest`, `/`, `/upload`, `/api/upload`, `/frame`
- Synchronous (reply contains the classification): `/api/frame`, or any route with `?sync=1`
- Formats: raw `image/jpeg` bytes; `multipart/form-data` (any field name); JSON `{"image": "<base64 or data URL>"}`
  (keys such as `image`, `frame`, `jpeg`, `img`, `data`, `file`, `b64`); form-urlencoded `frame=<base64>`; bare base64;
  PNG/BMP/WebP/GIF (transcoded to JPEG). Anything else is rejected with HTTP 400 and counted in telemetry (`stream.rejected`).

`GET /api/ingest` returns the accepted routes and the live stream counters. The receiver logs
`Live stream started: <client> POST <path> (<format>, ...)` when frames start and `Live stream idle` when they stop.

### Troubleshooting "Awaiting on-board camera"

1. The receiver log shows `[ONBOARD-CAM] Opened ...` or the open error; `curl -s localhost:5000/api/state` →
   `stream.camera.status` (`streaming`, `reconnecting`, `unavailable`) and `error`.
2. `ls /sys/class/video4linux/` + `cat /sys/class/video4linux/video*/name`: the Brio exposes two nodes (index 0 = video,
   index 1 = metadata); `--camera auto` picks the USB node with index 0. The user must be in the `video` group.
3. Only one process can hold the camera: stop any other capture program on the board.

Legacy `stream_cam.py` path:

1. `stream_cam.py` prints `[STREAM] receiver reachable at ...` at startup, then `[STREAM] POST ... -> HTTP 200` for the
   first 5 frames and every error. *not reachable* means the ADB forward is missing: run `adb forward tcp:5000 tcp:5000`.
   HTTP 409 means the on-board camera owns the stream (restart the receiver with `--camera none`).
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
| YOLO-World v2-S 640 (default) | ~19 ms   | —                            |
| DINOv2 ViT-S/14 224 (pose) | ~20 ms      | ~730 ms                      |
| YOLO26n 640 (fallback)     | ~6–8.5 ms   | ~200 ms                      |
| YOLO11n 640                | ~6.7 ms     | ~166 ms                      |
| EfficientDet-Lite2 int8    | ~3–6 ms     | ~84 ms                       |

End to end per frame (JPEG decode, letterbox, detect, crop, embed) is ~20–30 ms, dominated by image resizing on the CPU.
The default detector is YOLO-World with the domain vocabulary baked in at export (`export_yolo.sh world`), so cans are
reported as `beverage can`, never `cup`. With the COCO fallback (`KAMIKAZE_DETECTOR=yolo26n`) `cup`, and `bottle` boxes with
H/W 1.4–2.6, are relabelled `beverage can`. When nothing is found the center crop is used. A custom-trained YOLO can be
exported with `./export_yolo.sh best.pt` and selected with `KAMIKAZE_DETECTOR=best`.

### Camera focus

`v4l2_ctrl.py` drives V4L2 controls with raw `VIDIOC_QUERYCTRL` / `G_CTRL` / `S_CTRL` ioctls (no `v4l-utils` needed).
For an autofocus webcam, Manual Focus turns continuous AF off and pins `focus_absolute` (clamped and snapped to the
control's step) so AF hunting cannot blur the reticle crop; `KAMIKAZE_FOCUS=auto|<n>` applies it at every camera open.
The **Logitech Brio 101 on the board is fixed-focus**: it exposes no focus controls (only brightness, contrast,
saturation, white balance, gain, sharpness, backlight and exposure), so `/api/camera/focus` honestly reports
`supported: false` with the reason, and the dashboard disables the controls. Keep objects ≥ ~30 cm from it.

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
2. The wizard is category-specific (`taxonomy.WIZARD_VIEWS`) and every step card shows what it must capture:
   - **Smartphone (5 steps)**: Front (screen) → Back (camera island) → Right rail → Left rail → Top (tilt 30–45°). The
     rail prompts name the brand's real buttons.
   - **Beverage can / bottle (4 steps)**: Front (logo) → Back (nutrition panel) → Side (seam / barcode) → Top (pull-tab,
     tilt 30–45°). A can is rotationally near-symmetric, so a fifth "left" view adds no information.
   Each class records how many views its category expects (`view_expected`), shown as `views 4/4` / `5/5` in the
   classes table. For each view, hold the object inside the reticle and click **Capture [view] & Next**. The server samples the next
   10 frames, keeps only the embedding, DINOv2 pose and colour vectors (no images are written to disk) and advances.
3. Click **Done** to save. CAD retrieval runs in the background and never blocks enrollment.

For phones the four cardinal views make the yaw unambiguous over the full 360° (a single "side" view cannot tell left
from right); a can's yaw is resolved by front / side / back.
Databases enrolled with the older 4-view wizard still load: their `side` view is treated as `right` and `detail` as `top`.
Captures are paused by the lighting gate (below); a view whose frames keep failing it for 3 × frames-per-view (min 30)
frames is aborted with *capture failed: ...* so the operator can fix the light and retry.

Tips: plain background, even light, object filling about 60–70% of the reticle. Use the red **Delete** button (or
**Reset all classes**) to remove a bad run; its vectors, `enrolled_classes.pt` entry and cached CAD model are removed at once.

Classes enrolled with another embedder (for example the earlier torchvision backbone) are not mixed: the old database is
kept as `enrolled_classes.<embedder>.bak.pt` and the objects must be re-enrolled.

**PURGE DATABASE** (dashboard, or `POST /api/classes/purge`) deletes every class, removes `enrolled_classes.pt` (and its
`.tmp` / `.bak.pt` siblings) from disk and clears the cached CAD models: a clean slate before a fresh training run.

## Recognition logic

Every frame passes these stages in order (`receiver.py` `Pipeline.process_jpeg` / `_arbitrate`):

0. **Lighting gate.** Mean luma and the fraction of blown-out pixels (≥ 250) of the reticle crop (128 px grayscale,
   < 0.2 ms). Mean < 42 → *Lighting too low. Increase illumination.*; glare fraction > 18% → *Severe glare. Tilt object
   away from direct light.* The frame is not classified (amber guidance, LEDs amber) and not used for enrollment.
   `KAMIKAZE_LIGHT_GATE=0` disables it; `KAMIKAZE_LIGHT_MIN_LUMA` / `KAMIKAZE_GLARE_LEVEL` / `KAMIKAZE_GLARE_MAX` tune it.
1. **Empty-scene gate.** The 75% reticle crop is scored for structure (Laplacian variance and Canny edge density, Canny
   10/30 after a 3×3 blur so dim, soft outlines still register). Only when **both** are low — Laplacian below
   `KAMIKAZE_LAPLACIAN_MIN` (default 8, so smooth metallic cans pass) *and* edge density below `KAMIKAZE_EDGE_MIN` (0.002) — is the frame a blank wall /
   dark scene (a smooth phone has a low Laplacian variance but a clear outline, so a Laplacian-only test would reject it):
   detector and
   embedder are skipped, the state is **SEARCHING** with confidence 0% / margin 0%, the banner reads
   *Awaiting object inside center reticle...* and one cyan LED is lit `(0,100,150,15,1)`. Disable with `KAMIKAZE_SCENE_GATE=0`;
   tune edges with `KAMIKAZE_CANNY_LO` / `KAMIKAZE_CANNY_HI`. Live values are shown in the Edge-AI telemetry card.
2. **Detector = domain gate.** YOLO-World on the NPU only restricts *which classes may match*: `smartphone` ≥ 0.50 →
   smartphones only, `beverage can` / `soda can` / `bottle` ≥ 0.50 → beverage cans/bottles only. If the unrestricted top-1 is from the other domain the frame is
   **GUIDING** with *Verifying object geometry... Hold steady in center.*
3. **Max-over-views, calibrated confidence.** Each class keeps one centroid per enrolled view; its raw score is the
   **maximum** cosine over those views (`S = maxᵥ cos(x, μᵥ)`), so a can seen from its back is not penalised for not
   looking like its front (averaging views pulled every partial view toward ~0.5). `S` is mapped to a confidence by a
   monotone piecewise-linear curve through measured knots (`KAMIKAZE_CAL_KNOTS`, default 0.42 / 0.70 / 0.95):
   `S ≤ 0.42 → 0` (backgrounds and wrong objects measure 0.30–0.45 with MobileNetV3), `0.70 → 0.50`, `≥ 0.95 → 0.98`
   (`KAMIKAZE_CAL_CAP`: never a fake 100%). The gates are exact in raw terms: C = 0.70 at S ≈ 0.80, C = 0.25 at
   S ≈ 0.56. (A logistic `1/(1+e^(−12(S−0.45)))` was rejected: it reports empty backgrounds at 50–70%.) The confidence
   is smoothed with an EMA `C̄ₜ = 0.25·Cₜ + 0.75·C̄ₜ₋₁`. **IDENTIFIED** needs `C ≥ 0.70` *and* a top-1/top-2 margin
   `Δ ≥ 0.15` for 3 consecutive frames (green, LEDs 1–8 proportional to C / 0.98); `0.25 ≤ C < 0.70` or `Δ < 0.15` is
   **GUIDING** (amber `(255,120,0)`, 4 LEDs); `C < 0.25` is **SEARCHING** (dim cyan `(0,100,150)`, 1 LED).
   **Tri-modal fusion** (`ocr_fusion.fuse_calibrated`, top 5 candidates, in calibrated space):

       C_fused = (0.60·C_vision + 0.25·C_text + 0.15·C_color) / (sum of the weights of the modalities present)

   - **Text**: RapidOCR PP-OCRv4 reads the crop in its own worker at ~3 Hz (`KAMIKAZE_OCR_HZ`). The text **detector**
     runs on the Hexagon NPU (HTP, LiteRT + QNN delegate, `ocr_npu.py`, exported by `export_ppocr.sh`: ~17 ms vs ~110 ms
     on CPU); the recogniser stays on ONNX Runtime CPU (variable-width input, no HTP graph). A full read is still
     ~340 ms on 1080p crops, so OCR never runs in the frame path (`KAMIKAZE_OCR_NPU=0` forces the CPU detector). Words then pass a non-intrusive autocorrect (~0.2 ms): OCR glyph swaps (`c0ke → coke`,
     `5prite → sprite`) are kept only when they land on a vocabulary word, and fuzzy matching (`difflib`, ratio ≥
     `KAMIKAZE_OCR_FUZZY` = 0.8) applies only to words of ≥ 5 letters, against the canonical brand vocabulary plus every
     enrolled class's words. Short words are never fuzzed (`cake` stays `cake`). `C_text` is 1.0 for brand + detail,
     0.6 for brand only, 0.4 for detail only and 0.0 when another brand is read; it is absent when the text says
     nothing about the class.
   - **Colour** (`color_sig.py`, ~0.3 ms): a 21-float signature of the crop (18 saturation-weighted, soft-binned hue
     bins + dark / grey / bright) stored per view at enrollment (vectors only, no images). `C_color` is the best-view
     histogram intersection mapped from [0.30, 0.70] to [0, 1] (`KAMIKAZE_COLOR_I_LO/HI`). Coke red, Pepsi blue, Sprite
     green and a pink Kirkland can separate even when the embedding is unsure.
   - A missing modality drops out and the weights renormalise. Below the SEARCHING floor the vision score is returned
     unchanged, so text and colour can confirm or veto an object the camera sees but never conjure one out of
     background print. `KAMIKAZE_OCR=0` disables OCR.
4. **Smartphones.** Matching is the same max over views (the optional per-view reliability prior in
   `device_taxonomy.py`, which favours the rear shell, is off by default: `VIEW_RELIABILITY_KAPPA = 0`). If the live frame matches a phone's screen better than its back,
   confidence is capped at 35%, LEDs go amber and the banner says *Front screen detected. Flip phone 180° to expose camera
   island.* When two enrolled phones are within Δ 0.15, the tier taxonomy picks the cue that separates them
   (Pro LiDAR aperture, Ultra boxy corners + periscope, iPhone lens layout, Apple vs Samsung island).
5. **Pose.** A separate pose worker thread runs DINOv2 ViT-S/14 on the NPU (rate-capped at `KAMIKAZE_POSE_HZ`, 10 Hz) on
   the identified crop. During enrollment each view also stores a 768-d DINOv2 centroid ([CLS | mean patch]); live, the
   descriptor is compared with the view centroids after subtracting their mean (the shared object identity cancels, so
   neighbouring views separate), the similarities are soft-maxed and blended on the yaw circle (front 0°, right +90°,
   back 180°, left −90°): a continuous, unambiguous 360° yaw. Tilting the top toward the camera is positive pitch; the
   dashboard drives model-viewer's `orientation` (`roll pitch yaw`) with matching signs. Pitch comes from the Top-view weight plus the perspective taper of the object outline. Classes enrolled before the pose worker existed fall back to MobileNet view centroids
   (`pose.source` = `mobilenet-views` instead of `dinov2-npu`); re-enroll them for DINOv2 pose. `KAMIKAZE_POSE_DINO=0`
   disables the worker.
6. **Bifurcated routing** (`Pipeline._fuse`; the telemetry `route` field and the dashboard route chips show which
   branch decided). Pipeline: `raw frame -> YOLO-World NPU -> ROI crop (15% pad) -> routing switch -> label / confidence /
   bbox / yaw-pitch -> Modulino Pixels`.
   - **Smartphone branch**: the vision score becomes an ensemble `C = (1-w)·C_mobilenet + w·C_dino` (`KAMIKAZE_DINO_WEIGHT`,
     0.5). `C_dino` is the **maximum** raw cosine of the live DINOv2 descriptor over each class's per-view DINOv2
     centroids (rear camera island and screen are separate prototypes, never averaged), mapped through
     `KAMIKAZE_DINO_KNOTS` (0.45 → 0, 0.78 → 0.80, 0.92 → 0.95 cap). The knots are starting values, not yet measured on
     real phones; tune them after the first phone enrollments. The front-screen 35% cap still applies.
   - **Beverage branch**: an OCR keyword lexicon per class, learned at enrollment (metadata words + tokens read in ≥ 2
     OCR passes during the wizard, never another brand's words), matched against a rolling buffer of the last 5 OCR
     passes (≤ 3 s; `KAMIKAZE_OCR_ROLL`, `KAMIKAZE_OCR_ROLL_S`) so a word lost to cylindrical glare in one pass survives
     from its neighbours. A class is confirmed when a **discriminative** word (in no other enrolled can's lexicon, e.g.
     *lime* vs *lemon* vs *grapefruit* on otherwise identical Kirkland cans) plus a brand word or a second lexicon word
     were read; it then gets `C_fused ≥ 0.85` (+0.03 per extra word, cap 0.95). A competitor brand read vetoes, two
     flavours read is a tie (no override), and the route needs a detected can or vision ≥ the SEARCHING floor.

## Dashboard

The header carries an **LED HUD**: eight dots mirroring exactly what was last sent to the Modulino Pixels
(`set_pixels(r, g, b, brightness, count)`, colour and count) with the mode: `IDENTIFIED (6/8)`, `GUIDING (4/8)`,
`SEARCHING (1/8)`, `CAPTURING (n/8)` during enrollment, or a dimmed `BRIDGE DOWN` when the router RPC fails. Its
tooltip shows the RPC round-trip latency.

Under the camera, chips show the fused score (`label · fused % = V / T / C`), the dominant-colour swatch, the OCR words
and corrected words (~~c0ke~~ coke), and a **Focus** row (Auto / Manual + slider; disabled with the reason when the
camera has no focus control). Above the 3D stage, the **Online / Offline** toggle and source selector (online: Auto,
Sketchfab, Poly Pizza, Local Cache, Custom Scan Upload; offline: Local Cache and Custom Scan Upload only). Under the
stage, the **model frame** toolbar (Rotate X / Y / Z 90°, Auto-orient, Reset) fixes a downloaded model whose author
used another up-axis or front (see 3D models).

Two columns. **Left**, top to bottom: live camera (16:9) with the 75% corner-bracket reticle, status pill and guidance
banner (plus a lighting pill), the fusion chips and focus row; the category-specific training console (name + category,
4 or 5 step cards); a 2×2 grid
of telemetry cards — Host QCS8275 (junction °C, per-core GHz, load), Memory (16 GB LPDDR5 used/total), STM32H5F5 bridge
(`set_pixels` RPC ms, LEDs, router socket) and Edge-AI pipeline (NPU runtime, embed/detect ms, DINOv2 pose ms, fps,
calibrated similarity %, margin %, scene-gate values, active detector domain, OCR words read and the fused score).
**Right** (sticky): the interactive 3D CAD stage (`<model-viewer>`, ≥ 620 px tall, `environment-image="neutral"`,
`exposure="1.0"`, `shadow-intensity="1.2"`, `tone-mapping="aces"`; Auto Rotate / **Sync Physical Pose** / Re-Fetch CAD
for Active; author + licence badge), then the enrolled classes table (Name · Category · Views · CAD · Re-Fetch CAD ·
Delete) and **PURGE DATABASE**. Sync Physical Pose drives `camera-orbit` from the live yaw/pitch; the pill shows the
pose and its source. A class without a verified model shows an empty stage with a **3D MODEL NOT FOUND** overlay and
the reason, plus three explicit choices: **Browse options** (top ranked online candidates with thumbnail, author,
licence and score; one click downloads, validates and attaches the chosen one), **Upload GLB** (validated like any
download, credited as a user upload) and **Generate from specs** (below). Nothing is ever substituted automatically.

## 3D models and attribution

**Policy: no made-up models.** The stage only ever shows a model that is a verified match for the class, a model the
operator picked or uploaded, or a model generated from verified published dimensions (and labelled as such). Otherwise
the class is **NOT_FOUND**.

`cad_harvester.build_search_queries()` turns plain text into a cascade of queries, most specific first
(`s25 ultra` -> `Samsung Galaxy S25 Ultra`, `Samsung S25 Ultra`, `Galaxy S25 Ultra`, ...). The automatic harvest sends
only the **exact** level to Sketchfab (downloadable, non-NC licences) and Poly Pizza (`X-Auth-Token`); results are
pooled and ranked by

    RankScore = relevance(title, level) + popularity
    popularity = min(25, 2.5·ln(1 + likes) + 0.75·ln(1 + views) + 4·staff_pick)     (Sketchfab)
    popularity = 6 − 0.5·rank position                                               (Poly Pizza: no counts exposed)

Neither API returns download counts, and raw `likes × 3` would let a famous but wrong model beat an exact title match,
so popularity is log-scaled and capped. Relevance rejects any title, tag or description naming another brand or drink
type (a Kirkland sparkling water never becomes a Coca-Cola) and penalises another flavour or phone tier. The similar
and category levels (`Grapefruit Seltzer Can`, `Soda Can`, `Smartphone`) are only offered through **Browse options**.

Every download passes a **quality gate** (`validate_glb`): valid GLB 2.0 container, ≥ 200 unique vertices (rejects
boxes and placeholder primitives), ≤ 350k vertices, ≤ 35 MB, at least one material. Rejected files are deleted and the
next candidate is tried. A GLB without its attribution sidecar is never shown.

**Manufacturing and packaging sources.** Game-art sites rarely carry an exact iPhone 15 Pro chassis or a niche can
flavour, so the automatic harvest tries the sources where those assets actually live, in this order:

| Step | Phones | Cans |
|---|---|---|
| 1 | local model library (owner scans) | local model library |
| 2 | **1:1 CAD dummies** (`cad_dummies.py`): Printables GraphQL (no key) and Thingiverse (`THINGIVERSE_TOKEN`) | Sketchfab / Poly Pizza exact match |
| 3 | Sketchfab / Poly Pizza exact match | **Open Food Facts packaging photo** (`packaging.py`, no key) |

*Dummies* are the 1:1 replicas case makers print. Candidates are ranked by title (dummy / 1:1 words, exact tier) and
popularity. Licences must allow redistribution (Standard Digital File and NC are refused; `KAMIKAZE_CAD_ALLOW_NC=1`
admits NC for private demos), and ND files are never simplified. Every file must name this exact variant: a
`iphone_15_pro.stl` inside a *15 Pro / Pro Max* print is rejected, as is an unnamed file in a multi-variant print.
STL / OBJ / 3MF files are loaded (3MF XML is parsed with entities refused). They are oriented by their own axes or by
surface PCA, whichever bounds tighter, and must pass a **dimension gate**: height and width within the tolerance of the
GSMArena numbers, thickness between the body and body + camera bump, with mm / cm / m / inch auto-detected. The camera
bump goes to the back, top. The result is decimated (vertex clustering, ≤ 120k triangles) and written as a GLB in
metres. Live check: Printables *iPhone 15 Pro Max model* by jack@meow (public domain) passes at 159.5 × 77.7 mm.

*Packaging photos*: Open Food Facts is searched by brand + flavour. A hit must match brand, flavour, tier (zero /
diet), drink type and can volume, and it must have a front photo. The photo is segmented (GrabCut), checked to be an
upright side view whose aspect matches the Crown can, and unwrapped per row onto the visible arc of the cylinder.
The unseen back is filled with the label's edge colour and listed as an approximation; images are CC BY-SA 3.0,
credited to *Open Food Facts contributors* with the product link. Cans keep game art first because one photo
covers only about 140° of the label.

**360° label scan** (`label_stitch.py`, dashboard **Scan label 360°**, `POST /api/label_scan`). This captures
the real label of the can in your hand, including special editions and local variants. Before the scan starts, the
class must resolve to verified Crown dimensions, so its name needs a volume (`kirkland lime 355ml`). While the scan
runs, each frame's detected can is unwrapped and its low-glare central band is registered by phase correlation. The
band is then blended into a cylindrical texture. The Modulino shows amber, with the number of LEDs equal to the
coverage in eighths; it turns solid green at 98.5 %. **Finish & apply** wraps the stitched texture on the
manufacturer can dimensions and installs it, credited to the operator's scan. Frames and strips stay in RAM only, and
an abandoned scan expires after 180 s.

**Cited phone layouts.** `generate_spec` (and the MCP tool `assemble_phone_mesh`) accept
`{"layout": {"source": ..., "island": {...}, "lenses": [{"x", "y", "r", "h"}]}}`. The coordinates are in mm, measured
on the back from its top-left corner. Use them, for example, for lens centres from Apple's *Accessory Design
Guidelines* drawings. A layout without a `source`, or one that lies outside the published body, is refused, and the
source is added to the credit line. Without a layout, the generic per-maker arrangement is used and flagged as an
approximation.

**Generate from specs** (`spec_search.py` + `procedural_glb.py`) builds a dimension-calibrated GLB only from published
numbers: smartphones from the GSMArena spec page (height × width × depth in mm; Apple and Samsung), cans from Crown's
standard can table (the class must state its volume, e.g. `330ml`, snapped to a Crown size within 3%; 330 ml standard
= Ø 66 mm × 115.2 mm, neck 54 mm). Corner radius, bezels, lens layout and can neck shape are approximations and are
listed in the sidecar; the model is neutral (no brand graphics) and the stage shows the badge *Estimated Model:
Dimensionally Calibrated via Spec Search*. If no verified spec is found it stays NOT_FOUND.

**Offline mode is enforced, not just a preference.** `netguard.py` is a process-wide guard checked before every
external request (`mcp_server` Sketchfab / Poly Pizza / downloads, `spec_search`, model-viewer vendoring): while offline
any non-loopback host raises `NetworkBlocked` and is counted (`/api/settings/mode` → `network.blocked`, shown in the CAD
pill), so the receiver makes zero outbound requests. Offline, only the `local` and `upload` sources can be selected;
going online restores the last online source. A class with no cached model is **OFFLINE_MISSING** (not NOT_FOUND: a
model may exist online) and the dashboard asks *Object has no local CAD model. Switch to Online Search mode to query
repositories?* with **Switch to Online & Fetch**. The one-time NPU model download at first start (`npu.py`) is
provisioning, not retrieval, and is not guarded. Browse and spec generation need online mode.

**Model frame (orientation).** Repository models come in arbitrary frames (Z-up CAD exports, phones lying flat, cans on
their side), which breaks pose sync. `glb_orient.py` reads the GLB's node bounds and, per category, rotates the model so
+Y is up and the front faces +Z (a can's long axis up; a phone's thinnest axis toward the camera and its longest up).
The fix is a single wrapper node with a quaternion (mesh data untouched, idempotent, reversible with **Reset**), applied
automatically after each download and adjustable with the toolbar (`POST /api/classes/<id>/cad_orient`). Whether a
phone's screen or back faces +Z cannot be inferred from bounds; press **Rotate Y 90°** twice if it is reversed. A
rotated CC-BY model is a modification, so the sidecar records `modified` + `orientation_fix` and the credit line says
*(re-oriented)*.

Every model has a JSON sidecar with title, author, author URL, licence and source; credits are shown in the
dashboard and written to `static/models/CREDITS.md`. Third-party models, runtimes and libraries are credited in [ATTRIBUTION.md](ATTRIBUTION.md)
(also served at `/ATTRIBUTION.md` and linked from the dashboard footer).

**Local model library** (`model_library.py`, `static/models/library/`, git-ignored: owner scans are never committed).
Hand scans or other owned GLBs are added from the dashboard **Model library** panel (or `POST /api/models/library`),
pass the same quality gate, are auto-oriented and credited to their owner (*3D hand scan by Hershey-Bar*). Each entry
has a match key (category + brand first word + flavour / phone model), so a newly enrolled `kirkland lime sparkling
water` class binds to the lime scan **before any online search and also in offline mode**; an explicit Sketchfab /
Poly Pizza pick skips the library. The panel lists every library model and per-class cached model with **View**
(preview on the stage), **Use for class** and a trash-can **Delete** (deleting a library model also drops the
per-class copies made from it). Scans keep their scanned scale; model-viewer frames them automatically.

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
| POST | `/api/resolve`, `/api/cad/fetch` | plain-text resolution, CAD retrieval (`status: not_found` + `reason` when nothing verified exists) |
| GET, POST | `/api/settings/mode` | `{"mode": "online"\|"offline", "source": "auto"\|"sketchfab"\|"polypizza"\|"local"\|"upload"}`; returns the `sources` allowed in this mode and `network` guard stats (400 for an online source while offline) |
| POST | `/api/classes/<id>/cad_orient` | `{"auto": true}` \| `{"axis": "x"\|"y"\|"z", "deg": 90}` \| `{"reset": true}`: re-orient the cached GLB (404 without one) |
| GET, POST | `/api/camera/focus` | on-board camera focus `{"auto": true}` \| `{"auto": false, "value": n}`; 409 for HTTP ingest, 422 + status when the camera has no focus control |
| GET | `/api/classes/<id>/cad_candidates?source=&limit=4` | ranked online candidates, nothing downloaded (409 offline, 404 not enrolled) |
| POST | `/api/classes/<id>/cad_select` `{"uid": ...}` | download + validate + attach a browsed candidate |
| POST | `/api/classes/<id>/upload_cad?filename=x.glb` | raw GLB body (≤ `KAMIKAZE_CAD_UPLOAD_MB`, 35); 422 when it fails the quality gate |
| POST | `/api/classes/<id>/generate_spec` | dimension-calibrated GLB from verified specs, optional cited `{"layout": ...}` for phones (404 + reason when no spec is found) |
| GET, POST | `/api/label_scan` `{"action": "start"\|"status"\|"cancel"\|"finish", "label": ...}` | 360° can label scan (400 not a can / no verified volume, 404 not enrolled, 409 incomplete / offline / no scan) |
| GET | `/api/label_scan/texture.jpg` | stitched label preview (404 when no scan is running) |
| GET | `/api/credits`, `/api/db/export`; POST `/api/db/save` | attribution, database export/save |
| GET | `/api/models` | local model library + per-class cached models |
| POST | `/api/models/library?name=&category=&filename=&replace=1` | raw GLB body → quality gate → library (201; 409 exists; 422 rejected) |
| POST | `/api/models/library/<id>/attach` `{"label": ...}` | use a library model as a class's CAD |
| DELETE | `/api/models/library/<id>`, `/api/models/cache/<class>` | trash a library model / a cached per-class model |
| GET | `/ATTRIBUTION.md` | third-party credits |

`mcp_server.py` exposes the same capabilities as FastMCP tools over stdio (`search_3d_asset`, `actuate_modulino`,
`trigger_guidance`, `enroll_multiview`, `fetch_cad_asset`, `resolve_product`, `get_telemetry`, `get_pose_telemetry`, ...),
plus the manufacturing / packaging tools `search_cad_dummies` (rank, and with `install=true` harvest, a 1:1 phone dummy),
`get_product_packaging` (Open Food Facts hits, `install=true` builds the textured can), `stitch_camera_texture`
(drives the live 360° label scan on the running receiver) and `assemble_phone_mesh` (verified-spec phone, optional
cited `layout_json`).

## Tests

```bash
.venv/bin/python -W ignore -m unittest test_kamikaze test_comprehensive test_full_suite test_cad_sources
```

Hardware tests (LED RPC, Hexagon NPU, online CAD) skip themselves when the router socket, the NPU or credentials are absent.