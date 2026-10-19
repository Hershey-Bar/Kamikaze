# Contributing to Kamikaze

Thanks for hacking with us! This guide covers how to get set up on **Windows, Mac, or Linux** — both for local development (no board needed) and for live testing on the VENTUNO Q.

---

## Repo Layout

```
.
├── receiver.py          ← Board receiver: HTTP server + inference loop (runs on VENTUNO Q)
├── mcp_server.py        ← MCP tool server (runs on VENTUNO Q)
├── run_receiver.sh      ← Start/restart receiver on the board
├── requirements.txt     ← Board (aarch64) Python dependencies
├── requirements-local.txt  ← Local dev dependencies (any OS)
│
├── kamikaze/            ← Core Python package (inference, CAD, NPU, OCR, taxonomy…)
├── tools/               ← Client-side tools (stream_cam.py, local_test_server.py, replay_frames.py)
├── firmware/            ← Arduino sketch for the STM32H5 co-processor (modulino_bridge.ino)
├── scripts/             ← Export scripts (YOLO, DINOv2, PPOcr) and board setup helpers
├── tests/               ← Pytest test suite
├── models/              ← Model files (gitignored: .pte, npu/ binaries)
├── deployments/         ← Frozen dependency manifests + board deployment snapshots
├── static/              ← Web dashboard (index.html, GLB models — most gitignored)
└── iot_artifacts/       ← Reference guide + msgpack RPC test helpers from qc-executorch-guide
```

---

## Local Development (No Board Required)

This path works on **Windows, Mac, and Linux** and lets you test the full streamer → inference → LED pipeline without touching a VENTUNO Q.

### 1. Prerequisites

| Tool | Windows | Mac | Linux |
|------|---------|-----|-------|
| Git | `winget install Git.Git` | `xcode-select --install` | `sudo apt install git` |
| Python 3.12 | `winget install Python.Python.3.12` | `brew install python@3.12` | `sudo apt install python3.12` |
| GitHub CLI (optional, easiest auth) | `winget install GitHub.cli` | `brew install gh` | `sudo apt install gh` |

> **Mac webcam permission:** The first time you run `stream_cam.py`, macOS will prompt for Camera access.  
> Allow it in **System Settings → Privacy & Security → Camera**.

### 2. Clone & set up

```bash
# Accept the GitHub collaborator invitation first, then:
git clone https://github.com/Hershey-Bar/Kamikaze.git
cd Kamikaze

# Create a virtual environment
python3 -m venv venv

# Activate it:
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Mac / Linux:
source venv/bin/activate

# Install local-dev dependencies (no ARM/NPU board wheels)
pip install -r requirements-local.txt

# Copy and fill in the environment template
cp .env.example .env
# Edit .env — you only need SKETCHFAB_TOKEN for local CAD preview; other keys are optional.
```

### 3. Run the local test pipeline

**Terminal 1 — mock receiver:**
```bash
python tools/local_test_server.py
```
You'll see an ASCII LED bar in the terminal simulating what the Modulino Pixels would show on the board.

**Terminal 2 — webcam streamer:**
```bash
python tools/stream_cam.py --url http://127.0.0.1:5000
```
> No `adb forward` needed for local testing — the server runs on your own machine.

**Run the protocol test suite:**
```bash
pytest tests/
```

### 4. Work in your own branch

```bash
git checkout -b dev/<your-name>

# … make changes …

git add .
git commit -m "feat: describe what you did"
git push -u origin dev/<your-name>
```

Then open a Pull Request on GitHub to merge into `main`.

---

## Live Board Testing (VENTUNO Q)

When you're ready to test against the real hardware:

### Windows — ADB setup (one-time)
```powershell
# ADB is already installed via winget; add to PATH if needed:
winget install Google.PlatformTools
# Restart PowerShell, then plug in the board via USB-C and run:
adb devices   # should list 1103915714 or similar
```

### Forward the port and stream
```powershell
adb forward tcp:5000 tcp:5000
python tools/stream_cam.py   # defaults to --url http://127.0.0.1:5000/api/ingest
```

### Flash firmware changes
```bash
# Stop the router, flash, restart (run on the board via adb shell or SSH):
sudo systemctl stop arduino-router
arduino-cli compile --fqbn arduino:zephyr:ventunoq ~/modulino_bridge
arduino-cli upload  --fqbn arduino:zephyr:ventunoq ~/modulino_bridge
sudo systemctl start arduino-router
```
The same commands are in [`firmware/flash.sh`](firmware/flash.sh).

---

## Code style

- Python: no strict linter enforced yet — keep it readable, avoid bare `except:`.
- Arduino/C++: follow the existing style in [`firmware/modulino_bridge.ino`](firmware/modulino_bridge.ino).
- Commit messages: `type: short description` (e.g. `fix:`, `feat:`, `refactor:`, `docs:`).
