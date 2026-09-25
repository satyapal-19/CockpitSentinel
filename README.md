<h1 align="center">
  🚗 CockpitSentinel
</h1>

<p align="center">
  <strong>AI-Based Driver Monitoring System Using Computer Vision</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#team">Team</a>
</p>

---

## Features

| Module                  | What It Detects                    | How                                                           |
| ----------------------- | ---------------------------------- | ------------------------------------------------------------- |
| **Drowsiness**    | Eye closure, yawning, looking away | MediaPipe Face Landmarker → EAR, MAR, head yaw               |
| **Distraction**   | Phone use, smoking                 | YOLOv8 + YOLOWorld object detection                           |
| **Risk Scoring**  | 4-level risk assessment            | Weighted signal fusion → SAFE / CAUTION / WARNING / CRITICAL |
| **Stabilization** | Temporal window (2–3s analysis)   | Multi-second window prevents blink/flicker alarms; clears instantly on attentive return |

## Architecture

```
Camera / Video
      │
      ▼
  OpenCV Frame Capture
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
  DrowsinessDetector                DistractionDetector
  (MediaPipe Face Landmarker)       (YOLOv8 + YOLOWorld)
      │                                  │
      │  eyes_closed                     │  phone_detected
      │  yawning                         │  smoking_detected
      │  looking_away                    │
      ▼                                  ▼
  SignalStabilizer ──────► Signal Merge ◄──┘
                               │
                               ▼
                         AlertPolicy
                      (weighted scoring)
                               │
                               ▼
                   ┌───────────┴───────────┐
                   │    Risk Assessment    │
                   │  SAFE < CAUTION <     │
                   │  WARNING < CRITICAL   │
                   └───────────┬───────────┘
                               │
                               ▼
                     OpenCV Live Overlay
```

## Quick Start

> **Requires:** Python 3.11 or newer · Git · Webcam (for live monitoring)

### Windows

```powershell
git clone https://github.com/<your-org>/CockpitSentinel.git
cd CockpitSentinel

# Bootstrap (creates venv, installs deps, downloads models)
.\scripts\setup_environment.ps1

# Activate and verify
.\.venv\Scripts\Activate.ps1
python scripts\verify_environment.py

# Run tests
pytest

# Launch the monitor
cockpit-sentinel
```

### macOS / Linux

```bash
git clone https://github.com/<your-org>/CockpitSentinel.git
cd CockpitSentinel

# Bootstrap
bash scripts/setup.sh

# Activate and verify
source .venv/bin/activate
python scripts/verify_environment.py

# Run tests
pytest

# Launch the monitor
cockpit-sentinel
```

The setup script will:

- Create a `.venv` virtual environment using your system Python
- Copy `.env.example` → `.env` (if `.env` doesn't exist yet)
- Install all dependencies
- Download pretrained models (YOLOv8, YOLOWorld, MediaPipe Face Landmarker)
- Set up the data directory structure

**Everything stays inside the repo directory by default.** No external drives or paths needed.

## Usage

### Run the Live Monitor

```bash
# Using the installed command (recommended)
cockpit-sentinel

# Or with python explicitly
python scripts/run_drowsiness_monitor.py

# From a video file instead of webcam
cockpit-sentinel --source path/to/video.mp4

# Custom webcam index
cockpit-sentinel --source 1
```

> **⚠️ Windows users:** Do NOT run `.py` files directly (e.g., `.\scripts\run_drowsiness_monitor.py`).
> This may open your IDE instead of executing the script.
> Always use `python scripts\...` or the `cockpit-sentinel` command.

Press **Q** or **Esc** to stop the monitor.

### What You'll See

The monitor displays a live video feed with an interactive dashboard overlay:

- **Dashboard HUD**: Exact **Total Trip Time** (`TRIP: HH:MM:SS`), **Attention Index** (`XX.X% ATTENTIVE`), and real-time **FPS**.
- **Distraction Bounding Boxes**: Clear labeled bounding boxes with confidence tags around detected cell phones (cyan) and smoking items (orange).
- **Risk Level & Alarm Perimeter**: Color-coded status badge and perimeter alarm glow (Orange for Warning, Red for Critical) that clears immediately upon returning to attentive driving.
- **Telemetry Bar**: Real-time EAR, MAR, head yaw angle, and active duration counters for each distraction signal.

## Configuration

All runtime settings live in [`configs/`](configs/):

| File                                            | Controls                                                |
| ----------------------------------------------- | ------------------------------------------------------- |
| [`drowsiness.yaml`](configs/drowsiness.yaml)   | EAR/MAR thresholds, head yaw limit, frame stabilization |
| [`distraction.yaml`](configs/distraction.yaml) | Phone/smoking confidence thresholds, smoking labels     |
| [`alerts.yaml`](configs/alerts.yaml)           | Signal weights, risk levels, temporal window duration   |
| [`models.yaml`](configs/models.yaml)           | Model paths and registry                                |
| [`logging.yaml`](configs/logging.yaml)         | Log format, rotation, and output                        |

### Risk Scoring & Temporal Analysis

Alerts are evaluated over a **2.0 to 3.0 second temporal window** (configurable via `--time-window` or `alerts.yaml`). Transient eye blinks (<0.4s) or quick glances (<1.0s) will not trigger false alarms.

| Signal           | Weight | Threshold         | Temporal Confirmation |
| ---------------- | ------ | ----------------- | --------------------- |
| Eyes closed      | 3      | EAR < 0.22        | ≥ 2.0s sustained      |
| Yawning          | 2      | MAR > 0.60        | ≥ 2.5s sustained      |
| Looking away     | 2      | Head yaw > 30°   | ≥ 2.0s sustained      |
| Phone detected   | 4      | Confidence > 0.40 | ≥ 2.0s sustained      |
| Smoking detected | 3      | Confidence > 0.35 | ≥ 2.0s sustained      |

**Risk levels:** SAFE (< 2) · CAUTION (≥ 2) · WARNING (≥ 4) · CRITICAL (≥ 6)

## Data & Models

By default, all data and models are stored inside the repo directory (git-ignored):

```
CockpitSentinel/
├── data/                  # Datasets (git-ignored)
│   ├── external/          #   Downloaded datasets (NTHU, YawDD, State Farm, etc.)
│   ├── processed/         #   Feature extractions and train/test splits
│   └── raw/
├── models/                # Model weights (git-ignored)
│   ├── pretrained/        #   YOLOv8, YOLOWorld, Face Landmarker
│   └── trained/           #   Project-specific trained models
└── experiments/           # MLflow tracking (git-ignored)
    └── mlruns/
```

To use a custom storage location, edit `.env`:

```ini
COCKPIT_DATA_ROOT=/path/to/your/storage
COCKPIT_MODELS_ROOT=/path/to/your/storage/models
MLFLOW_TRACKING_URI=file:///path/to/your/storage/experiments/mlruns
```

Leave these empty to use the repo directory (the default).

## Project Structure

```
CockpitSentinel/
├── src/cockpit_sentinel/         # Main package
│   ├── alerts/                   #   Alert policy and scoring
│   ├── detection/                #   Phone & smoking detection (YOLO)
│   ├── drowsiness/               #   Eye/mouth/head analysis (MediaPipe)
│   ├── pipeline/                 #   Live monitor and video processing
│   ├── utils/                    #   Shared helpers
│   └── domain.py                 #   Core data contracts and risk logic
├── configs/                      # YAML configuration files
├── scripts/                      # Setup, verification, and runner scripts
│   ├── setup_environment.ps1     #   Windows bootstrap
│   ├── setup.sh                  #   macOS / Linux bootstrap
│   ├── setup_environment.py      #   Cross-platform setup logic
│   └── verify_environment.py     #   Environment validator
├── tests/
│   ├── unit/                     #   17 unit tests
│   └── integration/              #   Integration tests (planned)
├── notebooks/                    # Jupyter notebooks for experiments
├── docs/                         # Setup guides and architecture decisions
├── .github/workflows/ci.yml      # GitHub Actions CI (multi-OS, multi-Python)
├── pyproject.toml                # Project metadata and tool config
├── requirements.in               # Direct dependencies
└── requirements.txt              # Pinned dependencies (pip-compile)
```

## Environment Variables

The `.env` file is optional. If unset, everything defaults to the repo directory.

| Variable                | Purpose                                 | Default                                 |
| ----------------------- | --------------------------------------- | --------------------------------------- |
| `COCKPIT_DATA_ROOT`   | Root directory for datasets and storage | _(repo directory)_                    |
| `COCKPIT_MODELS_ROOT` | Root directory for model files          | _(repo directory)/models_             |
| `MLFLOW_TRACKING_URI` | MLflow experiment store                 | _(repo directory)/experiments/mlruns_ |
| `LOG_LEVEL`           | Application log level                   | `INFO`                                |
| `WEBCAM_INDEX`        | Default webcam device                   | `0`                                   |
| `KAGGLE_USERNAME`     | Kaggle API username                     | _(empty)_                             |
| `KAGGLE_KEY`          | Kaggle API key                          | _(empty)_                             |

## NVIDIA GPU Setup

For machines with an NVIDIA GPU:

```powershell
.\scripts\setup_nvidia_environment.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu121
```

See [`docs/setup/team-environment.md`](docs/setup/team-environment.md) for details.

## Team

**Walchand College of Engineering, Sangli** — Department of Computer Science & Engineering

Mini Project 2026–27

- **Satyapal Gaikwad**
- **Darshan Patil**
- **Yashodhan Jadhav**

---

<p align="center">
  <sub>Do not commit datasets, model weights, virtual environments, logs, or <code>.env</code> files.</sub>
</p>
