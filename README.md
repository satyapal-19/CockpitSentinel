<h1 align="center">
  🚗 CockpitSentinel
</h1>

<p align="center">
  <strong>AI-Based Driver Monitoring System Using Computer Vision</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#in-cabin-face-recognition">Face ID</a> •
  <a href="#dashboard--telemetry">Dashboard</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#team">Team</a>
</p>

---

## Features

| Module                       | What It Detects / Provides          | How                                                                                         |
| ---------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------- |
| **Face Recognition**   | In-cabin driver auto-identification | **FaceNet (InceptionResnetV1)** + MTCNN / Haar alignment → auto-loads driver profile |
| **Drowsiness**         | Eye closure, yawning, looking away  | MediaPipe Face Landmarker → EAR, MAR, head yaw                                             |
| **Distraction**        | Phone use, smoking                  | Fine-tuned **YOLO11s** balanced detector (95.05% phone precision, 96.89% smoke mAP50) + OpenVINO (32.6 FPS) / TensorRT |
| **Risk Scoring**       | 4-level risk assessment             | Weighted signal fusion → SAFE / CAUTION / WARNING / CRITICAL                               |
| **Stabilization**      | Temporal window (2–3s analysis)    | Multi-second window prevents blink/flicker alarms; clears instantly on attentive return     |
| **Web Dashboard**      | Fleet UI & calibration sliders      | FastAPI responsive cockpit dashboard with Chart.js analytics & face biometrics modal        |
| **Driver Profiles**    | Personalized detection thresholds   | Custom EAR, MAR, Yaw, Phone/Smoking conf stored with 512-D face embeddings                  |
| **Route Management**   | Expressways & corridor tracking     | Distance, duration, risk profile, and assigned driver pilots                                |
| **Trip Telemetry**     | Session logs & attention scoring    | Tracks duration, % attentive, frame telemetry, safety rating                                |
| **Alert History**      | Incident audit log feed             | Timestamped alerts with signals, duration, and severity level                               |
| **Safety Leaderboard** | Driver ratings & ranking            | 1–5 star ratings, 0–100 safety score, Gold/Silver badges                                  |
| **Hybrid Database**    | Cloud sync with offline fallback    | **MongoDB Atlas** with seamless local **SQLite** fallback                       |

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
  (MediaPipe Face Landmarker)       (Unified YOLO11s / OpenVINO)
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
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
 OpenCV Live Overlay HUD                Database & Web Dashboard
 (Real-time in-cabin display)           (FastAPI + MongoDB / SQLite)
                                        ├── Driver Profiles & Thresholds
                                        ├── Route Corridor Tracking
                                        ├── Trip Session Telemetry
                                        ├── Alert Incident Audit Feed
                                        └── Driver Safety Leaderboard
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

```powershell
# 1-Click Launch (Auto-detects OpenVINO / YOLO11s distraction model)
.\scripts\run_webcam_monitor.ps1

# Launch with Driver Profile & Route (loads personalized thresholds & logs trip)
.\scripts\run_webcam_monitor.ps1 -DriverId "drv-satyapal-001" -RouteId "rt-pune-mum-exp"

# Launch with Balanced Distraction Model (95.05% phone precision, 32.6 FPS on CPU)
cockpit-sentinel --source 0 --distraction-model weights/trained/yolo11s_balanced_distraction/best_openvino_model

# Or with PyTorch checkpoint
cockpit-sentinel --source 0 --distraction-model weights/trained/yolo11s_balanced_distraction/best.pt

# From a video file instead of webcam
cockpit-sentinel --source path/to/video.mp4

# Run with demo YawDD video
.\scripts\run_viva_demo.ps1
```

> **⚠️ Windows users:** Do NOT run `.py` files directly (e.g., `.\scripts\run_drowsiness_monitor.py`).
> Always use `python scripts\...` or the PowerShell launcher `.\scripts\run_webcam_monitor.ps1`.

Press **Q** or **Esc** to stop the monitor.

---

### Dashboard & Telemetry

CockpitSentinel includes a full-featured web dashboard with persistent database storage, customizable driver sensitivity thresholds, route telemetry, alert audit feeds, and driver safety leaderboards.

#### 1-Click Dashboard Launch

```powershell
# Launches FastAPI server on http://localhost:8000 and opens your browser
.\scripts\run_dashboard.ps1

# Custom port
.\scripts\run_dashboard.ps1 -Port 8080
```

Or run via Python directly:

```bash
uvicorn cockpit_sentinel.dashboard.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Dashboard Features:

1. **Fleet Overview & KPI Metrics**: Total active drivers, total logged trips, fleet mileage, average attentiveness index %, incident counts, and safety ratings.
2. **Driver Profiles & Biometric Calibration**:
   - Create and manage driver profiles (name, license number, experience, status, avatar, contact).
   - **Interactive Calibration Sliders**: Adjust Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR yawn), Head Yaw turn limit (degrees), Phone Detection Confidence, Smoking Detection Confidence, and Temporal Smoothing Window (seconds). All changes save immediately to MongoDB Atlas / SQLite.
3. **Route Management**:
   - Register commercial highways and urban transit corridors (origin, destination, distance in km, estimated travel duration, speed limits, and risk levels: low, moderate, high ghat section, or high-speed expressway).
   - Assign pilot drivers and define checkpoint milestones.
4. **Trip Session Telemetry**:
   - Logs attentive vs. inattentive time, % attention score, total processed frames, average FPS, alert counts, and safety star rating per trip.
   - Interactive Chart.js trend curves for fleet attention monitoring.
5. **Alert Incident Audit Feed**:
   - Audit trail of past distraction & fatigue alerts.
   - Filter by Driver and Risk Level (Caution, Warning, Critical).
   - View exact triggered signals (`eyes_closed`, `yawning`, `looking_away`, `phone_detected`, `smoking_detected`), alert duration, and timestamp.
6. **Driver Safety Ratings & Leaderboard**:
   - Safety Index (0–100%) and 1–5 star driver ratings.
   - Dynamic leaderboard with Gold, Silver, and Bronze badges for the safest drivers.
7. **Hybrid Cloud & Edge Database**:
   - Connects to **MongoDB Atlas** for centralized fleet telemetry.
   - Features automatic local **SQLite fallback** (`data/cockpit_sentinel.db`) ensuring vehicular operation runs without disruption even in zero-connectivity tunnels and remote highway zones.

---

### In-Cabin Face Recognition & Auto Profile Loading

CockpitSentinel features automatic biometric face recognition for seamless in-cabin driver identification:

- **Continuous In-Cabin Verification**: When a driver enters the vehicle cabin, the system continuously scans their face using **FaceNet (InceptionResnetV1)** with MTCNN and Haar cascade fallback.
- **Strict Anti-Impersonation Rejection**: Operates with a strict cosine similarity threshold ($\ge 65\%$, configurable via `--face-threshold`). Other individuals, passengers, or unrecognized persons are strictly rejected and will never be identified as an enrolled driver.
- **Mismatch Alerts & Guest Mode**:
  - If a driver was scheduled or expected (`--driver-id`) but an unverified person is driving, the HUD raises an alert: `CABIN ALERT: UNVERIFIED DRIVER (EXP: <NAME>)` in red.
  - If an unrecognized person sits in the driver's seat without assignment, the HUD switches to `DRIVER: UNRECOGNIZED PERSON [GUEST MODE]` in amber, reverting to safe baseline defaults.
  - When the genuine driver is verified, the HUD displays `DRIVER: <NAME> [VERIFIED <conf>%]` in emerald green and applies personalized EAR, MAR, yaw, and distraction thresholds.
- **Automatic Telemetry Binding**: Trips and safety incident logs are automatically recorded to the database under the verified driver's ID (or attributed as unverified/guest if unauthenticated).

#### Enrolling Driver Facial Biometrics

You can enroll a driver's facial data in two ways:

1. **Via Web Dashboard (Recommended)**:

   - Go to the **Drivers** tab in the dashboard (`http://localhost:8000`).
   - Click the **Face** button on any driver card.
   - Click **Start Camera** and **Capture Snapshot**, or choose **Upload Photo**.
   - Click **Save Face to Profile** to compute and persist the 512-dimensional embedding to the database.
2. **Via Command-Line Interface (CLI)**:

   ```powershell
   # Enroll via photo file
   python scripts/enroll_driver_face.py --driver-id "drv-satyapal-001" --image "path/to/driver_photo.jpg"

   # Enroll via interactive webcam countdown capture
   python scripts/enroll_driver_face.py --driver-id "drv-satyapal-001" --camera-index 0
   ```

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
| Eyes closed      | 3      | EAR < 0.22        | ≥ 2.0s sustained     |
| Yawning          | 2      | MAR > 0.60        | ≥ 2.5s sustained     |
| Looking away     | 2      | Head yaw > 30°   | ≥ 2.0s sustained     |
| Phone detected   | 4      | Confidence > 0.40 | ≥ 2.0s sustained     |
| Smoking detected | 3      | Confidence > 0.35 | ≥ 2.0s sustained     |

**Risk levels:** SAFE (< 2) · CAUTION (≥ 2) · WARNING (≥ 4) · CRITICAL (≥ 6)

---

### YOLO11s Distraction Model Training & Performance (Google Colab)

CockpitSentinel employs a custom **YOLO11s** unified distraction detection model trained with transfer learning on Google Colab (Tesla T4 GPU). It fuses our in-cabin smoking dataset with a dedicated driver phone interaction dataset (`using-phone-while-driving-qcvye`, 2,757 annotations) into a balanced multi-class detector.

| Metric | Version 1 (Base Smoking Model) | Version 2 (Balanced Distraction Model) | Improvement |
| :--- | :---: | :---: | :---: |
| **Phone Detection Precision ($P$)** | 90.71% | **95.05%** | **+4.34% precision surge** |
| **Smoking Detection ($mAP@50$)** | **96.89%** | **89.71%** | Outstanding baseline anchor |
| **Validation Bounding Box Loss** | 1.9108 | **1.3059** | **-31.6% tighter localization** |
| **Merged Corpus $mAP@50-95$** | 45.86% | **53.09%** | **+7.23% overall accuracy** |
| **Intel CPU Latency (OpenVINO FP16)** | 44.0 ms | **30.6 ms** | **32.6 FPS (11.2× speedup)** |

### Standard Dataset Benchmarks & PDF Test Reports

CockpitSentinel models and algorithms have been rigorously evaluated across **5 standard public benchmarks**:

| Benchmark Dataset | Evaluation Scope | Key Metric & Result | Findings |
| :--- | :--- | :---: | :--- |
| **YawDD** (Yawning Detection) | 10 driver video sessions | **100% Accuracy / Recall / Precision** | Perfect MAR separation between normal ($0.015$), talking ($0.148$), and yawning ($0.95$–$1.09$). Distraction false-alarm rejection: **93.8% Specificity**. |
| **Kaggle NTHU DDD** | 250 evaluation frames | **86.0% Nodding Accuracy** | MediaPipe EAR reliably detects microsleeps (mean EAR $0.151$). YOLO distraction false alarm rejection: **99.2% Specificity**. |
| **Kaggle State Farm DDD** | 1,065 real-cabin images | **99.70% Phone Recall** | Detected cell phone in **331 of 332** phone interaction images ($0.715$ mean confidence). |
| **Roboflow Distraction Corpus** | 2,757 annotations | **90.78% mAP@50 / 96.89% Smoking** | Baseline v1 model: $91.35\%$ precision, $85.91\%$ recall, $88.55\%$ F1-score. |
| **Intel OpenVINO Edge Benchmark** | 640×640 on Core i5 | **22.7 – 32.6 FPS (30.6 ms)** | Full real-time edge vehicular deployment feasibility. |

All comprehensive technical reports are compiled as standalone, publication-quality PDF documents located in [`test_reports/`](test_reports/):

- [`01_Smoking_Detection_Colab_Training_Report.pdf`](test_reports/01_Smoking_Detection_Colab_Training_Report.pdf) — Colab T4 training setup, loss curves, and OpenVINO quantization.
- [`02_Distraction_Model_Ablation_and_Accuracy_Report.pdf`](test_reports/02_Distraction_Model_Ablation_and_Accuracy_Report.pdf) — Head-to-head v1 vs. v2 ablation study, precision gains, and domain shift analysis.
- [`03_YawDD_and_Kaggle_External_Benchmarks_Report.pdf`](test_reports/03_YawDD_and_Kaggle_External_Benchmarks_Report.pdf) — Video session results on YawDD, NTHU drowsy blinks, and State Farm phone sensitivity.
- [`CockpitSentinel_Master_Model_Evaluation_Report.pdf`](test_reports/CockpitSentinel_Master_Model_Evaluation_Report.pdf) — Master technical evaluation synthesizing all 5 datasets and vehicle edge deployments.

To regenerate the PDF reports at any time:
```powershell
python scripts/generate_pdf_reports.py
```

---

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
├── test_reports/          # Generated PDF evaluation reports
│   ├── 01_Smoking_Detection_Colab_Training_Report.pdf
│   ├── 02_Distraction_Model_Ablation_and_Accuracy_Report.pdf
│   ├── 03_YawDD_and_Kaggle_External_Benchmarks_Report.pdf
│   └── CockpitSentinel_Master_Model_Evaluation_Report.pdf
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
│   ├── dashboard/                #   FastAPI web dashboard & HTML telemetry UI
│   ├── database/                 #   MongoDB Atlas & SQLite persistence models
│   ├── detection/                #   Phone & smoking detection (YOLO)
│   ├── drowsiness/               #   Eye/mouth/head analysis (MediaPipe)
│   ├── pipeline/                 #   Live monitor and video processing
│   ├── utils/                    #   Shared helpers
│   └── domain.py                 #   Core data contracts and risk logic
├── configs/                      # YAML configuration files
├── scripts/                      # Setup, training, and runner scripts
│   ├── run_dashboard.ps1         #   1-click telemetry dashboard launcher (port 8000)
│   ├── run_webcam_monitor.ps1    #   1-click webcam runner (with OpenVINO auto-detect)
│   ├── run_viva_demo.ps1         #   Demo runner on YawDD video
│   ├── evaluate_yawdd.py         #   YawDD multi-video benchmark evaluator
│   ├── evaluate_kaggle_nthu.py   #   Kaggle NTHU drowsiness evaluator
│   ├── benchmark_state_farm_phone.py # Out-of-domain State Farm evaluation benchmark
│   ├── evaluate_distraction_model.py # Distraction validation evaluator
│   ├── merge_phone_smoking_datasets.py # Balanced dataset merger & canonical label remapper
│   ├── download_distraction_dataset.py # Roboflow dataset downloader
│   ├── train_yolo11_distraction.py     # YOLO11 distraction trainer
│   ├── generate_pdf_reports.py   #   Publication-quality PDF report generator
│   ├── export_openvino_model.py  #   Intel OpenVINO export & benchmark tool
│   ├── export_jetson_tensorrt.py #   NVIDIA Jetson TensorRT export script
│   ├── setup_environment.ps1     #   Windows bootstrap
│   ├── setup.sh                  #   macOS / Linux bootstrap
│   ├── setup_environment.py      #   Cross-platform setup logic
│   └── verify_environment.py     #   Environment validator
├── test_reports/                 # Compiled PDF test & benchmark reports
│   ├── 01_Smoking_Detection_Colab_Training_Report.pdf
│   ├── 02_Distraction_Model_Ablation_and_Accuracy_Report.pdf
│   ├── 03_YawDD_and_Kaggle_External_Benchmarks_Report.pdf
│   └── CockpitSentinel_Master_Model_Evaluation_Report.pdf
├── tests/
│   ├── unit/                     #   61 unit tests (100% pass)
│   └── integration/              #   Integration tests (planned)
├── notebooks/
│   ├── train_yolo11_distraction_colab.ipynb # Base Colab training notebook
│   └── train_combined_phone_smoking_colab.ipynb # Balanced phone & smoking Colab notebook
├── docs/                         # Guides and architecture documentation
│   ├── cloud_deployment_guide.md #   AWS and cloud provider deployment, CI/CD & telemetry guide
│   ├── colab_training_guide.md   #   Google Colab training, dataset fusion & metrics guide
│   ├── datasets.md               #   Dataset registry (YawDD, NTHU, Roboflow, State Farm)
│   └── vehicle_deployment.md     #   In-vehicle edge hardware deployment (Jetson/TensorRT)
├── Dockerfile                    # Production cloud and dashboard container
├── docker-compose.yml            # Local and cloud 1-click Docker orchestration
├── .github/workflows/ci.yml      # GitHub Actions CI (multi-OS, multi-Python)
├── pyproject.toml                # Project metadata and tool config
├── requirements.in               # Direct dependencies
└── requirements.txt              # Pinned dependencies (pip-compile)
```

## Environment Variables

The `.env` file is optional. If unset, sensible defaults are used:

| Variable                | Purpose                                              | Default                                   |
| ----------------------- | ---------------------------------------------------- | ----------------------------------------- |
| `MONGODB_URI`         | MongoDB Atlas connection string                      | _(cluster URI or auto fallback)_        |
| `DATABASE_BACKEND`    | Database backend (`mongodb`, `sqlite`, `auto`) | `auto` (Atlas primary, SQLite fallback) |
| `DASHBOARD_PORT`      | Local port for web dashboard                         | `8000`                                  |
| `COCKPIT_DATA_ROOT`   | Root directory for datasets and storage              | _(repo directory)_                      |
| `COCKPIT_MODELS_ROOT` | Root directory for model files                       | _(repo directory)/models_               |
| `MLFLOW_TRACKING_URI` | MLflow experiment store                              | _(repo directory)/experiments/mlruns_   |
| `LOG_LEVEL`           | Application log level                                | `INFO`                                  |
| `WEBCAM_INDEX`        | Default webcam device                                | `0`                                     |
| `KAGGLE_USERNAME`     | Kaggle API username                                  | _(empty)_                               |
| `KAGGLE_KEY`          | Kaggle API key                                       | _(empty)_                               |

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
