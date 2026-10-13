# Dataset Registry

All datasets live at `D:\CockpitSentinel\data\external\` (controlled via `COCKPIT_DATA_ROOT` in `.env`).
Raw data is **immutable** — never modify the originals; preprocessing outputs go to `data/processed/`.

---

## Summary Table

| Dataset                 | Local Path                             | Purpose                                   | Status                             |
| ----------------------- | -------------------------------------- | ----------------------------------------- | ---------------------------------- |
| **Kaggle DDD**    | `external/kaggle_driver_monitoring/` | Drowsy vs Non-Drowsy image classification | Sample extracted (100+100 images)  |
| **NTHU DDD**      | `external/nthu/`                     | EAR/MAR threshold calibration with video  | Folders ready                      |
| **YawDD**         | `external/yawdd/`                    | Yawning / fatigue video validation        | Folders ready — download required |
| **State Farm**    | `external/state_farm/`               | 10-class distraction detection benchmark  | Folders ready — download required |
| **Phone & Smoke** | `external/distraction_phone_smoking/`| YOLO11 bounding boxes for phone & smoking| Script ready (`download_distraction_dataset.py`) |
| **Custom Webcam** | `external/custom/`                   | Local capture for integration testing     | Empty placeholder                  |

---

## 1. Kaggle Driver Drowsiness Dataset (DDD)

**Source:** https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd
**Full size:** 41,793 images (22,348 Drowsy + 19,445 Non-Drowsy)

### Folder Structure

```text
kaggle_driver_monitoring/
├── images/
│   ├── drowsy/           <- 100 sample Drowsy images (extracted)
│   └── non_drowsy/       <- 100 sample Non-Drowsy images (extracted)
├── kaggle_dataset_zip.zip  <- Full 2.7 GB archive
└── README.md
```

### Download Full Dataset

```powershell
kaggle datasets download -d ismailnasri20/driver-drowsiness-dataset-ddd `
  -p D:\CockpitSentinel\data\external\kaggle_driver_monitoring --unzip
```

---

## 2. NTHU Driver Drowsiness Detection (NTHU DDD)

**Source:** https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
**Kaggle mirror:** https://www.kaggle.com/datasets/taweilo/driver-drowsiness-dataset-ddd
**Type:** Videos - multi-subject, varied lighting (day + night, glasses/no glasses)

### Class Labels

- `normal` - attentive, eyes open
- `yawning` - mouth wide open, fatigue indicator
- `eye_closure` - drowsy, micro-sleep

### Folder Structure

```text
nthu/
├── Training_Evaluation_Dataset/
│   ├── Training Dataset/    <- Per-subject subfolders (001/, 002/ ...)
│   └── Evaluation Dataset/
├── Testing_Dataset/
└── Nighttime_Dataset/
```

### Download Instructions

```powershell
kaggle datasets download -d taweilo/driver-drowsiness-dataset-ddd `
  -p D:\CockpitSentinel\data\external\nthu --unzip
```

---

## 3. YawDD - Yawning Detection Dataset

**Source:** https://www.site.uottawa.ca/~shervin/yawning/
**Type:** AVI Videos - 107 subjects, dashboard + IR mirror cameras

### Class Labels

- `normal` - no yawning or talking
- `talking` - mouth moving, not yawning
- `yawning` - full yawn detected

### Folder Structure

```text
yawdd/
├── Driver_Drowsiness_Dataset/
│   ├── Camera_Dash/     <- Dashboard-cam AVI files
│   └── Camera_IR/       <- IR mirror-cam AVI files
└── Mirror_Dataset/
    ├── Camera_Dash/
    └── Camera_IR/
```

---

## 4. State Farm Distracted Driver Detection

**Source:** https://www.kaggle.com/c/state-farm-distracted-driver-detection
**Type:** JPEG images - 10 distraction classes, in-cab front-facing camera

### Class Labels

| Folder                   | Description             |
| ------------------------ | ----------------------- |
| `c0_normal`            | Safe driving            |
| `c1_texting_right`     | Texting - right hand    |
| `c2_phone_right`       | Phone call - right hand |
| `c3_texting_left`      | Texting - left hand     |
| `c4_phone_left`        | Phone call - left hand  |
| `c5_radio`             | Operating radio         |
| `c6_drinking`          | Drinking                |
| `c7_reach_behind`      | Reaching behind seat    |
| `c8_hair_makeup`       | Hair and makeup         |
| `c9_talking_passenger` | Talking to passenger    |

### Folder Structure

```text
state_farm/
├── train/
│   ├── c0_normal/
│   ├── c1_texting_right/
│   ├── c2_phone_right/
│   ├── c3_texting_left/
│   ├── c4_phone_left/
│   ├── c5_radio/
│   ├── c6_drinking/
│   ├── c7_reach_behind/
│   ├── c8_hair_makeup/
│   └── c9_talking_passenger/
└── test/  (same structure)
```

### Download Instructions

```powershell
# Must accept competition rules at: https://www.kaggle.com/c/state-farm-distracted-driver-detection
kaggle competitions download -c state-farm-distracted-driver-detection `
  -p D:\CockpitSentinel\data\external\state_farm --unzip
```

---

## 5. Driver Distraction Dataset (Phone & Smoking - YOLO Format)

**Source:** [Roboflow Universe - Driver Distraction](https://universe.roboflow.com/1539379769-qq-com/driver-smoking-phoneuse-earphone-detection-9nlii)
**Type:** Bounding box object detection formatted for YOLO11 / YOLOv8
**Target Classes:** `smoke`, `using_phone`, `earphone`, `without_using_phone`

### Automated Download
```powershell
# Set Roboflow API key (free from app.roboflow.com)
python scripts/download_distraction_dataset.py --api-key YOUR_API_KEY
```

### Folder Structure
```text
distraction_phone_smoking/
├── data.yaml           <- YOLO dataset configuration
├── train/
│   ├── images/         <- Train JPEG/PNG frames
│   └── labels/         <- YOLO format TXT bounding boxes
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## Full Storage Layout

```text
D:\CockpitSentinel\
├── data\
│   ├── external\           <- Raw datasets (immutable after download)
│   │   ├── kaggle_driver_monitoring\
│   │   ├── nthu\
│   │   ├── yawdd\
│   │   ├── state_farm\
│   │   └── custom\
│   ├── processed\          <- Preprocessed splits (generated)
│   │   ├── features\
│   │   └── splits\
│   └── raw\               <- Unextracted archives
├── models\
│   ├── pretrained\        <- face_landmarker.task, yolov8n.pt
│   └── trained\           <- Your trained model checkpoints
└── experiments\
    └── mlruns\            <- MLflow run tracking
```
