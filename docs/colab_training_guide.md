# CockpitSentinel — Google Colab Distraction Model Training Guide

This guide details the complete training workflow, dataset fusion pipeline, and edge optimization steps used to train the **YOLO11s Unified Distraction Model** for CockpitSentinel.

---

## 1. Overview & Architecture

CockpitSentinel replaces separate detector models with a **single unified YOLO11s detector** capable of identifying mobile phone use, smoking, and earphone distractions in one forward pass.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATASET FUSION PIPELINE                         │
│                                                                        │
│   [Base Smoking Dataset]                  [Driver Phone Dataset]       │
│   (satyapal-gaikwad / v1)                 (long-itwjh / v2)            │
│   1,253 images │ 613 smoke, 193 phone     886 images │ 2,757 phones    │
│            │                                       │                   │
│            └───────────────┬───────────────────────┘                   │
│                            ▼                                           │
│             Standardized Label Remapper                                │
│       [0: earphone, 1: person, 2: smoke, 3: using_phone]               │
│                            │                                           │
│                            ▼                                           │
│              Merged Balanced Training Corpus                           │
│             >2,100 Images │ >3,400 Annotations                         │
│                            │                                           │
│                            ▼                                           │
│                 YOLO11s Transfer Learning                              │
│              (Google Colab Tesla T4 GPU)                               │
│                            │                                           │
│             ┌──────────────┴──────────────┐                            │
│             ▼                             ▼                            │
│     Intel OpenVINO FP16             ONNX / TensorRT                    │
│      (30.6 ms / 32.6 FPS)       (Vehicle Edge Computer)                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Sources & Specifications

### Dataset A: Base In-Cabin Distraction Dataset (Smoking Anchor)
- **Source:** [Roboflow Universe — satyapal-gaikwad/driver-smoking-phoneuse-earphone-detection-9nlii-oscoe](https://universe.roboflow.com/satyapal-gaikwad/driver-smoking-phoneuse-earphone-detection-9nlii-oscoe/1)
- **Volume:** 1,253 images (1,052 train / 201 val)
- **Annotations:** 1,552 bounding boxes
- **Key Contribution:** High-recall cigarette and vape annotations (613 train instances).

### Dataset B: Dedicated Driver Phone Dataset (Phone Precision Anchor)
- **Source:** [Roboflow Universe — long-itwjh/using-phone-while-driving-qcvye](https://universe.roboflow.com/long-itwjh/using-phone-while-driving-qcvye/2)
- **Volume:** 886 in-cabin images
- **Annotations:** 2,757 phone bounding boxes, 2,707 steering wheel instances
- **Key Contribution:** Eliminates false-positives from steering wheel grip and chin-resting postures.

### Canonical Class Schema
All labels are standardized into 4 core classes:

| Class ID | Canonical Name | Mapped Raw Labels |
| :---: | :--- | :--- |
| **0** | `earphone` | `earphone`, `headphone`, `earphones` |
| **1** | `person` | `person`, `driver` |
| **2** | `smoke` | `smoke`, `smoking`, `cigarette`, `vape` |
| **3** | `using_phone` | `phone`, `using_phone`, `cellphone`, `mobile`, `mobile-use`, `cell phone` |

---

## 3. Step-by-Step Google Colab Training Workflow

The entire pipeline is automated in [`notebooks/train_combined_phone_smoking_colab.ipynb`](../notebooks/train_combined_phone_smoking_colab.ipynb).

### Step 1: Environment Setup on Colab
Select **Runtime > Change runtime type > T4 GPU**.

```python
!nvidia-smi
!pip install --upgrade ultralytics roboflow openvino onnx pyyaml
```

### Step 2: Download Both Datasets via Roboflow API
```python
from roboflow import Roboflow

rf = Roboflow(api_key="PB8gWnTTkNlkxmSOpsXH")

# 1. Base smoking dataset
base_proj = rf.workspace("satyapal-gaikwad").project("driver-smoking-phoneuse-earphone-detection-9nlii-oscoe")
base_proj.version(1).download(model_format="yolov11", location="/content/base_dataset")

# 2. Dedicated phone dataset (version 2)
phone_proj = rf.workspace("long-itwjh").project("using-phone-while-driving-qcvye")
phone_proj.version(2).download(model_format="yolov11", location="/content/phone_dataset")
```

### Step 3: Dataset Merging & Label Remapping
Run the automated remapping script to merge annotations into `/content/merged_dataset` with standardized class IDs:

```python
import shutil, yaml
from pathlib import Path
from collections import Counter

STANDARD_NAMES = ['earphone', 'person', 'smoke', 'using_phone']
STANDARD_MAP = {
    'earphone': 0, 'headphone': 0, 'earphones': 0,
    'person': 1, 'driver': 1,
    'smoke': 2, 'smoking': 2, 'cigarette': 2, 'vape': 2,
    'using_phone': 3, 'phone': 3, 'cellphone': 3, 'mobile': 3, 'mobile-use': 3, 'cell phone': 3
}

# (Merges images and remapped labels for train and valid splits)
```

### Step 4: Fine-Tuning YOLO11s with Transfer Learning
```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.train(
    data="/content/merged_dataset/data.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    patience=15,
    optimizer="AdamW",
    lr0=0.001,
    name="yolo11s_balanced_distraction",
    project="runs"
)
```

### Step 5: Exporting for Vehicle Deployment
```python
best_weights = f"{results.save_dir}/weights/best.pt"
trained_model = YOLO(best_weights)

# 1. Export Intel OpenVINO FP16 (CPU acceleration)
ov_dir = trained_model.export(format="openvino", half=True)

# 2. Export ONNX (Edge accelerator / Jetson)
onnx_file = trained_model.export(format="onnx", dynamic=True)
```

---

## 4. Quantitative Results & Comparison

### Loss Convergence
- **Validation Box Loss:** Decreased from **$1.9108 \rightarrow 1.3059$** ($-31.6\%$ error reduction)
- **Validation Class Loss:** Decreased from **$0.9673 \rightarrow 0.8509$** ($-12.0\%$ error reduction)
- **Validation DFL Loss:** Decreased from **$1.5453 \rightarrow 1.2036$** ($-22.1\%$ coordinate error reduction)

### Detection Performance
- **Phone Precision:** Jumped from **$90.71\% \rightarrow 95.05\%$** ($+4.34\%$ increase).
- **Smoking mAP@50:** Sustained top performance at **$89.71\%–96.89\%$**.
- **Merged mAP@50-95:** Rose to **$53.09\%$** across all bounding boxes.

### CPU Hardware Benchmark (Intel Core i5 CPU)
- **PyTorch CPU (`best.pt`):** 343.4 ms/frame (2.9 FPS)
- **Intel OpenVINO FP16:** **30.6 ms/frame (32.6 FPS)** $\rightarrow$ **11.2× speedup**

---

## 5. Local Deployment in CockpitSentinel

Extracted weights are registered under [`weights/trained/yolo11s_balanced_distraction/`](../weights/trained/yolo11s_balanced_distraction/):
- `best.pt` (PyTorch)
- `best.onnx` (ONNX)
- `best_openvino_model/` (OpenVINO FP16)

Run live monitoring with the balanced OpenVINO model:
```powershell
cockpit-sentinel --source 0 --distraction-model weights/trained/yolo11s_balanced_distraction/best_openvino_model
```
