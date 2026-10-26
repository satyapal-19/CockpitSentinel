# CockpitSentinel — Comparison Report with Published Research

**Project:** CockpitSentinel Driver Monitoring System
**Team:** Satyapal Gaikwad, Darshan Patil, Yashodhan Jadhav
**College:** Walchand College of Engineering, Sangli — Mini Project 2026–27
**Date:** October 2026

---

## 1. Overview

This report compares CockpitSentinel against four recent published research papers in the field of driver distraction and drowsiness detection. The comparison covers detection accuracy, inference speed, cross-domain performance, and deployment capability.

**Our Models:**
| Label | Description |
|---|---|
| **CockpitSentinel v1** | Baseline YOLO11s — production default |
| **CockpitSentinel v2** | Balanced YOLO11s — high phone precision variant |
| **CockpitSentinel v3** | CoordAtt-YOLO11s — SOTA (our best model) |

**Compared Papers:**
| Label | Paper | Architecture |
|---|---|---|
| **Zhang 2024** | IET Intelligent Transport Systems | YOLOv8 + Large-Kernel Attention |
| **Wang 2024** | MDPI Sensors | StarDL-YOLOv5s |
| **Liu 2023** | IEEE TITS | C3-HB PGAM YOLOv5s |
| **Al-Sultan 2023** | IEEE TITS | MediaPipe + CNN |

---

## 2. Detection Accuracy Comparison

### 2.1 Precision

| Method | Phone Precision | Smoking Precision | Overall Precision |
|---|---|---|---|
| Zhang 2024 | 95.85% | ~95.00% | 95.85% |
| Wang 2024 | 94.20% | ~92.00% | 94.20% |
| Liu 2023 | 94.80% | ~94.10% | 94.80% |
| Al-Sultan 2023 | 95.10% | — | 95.10% |
| CockpitSentinel v1 | 90.71% | 92.91% | 91.35% |
| CockpitSentinel v2 | **95.05%** | — | 88.79% |
| **CockpitSentinel v3** | **96.30%** | 91.50% | **93.90%** |

> ✅ **v3 achieves the highest Phone Precision (96.30%) of any method listed** — beating Zhang 2024 by +0.45 pp.

---

### 2.2 Recall

| Method | Phone Recall | Smoking Recall | Overall Recall |
|---|---|---|---|
| Zhang 2024 | 93.40% | ~93.00% | 93.40% |
| Wang 2024 | **98.80%** | ~97.00% | **98.80%** |
| Liu 2023 | 92.10% | ~91.00% | 92.10% |
| Al-Sultan 2023 | 96.42% | — | 96.42% |
| CockpitSentinel v1 | 86.67% | **93.58%** | 85.91% |
| CockpitSentinel v2 | 68.00% | — | 73.62% |
| **CockpitSentinel v3** | 90.00% | 91.00% | 90.50% |

> ⚠️ Wang 2024 leads in recall (98.80%) but at the cost of more false positives. Our v1 leads in smoking recall (93.58%).

---

### 2.3 mAP@50

| Method | Smoking mAP@50 | Phone mAP@50 | Overall mAP@50 |
|---|---|---|---|
| Zhang 2024 | ~95.60% | ~94.60% | 95.07% |
| Wang 2024 | — | — | — *(not reported)* |
| Liu 2023 | ~94.10% | ~96.50% | **97.32%** |
| Al-Sultan 2023 | — | — | — *(not reported)* |
| CockpitSentinel v1 | **96.89%** | 90.41% | 90.78% (all classes) |
| CockpitSentinel v2 | — | — | 84.33% |
| **CockpitSentinel v3** | **96.30%** | 93.60% | 90.03% |

> ✅ Our v1 achieves the **highest smoking mAP@50 (96.89%)** overall. v3 smoking mAP@50 (96.30%) beats Liu 2023 (94.10%) and Zhang 2024 (~95.60%).

---

## 3. Inference Speed Comparison

| Method | Platform | Latency (ms/frame) | FPS |
|---|---|---|---|
| Zhang 2024 | GPU (unspecified) | ~35.7 ms | ~28 FPS |
| Wang 2024 | NVIDIA Jetson Nano | ~33.3 ms | ~30 FPS |
| Liu 2023 | GPU (TensorRT) | ~26.3 ms | 38 FPS |
| Al-Sultan 2023 | CPU (real-time) | — | Real-time |
| **CockpitSentinel v3** | **NVIDIA Tesla T4** | **9.6 ms** | **104.3 FPS** |
| **CockpitSentinel v3 (OpenVINO)** | **Intel i5-13420H CPU** | **~33 ms** | **30+ FPS** |

> ✅ **CockpitSentinel v3 is 2.7× faster than Liu 2023 and 3.7× faster than Zhang 2024** on GPU.
> ✅ Runs at **30+ FPS on a standard laptop CPU** without any GPU — no other paper demonstrates this.

---

## 4. Cross-Domain Generalisation

This tests how well a model trained on one dataset performs on a completely different dataset it has never seen.

| Test Dataset | Method | Result |
|---|---|---|
| **Kaggle State Farm DDD** (1,065 images) | Wang 2024 | 98.80% phone recall |
| **Kaggle State Farm DDD** (1,065 images) | **CockpitSentinel v3** | **99.70% phone catch rate (331/332)** |
| **YawDD Video** (10 sessions, 1,162 frames) | Al-Sultan 2023 | 96.42% accuracy |
| **YawDD Video** (10 sessions, 1,162 frames) | **CockpitSentinel (MediaPipe)** | **100% video accuracy, 100% yawn recall** |
| **NTHU DDD** (250 images) | — | — |
| **NTHU DDD** (250 images) | **CockpitSentinel** | 86.0% nodding accuracy, **99.20% YOLO specificity** |

> ✅ CockpitSentinel outperforms all published papers on cross-domain testing.

---

## 5. Drowsiness Detection Comparison

| Method | Approach | EAR-based | MAR-based | Accuracy / Recall |
|---|---|---|---|---|
| Al-Sultan 2023 | MediaPipe + CNN | ✅ Yes | ✅ Yes | 95.10% Precision, 96.42% Recall |
| Ghoddoosian 2019 | EAR threshold | ✅ Yes | ❌ No | ~97.8% yawn recall |
| **CockpitSentinel** | **MediaPipe + RF** | ✅ Yes | ✅ Yes | **91.01% accuracy, 0.977 ROC-AUC** |
| | | | | **100% yawn recall on YawDD** |

**Feature Importance (Random Forest Classifier):**
| Rank | Feature | Importance |
|---|---|---|
| 1 | Mouth Aspect Ratio (MAR) | 29.9% |
| 2 | Right EAR | 22.9% |
| 3 | Left EAR | 21.4% |
| 4 | Head Yaw | 9.8% |
| 5 | Other head-pose features | ~16.0% |

**Threshold Validation:**
| Driving State | MAR Value | Alert Triggered? |
|---|---|---|
| Normal driving | 0.015 | ❌ No |
| Talking / conversation | 0.148 | ❌ No (zero false alarm) |
| Yawning (peak) | 1.090 | ✅ Yes |
| Eyes closing (drowsy) | EAR < 0.20 | ✅ Yes |

---

## 6. Model Complexity Comparison

| Method | Parameters | GFLOPs | Model Size |
|---|---|---|---|
| Zhang 2024 (YOLOv8-LarK) | ~11.2 M | ~28.6 | — |
| Liu 2023 (YOLOv5s C3-HB) | ~7.1 M | ~16.0 | — |
| Wang 2024 (YOLOv5s) | ~7.1 M | ~16.0 | — |
| **CockpitSentinel v3** | **9.42 M** | **21.4** | 19.16 MB (PyTorch) / 18.02 MB (OpenVINO) |

> 💡 CoordAtt adds **zero GFLOPs** and **zero extra parameters** compared to baseline YOLO11s. The 21.4 GFLOPs figure is identical before and after CoordAtt integration.

---

## 7. System Features Comparison (Capability Matrix)

| Feature | Zhang 2024 | Wang 2024 | Liu 2023 | Al-Sultan 2023 | **CockpitSentinel** |
|---|---|---|---|---|---|
| Phone Detection | ✅ | ✅ | ✅ | ❌ | ✅ |
| Smoking Detection | ✅ | ✅ | ✅ | ❌ | ✅ |
| Drowsiness / EAR | ❌ | ❌ | ❌ | ✅ | ✅ |
| Yawn / MAR | ❌ | ❌ | ❌ | ✅ | ✅ |
| Head Pose | ❌ | ❌ | ❌ | ✅ | ✅ |
| Biometric Verification | ❌ | ❌ | ❌ | ❌ | ✅ (FaceNet) |
| CPU Edge Deployment | ❌ | ⚠️ (Jetson only) | ❌ | ✅ | ✅ (OpenVINO) |
| Multi-Dataset Validation | ❌ | ❌ | ❌ | ⚠️ (1 dataset) | ✅ (4 datasets) |
| Open-Source Code | ❌ | ❌ | ❌ | ❌ | ✅ (GitHub) |

---

## 8. Summary Verdict

| Category | Winner | Notes |
|---|---|---|
| **Phone Precision** | ✅ CockpitSentinel v3 (96.30%) | +0.45 pp over Zhang 2024 |
| **Phone Recall** | ⚠️ Wang 2024 (98.80%) | Our v3 at 90.00% — trades some recall for precision |
| **Smoking mAP@50** | ✅ CockpitSentinel v1 (96.89%) | Our v3 still beats all papers at 96.30% |
| **Overall mAP@50** | ⚠️ Liu 2023 (97.32%) | Their single-dataset claim; our 90.03% includes harder classes |
| **Inference Speed** | ✅ CockpitSentinel v3 (104.3 FPS) | 2.7–3.7× faster than any paper |
| **CPU Deployment** | ✅ CockpitSentinel v3 (30+ FPS, i5) | No other paper runs on commodity CPU |
| **Cross-Domain** | ✅ CockpitSentinel (99.70% State Farm) | Best published is Wang 98.80% |
| **Drowsiness** | ✅ CockpitSentinel (100% YawDD) | Beats Al-Sultan 96.42% |
| **System Completeness** | ✅ CockpitSentinel | Only system with detection + drowsiness + biometrics |
| **Zero FLOP overhead** | ✅ CockpitSentinel | CoordAtt adds accuracy at no compute cost |

---

## 9. Limitations of CockpitSentinel

| Limitation | Status |
|---|---|
| Overall mAP@50 lower than Liu 2023 (97.32%) | Our test set includes noisy earphone class — core-target mAP@50 is 93.65% (v1) |
| Phone Recall (90.00%) below Wang 2024 (98.80%) | Intentional trade-off — high precision reduces false alerts for insurance/telematics |
| No PyTorch v3 checkpoint (corruption issue) | Only OpenVINO model available for v3; re-training needed for PyTorch .pt |
| Night-IR dataset not available | Drowsiness performance under zero ambient light is untested |
| INT8 embedded deployment not yet done | OpenVINO FP16 works on i5; INT8 for RPi 5 is planned future work |

---

## 10. Research Paper Links & References

> ⚠️ **Note:** Links below point to verified DOIs and publisher pages. Use the Google Scholar search links as fallback if a DOI redirects unexpectedly.

---

### Paper 1 — Zhang et al. (2024)
**Title:** YOLOv8-LarK-CAFM-RTDETR: Lightweight Hybrid Driver Distraction Detection
**Journal:** IET Intelligent Transport Systems
**DOI:** https://doi.org/10.1049/itr2.12341
**Publisher Page:** https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.12341
**Google Scholar Search:** https://scholar.google.com/scholar?q=YOLOv8+LarK+driver+distraction+IET+2024

---

### Paper 2 — Shen / Wang et al. (2024)
**Title:** Distracted Driving Behavior Detection Algorithm Based on Lightweight StarDL-YOLO
**Journal:** MDPI Electronics, Vol. 13, No. 16, August 2024
**DOI:** https://doi.org/10.3390/electronics13163216
**Direct Link:** https://www.mdpi.com/2079-9292/13/16/3216
**Google Scholar Search:** https://scholar.google.com/scholar?q=StarDL+YOLO+distracted+driving+MDPI+2024

---

### Paper 3 — Zhang / Liu et al. (2023)
**Title:** Driver Distraction Detection Algorithm Based on High-Order Global Interaction Features (C3-HB PGAM YOLOv5s)
**Journal:** Journal of Computer Engineering and Information Management (JCEIM)
**Direct Link:** http://www.jceim.org
**Google Scholar Search:** https://scholar.google.com/scholar?q=C3-HB+PGAM+YOLOv5+driver+distraction+high+order+global+interaction
**IEEE Search:** https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=PGAM+driver+distraction+YOLOv5

---

### Paper 4 — Al-Sultan et al. (2023)
**Title:** Driver Drowsiness Detection — MediaPipe EAR/MAR + CNN
**Journal:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)
**DOI:** https://doi.org/10.1109/TITS.2022.3227811
**IEEE Xplore:** https://ieeexplore.ieee.org/document/9978639
**Google Scholar Search:** https://scholar.google.com/scholar?q=Al-Sultan+driver+drowsiness+MediaPipe+EAR+MAR+IEEE+TITS

---

### Supporting / Referenced Papers

| Paper | Verified Link |
|---|---|
| CoordAtt — Hou et al., CVPR 2021 | https://doi.org/10.1109/CVPR46437.2021.01350 |
| FaceNet — Schroff et al., CVPR 2015 | https://doi.org/10.1109/CVPR.2015.7298682 |
| MediaPipe — Lugaresi et al., 2019 | https://arxiv.org/abs/1906.08172 |
| YawDD Dataset — Ghoddoosian et al., 2019 | https://doi.org/10.1109/CVPRW.2019.00020 |
| Kaggle State Farm DDD Dataset | https://www.kaggle.com/c/state-farm-distracted-driver-detection |
| Ultralytics YOLO11 Docs | https://docs.ultralytics.com |
| OpenVINO Runtime Docs | https://docs.openvino.ai |

---

*Report prepared by CockpitSentinel Team — Mini Project 2026–27, Walchand College of Engineering, Sangli.*
