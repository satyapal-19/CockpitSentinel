"""Benchmark CockpitSentinel on the Kaggle NTHU Driver Drowsiness Dataset (DDD).

Evaluates:
- Drowsiness / Fatigue detection across 4 categories:
    1. notdrowsy (100 images) - Attentive driving negative control
    2. drowsy/sleepyCombination (50 images) - Micro-sleep and eye closure
    3. drowsy/slowBlinkWithNodding (50 images) - Slow blinking and head pitch/nodding
    4. drowsy/yawning (50 images) - Fatigue yawning
- YOLO11s distraction model out-of-domain cross-validation (evaluating false alarms).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from cockpit_sentinel.drowsiness.detector import DrowsinessConfig, DrowsinessDetector


def evaluate_nthu(
    nthu_dir: Path,
    landmarker_path: Path,
    distraction_model_path: Path | None = None,
    ear_thresh: float = 0.22,
    mar_thresh: float = 0.60,
    yaw_thresh: float = 30.0,
) -> dict[str, float]:
    print("=" * 80)
    print("      KAGGLE NTHU DRIVER DROWSINESS DATASET (DDD) ACCURACY BENCHMARK")
    print("=" * 80)
    print(f"Dataset Root     : {nthu_dir}")
    print(f"Face Landmarker  : {landmarker_path}")
    if distraction_model_path:
        print(f"Distraction Model: {distraction_model_path}")
    print(f"Thresholds       : EAR < {ear_thresh:.2f}, MAR > {mar_thresh:.2f}, Yaw > {yaw_thresh:.1f}°")
    print("-" * 80)

    cfg = DrowsinessConfig(
        eye_aspect_ratio_threshold=ear_thresh,
        mouth_aspect_ratio_threshold=mar_thresh,
        head_yaw_threshold_degrees=yaw_thresh,
        minimum_consecutive_frames=1,
    )
    detector = DrowsinessDetector(landmarker_path, config=cfg)
    yolo_model = YOLO(str(distraction_model_path)) if distraction_model_path and distraction_model_path.exists() else None

    categories = {
        "notdrowsy": {
            "path": nthu_dir / "notdrowsy",
            "is_drowsy": False,
            "description": "Safe Attentive Driving (Negative Control)",
        },
        "sleepyCombination": {
            "path": nthu_dir / "drowsy" / "sleepyCombination",
            "is_drowsy": True,
            "description": "Drowsy (Eye Closure & Fatigue)",
        },
        "slowBlinkWithNodding": {
            "path": nthu_dir / "drowsy" / "slowBlinkWithNodding",
            "is_drowsy": True,
            "description": "Drowsy (Slow Blink & Head Nodding)",
        },
        "yawning": {
            "path": nthu_dir / "drowsy" / "yawning",
            "is_drowsy": True,
            "description": "Drowsy (Fatigue Yawning)",
        },
    }

    results = {}
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_tn = 0
    total_images_tested = 0
    yolo_false_alarms = 0

    for cat_name, cat_info in categories.items():
        folder = cat_info["path"]
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if not images:
            continue

        cat_tp = 0
        cat_fn = 0
        cat_fp = 0
        cat_tn = 0
        ears = []
        mars = []
        yaws = []
        cat_yolo_detections = 0

        for img_p in images:
            frame = cv2.imread(str(img_p))
            if frame is None:
                continue

            total_images_tested += 1
            res = detector.analyze(frame)

            if res.eye_aspect_ratio is not None:
                ears.append(res.eye_aspect_ratio)
            if res.mouth_aspect_ratio is not None:
                mars.append(res.mouth_aspect_ratio)
            if res.head_yaw_degrees is not None:
                yaws.append(abs(res.head_yaw_degrees))

            # Drowsiness flag: eyes closed, yawning, or looking away (head nodding/turn)
            is_flagged = res.signals.eyes_closed or res.signals.yawning or res.signals.looking_away

            if cat_info["is_drowsy"]:
                if is_flagged:
                    cat_tp += 1
                    total_tp += 1
                else:
                    cat_fn += 1
                    total_fn += 1
            else:
                if is_flagged:
                    cat_fp += 1
                    total_fp += 1
                else:
                    cat_tn += 1
                    total_tn += 1

            # YOLO distraction cross-check
            if yolo_model:
                preds = yolo_model(frame, conf=0.35, verbose=False)
                for r in preds:
                    if hasattr(r, "boxes") and len(r.boxes) > 0:
                        for b in r.boxes:
                            cname = r.names.get(int(b.cls[0].item()), "").lower()
                            if "phone" in cname or "smoke" in cname:
                                cat_yolo_detections += 1
                                yolo_false_alarms += 1

        tested = len(images)
        correct = cat_tp if cat_info["is_drowsy"] else cat_tn
        acc = (correct / tested) * 100 if tested > 0 else 0.0

        results[cat_name] = {
            "description": cat_info["description"],
            "tested": tested,
            "correct": correct,
            "accuracy_pct": acc,
            "avg_ear": float(np.mean(ears)) if ears else 0.0,
            "avg_mar": float(np.mean(mars)) if mars else 0.0,
            "avg_yaw": float(np.mean(yaws)) if yaws else 0.0,
            "yolo_distract_det": cat_yolo_detections,
        }

    # Print breakdown table
    print(f"{'Category':<22} | {'Description':<35} | {'Tested':<6} | {'Correct':<7} | {'Accuracy':<8} | {'Avg EAR':<7} | {'Avg MAR':<7}")
    print("-" * 105)
    for c_name, stats in results.items():
        print(
            f"{c_name:<22} | {stats['description']:<35} | {stats['tested']:<6} | {stats['correct']:<7} | {stats['accuracy_pct']:>7.2f}% | {stats['avg_ear']:<7.3f} | {stats['avg_mar']:<7.3f}"
        )
    print("=" * 105)

    total_drowsy = total_tp + total_fn
    total_attentive = total_fp + total_tn
    accuracy = (total_tp + total_tn) / total_images_tested if total_images_tested > 0 else 0.0
    recall = total_tp / total_drowsy if total_drowsy > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = total_tn / total_attentive if total_attentive > 0 else 0.0

    print("\n--- OVERALL KAGGLE NTHU BENCHMARK METRICS ---")
    print(f"Total Evaluated Benchmark Images : {total_images_tested} ({total_drowsy} Drowsy + {total_attentive} Attentive)")
    print(f"True Positives (TP)              : {total_tp} / {total_drowsy} drowsy correctly flagged")
    print(f"True Negatives (TN)              : {total_tn} / {total_attentive} attentive correctly cleared")
    print(f"False Positives (FP)             : {total_fp}")
    print(f"False Negatives (FN)             : {total_fn}")
    print("-" * 50)
    print(f"Accuracy                         : {accuracy*100:.2f}%")
    print(f"Sensitivity / Recall (TP Rate)   : {recall*100:.2f}%")
    print(f"Precision                        : {precision*100:.2f}%")
    print(f"Specificity (TN Rate)            : {specificity*100:.2f}%")
    print(f"F1-Score                         : {f1*100:.2f}%")
    print("-" * 50)

    if yolo_model and total_images_tested > 0:
        yolo_fpr = (yolo_false_alarms / total_images_tested) * 100
        print(f"YOLO Distraction Model Specificity: {100-yolo_fpr:.2f}% (False Alarm Rate: {yolo_fpr:.2f}%)")
    print("=" * 50 + "\n")

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": specificity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on Kaggle NTHU DDD benchmark.")
    parser.add_argument(
        "--nthu-dir",
        type=Path,
        default=Path("D:/CockpitSentinel/data/external/nthu/train"),
        help="Path to NTHU train folder.",
    )
    parser.add_argument(
        "--landmarker",
        type=Path,
        default=Path("D:/CockpitSentinel/models/pretrained/face_landmarker.task"),
        help="Path to face landmarker model.",
    )
    parser.add_argument(
        "--distraction-model",
        type=Path,
        default=Path("weights/trained/yolo11s_distraction/best.pt"),
        help="Path to trained YOLO distraction model.",
    )
    parser.add_argument("--ear-thresh", type=float, default=0.22, help="EAR threshold.")
    parser.add_argument("--mar-thresh", type=float, default=0.60, help="MAR threshold.")
    parser.add_argument("--yaw-thresh", type=float, default=30.0, help="Yaw threshold.")
    args = parser.parse_args()

    evaluate_nthu(
        nthu_dir=args.nthu_dir,
        landmarker_path=args.landmarker,
        distraction_model_path=args.distraction_model,
        ear_thresh=args.ear_thresh,
        mar_thresh=args.mar_thresh,
        yaw_thresh=args.yaw_thresh,
    )


if __name__ == "__main__":
    main()
