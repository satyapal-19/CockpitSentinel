"""Evaluate CockpitSentinel on the YawDD (Yawning Detection Dataset) benchmark.

Evaluates:
1. Drowsiness / Fatigue (Yawn detection via MAR & MediaPipe Face Landmarker):
   - Positive class: *-Yawning.avi
   - Negative control: *-Normal.avi
   - Confusion control: *-Talking.avi (talking mouth movement should not trigger yawn)
2. Distraction False Alarm Rate (YOLO distraction detector on YawDD):
   - Verifies phone and smoking false positive rate on benchmark driving videos.

Computes:
- Video-level and frame-level Accuracy, Precision, Recall, and F1-score
- Mean MAR during Normal vs Talking vs Yawning
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from cockpit_sentinel.drowsiness.detector import DrowsinessConfig, DrowsinessDetector


def evaluate_yawdd(
    yawdd_dir: Path,
    landmarker_path: Path,
    distraction_model_path: Path | None = None,
    mar_thresh: float = 0.60,
    ear_thresh: float = 0.22,
    frame_stride: int = 5,
) -> dict[str, Any]:
    print("=" * 80)
    print("       COCKPITSENTINEL YAWDD BENCHMARK ACCURACY & RECALL EVALUATION")
    print("=" * 80)
    print(f"YawDD Dataset Path : {yawdd_dir}")
    print(f"Face Landmarker    : {landmarker_path}")
    if distraction_model_path:
        print(f"Distraction Model  : {distraction_model_path}")
    print(f"Thresholds         : MAR={mar_thresh:.2f}, EAR={ear_thresh:.2f}")
    print(f"Frame Stride       : {frame_stride} (evaluating every {frame_stride}th frame)")
    print("-" * 80)

    cfg = DrowsinessConfig(
        mouth_aspect_ratio_threshold=mar_thresh,
        eye_aspect_ratio_threshold=ear_thresh,
    )
    detector = DrowsinessDetector(landmarker_path, config=cfg)
    
    yolo_model = YOLO(str(distraction_model_path)) if distraction_model_path and distraction_model_path.exists() else None

    # Discover videos
    videos = sorted(yawdd_dir.glob("**/*.avi"))
    if not videos:
        print(f"[ERROR] No .avi videos found in {yawdd_dir}")
        return {}

    video_results = []
    
    total_yawn_frames = 0
    detected_yawn_frames = 0
    total_normal_frames = 0
    false_yawn_frames_normal = 0
    total_talking_frames = 0
    false_yawn_frames_talking = 0
    
    distraction_frames_tested = 0
    distraction_false_alarms = 0

    mar_normal_vals = []
    mar_talking_vals = []
    mar_yawning_vals = []

    for vid_path in videos:
        fname = vid_path.name
        # Determine ground truth from filename
        if "Yawning" in fname:
            gt_type = "yawning"
        elif "Talking" in fname:
            gt_type = "talking"
        elif "Normal" in fname:
            gt_type = "normal"
        else:
            # Skip unclassified or multi-behavior long dash videos for strict ground-truth benchmark
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        vid_mar_list = []
        yawn_frames_in_vid = 0
        total_eval_frames_in_vid = 0
        distract_in_vid = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                total_eval_frames_in_vid += 1
                analysis = detector.analyze(frame)
                
                if analysis.mouth_aspect_ratio is not None:
                    mar_val = analysis.mouth_aspect_ratio
                    vid_mar_list.append(mar_val)
                    if gt_type == "normal":
                        mar_normal_vals.append(mar_val)
                    elif gt_type == "talking":
                        mar_talking_vals.append(mar_val)
                    elif gt_type == "yawning":
                        mar_yawning_vals.append(mar_val)

                if analysis.signals.yawning:
                    yawn_frames_in_vid += 1

                # Distraction model test (evaluating false positive rate on driver faces)
                if yolo_model is not None:
                    distraction_frames_tested += 1
                    preds = yolo_model(frame, conf=0.35, verbose=False)
                    for r in preds:
                        if hasattr(r, "boxes") and len(r.boxes) > 0:
                            for b in r.boxes:
                                c_name = r.names.get(int(b.cls[0].item()), "").lower()
                                if "phone" in c_name or "smoke" in c_name:
                                    distract_in_vid += 1
                                    distraction_false_alarms += 1

            frame_idx += 1
        cap.release()

        max_mar = max(vid_mar_list) if vid_mar_list else 0.0
        avg_mar = float(np.mean(vid_mar_list)) if vid_mar_list else 0.0
        yawn_detected_in_vid = yawn_frames_in_vid > 0

        if gt_type == "yawning":
            total_yawn_frames += total_eval_frames_in_vid
            detected_yawn_frames += yawn_frames_in_vid
        elif gt_type == "normal":
            total_normal_frames += total_eval_frames_in_vid
            false_yawn_frames_normal += yawn_frames_in_vid
        elif gt_type == "talking":
            total_talking_frames += total_eval_frames_in_vid
            false_yawn_frames_talking += yawn_frames_in_vid

        video_results.append({
            "file": fname,
            "ground_truth": gt_type,
            "frames_tested": total_eval_frames_in_vid,
            "yawn_frames": yawn_frames_in_vid,
            "detected_yawn": yawn_detected_in_vid,
            "avg_mar": avg_mar,
            "max_mar": max_mar,
            "distract_false_pos": distract_in_vid,
        })

    # Summary calculations (Video-level metrics)
    yawn_vids = [v for v in video_results if v["ground_truth"] == "yawning"]
    normal_vids = [v for v in video_results if v["ground_truth"] == "normal"]
    talking_vids = [v for v in video_results if v["ground_truth"] == "talking"]
    negative_vids = normal_vids + talking_vids

    tp = sum(1 for v in yawn_vids if v["detected_yawn"])
    fn = len(yawn_vids) - tp
    fp = sum(1 for v in negative_vids if v["detected_yawn"])
    tn = len(negative_vids) - fp

    vid_accuracy = (tp + tn) / len(video_results) if video_results else 0.0
    vid_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    vid_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    vid_f1 = (2 * vid_precision * vid_recall) / (vid_precision + vid_recall) if (vid_precision + vid_recall) > 0 else 0.0
    vid_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n{'Video Name':<38} | {'GT Class':<9} | {'Tested':<6} | {'Yawn Det':<8} | {'Avg MAR':<7} | {'Max MAR':<7} | {'Verdict':<8}")
    print("-" * 95)
    for v in video_results:
        verdict = "CORRECT" if (v["detected_yawn"] and v["ground_truth"] == "yawning") or (not v["detected_yawn"] and v["ground_truth"] != "yawning") else "MISMATCH"
        print(f"{v['file']:<38} | {v['ground_truth']:<9} | {v['frames_tested']:<6} | {v['yawn_frames']:<8} | {v['avg_mar']:<7.3f} | {v['max_mar']:<7.3f} | {verdict:<8}")

    print("=" * 95)
    print("\n--- YAWDD BENCHMARK STATISTICAL METRICS ---")
    print(f"Total Labeled Videos Evaluated : {len(video_results)}")
    print(f"True Positives (TP)            : {tp} / {len(yawn_vids)} fatigue videos detected")
    print(f"True Negatives (TN)            : {tn} / {len(negative_vids)} normal/talking videos correct")
    print(f"False Positives (FP)           : {fp}")
    print(f"False Negatives (FN)           : {fn}")
    print("-" * 50)
    print(f"Video-Level Accuracy           : {vid_accuracy*100:.2f}%")
    print(f"Sensitivity / Recall (TP Rate) : {vid_recall*100:.2f}%")
    print(f"Precision                      : {vid_precision*100:.2f}%")
    print(f"Specificity (TN Rate)          : {vid_specificity*100:.2f}%")
    print(f"F1-Score                       : {vid_f1*100:.2f}%")
    print("-" * 50)
    print(f"Mean MAR in Normal Driving     : {np.mean(mar_normal_vals):.3f}")
    print(f"Mean MAR while Talking         : {np.mean(mar_talking_vals):.3f}")
    print(f"Mean MAR during Yawning        : {np.mean(mar_yawning_vals):.3f}")

    if distraction_frames_tested > 0:
        distract_fpr = (distraction_false_alarms / distraction_frames_tested) * 100
        print("-" * 50)
        print("Distraction Model Cross-Validation on YawDD Driver Videos:")
        print(f"  Frames Evaluated for False Phone/Smoking : {distraction_frames_tested}")
        print(f"  False Distraction Detections            : {distraction_false_alarms}")
        print(f"  Distraction False Positive Rate (FPR)   : {distract_fpr:.2f}% (Specificity: {100-distract_fpr:.2f}%)")
    print("=" * 95 + "\n")

    return {
        "accuracy": vid_accuracy,
        "recall": vid_recall,
        "precision": vid_precision,
        "f1": vid_f1,
        "specificity": vid_specificity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CockpitSentinel on YawDD.")
    parser.add_argument(
        "--yawdd-dir",
        type=Path,
        default=Path("data/external/yawdd/Mirror"),
        help="Path to YawDD Mirror folder with ground-truth labeled videos.",
    )
    parser.add_argument(
        "--landmarker",
        type=Path,
        default=Path("D:/CockpitSentinel/models/pretrained/face_landmarker.task"),
        help="Path to MediaPipe face landmarker task file.",
    )
    parser.add_argument(
        "--distraction-model",
        type=Path,
        default=Path("weights/trained/yolo11s_distraction/best.pt"),
        help="Path to YOLO distraction model checkpoint.",
    )
    parser.add_argument("--mar-thresh", type=float, default=0.60, help="MAR threshold for yawning.")
    parser.add_argument("--ear-thresh", type=float, default=0.22, help="EAR threshold for eye closure.")
    parser.add_argument("--stride", type=int, default=5, help="Frame stride for speed.")
    args = parser.parse_args()

    evaluate_yawdd(
        yawdd_dir=args.yawdd_dir,
        landmarker_path=args.landmarker,
        distraction_model_path=args.distraction_model,
        mar_thresh=args.mar_thresh,
        ear_thresh=args.ear_thresh,
        frame_stride=args.stride,
    )


if __name__ == "__main__":
    main()
