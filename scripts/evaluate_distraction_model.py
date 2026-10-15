"""
Evaluate CockpitSentinel YOLO Distraction Detection Model on the validation benchmark dataset.

Computes:
  - Overall mAP@50 and mAP@50-95
  - Precision, Recall, and F1-score
  - Per-class detection metrics (using_phone, smoke, earphone, person)
  - Inference latency statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate_distraction_model(
    weights_path: Path,
    data_yaml: Path,
    device: str = "cpu",
    batch_size: int = 16,
    imgsz: int = 640,
    save_json: bool = True,
) -> dict:
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    print(f"\n[INFO] Loading YOLO distraction model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[INFO] Running validation on {data_yaml} using device={device}...")
    metrics = model.val(
        data=str(data_yaml),
        device=device,
        batch=batch_size,
        imgsz=imgsz,
        plots=False,
    )

    names = model.names
    class_metrics = {}
    for idx, class_name in names.items():
        if idx < len(metrics.box.maps):
            class_metrics[class_name] = {
                "mAP50_95": float(metrics.box.maps[idx]),
            }

    overall_metrics = {
        "model_path": str(weights_path),
        "dataset_yaml": str(data_yaml),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "mAP75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1_score": float(
            2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-16)
        ),
        "inference_speed_ms": {
            "preprocess": float(metrics.speed.get("preprocess", 0.0)),
            "inference": float(metrics.speed.get("inference", 0.0)),
            "postprocess": float(metrics.speed.get("postprocess", 0.0)),
        },
        "per_class": class_metrics,
    }

    print("\n" + "=" * 65)
    print("      COCKPITSENTINEL YOLO11 DISTRACTION MODEL ACCURACY REPORT")
    print("=" * 65)
    print(f"Overall Precision (P) : {overall_metrics['precision']*100:.2f}%")
    print(f"Overall Recall (R)    : {overall_metrics['recall']*100:.2f}%")
    print(f"Overall F1-Score      : {overall_metrics['f1_score']*100:.2f}%")
    print(f"Overall mAP@50        : {overall_metrics['mAP50']*100:.2f}%")
    print(f"Overall mAP@50-95     : {overall_metrics['mAP50_95']*100:.2f}%")
    print("-" * 65)
    print(f"{'Class Name':<20} {'mAP@50-95':<15}")
    print("-" * 65)
    for c_name, data in class_metrics.items():
        print(f"{c_name:<20} {data['mAP50_95']*100:>8.2f}%")
    print("=" * 65 + "\n")

    if save_json:
        reports_dir = REPO_ROOT / "experiments"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / "distraction_model_accuracy_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(overall_metrics, f, indent=2)
        print(f"[OK] Report saved to: {report_file}")

    return overall_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO Distraction Model Accuracy.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "weights" / "trained" / "yolo11s_distraction" / "best.pt",
        help="Path to YOLO .pt checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "external" / "distraction_phone_smoking" / "data.yaml",
        help="Path to data.yaml dataset definition",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or '0')")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    evaluate_distraction_model(
        weights_path=args.weights,
        data_yaml=args.data,
        device=args.device,
        batch_size=args.batch,
        imgsz=args.imgsz,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
