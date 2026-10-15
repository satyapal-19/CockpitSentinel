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

    present_indices = [i for i in range(len(model.names)) if i in metrics.box.ap_class_index] if hasattr(metrics.box, 'ap_class_index') else list(range(len(metrics.box.p)))
    
    class_metrics = {}
    for list_idx, cls_id in enumerate(metrics.box.ap_class_index if hasattr(metrics.box, 'ap_class_index') else range(len(metrics.box.p))):
        c_name = model.names.get(int(cls_id), f"class_{cls_id}")
        p = float(metrics.box.p[list_idx]) if list_idx < len(metrics.box.p) else 0.0
        r = float(metrics.box.r[list_idx]) if list_idx < len(metrics.box.r) else 0.0
        f1 = float(metrics.box.f1[list_idx]) if hasattr(metrics.box, 'f1') and list_idx < len(metrics.box.f1) else (2 * p * r / (p + r + 1e-16))
        ap50 = float(metrics.box.ap50[list_idx]) if list_idx < len(metrics.box.ap50) else 0.0
        ap = float(metrics.box.ap[list_idx]) if list_idx < len(metrics.box.ap) else 0.0

        class_metrics[c_name] = {
            "precision": p,
            "recall": r,
            "f1_score": f1,
            "mAP50": ap50,
            "mAP50_95": ap,
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

    print("\n" + "=" * 85)
    print("      COCKPITSENTINEL YOLO11 DISTRACTION MODEL ACCURACY & PERFORMANCE REPORT")
    print("=" * 85)
    print(f"Overall Precision (P) : {overall_metrics['precision']*100:.2f}%")
    print(f"Overall Recall (R)    : {overall_metrics['recall']*100:.2f}%")
    print(f"Overall F1-Score      : {overall_metrics['f1_score']*100:.2f}%")
    print(f"Overall mAP@50        : {overall_metrics['mAP50']*100:.2f}%")
    print(f"Overall mAP@50-95     : {overall_metrics['mAP50_95']*100:.2f}%")
    print("-" * 85)
    print(f"{'Class Name':<16} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'mAP@50':<12} {'mAP@50-95':<12}")
    print("-" * 85)
    for c_name, data in class_metrics.items():
        print(
            f"{c_name:<16} "
            f"{data['precision']*100:>9.2f}%  "
            f"{data['recall']*100:>9.2f}%  "
            f"{data['f1_score']*100:>9.2f}%  "
            f"{data['mAP50']*100:>9.2f}%  "
            f"{data['mAP50_95']*100:>9.2f}%"
        )
    print("=" * 85 + "\n")

    if save_json:
        reports_dir = REPO_ROOT / "experiments"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"{weights_path.parent.name}_accuracy_report.json"
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
