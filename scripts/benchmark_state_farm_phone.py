"""Benchmark trained distraction detector against the State Farm Distracted Driver dataset.

Evaluates out-of-domain generalization specifically for mobile phone interactions:
- c1_texting_right: Texting with right hand
- c2_phone_right:   Talking on phone with right hand
- c3_texting_left:  Texting with left hand
- c4_phone_left:    Talking on phone with left hand
- c0_normal:        Safe driving (measures false positive rate)

Usage:
    python scripts/benchmark_state_farm_phone.py --data-dir data/external/state_farm/train
    python scripts/benchmark_state_farm_phone.py --data-dir path/to/state_farm --max-samples 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


PHONE_CLASSES = {
    "c1_texting_right": "Texting (Right Hand)",
    "c2_phone_right": "Calling (Right Hand)",
    "c3_texting_left": "Texting (Left Hand)",
    "c4_phone_left": "Calling (Left Hand)",
}

NORMAL_CLASS = "c0_normal"


def benchmark_state_farm(
    model_path: Path,
    data_dir: Path,
    conf_thresh: float = 0.25,
    max_samples: int | None = 200,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run model inference across State Farm phone and normal classes."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is required: pip install ultralytics")
        sys.exit(1)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"State Farm data directory not found: {data_dir}")

    print(f"\n[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))

    target_classes = dict(PHONE_CLASSES)
    target_classes[NORMAL_CLASS] = "Safe Driving (Negative Control)"

    results: dict[str, dict[str, Any]] = {}

    print(f"[INFO] Evaluating classes with conf threshold >= {conf_thresh}...")
    print(f"[INFO] Device: {device} | Max samples per class: {max_samples or 'All'}\n")

    for class_folder, class_desc in target_classes.items():
        folder_path = data_dir / class_folder
        if not folder_path.exists():
            print(f"[WARN] Folder not found: {folder_path} (skipping)")
            continue

        images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        if not images:
            print(f"[WARN] No images found in {folder_path}")
            continue

        if max_samples and len(images) > max_samples:
            images = images[:max_samples]

        detected_count = 0
        confidences: list[float] = []

        for img_path in images:
            preds = model(str(img_path), conf=conf_thresh, device=device, verbose=False)
            has_phone = False
            top_conf = 0.0

            for r in preds:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = r.names.get(cls_id, "").lower()
                    box_conf = float(box.conf[0].item())

                    if "phone" in cls_name or cls_name in {"using_phone", "cell phone"}:
                        has_phone = True
                        if box_conf > top_conf:
                            top_conf = box_conf

            if has_phone:
                detected_count += 1
                confidences.append(top_conf)

        total_tested = len(images)
        rate = (detected_count / total_tested) * 100 if total_tested > 0 else 0.0
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        results[class_folder] = {
            "description": class_desc,
            "total": total_tested,
            "detected": detected_count,
            "rate_pct": rate,
            "avg_conf": avg_conf,
        }

    # Print summary table
    print("=" * 80)
    print(f"{'State Farm Class':<22} | {'Description':<28} | {'Tested':<7} | {'Phone Det':<9} | {'Rate %':<8} | {'Avg Conf':<8}")
    print("-" * 80)

    for c_id, stats in results.items():
        desc = stats["description"]
        tested = stats["total"]
        det = stats["detected"]
        rate = stats["rate_pct"]
        avg_c = stats["avg_conf"]
        print(f"{c_id:<22} | {desc:<28} | {tested:<7} | {det:<9} | {rate:>6.1f}% | {avg_c:>7.3f}")

    print("=" * 80)

    # Key takeaways
    phone_tested = sum(stats["total"] for c_id, stats in results.items() if c_id != NORMAL_CLASS)
    phone_detected = sum(stats["detected"] for c_id, stats in results.items() if c_id != NORMAL_CLASS)
    overall_phone_sensitivity = (phone_detected / phone_tested * 100) if phone_tested > 0 else 0.0

    normal_stats = results.get(NORMAL_CLASS)
    fpr = normal_stats["rate_pct"] if normal_stats else 0.0

    print("\n--- Summary Benchmark Metrics ---")
    print(f"Overall Phone Distraction Sensitivity: {overall_phone_sensitivity:.1f}% ({phone_detected}/{phone_tested})")
    print(f"Safe Driving False Positive Rate (FPR): {fpr:.1f}%")
    print("---------------------------------\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLO model on State Farm Distracted Driver.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("weights/trained/yolo11s_distraction/best.pt"),
        help="Path to trained YOLO model checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/external/state_farm/train"),
        help="Path to State Farm train folder containing c0_normal, c1_texting_right, etc.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Phone detection confidence threshold.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum images per class to test (default: 200). Use 0 for all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or '0' for GPU.",
    )

    args = parser.parse_args()
    samples = None if args.max_samples <= 0 else args.max_samples
    benchmark_state_farm(
        model_path=args.model,
        data_dir=args.data_dir,
        conf_thresh=args.conf,
        max_samples=samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
