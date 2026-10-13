"""Export trained YOLO model to Intel OpenVINO (FP16) and benchmark on Intel CPU.

Usage:
    python scripts/export_openvino_model.py --model yolo11s.pt
    python scripts/export_openvino_model.py --model weights/trained/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def export_and_benchmark(model_path: Path, half: bool = True) -> Path:
    """Export YOLO model to OpenVINO format and benchmark speedup."""
    if not model_path.exists():
        # If it's a known standard model name like yolo11s.pt, ultralytics will auto-download
        if not model_path.name.endswith(".pt"):
            raise FileNotFoundError(f"Model file not found: {model_path}")

    print("\n" + "=" * 65)
    print("  COCKPIT SENTINEL — INTEL OPENVINO EXPORT & BENCHMARK")
    print("=" * 65)
    print(f"Input model: {model_path}")
    print(f"Precision:   {'FP16 (half=True)' if half else 'FP32'}")

    model = YOLO(str(model_path))

    # Export to OpenVINO
    print("\n[INFO] Exporting to OpenVINO format...")
    openvino_dir_str = model.export(format="openvino", half=half)
    openvino_dir = Path(openvino_dir_str)
    print(f"[SUCCESS] Exported OpenVINO directory: {openvino_dir.resolve()}")

    # Benchmark: PyTorch vs OpenVINO
    print("\n[INFO] Running CPU benchmark on 10 test frames (640x640)...")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    # 1. PyTorch CPU
    pt_model = YOLO(str(model_path))
    for _ in range(2):
        pt_model(dummy_frame, verbose=False)
    t0 = time.perf_counter()
    for _ in range(10):
        pt_model(dummy_frame, verbose=False)
    pt_dt = (time.perf_counter() - t0) / 10
    pt_fps = 1.0 / pt_dt

    # 2. OpenVINO CPU
    ov_model = YOLO(str(openvino_dir))
    for _ in range(2):
        ov_model(dummy_frame, verbose=False)
    t0 = time.perf_counter()
    for _ in range(10):
        ov_model(dummy_frame, verbose=False)
    ov_dt = (time.perf_counter() - t0) / 10
    ov_fps = 1.0 / ov_dt

    speedup = pt_dt / ov_dt if ov_dt > 0 else 1.0

    print("\n--- Benchmark Results on Your Intel CPU ---")
    print(f"  PyTorch CPU:   {pt_dt * 1000:6.1f} ms/frame  ({pt_fps:5.1f} FPS)")
    print(f"  OpenVINO CPU:  {ov_dt * 1000:6.1f} ms/frame  ({ov_fps:5.1f} FPS)")
    print(f"  Speedup:       {speedup:6.2f}x faster with OpenVINO!")
    print("=" * 65)

    print("\nTo use this model in CockpitSentinel:")
    print("1. Copy the exported folder to your weights directory:")
    print(f"   weights/trained/{openvino_dir.name}")
    print("2. Point configs/models.yaml to the folder path.")
    print("=" * 65 + "\n")

    return openvino_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO model to Intel OpenVINO and benchmark."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo11s.pt"),
        help="Path to YOLO .pt model (e.g. weights/trained/best.pt or yolo11s.pt).",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Export in FP32 instead of default FP16 (half).",
    )

    args = parser.parse_args()
    export_and_benchmark(args.model, half=not args.fp32)


if __name__ == "__main__":
    main()
