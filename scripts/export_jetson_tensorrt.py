"""Export trained YOLO11 model to NVIDIA TensorRT engine for in-vehicle edge deployment.

Usage (on NVIDIA Jetson Orin Nano / Orin NX / Linux GPU):
    python scripts/export_jetson_tensorrt.py --model weights/trained/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_tensorrt(model_path: Path, half: bool = True, device: int = 0) -> Path:
    """Export YOLO model to TensorRT .engine format for automotive edge hardware."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    print("\n" + "=" * 65)
    print("  COCKPIT SENTINEL — VEHICLE EDGE DEPLOYMENT (TENSORRT)")
    print("=" * 65)
    print(f"Input model: {model_path.resolve()}")
    print(f"Device:      CUDA GPU #{device}")
    print(f"Precision:   {'FP16 (half=True)' if half else 'FP32'}")
    print("\n[INFO] Compiling model into optimized TensorRT engine...")

    model = YOLO(str(model_path))
    engine_path_str = model.export(
        format="engine",
        half=half,
        device=device,
        workspace=4,  # 4GB workspace memory for TensorRT optimization
    )
    engine_path = Path(engine_path_str)

    print(f"\n[SUCCESS] TensorRT Engine created at: {engine_path.resolve()}")
    print("=" * 65)
    print("To run in-vehicle with CockpitSentinel:")
    print(f"  python -m cockpit_sentinel.pipeline.live_monitor --source 0 \\")
    print(f"         --phone-model-path {engine_path} \\")
    print(f"         --smoking-model-path {engine_path} --device 0")
    print("=" * 65 + "\n")
    return engine_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO model to TensorRT engine for NVIDIA Jetson."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained PyTorch weights (e.g. weights/trained/best.pt).",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Export in FP32 instead of default FP16.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )

    args = parser.parse_args()
    export_tensorrt(args.model, half=not args.fp32, device=args.device)


if __name__ == "__main__":
    main()
