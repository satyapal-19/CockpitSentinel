"""Train YOLO11 with Coordinate Attention (CoordAtt) for Driver Distraction.

Integrates Coordinate Attention into P3 and P4 detection heads to dramatically
improve small-object sensitivity and hand-to-ear phone boundary detection.

Usage:
    python scripts/train_yolo11_coordatt.py --data data/custom_distraction/data.yaml
    python scripts/train_yolo11_coordatt.py --data data/custom_distraction/data.yaml --epochs 60 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from cockpit_sentinel.detection.attention import register_attention_modules


def train_coordatt(
    data_yaml: Path,
    cfg_yaml: Path = Path("configs/models/yolo11-coordatt.yaml"),
    pretrained_weights: str = "yolo11s.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",
    project: str = "experiments/yolo11_coordatt",
    name: str = "phone_smoking_detector",
) -> Path:
    """Train Coordinate Attention YOLO11 model on distraction dataset."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")
    if not cfg_yaml.exists():
        raise FileNotFoundError(f"Model config not found at: {cfg_yaml}")

    # 1. Register CoordAtt and CBAM into Ultralytics runtime
    print("\n[INFO] Registering Coordinate Attention modules into Ultralytics...")
    register_attention_modules()

    # 2. Build model from custom architecture YAML with transfer learning
    print(f"[INFO] Initializing YOLO11-CoordAtt from: {cfg_yaml}")
    model = YOLO(str(cfg_yaml.resolve()))

    # Transfer pretrained backbone weights if available
    try:
        print(f"[INFO] Transferring pretrained weights from: {pretrained_weights}")
        model.load(pretrained_weights)
        print("[SUCCESS] Pretrained backbone weights transferred successfully.")
    except Exception as exc:
        print(f"[NOTICE] Training from scratch (backbone transfer notice: {exc})")

    # 3. Train model
    print(f"[INFO] Starting training on {data_yaml} (Epochs: {epochs}, Imgsz: {imgsz})...")
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        device=device,
        project=project,
        name=name,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n[SUCCESS] Coordinate Attention training complete! Best weights: {best_weights.resolve()}")
    return best_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO11-CoordAtt Phone & Smoking Distraction Detector."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data.yaml dataset definition file.",
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("configs/models/yolo11-coordatt.yaml"),
        help="Path to YOLO11-CoordAtt architecture YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs (default: 60).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to train on: '0', 'cpu', 'cuda:0'.",
    )
    args = parser.parse_args()

    train_coordatt(
        data_yaml=args.data,
        cfg_yaml=args.cfg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
