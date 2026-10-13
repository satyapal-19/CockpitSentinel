"""Train a custom YOLO11s/YOLO11m distraction model (Phone & Smoking detection).

Usage:
    python scripts/train_yolo11_distraction.py --data data/external/distraction_phone_smoking/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml: Path,
    model_size: str = "yolo11s.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",
    project: str = "experiments/yolo11",
    name: str = "distraction_detector",
) -> Path:
    """Train YOLO11 model on phone and smoking dataset."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    print(f"\n[INFO] Initializing YOLO11 model: {model_size}")
    model = YOLO(model_size)

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
    print(f"\n[SUCCESS] Training finished! Best weights: {best_weights.resolve()}")
    return best_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO11 Phone and Smoking Distraction Detector."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data.yaml dataset definition file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"],
        help="Base pretrained YOLO11 model to fine-tune.",
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
        help="Image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Hardware device ('0' for GPU, 'cpu' for CPU).",
    )

    args = parser.parse_args()
    train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
