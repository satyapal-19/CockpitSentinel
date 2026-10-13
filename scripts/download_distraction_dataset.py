"""Download and prepare driver phone and smoking detection datasets for YOLO training.

Supports:
1. Roboflow Universe API download (YOLOv8/YOLO11 format).
2. Direct Kaggle download.
3. Verification and inspection of dataset data.yaml.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("data/external/distraction_phone_smoking")

# Curated high-quality public Roboflow Universe datasets for driver distraction:
RECOMMENDED_DATASETS = {
    "driver_distraction": {
        "workspace": "1539379769-qq-com",
        "project": "driver-smoking-phoneuse-earphone-detection-9nlii",
        "version": 1,
        "description": "Driver smoking, phone use, and earphone detection (in-cabin driver view).",
    },
    "driver_behavior": {
        "workspace": "smoking-wislh",
        "project": "driver-behavior-ckmss",
        "version": 1,
        "description": "Driver behavior: Smoking, Mobile-Use, Drinking, Eating.",
    },
}


def download_from_roboflow(
    api_key: str,
    output_dir: Path,
    dataset_key: str = "driver_distraction",
) -> Path:
    """Download dataset from Roboflow Universe into output_dir in YOLO format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] 'roboflow' package is not installed.")
        print("Please install it: pip install roboflow")
        sys.exit(1)

    info = RECOMMENDED_DATASETS.get(dataset_key, RECOMMENDED_DATASETS["driver_distraction"])
    print(f"\n[INFO] Connecting to Roboflow Universe...")
    print(f"       Project: {info['workspace']}/{info['project']} (v{info['version']})")
    print(f"       Purpose: {info['description']}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(info["workspace"]).project(info["project"])
    version = project.version(info["version"])

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = version.download(model_format="yolov11", location=str(output_dir.resolve()))
    print(f"\n[SUCCESS] Dataset downloaded to: {dataset.location}")
    return Path(dataset.location)


def verify_dataset(dataset_dir: Path) -> bool:
    """Check dataset structure and data.yaml."""
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"[WARN] data.yaml not found at: {data_yaml}")
        return False

    import yaml

    with data_yaml.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("\n--- Dataset Summary ---")
    print(f"Location: {dataset_dir.resolve()}")
    print(f"Classes:  {cfg.get('names', 'Unknown')}")
    print(f"NC:       {cfg.get('nc', 'Unknown')}")

    for split in ["train", "val", "test"]:
        split_path = dataset_dir / split / "images"
        if not split_path.exists():
            split_path = dataset_dir / split
        if split_path.exists() and split_path.is_dir():
            count = len(list(split_path.glob("*.jpg")) + list(split_path.glob("*.png")))
            print(f"  {split.upper()}: {count} images found")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download phone & smoking dataset for YOLO11 training."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var). Free from roboflow.com",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(RECOMMENDED_DATASETS.keys()),
        default="driver_distraction",
        help="Dataset preset to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save downloaded dataset.",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        help="Verify an existing downloaded dataset folder.",
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset(args.verify_only)
        return

    if not args.api_key:
        print("\n" + "=" * 60)
        print("  ROBOFLOW API KEY REQUIRED FOR DIRECT DOWNLOAD")
        print("=" * 60)
        print("1. Get your free API key at: https://app.roboflow.com")
        print("2. Run:")
        print(f"   python scripts/download_distraction_dataset.py --api-key YOUR_KEY")
        print("   OR export ROBOFLOW_API_KEY=YOUR_KEY")
        print("\nAlternatively, download manually via browser from:")
        for k, v in RECOMMENDED_DATASETS.items():
            print(f"  - https://universe.roboflow.com/{v['workspace']}/{v['project']}")
        print(f"and unzip into: {args.output_dir.resolve()}\n")
        return

    downloaded_dir = download_from_roboflow(args.api_key, args.output_dir, args.dataset)
    verify_dataset(downloaded_dir)


if __name__ == "__main__":
    main()
