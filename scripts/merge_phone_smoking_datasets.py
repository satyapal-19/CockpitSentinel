"""Merge existing smoking dataset with a dedicated driver phone dataset.

Creates a balanced super-dataset for YOLO11 training:
- Preserves high-recall smoking annotations
- Boosts phone use instances to 600+
- Remaps labels into a standardized unified schema:
    0: earphone
    1: person
    2: smoke
    3: using_phone

Usage:
    python scripts/merge_phone_smoking_datasets.py \
        --base-dir data/external/distraction_phone_smoking \
        --phone-dir data/external/phone_dataset \
        --output-dir data/processed/merged_phone_smoking
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


STANDARD_NAMES = ["earphone", "person", "smoke", "using_phone"]
STANDARD_MAP = {
    "earphone": 0,
    "headphone": 0,
    "earphones": 0,
    "person": 1,
    "driver": 1,
    "smoke": 2,
    "smoking": 2,
    "cigarette": 2,
    "cigar": 2,
    "vape": 2,
    "using_phone": 3,
    "phone": 3,
    "cellphone": 3,
    "mobile": 3,
    "mobile-use": 3,
    "cell phone": 3,
}


def load_dataset_config(dataset_dir: Path) -> dict[str, Any]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml missing in {dataset_dir}")
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_split(
    split_name: str,
    source_dir: Path,
    source_names: list[str] | dict[int, str],
    target_img_dir: Path,
    target_lbl_dir: Path,
    prefix: str,
    stats: Counter[int],
) -> int:
    img_dir = source_dir / split_name / "images"
    if not img_dir.exists():
        img_dir = source_dir / split_name
    lbl_dir = source_dir / split_name / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[WARN] Split {split_name} not found in {source_dir}")
        return 0

    if isinstance(source_names, dict):
        source_names_list = [source_names[i] for i in sorted(source_names.keys())]
    else:
        source_names_list = list(source_names)

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    copied_count = 0

    for img_path in images:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        with lbl_path.open("r", encoding="utf-8") as f:
            lines = [l.strip().split() for l in f.readlines() if l.strip()]

        remapped_lines = []
        for parts in lines:
            if not parts:
                continue
            src_cls_id = int(parts[0])
            if src_cls_id >= len(source_names_list):
                continue
            src_cls_name = str(source_names_list[src_cls_id]).lower().strip()

            if src_cls_name in STANDARD_MAP:
                target_cls_id = STANDARD_MAP[src_cls_name]
                remapped_lines.append(f"{target_cls_id} " + " ".join(parts[1:]))
                stats[target_cls_id] += 1

        if remapped_lines:
            dest_img_name = f"{prefix}_{img_path.name}"
            dest_lbl_name = f"{prefix}_{stem}.txt"

            shutil.copy(img_path, target_img_dir / dest_img_name)
            with (target_lbl_dir / dest_lbl_name).open("w", encoding="utf-8") as f:
                f.write("\n".join(remapped_lines) + "\n")
            copied_count += 1

    return copied_count


def merge_datasets(
    base_dir: Path,
    phone_dir: Path,
    output_dir: Path,
) -> None:
    print(f"\n[INFO] Loading base dataset: {base_dir}")
    base_cfg = load_dataset_config(base_dir)
    print(f"[INFO] Loading phone dataset: {phone_dir}")
    phone_cfg = load_dataset_config(phone_dir)

    train_stats: Counter[int] = Counter()
    val_stats: Counter[int] = Counter()

    for split in ["train", "valid"]:
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        target_stats = train_stats if split == "train" else val_stats

        # 1. Base dataset
        c1 = merge_split(
            split, base_dir, base_cfg["names"], img_out, lbl_out, prefix="base", stats=target_stats
        )
        # 2. Phone dataset
        c2 = merge_split(
            split, phone_dir, phone_cfg["names"], img_out, lbl_out, prefix="phone", stats=target_stats
        )
        print(f"[INFO] Split '{split}' merged: {c1} images from base, {c2} images from phone (Total: {c1 + c2})")

    # Write merged data.yaml
    merged_yaml = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(STANDARD_NAMES),
        "names": STANDARD_NAMES,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(merged_yaml, f)

    print("\n" + "=" * 65)
    print("  MERGED DATASET CREATED SUCCESSFULLY")
    print("=" * 65)
    print(f"Output Directory: {output_dir.resolve()}")
    print(f"Config File:      {output_dir / 'data.yaml'}")
    print("\n--- Train Annotations Breakdown ---")
    for idx, name in enumerate(STANDARD_NAMES):
        print(f"  Class {idx} ({name:<12s}): {train_stats[idx]:5d} boxes")

    print("\n--- Valid Annotations Breakdown ---")
    for idx, name in enumerate(STANDARD_NAMES):
        print(f"  Class {idx} ({name:<12s}): {val_stats[idx]:5d} boxes")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge distraction datasets.")
    parser.add_argument("--base-dir", type=Path, required=True, help="Base dataset path.")
    parser.add_argument("--phone-dir", type=Path, required=True, help="Phone dataset path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/merged_phone_smoking"),
        help="Destination directory for merged dataset.",
    )

    args = parser.parse_args()
    merge_datasets(args.base_dir, args.phone_dir, args.output_dir)


if __name__ == "__main__":
    main()
