"""Automated Dataset Pipeline for CockpitSentinel.

Converts mobile phone videos and photos into a fully annotated YOLO11 dataset
with zero manual labeling required.

Workflow:
1. Place raw phone videos/photos in:
   - data/raw_input/normal/    (Clean attentive driving, hands on wheel)
   - data/raw_input/phone/     (Calling, texting, holding mobile phone)
   - data/raw_input/smoking/   (Cigarette, vaping, smoking actions)

2. Run:
   python scripts/auto_build_dataset.py

3. Output:
   data/custom_distraction/
     ├── data.yaml
     ├── train/ (images, labels)
     ├── valid/ (images, labels)
     └── test/  (images, labels)
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

# Standard 2-class distraction mapping
CLASS_MAP = {
    "smoke": 0,
    "using_phone": 1,
}


def is_blurry(image: cv2.typing.MatLike, threshold: float = 60.0) -> bool:
    """Detect if an extracted video frame is motion-blurred."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def extract_frames_from_video(
    video_path: Path, output_dir: Path, target_fps: float = 2.0
) -> list[Path]:
    """Extract de-duplicated, non-blurry frames from a video at target_fps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f" [WARN] Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps / target_fps)))

    extracted_paths: list[Path] = []
    frame_idx = 0
    saved_idx = 0

    stem = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0 and not is_blurry(frame):
            out_path = output_dir / f"{stem}_fr{saved_idx:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_paths.append(out_path)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return extracted_paths


def process_raw_inputs(input_root: Path, staging_dir: Path, target_fps: float = 2.0) -> dict[str, list[Path]]:
    """Scan raw inputs (images & videos) and gather clean frames by category."""
    categories = ["normal", "phone", "smoking"]
    categorized_frames: dict[str, list[Path]] = {c: [] for c in categories}

    for cat in categories:
        cat_dir = input_root / cat
        if not cat_dir.exists():
            cat_dir.mkdir(parents=True, exist_ok=True)
            continue

        stage_cat_dir = staging_dir / cat
        stage_cat_dir.mkdir(parents=True, exist_ok=True)

        # 1. Process existing photos (.jpg, .png, .jpeg, .webp)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for img_file in cat_dir.glob(ext):
                dest = stage_cat_dir / img_file.name
                shutil.copy2(img_file, dest)
                categorized_frames[cat].append(dest)

        # 2. Extract frames from video files (.mp4, .mov, .avi, .mkv)
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.m4v"):
            for vid_file in cat_dir.glob(ext):
                print(f" -> Extracting frames from: {vid_file.name} ({cat})...")
                frames = extract_frames_from_video(vid_file, stage_cat_dir, target_fps=target_fps)
                categorized_frames[cat].extend(frames)
                print(f"    Extracted {len(frames)} clean frames.")

    return categorized_frames


def auto_annotate(
    categorized_frames: dict[str, list[Path]],
    output_root: Path,
    model_path: Path,
    conf_thresh: float = 0.20,
) -> None:
    """Auto-label frames with YOLO and split into train/val/test."""
    print(f"\n[AI AUTO-LABELER] Loading pre-trained model: {model_path}...")
    model = YOLO(str(model_path))

    # Identify model class indices for phone and smoke
    phone_indices = [
        idx for idx, name in model.names.items()
        if any(term in str(name).lower() for term in ("phone", "cell", "calling", "mobile"))
    ]
    smoke_indices = [
        idx for idx, name in model.names.items()
        if any(term in str(name).lower() for term in ("smoke", "smoking", "cigarette", "vape"))
    ]

    print(f" -> Detected model indices: Phone={phone_indices}, Smoke={smoke_indices}")

    # Build dataset splits
    splits = ["train", "valid", "test"]
    for s in splits:
        (output_root / s / "images").mkdir(parents=True, exist_ok=True)
        (output_root / s / "labels").mkdir(parents=True, exist_ok=True)

    total_images = 0
    phone_box_count = 0
    smoke_box_count = 0
    negative_frame_count = 0

    # Process all categories
    all_samples: list[tuple[Path, str]] = []
    for cat, paths in categorized_frames.items():
        for p in paths:
            all_samples.append((p, cat))

    if not all_samples:
        print("[NOTICE] No input files found! Please add videos/photos to data/raw_input/.")
        return

    # Shuffle for split
    random.seed(42)
    random.shuffle(all_samples)

    for p, cat in all_samples:
        # Determine split (70% train, 20% val, 10% test)
        r = random.random()
        split = "train" if r < 0.70 else ("valid" if r < 0.90 else "test")

        dest_img = output_root / split / "images" / p.name
        dest_lbl = output_root / split / "labels" / f"{p.stem}.txt"

        shutil.copy2(p, dest_img)

        lines: list[str] = []

        if cat == "normal":
            # Normal driving is pure negative background: empty label file
            negative_frame_count += 1
        else:
            # Run inference to detect bounding boxes automatically
            results = model.predict(source=str(p), conf=conf_thresh, verbose=False)
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        xywhn = box.xywhn[0].tolist()  # [cx, cy, w, h] normalized

                        target_class: int | None = None
                        if cls_id in smoke_indices:
                            target_class = CLASS_MAP["smoke"]  # 0
                            smoke_box_count += 1
                        elif cls_id in phone_indices:
                            target_class = CLASS_MAP["using_phone"]  # 1
                            phone_box_count += 1

                        if target_class is not None:
                            cx, cy, w, h = xywhn
                            lines.append(f"{target_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # If model missed but image was explicitly in phone or smoking folder,
            # we keep it as a hard negative or let user verify
            if not lines:
                negative_frame_count += 1

        # Write YOLO label file
        with dest_lbl.open("w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        total_images += 1

    # Write data.yaml
    yaml_content = f"""# CockpitSentinel Custom 2-Class Distraction Dataset
path: {output_root.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: smoke
  1: using_phone
"""
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print("\n" + "=" * 65)
    print("  AUTOMATED DATASET GENERATION COMPLETE!")
    print("=" * 65)
    print(f" Total Frames Processed: {total_images}")
    print(f" Phone Distractions (Class 1): {phone_box_count} boxes")
    print(f" Smoking Distractions (Class 0): {smoke_box_count} boxes")
    print(f" Attentive Negative Background Frames: {negative_frame_count}")
    print(f" data.yaml created at: {yaml_path.resolve()}")
    print("=" * 65)
    print("\nYou can now start training immediately with:")
    print(f"  python scripts/train_yolo11_distraction.py --data {yaml_path.as_posix()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-build YOLO dataset from mobile phone photos and videos."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw_input"),
        help="Directory with normal/, phone/, and smoking/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/custom_distraction"),
        help="Output destination for train/val/test and data.yaml.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("weights/trained/yolo11s_balanced_distraction/best.pt"),
        help="Pre-trained distraction model to use for auto-labeling.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frame extraction rate from videos (frames per second).",
    )
    args = parser.parse_args()

    # Create default input folders if they don't exist
    for sub in ("normal", "phone", "smoking"):
        (args.input_dir / sub).mkdir(parents=True, exist_ok=True)

    staging = Path("data/.staging_frames")
    staging.mkdir(parents=True, exist_ok=True)

    categorized = process_raw_inputs(args.input_dir, staging, target_fps=args.fps)

    auto_annotate(
        categorized_frames=categorized,
        output_root=args.output_dir,
        model_path=args.model,
    )

    # Clean up staging directory
    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
