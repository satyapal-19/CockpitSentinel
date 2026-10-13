"""Enroll a driver's facial biometric profile from webcam or image file."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cockpit_sentinel.database.client import get_db_manager
from cockpit_sentinel.recognition.face_identifier import get_face_identifier


def enroll_from_image(driver_id: str, image_path: Path) -> None:
    """Enroll a driver's face from a static image file."""
    db = get_db_manager()
    driver = db.get_driver(driver_id)
    if not driver:
        print(f"[ERROR] Driver with ID '{driver_id}' not found in database.")
        sys.exit(1)

    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Could not decode image file: {image_path}")
        sys.exit(1)

    print(f"Extracting face embedding for {driver.name}...")
    identifier = get_face_identifier()
    try:
        updated_driver = identifier.enroll_driver_face(driver, image)
        db.save_driver(updated_driver)
        print(f"\n[SUCCESS] Facial profile successfully registered for: {updated_driver.name}")
        print(f"Driver ID: {updated_driver.id} | License: {updated_driver.license_number}")
        print("Ready for automated in-cabin login & threshold configuration.")
    except Exception as exc:
        print(f"[ERROR] Enrollment failed: {exc}")
        sys.exit(1)


def enroll_from_webcam(driver_id: str, camera_index: int = 0) -> None:
    """Capture multiple face samples from webcam and register driver face."""
    db = get_db_manager()
    driver = db.get_driver(driver_id)
    if not driver:
        print(f"[ERROR] Driver with ID '{driver_id}' not found in database.")
        sys.exit(1)

    identifier = get_face_identifier()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {camera_index}")
        sys.exit(1)

    window_name = f"CockpitSentinel — Face Enrollment: {driver.name}"
    print(f"\n============================================================")
    print(f" Enrolling Driver Face: {driver.name} ({driver.license_number})")
    print(f" Look directly into camera. Press 'SPACE' to capture samples.")
    print(f" Press 'Q' or 'Esc' to cancel.")
    print(f"============================================================\n")

    captured_frames: list[np.ndarray] = []
    capture_samples_count = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            display = frame.copy()

            # Center guide reticle
            center_x, center_y = w // 2, h // 2
            axes = (int(w * 0.18), int(h * 0.28))
            cv2.ellipse(display, (center_x, center_y), axes, 0, 0, 360, (0, 220, 255), 2)

            # Header text
            cv2.rectangle(display, (0, 0), (w, 65), (20, 20, 20), -1)
            cv2.putText(
                display,
                f"ENROLLING: {driver.name.upper()}",
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 240, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Align face in center ellipse and press SPACE to capture ({len(captured_frames)}/{capture_samples_count})",
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Enrollment cancelled by user.")
                return

            if key == 32:  # Spacebar pressed
                print(f"Capturing face sample {len(captured_frames) + 1} of {capture_samples_count}...")
                captured_frames.append(frame.copy())
                if len(captured_frames) >= capture_samples_count:
                    break

        if len(captured_frames) < capture_samples_count:
            print("Insufficient samples captured. Enrollment aborted.")
            return

        print("Computing and averaging 512-D biometric embeddings...")
        updated_driver = identifier.enroll_driver_face(driver, captured_frames)
        db.save_driver(updated_driver)
        print(f"\n[SUCCESS] Driver facial profile successfully saved to database!")
        print(f"Driver Name: {updated_driver.name}")
        print(f"Status: Face Enrolled (Autonomous Cabin Profile Switch Enabled)")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a driver's face for automated cabin profile loading.")
    parser.add_argument("--driver-id", default=None, help="Driver Profile ID to enroll (e.g. 'drv-satyapal-001').")
    parser.add_argument("--image", type=Path, default=None, help="Optional image file path to enroll from.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0).")
    args = parser.parse_args()

    db = get_db_manager()
    drivers = db.list_drivers()
    if not drivers:
        print("[ERROR] No drivers found in database. Please create a driver first.")
        sys.exit(1)

    driver_id = args.driver_id
    if not driver_id:
        print("\nAvailable Driver Profiles:")
        for idx, d in enumerate(drivers, 1):
            face_status = "✅ Enrolled" if d.face_registered else "⚠️ Pending"
            print(f"  [{idx}] {d.name} ({d.license_number}) - ID: {d.id} [{face_status}]")
        choice = input(f"\nSelect driver number [1-{len(drivers)}] (default 1): ").strip()
        selected_idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(drivers) else 0
        driver_id = drivers[selected_idx].id

    if args.image:
        enroll_from_image(driver_id, args.image)
    else:
        enroll_from_webcam(driver_id, camera_index=args.camera_index)


if __name__ == "__main__":
    main()
