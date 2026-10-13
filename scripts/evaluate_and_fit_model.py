"""
Evaluate CockpitSentinel models and fit machine learning classifiers on benchmark datasets.

Datasets evaluated:
  - Kaggle Driver Drowsiness Dataset (DDD)
  - NTHU Driver Drowsiness Dataset (DDD)

Tasks:
  1. Extract facial landmark geometry (EAR, MAR, Head Yaw) using MediaPipe.
  2. Benchmark the existing baseline geometric rule-based detector.
  3. Fit and evaluate ML classifiers (Random Forest, SVM, Logistic Regression).
  4. Save the fitted model and performance report for viva demonstration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add src to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cockpit_sentinel.drowsiness.detector import (
    DrowsinessDetector,
    eye_aspect_ratio,
    load_drowsiness_config,
    mouth_aspect_ratio,
    _estimate_head_yaw,
    _pixel_points,
    LEFT_EYE,
    RIGHT_EYE,
    MOUTH,
    HEAD_POSE,
)


def resolve_paths() -> tuple[Path, Path, Path]:
    env_values = dotenv_values(REPO_ROOT / ".env")
    data_root = Path(
        os.environ.get("COCKPIT_DATA_ROOT")
        or env_values.get("COCKPIT_DATA_ROOT")
        or str(REPO_ROOT / "data")
    )
    models_root = Path(
        os.environ.get("COCKPIT_MODELS_ROOT")
        or env_values.get("COCKPIT_MODELS_ROOT")
        or str(data_root / "models")
    )
    external_data = data_root / "data" / "external" if (data_root / "data" / "external").exists() else data_root / "external"
    if not external_data.exists():
        external_data = REPO_ROOT / "data" / "external"
    return data_root, models_root, external_data


def collect_dataset_samples(external_dir: Path) -> list[tuple[Path, int, str]]:
    """Collect (image_path, label, dataset_name) tuples. Label: 1 = Drowsy, 0 = Alert."""
    samples: list[tuple[Path, int, str]] = []

    # 1. Kaggle DDD
    kaggle_drowsy = external_dir / "kaggle_driver_monitoring" / "images" / "drowsy"
    kaggle_alert = external_dir / "kaggle_driver_monitoring" / "images" / "non_drowsy"

    if kaggle_drowsy.exists():
        for p in kaggle_drowsy.glob("*.png"):
            samples.append((p, 1, "Kaggle DDD (Drowsy)"))
    if kaggle_alert.exists():
        for p in kaggle_alert.glob("*.png"):
            samples.append((p, 0, "Kaggle DDD (Alert)"))

    # 2. NTHU DDD
    nthu_train = external_dir / "nthu" / "train"
    if nthu_train.exists():
        for sub_dir in (nthu_train / "drowsy").glob("*"):
            if sub_dir.is_dir():
                for p in sub_dir.glob("*.jpg"):
                    samples.append((p, 1, f"NTHU ({sub_dir.name})"))
        nthu_alert = nthu_train / "notdrowsy"
        if nthu_alert.exists():
            for p in nthu_alert.glob("*.jpg"):
                samples.append((p, 0, "NTHU (Alert)"))

    return samples


def extract_features(
    samples: list[tuple[Path, int, str]],
    model_path: Path,
    config_path: Path,
) -> pd.DataFrame:
    """Extract EAR, MAR, Head Yaw, and facial landmarks across images."""
    config = load_drowsiness_config(config_path)
    records = []

    print(f"\n[INFO] Extracting features from {len(samples)} images using MediaPipe...")
    start_time = time.time()
    detected_count = 0

    with DrowsinessDetector(model_path, config) as detector:
        for idx, (img_path, label, source) in enumerate(samples, 1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            # Run detection directly
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_mp = detector._landmarker.detect(
                cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if False else __import__("mediapipe").Image(
                    image_format=__import__("mediapipe").ImageFormat.SRGB,
                    data=rgb_frame,
                )
            )

            if not image_mp.face_landmarks:
                continue

            detected_count += 1
            landmarks = image_mp.face_landmarks[0]
            height, width = frame.shape[:2]

            left_ear = eye_aspect_ratio(_pixel_points(landmarks, LEFT_EYE, width, height))
            right_ear = eye_aspect_ratio(_pixel_points(landmarks, RIGHT_EYE, width, height))
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(_pixel_points(landmarks, MOUTH, width, height))
            yaw = _estimate_head_yaw(landmarks, width, height)

            records.append({
                "path": str(img_path),
                "source": source,
                "label": label,
                "ear": ear,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "mar": mar,
                "head_yaw": abs(yaw),
            })

            if idx % 50 == 0 or idx == len(samples):
                print(f"  Processed {idx}/{len(samples)} images ({detected_count} faces detected)")

    elapsed = time.time() - start_time
    print(f"[OK] Feature extraction complete in {elapsed:.2f}s ({detected_count}/{len(samples)} faces found)")
    return pd.DataFrame(records)


def evaluate_baseline_rule(df: pd.DataFrame, ear_threshold: float, mar_threshold: float) -> dict[str, float]:
    """Evaluate baseline rule: pred = 1 if (ear < ear_threshold or mar > mar_threshold) else 0."""
    y_true = df["label"].to_numpy()
    y_pred = ((df["ear"] < ear_threshold) | (df["mar"] > mar_threshold)).astype(int).to_numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def fit_and_evaluate_ml_models(df: pd.DataFrame) -> tuple[dict[str, dict[str, any]], Pipeline]:
    """Train and evaluate Random Forest, Logistic Regression, and SVM."""
    feature_cols = ["ear", "left_ear", "right_ear", "mar", "head_yaw"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
    }

    results: dict[str, dict[str, any]] = {}
    best_f1 = -1.0
    best_pipeline: Pipeline | None = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
        }

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipe

    # Feature importances from Random Forest
    rf_pipe = Pipeline([("scaler", StandardScaler()), ("classifier", models["Random Forest"])])
    rf_pipe.fit(X, y)
    importances = rf_pipe.named_steps["classifier"].feature_importances_
    results["feature_importances"] = dict(zip(feature_cols, [float(x) for x in importances]))

    return results, best_pipeline


def print_report(
    total_samples: int,
    detected_faces: int,
    baseline_metrics: dict[str, any],
    ml_results: dict[str, any],
) -> None:
    print("\n" + "=" * 70)
    print("           COCKPITSENTINEL MODEL ACCURACY & EVALUATION REPORT")
    print("=" * 70)
    print(f"Total dataset images found     : {total_samples}")
    print(f"Faces detected by MediaPipe    : {detected_faces} ({detected_faces/total_samples*100:.1f}%)")
    print("-" * 70)
    print(f"{'Model / Architecture':<26} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)

    # Baseline rule
    b = baseline_metrics
    print(f"{'Rule-Based Baseline (EAR/MAR)':<26} {b['accuracy']*100:>7.2f}%   {b['precision']*100:>7.2f}%   {b['recall']*100:>7.2f}%   {b['f1']*100:>7.2f}%")

    # ML models
    for name in ["Logistic Regression", "Support Vector Machine", "Random Forest"]:
        m = ml_results[name]
        print(f"{name:<26} {m['test_accuracy']*100:>7.2f}%   {m['precision']*100:>7.2f}%   {m['recall']*100:>7.2f}%   {m['f1_score']*100:>7.2f}%")

    print("-" * 70)
    print("\n[+] Confusion Matrix (Best ML Model — Random Forest):")
    cm = np.array(ml_results["Random Forest"]["confusion_matrix"])
    print(f"    True Negative (Alert):    {cm[0, 0]:>4}    False Positive: {cm[0, 1]:>4}")
    print(f"    False Negative:           {cm[1, 0]:>4}    True Positive (Drowsy): {cm[1, 1]:>4}")

    print("\n[+] Feature Importances (Random Forest):")
    for feat, imp in sorted(ml_results["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(imp * 30)
        print(f"    {feat:<12} : {imp*100:>5.1f}%  |{bar}")
    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and fit CockpitSentinel models.")
    parser.add_argument("--save-model", action="store_true", default=True, help="Save best model checkpoint")
    args = parser.parse_args()

    data_root, models_root, external_dir = resolve_paths()
    model_asset = models_root / "pretrained" / "face_landmarker.task"
    if not model_asset.exists():
        model_asset = Path("D:/CockpitSentinel/models/pretrained/face_landmarker.task")
    config_file = REPO_ROOT / "configs" / "drowsiness.yaml"

    if not model_asset.exists():
        print(f"[ERROR] Face landmarker model not found at: {model_asset}")
        return 1

    # 1. Collect samples
    samples = collect_dataset_samples(external_dir)
    if not samples:
        print(f"[ERROR] No dataset samples found in: {external_dir}")
        return 1

    drowsy_count = sum(1 for _, lbl, _ in samples if lbl == 1)
    alert_count = sum(1 for _, lbl, _ in samples if lbl == 0)
    print(f"[INFO] Collected {len(samples)} benchmark samples:")
    print(f"       - Drowsy: {drowsy_count}")
    print(f"       - Alert:  {alert_count}")

    # 2. Extract features
    df = extract_features(samples, model_asset, config_file)
    if df.empty or len(df) < 10:
        print("[ERROR] Insufficient facial features extracted.")
        return 1

    # Save extracted features
    processed_dir = data_root / "data" / "processed" / "features" if (data_root / "data" / "processed").exists() else REPO_ROOT / "data" / "processed" / "features"
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_csv = processed_dir / "drowsiness_extracted_features.csv"
    df.to_csv(features_csv, index=False)
    print(f"[OK] Saved extracted features dataset to: {features_csv}")

    # 3. Evaluate baseline rule
    cfg = load_drowsiness_config(config_file)
    baseline_metrics = evaluate_baseline_rule(
        df,
        ear_threshold=cfg.eye_aspect_ratio_threshold,
        mar_threshold=cfg.mouth_aspect_ratio_threshold,
    )

    # 4. Train & evaluate ML models
    ml_results, best_pipeline = fit_and_evaluate_ml_models(df)

    # 5. Print comprehensive report
    print_report(len(samples), len(df), baseline_metrics, ml_results)

    # 6. Save model checkpoint
    if args.save_model and best_pipeline is not None:
        trained_dir = models_root / "trained" / "drowsiness"
        trained_dir.mkdir(parents=True, exist_ok=True)
        model_out = trained_dir / "drowsiness_classifier.joblib"
        joblib.dump(best_pipeline, model_out)
        print(f"[SUCCESS] Trained model checkpoint saved to: {model_out}")

        # Save metrics JSON
        exp_dir = data_root / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        report_out = exp_dir / "model_accuracy_report.json"
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_samples": len(samples),
                "detected_faces": len(df),
                "baseline_rule": baseline_metrics,
                "ml_models": ml_results,
            }, f, indent=2)
        print(f"[SUCCESS] Full metrics JSON exported to: {report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
