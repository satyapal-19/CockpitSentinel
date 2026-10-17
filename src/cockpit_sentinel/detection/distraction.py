"""Object-detection signals for phone use and smoking-related distractions."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import yaml
from ultralytics import YOLO, YOLOWorld

from cockpit_sentinel.domain import DriverSignals
from cockpit_sentinel.utils.device import resolve_device


class ObjectDetector(Protocol):
    """Minimal callable interface shared by Ultralytics detection models."""

    def __call__(self, frame: np.ndarray, **kwargs: Any) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class DetectionBox:
    """Bounding box coordinates, label, and confidence score."""

    box: tuple[int, int, int, int]
    label: str
    confidence: float


@dataclass(frozen=True, slots=True)
class DistractionConfig:
    """Confidence thresholds and text labels for distraction detection."""

    phone_confidence_threshold: float = 0.40
    smoking_confidence_threshold: float = 0.35
    smoking_labels: tuple[str, ...] = ("smoke", "cigarette", "smoking", "vape")

    def __post_init__(self) -> None:
        if not 0 < self.phone_confidence_threshold <= 1:
            raise ValueError("phone_confidence_threshold must be between 0 and 1.")
        if not 0 < self.smoking_confidence_threshold <= 1:
            raise ValueError("smoking_confidence_threshold must be between 0 and 1.")
        if not self.smoking_labels or not all(self.smoking_labels):
            raise ValueError("smoking_labels must contain at least one non-empty label.")


@dataclass(frozen=True, slots=True)
class DistractionAnalysis:
    """Per-frame object-detection output for distraction signals."""

    signals: DriverSignals
    phone_confidence: float | None = None
    smoking_confidence: float | None = None
    detections: tuple[DetectionBox, ...] = ()


def load_distraction_config(path: Path) -> DistractionConfig:
    """Load object-detection thresholds and prompts from YAML."""

    with path.open(encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError("Distraction configuration must be a YAML mapping.")

    expected_keys = set(DistractionConfig.__dataclass_fields__)
    if set(raw_config) != expected_keys:
        raise ValueError("Distraction configuration must define every setting.")
    labels = raw_config["smoking_labels"]
    if not isinstance(labels, list):
        raise ValueError("smoking_labels must be a YAML list.")

    return DistractionConfig(
        phone_confidence_threshold=raw_config["phone_confidence_threshold"],
        smoking_confidence_threshold=raw_config["smoking_confidence_threshold"],
        smoking_labels=tuple(labels),
    )


DEFAULT_PHONE_LABELS = frozenset(
    {"cell phone", "phone", "using_phone", "mobile_phone", "calling", "texting"}
)


@dataclass(slots=True)
class _OnnxBox:
    cls: list[int]
    conf: list[float]
    xyxy: list[list[int]]


@dataclass(slots=True)
class _OnnxResult:
    boxes: list[_OnnxBox]
    names: dict[int, str]


class OnnxYOLO11Detector:
    """Sub-100ms ultra-low memory YOLO11 distraction detector using OpenCV DNN."""

    def __init__(self, onnx_path: Path) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model was not found: {onnx_path}")
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        self.names = {0: "earphone", 1: "person", 2: "smoke", 3: "using_phone"}

    def __call__(self, frame: np.ndarray, conf: float = 0.25, **kwargs: Any) -> list[Any]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()  # (1, 8, 8400)
        preds = out[0].T  # (8400, 8)

        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= conf
        if not np.any(mask):
            return [_OnnxResult(boxes=[], names=self.names)]

        f_boxes = preds[:, :4][mask]
        f_confs = confidences[mask]
        f_classes = class_ids[mask]

        boxes_xywh: list[list[int]] = []
        confs_list: list[float] = []
        class_list: list[int] = []

        scale_x = w / 640.0
        scale_y = h / 640.0

        for i in range(len(f_boxes)):
            cx, cy, bw, bh = f_boxes[i]
            x1 = int((cx - bw / 2.0) * scale_x)
            y1 = int((cy - bh / 2.0) * scale_y)
            x2 = int((cx + bw / 2.0) * scale_x)
            y2 = int((cy + bh / 2.0) * scale_y)
            boxes_xywh.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
            confs_list.append(float(f_confs[i]))
            class_list.append(int(f_classes[i]))

        indices = cv2.dnn.NMSBoxes(boxes_xywh, confs_list, conf, 0.45)
        boxes_out: list[_OnnxBox] = []
        if len(indices) > 0:
            for idx in indices:
                x, y, bw, bh = boxes_xywh[idx]
                boxes_out.append(
                    _OnnxBox(
                        cls=[class_list[idx]],
                        conf=[confs_list[idx]],
                        xyxy=[[x, y, x + bw, y + bh]],
                    )
                )

        return [_OnnxResult(boxes=boxes_out, names=self.names)]


class DistractionDetector:
    """Detect phone use and smoking-related objects from BGR video frames."""

    def __init__(
        self,
        phone_model_path: Path,
        smoking_model_path: Path | None = None,
        config: DistractionConfig | None = None,
        *,
        phone_model: ObjectDetector | None = None,
        smoking_model: ObjectDetector | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or DistractionConfig()
        self.device = resolve_device(device) if device is not None else None

        # Check if caller wants a single unified model (same model path or explicitly identical model)
        is_same_path = (smoking_model_path is None) or (phone_model_path == smoking_model_path)
        is_same_object = (phone_model is not None) and (phone_model is smoking_model)

        if is_same_path or is_same_object:
            self._phone_model = phone_model or self._load_phone_model(phone_model_path, self.device)
            self._smoking_model = self._phone_model
            self._is_unified = True
        else:
            self._phone_model = phone_model or self._load_phone_model(phone_model_path, self.device)
            self._smoking_model = smoking_model or self._load_smoking_model(
                smoking_model_path, self.device  # type: ignore[arg-type]
            )
            self._is_unified = False

    def close(self) -> None:
        """Release any device or model resources."""
        pass

    def __enter__(self) -> DistractionDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def analyze(self, frame: np.ndarray) -> DistractionAnalysis:
        """Run object detector(s) and return the relevant risk signals."""

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR frame.")

        predict_kwargs: dict[str, Any] = {"verbose": False}
        if self.device is not None:
            predict_kwargs["device"] = self.device

        smoking_targets = set(self.config.smoking_labels)

        if self._is_unified:
            # Single forward pass for both phone and smoking detection
            min_conf = min(
                self.config.phone_confidence_threshold, self.config.smoking_confidence_threshold
            )
            results = self._phone_model(frame, conf=min_conf, **predict_kwargs)
            phone_confidence, phone_boxes = _extract_detections(
                results,
                DEFAULT_PHONE_LABELS,
                frame_shape=frame.shape[:2],
                min_confidence=self.config.phone_confidence_threshold,
            )
            smoking_confidence, smoking_boxes = _extract_detections(
                results,
                smoking_targets,
                frame_shape=frame.shape[:2],
                min_confidence=self.config.smoking_confidence_threshold,
            )
        else:
            # Dual model pass (legacy separate phone + YOLO-World smoking)
            phone_confidence, phone_boxes = _extract_detections(
                self._phone_model(
                    frame, conf=self.config.phone_confidence_threshold, **predict_kwargs
                ),
                DEFAULT_PHONE_LABELS,
                frame_shape=frame.shape[:2],
            )
            smoking_confidence, smoking_boxes = _extract_detections(
                self._smoking_model(
                    frame, conf=self.config.smoking_confidence_threshold, **predict_kwargs
                ),
                smoking_targets,
                frame_shape=frame.shape[:2],
            )

        return DistractionAnalysis(
            signals=DriverSignals(
                phone_detected=phone_confidence is not None,
                smoking_detected=smoking_confidence is not None,
            ),
            phone_confidence=phone_confidence,
            smoking_confidence=smoking_confidence,
            detections=tuple(phone_boxes + smoking_boxes),
        )

    @staticmethod
    def _load_phone_model(path: Path, device: str | None = None) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Phone detection model was not found: {path}")
        if path.suffix.lower() == ".onnx":
            return OnnxYOLO11Detector(path)
        import torch
        torch.set_num_threads(1)
        model = YOLO(path)
        if device is not None:
            with contextlib.suppress(Exception):
                model.to(device)
        return model

    def _load_smoking_model(self, path: Path, device: str | None = None) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Smoking detection model was not found: {path}")
        if "world" in str(path).lower():
            model = YOLOWorld(path)
            with contextlib.suppress(Exception):
                model.set_classes(list(self.config.smoking_labels))
        else:
            model = YOLO(path)
        if device is not None:
            with contextlib.suppress(Exception):
                model.to(device)
        return model


def _extract_detections(
    results: Sequence[Any],
    target_labels: set[str] | frozenset[str],
    frame_shape: tuple[int, int] | None = None,
    min_confidence: float | None = None,
) -> tuple[float | None, list[DetectionBox]]:
    """Extract highest confidence score and bounding boxes for matching target labels."""
    highest: float | None = None
    detections: list[DetectionBox] = []
    normalized_targets = {label.casefold() for label in target_labels}

    frame_h, frame_w = frame_shape if frame_shape is not None else (0, 0)
    frame_area = frame_h * frame_w

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        names = getattr(result, "names", {})
        for box in boxes:
            class_index = int(box.cls[0])
            raw_label = str(names.get(class_index, class_index))
            if raw_label.casefold() not in normalized_targets:
                continue

            confidence = float(box.conf[0])
            if min_confidence is not None and confidence < min_confidence:
                continue

            coords = (0, 0, 0, 0)
            xyxy = getattr(box, "xyxy", None)
            if xyxy is not None and len(xyxy) > 0:
                coords_raw = xyxy[0]
                if hasattr(coords_raw, "tolist"):
                    coords_raw = coords_raw.tolist()
                coords = (
                    int(coords_raw[0]),
                    int(coords_raw[1]),
                    int(coords_raw[2]),
                    int(coords_raw[3]),
                )

                # Heuristic: filter large smoking boxes (>25% frame) which are false positives
                if frame_area > 0 and raw_label.casefold() in {"smoke", "cigarette", "smoking", "vape"}:
                    box_w = max(0, coords[2] - coords[0])
                    box_h = max(0, coords[3] - coords[1])
                    if (box_w * box_h) > (0.25 * frame_area):
                        continue

            if highest is None or confidence > highest:
                highest = confidence

            detections.append(DetectionBox(box=coords, label=raw_label, confidence=confidence))

    return highest, detections


def _highest_confidence(results: Sequence[Any], target_labels: set[str]) -> float | None:
    highest, _ = _extract_detections(results, target_labels)
    return highest
