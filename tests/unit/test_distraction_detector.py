"""Tests for object-detection distraction signals without model hardware."""

from pathlib import Path

import numpy as np
import pytest

from cockpit_sentinel.detection import DistractionConfig, DistractionDetector


class FakeBox:
    def __init__(self, class_index: int, confidence: float) -> None:
        self.cls = np.array([class_index])
        self.conf = np.array([confidence])


class FakeResult:
    def __init__(self, names: dict[int, str], boxes: list[FakeBox]) -> None:
        self.names = names
        self.boxes = boxes


class FakeModel:
    def __init__(self, result: FakeResult) -> None:
        self._result = result

    def __call__(self, frame: np.ndarray, **kwargs: object) -> list[FakeResult]:
        del frame, kwargs
        return [self._result]


def test_distraction_detector_combines_phone_and_smoking_detections():
    phone_model = FakeModel(FakeResult({67: "cell phone"}, [FakeBox(67, 0.89)]))
    smoking_model = FakeModel(FakeResult({0: "cigarette"}, [FakeBox(0, 0.63)]))
    detector = DistractionDetector(
        Path("phone.pt"),
        Path("smoking.pt"),
        phone_model=phone_model,
        smoking_model=smoking_model,
    )

    analysis = detector.analyze(np.zeros((120, 160, 3), dtype=np.uint8))

    assert analysis.signals.phone_detected
    assert analysis.signals.smoking_detected
    assert analysis.phone_confidence == pytest.approx(0.89)
    assert analysis.smoking_confidence == pytest.approx(0.63)


def test_distraction_config_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="between 0 and 1"):
        DistractionConfig(phone_confidence_threshold=0)


def test_distraction_detector_supports_device_and_context_manager():
    phone_model = FakeModel(FakeResult({67: "cell phone"}, []))
    smoking_model = FakeModel(FakeResult({0: "cigarette"}, []))

    with DistractionDetector(
        Path("phone.pt"),
        Path("smoking.pt"),
        phone_model=phone_model,
        smoking_model=smoking_model,
        device="cpu",
    ) as detector:
        assert detector.device == "cpu"
        analysis = detector.analyze(np.zeros((120, 160, 3), dtype=np.uint8))
        assert not analysis.signals.phone_detected
        assert not analysis.signals.smoking_detected


class FakeBoxWithCoords:
    def __init__(
        self, class_index: int, confidence: float, xyxy: tuple[int, int, int, int]
    ) -> None:
        self.cls = np.array([class_index])
        self.conf = np.array([confidence])
        self.xyxy = np.array([xyxy])


def test_distraction_detector_extracts_bounding_boxes():
    phone_box = FakeBoxWithCoords(67, 0.85, (10, 20, 100, 150))
    phone_model = FakeModel(FakeResult({67: "cell phone"}, [phone_box]))
    smoking_model = FakeModel(FakeResult({0: "cigarette"}, []))

    detector = DistractionDetector(
        Path("phone.pt"),
        Path("smoking.pt"),
        phone_model=phone_model,
        smoking_model=smoking_model,
    )

    analysis = detector.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
    assert analysis.signals.phone_detected
    assert len(analysis.detections) == 1
    assert analysis.detections[0].box == (10, 20, 100, 150)
    assert analysis.detections[0].label == "cell phone"
    assert analysis.detections[0].confidence == pytest.approx(0.85)


def test_distraction_detector_single_unified_model():
    # Single model that outputs both phone and cigarette in a single forward pass
    boxes = [
        FakeBoxWithCoords(0, 0.92, (10, 20, 100, 150)),
        FakeBoxWithCoords(1, 0.78, (200, 250, 240, 280)),
    ]
    unified_model = FakeModel(FakeResult({0: "phone", 1: "cigarette"}, boxes))

    detector = DistractionDetector(
        Path("unified_distraction.pt"),
        phone_model=unified_model,
        smoking_model=unified_model,
    )

    analysis = detector.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
    assert analysis.signals.phone_detected
    assert analysis.signals.smoking_detected
    assert analysis.phone_confidence == pytest.approx(0.92)
    assert analysis.smoking_confidence == pytest.approx(0.78)
    assert len(analysis.detections) == 2
