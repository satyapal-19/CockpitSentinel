"""Interactive webcam and video runner for the fatigue monitoring pipeline."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from cockpit_sentinel.alerts.policy import AlertPolicy, load_alert_policy
from cockpit_sentinel.detection import (
    DistractionAnalysis,
    DistractionDetector,
    load_distraction_config,
)
from cockpit_sentinel.domain import DriverSignals, RiskAssessment, RiskLevel
from cockpit_sentinel.drowsiness import (
    DrowsinessAnalysis,
    DrowsinessDetector,
    load_drowsiness_config,
)
from cockpit_sentinel.utils.device import resolve_device
from cockpit_sentinel.utils.paths import (
    find_project_root,
    resolve_config_path,
)
from cockpit_sentinel.utils.paths import (
    resolve_model_path as _resolve_model_path,
)

REPO_ROOT = find_project_root()
WINDOW_TITLE = "CockpitSentinel - Drowsiness Monitor"
RISK_COLORS = {
    RiskLevel.SAFE: (61, 153, 112),
    RiskLevel.CAUTION: (33, 180, 241),
    RiskLevel.WARNING: (22, 137, 243),
    RiskLevel.CRITICAL: (49, 69, 220),
}


class FrameAnalyzer(Protocol):
    """Detector interface used by the live monitoring loop."""

    def analyze(self, frame: np.ndarray) -> DrowsinessAnalysis: ...


class DistractionAnalyzer(Protocol):
    """Object detector interface used by the live monitoring loop."""

    def analyze(self, frame: np.ndarray) -> DistractionAnalysis: ...


@dataclass(frozen=True, slots=True)
class TripStats:
    """Accurately records total trip time and attention statistics."""

    total_seconds: float = 0.0
    attentive_seconds: float = 0.0
    inattentive_seconds: float = 0.0
    fps: float = 0.0
    total_frames: int = 0

    @property
    def formatted_trip_time(self) -> str:
        """Formatted trip time string (HH:MM:SS)."""
        total = int(self.total_seconds)
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def attention_score(self) -> float:
        """Percentage of total trip time spent attentive [0.0 - 100.0]."""
        if self.total_seconds <= 0.0:
            return 100.0
        return min(100.0, max(0.0, (self.attentive_seconds / self.total_seconds) * 100.0))


class TripTracker:
    """Tracks elapsed trip duration, attentive state, and FPS."""

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._last_time: float | None = None
        self._total_seconds: float = 0.0
        self._attentive_seconds: float = 0.0
        self._inattentive_seconds: float = 0.0
        self._fps: float = 0.0
        self._total_frames: int = 0

    def update(self, is_attentive: bool, timestamp: float | None = None) -> TripStats:
        now = time.monotonic() if timestamp is None else timestamp
        if self._start_time is None:
            self._start_time = now
            self._last_time = now

        dt = max(0.0, now - (self._last_time or now))
        self._last_time = now
        self._total_seconds = max(0.0, now - self._start_time)
        self._total_frames += 1

        if dt > 0.0:
            instant_fps = 1.0 / dt
            self._fps = 0.9 * self._fps + 0.1 * instant_fps if self._fps > 0 else instant_fps

        if is_attentive:
            self._attentive_seconds += dt
        else:
            self._inattentive_seconds += dt

        return TripStats(
            total_seconds=self._total_seconds,
            attentive_seconds=self._attentive_seconds,
            inattentive_seconds=self._inattentive_seconds,
            fps=self._fps,
            total_frames=self._total_frames,
        )

    def reset(self) -> None:
        self._start_time = None
        self._last_time = None
        self._total_seconds = 0.0
        self._attentive_seconds = 0.0
        self._inattentive_seconds = 0.0
        self._fps = 0.0
        self._total_frames = 0


class TemporalSignalStabilizer:
    """Track signal durations in seconds over a sliding time window (2.0-3.0s).

    A signal must persist continuously for at least its configured duration threshold
    (e.g., 2.0 to 3.0 seconds) to be flagged as active. As soon as the condition clears
    (driver returns to attentive state), the signal is immediately cleared.
    """

    def __init__(
        self,
        window_seconds: float = 2.0,
        signal_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        self.window_seconds = max(0.0, window_seconds)
        self.thresholds: dict[str, float] = {
            "eyes_closed": 2.0,
            "yawning": 2.5,
            "looking_away": 2.0,
            "phone_detected": 2.0,
            "smoking_detected": 2.0,
        }
        if window_seconds == 0.0:
            for k in self.thresholds:
                self.thresholds[k] = 0.0
        elif window_seconds != 2.0:
            for k in self.thresholds:
                self.thresholds[k] = window_seconds

        if signal_thresholds:
            self.thresholds.update(signal_thresholds)

        self._active_durations: dict[str, float] = {
            name: 0.0 for name in DriverSignals.__dataclass_fields__
        }
        self._last_timestamp: float | None = None

    def update(
        self, raw_signals: DriverSignals, timestamp: float | None = None
    ) -> tuple[DriverSignals, dict[str, float]]:
        current_time = time.monotonic() if timestamp is None else timestamp
        dt = (
            0.0
            if self._last_timestamp is None
            else max(0.0, current_time - self._last_timestamp)
        )
        self._last_timestamp = current_time

        stabilized: dict[str, bool] = {}
        for name in DriverSignals.__dataclass_fields__:
            is_active = bool(getattr(raw_signals, name, False))
            thresh = self.thresholds.get(name, self.window_seconds)

            if is_active:
                self._active_durations[name] += dt
                stabilized[name] = self._active_durations[name] >= thresh
            else:
                # Promptly clear active duration and reset signal on attentive return
                self._active_durations[name] = 0.0
                stabilized[name] = False

        return DriverSignals(**stabilized), dict(self._active_durations)

    def reset(self) -> None:
        for name in self._active_durations:
            self._active_durations[name] = 0.0
        self._last_timestamp = None


@dataclass(frozen=True, slots=True)
class MonitorFrame:
    """One processed frame with detector output, risk result, and display image."""

    analysis: DrowsinessAnalysis
    distraction: DistractionAnalysis | None
    assessment: RiskAssessment
    image: np.ndarray
    trip_stats: TripStats | None = None
    active_durations: dict[str, float] | None = None


class LiveMonitor:
    """Combine face analysis, distraction detection, and risk scoring for video frames."""

    def __init__(
        self,
        detector: FrameAnalyzer,
        policy: AlertPolicy,
        distraction_detector: DistractionAnalyzer | None = None,
        time_window_seconds: float | None = None,
        signal_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        self._detector = detector
        self._policy = policy
        self._distraction_detector = distraction_detector

        if time_window_seconds is not None:
            self._time_window_seconds = max(0.0, time_window_seconds)
        else:
            self._time_window_seconds = getattr(policy, "time_window_seconds", 0.0)

        self._temporal_stabilizer = TemporalSignalStabilizer(
            window_seconds=self._time_window_seconds,
            signal_thresholds=signal_thresholds,
        )
        self._trip_tracker = TripTracker()

    def process(self, frame: np.ndarray, timestamp: float | None = None) -> MonitorFrame:
        analysis = self._detector.analyze(frame)
        distraction = None
        if self._distraction_detector is not None:
            distraction = self._distraction_detector.analyze(frame)
        distraction_signals = distraction.signals if distraction else DriverSignals()
        raw_signals = merge_signals(analysis.signals, distraction_signals)

        if self._time_window_seconds > 0.0:
            signals, active_durations = self._temporal_stabilizer.update(
                raw_signals, timestamp=timestamp
            )
        else:
            signals = raw_signals
            active_durations = {name: 0.0 for name in DriverSignals.__dataclass_fields__}

        assessment = self._policy.assess(signals)
        is_attentive = assessment.level == RiskLevel.SAFE
        trip_stats = self._trip_tracker.update(is_attentive, timestamp=timestamp)

        image = draw_monitor_overlay(
            frame,
            analysis,
            assessment,
            distraction=distraction,
            trip_stats=trip_stats,
            active_durations=active_durations,
        )
        return MonitorFrame(
            analysis=analysis,
            distraction=distraction,
            assessment=assessment,
            image=image,
            trip_stats=trip_stats,
            active_durations=active_durations,
        )


def merge_signals(drowsiness: DriverSignals, distraction: DriverSignals) -> DriverSignals:
    """Preserve fatigue signals while adding object-detection distraction signals."""

    return DriverSignals(
        eyes_closed=drowsiness.eyes_closed,
        yawning=drowsiness.yawning,
        looking_away=drowsiness.looking_away,
        phone_detected=distraction.phone_detected,
        smoking_detected=distraction.smoking_detected,
    )


def draw_monitor_overlay(
    frame: np.ndarray,
    analysis: DrowsinessAnalysis,
    assessment: RiskAssessment,
    distraction: DistractionAnalysis | None = None,
    trip_stats: TripStats | None = None,
    active_durations: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Draw readable dashboard status, telemetry, trip timer, and bounding boxes."""

    image = frame.copy()
    height, width = image.shape[:2]
    color = RISK_COLORS[assessment.level]

    # 1. Distraction bounding boxes (Phone & Smoking) with labels and confidence
    if distraction is not None and getattr(distraction, "detections", None):
        for det in distraction.detections:
            x1, y1, x2, y2 = det.box
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            is_phone = "phone" in det.label.casefold()
            box_color = (255, 180, 0) if is_phone else (0, 140, 255)
            tag_name = "PHONE" if is_phone else det.label.upper()
            tag_text = f"{tag_name} {int(det.confidence * 100)}%"

            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(image, (x1, max(0, y1 - 20)), (x1 + tw + 6, y1), box_color, -1)
            cv2.putText(
                image,
                tag_text,
                (x1 + 3, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # 2. Warning / Critical perimeter alert border
    if assessment.level is RiskLevel.CRITICAL:
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (49, 69, 220), 4)
    elif assessment.level is RiskLevel.WARNING:
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (22, 137, 243), 3)

    # 3. Top Dashboard Header
    header_h = 75 if height < 300 else 85
    cv2.rectangle(image, (0, 0), (width, header_h), (22, 22, 22), thickness=-1)
    cv2.line(image, (0, header_h), (width, header_h), (60, 60, 60), 1)

    # Risk Level & Status
    cv2.putText(
        image,
        f"RISK: {assessment.level.upper()} ({assessment.score})",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    status_str = ", ".join(assessment.reasons).upper() if assessment.reasons else "ATTENTIVE"
    status_color = (80, 220, 120) if assessment.level is RiskLevel.SAFE else color
    cv2.putText(
        image,
        f"STATUS: {status_str}",
        (14, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # Right side: Trip Time, Attention %, FPS
    if trip_stats is not None:
        right_x = max(width // 2, width - 230)
        cv2.putText(
            image,
            f"TRIP: {trip_stats.formatted_trip_time}",
            (right_x, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"ATTENTION: {trip_stats.attention_score:.1f}%",
            (right_x, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 220, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{trip_stats.fps:.1f} FPS",
            (right_x, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    # 4. Bottom Telemetry Bar (only rendered if height >= 240)
    if height >= 240:
        bot_h = 30
        bot_y = height - bot_h
        cv2.rectangle(image, (0, bot_y), (width, height), (22, 22, 22), -1)
        cv2.line(image, (0, bot_y), (width, bot_y), (60, 60, 60), 1)

        if analysis.face_detected:
            metrics_str = (
                f"EAR: {analysis.eye_aspect_ratio:.2f} | "
                f"MAR: {analysis.mouth_aspect_ratio:.2f} | "
                f"Yaw: {analysis.head_yaw_degrees:.1f} deg"
            )
        else:
            metrics_str = "Face not detected"

        cv2.putText(
            image,
            metrics_str,
            (14, bot_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        phone_txt = "PHONE: OK"
        if active_durations and active_durations.get("phone_detected", 0.0) > 0:
            phone_txt = f"PHONE: {active_durations['phone_detected']:.1f}s"
        smoke_txt = "SMOKE: OK"
        if active_durations and active_durations.get("smoking_detected", 0.0) > 0:
            smoke_txt = f"SMOKE: {active_durations['smoking_detected']:.1f}s"

        distract_str = f"{phone_txt} | {smoke_txt}"
        right_distract_x = max(width // 2, width - 240)
        cv2.putText(
            image,
            distract_str,
            (right_distract_x, bot_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    return image


def run_live_monitor(
    source: int | str,
    detector: FrameAnalyzer,
    policy: AlertPolicy,
    distraction_detector: DistractionAnalyzer | None = None,
    time_window_seconds: float = 2.0,
) -> None:
    """Open a webcam or video file and show processed frames until Q or Esc is pressed."""

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    monitor = LiveMonitor(
        detector,
        policy,
        distraction_detector,
        time_window_seconds=time_window_seconds,
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            processed = monitor.process(frame)
            cv2.imshow(WINDOW_TITLE, processed.image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_source(value: str) -> int | str:
    """Treat numeric inputs as webcam indexes and all other inputs as file paths."""

    return int(value) if value.isdigit() else value


def resolve_model_path(filename: str, models_root: Path | None = None) -> Path:
    """Resolve a shared model path from the environment or local .env file."""
    return _resolve_model_path(filename, models_root=models_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the drowsiness webcam/video monitor.")
    parser.add_argument(
        "--source", default="0", help="Webcam index or video file path (default: 0)."
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("COCKPIT_DEVICE", "auto"),
        help="Inference compute device: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps' (default: auto).",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        default=2.0,
        help="Temporal analysis window in seconds (default: 2.0, recommended: 2.0-3.0).",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Optional root directory containing pretrained models.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to MediaPipe face landmarker model (.task).",
    )
    parser.add_argument(
        "--phone-model-path",
        type=Path,
        default=None,
        help="Path to YOLO phone detection model (.pt).",
    )
    parser.add_argument(
        "--smoking-model-path",
        type=Path,
        default=None,
        help="Path to YOLOWorld smoking detection model (.pt).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing YAML configuration files.",
    )
    parser.add_argument(
        "--drowsiness-config",
        type=Path,
        default=None,
        help="Path to drowsiness YAML configuration file.",
    )
    parser.add_argument(
        "--distraction-config",
        type=Path,
        default=None,
        help="Path to distraction YAML configuration file.",
    )
    parser.add_argument(
        "--alert-policy",
        type=Path,
        default=None,
        help="Path to alert policy YAML configuration file.",
    )
    args = parser.parse_args()

    model_path = args.model_path or resolve_model_path("face_landmarker.task", args.models_root)
    phone_model_path = args.phone_model_path or resolve_model_path("yolov8n.pt", args.models_root)
    smoking_model_path = args.smoking_model_path or resolve_model_path(
        "yolov8s-worldv2.pt", args.models_root
    )

    missing_models: list[str] = []
    for name, path in [
        ("Face Landmarker", model_path),
        ("Phone Detector", phone_model_path),
        ("Smoking Detector", smoking_model_path),
    ]:
        if not path.exists():
            missing_models.append(f"  - {name}: {path}")

    if missing_models:
        print("\n[ERROR] Required model file(s) not found:")
        for line in missing_models:
            print(line)
        print("\nPlease run the setup script to download models:")
        print("  python scripts/setup_environment.py")
        print(
            "Or specify custom model paths via CLI flags "
            "(--model-path, --phone-model-path, etc.).\n"
        )
        raise SystemExit(1)

    drowsiness_cfg_path = resolve_config_path(
        "drowsiness.yaml", custom_dir=args.config_dir, explicit_file=args.drowsiness_config
    )
    distraction_cfg_path = resolve_config_path(
        "distraction.yaml", custom_dir=args.config_dir, explicit_file=args.distraction_config
    )
    alerts_cfg_path = resolve_config_path(
        "alerts.yaml", custom_dir=args.config_dir, explicit_file=args.alert_policy
    )

    drowsiness_config = load_drowsiness_config(drowsiness_cfg_path)
    distraction_config = load_distraction_config(distraction_cfg_path)
    alert_policy = load_alert_policy(alerts_cfg_path)

    target_device = resolve_device(args.device)
    time_window_sec = (
        args.time_window
        if args.time_window is not None
        else getattr(alert_policy, "time_window_seconds", 2.0)
    )

    print(
        f"CockpitSentinel starting on device: {target_device} "
        f"(temporal window: {time_window_sec:.1f}s)"
    )
    print("Press Q or Esc in the video window to stop.")

    try:
        with (
            DrowsinessDetector(model_path, drowsiness_config, delegate=target_device) as detector,
            DistractionDetector(
                phone_model_path,
                smoking_model_path,
                distraction_config,
                device=target_device,
            ) as distraction_detector,
        ):
            run_live_monitor(
                parse_source(args.source),
                detector,
                alert_policy,
                distraction_detector,
                time_window_seconds=time_window_sec,
            )
    except KeyboardInterrupt:
        print("\nMonitor interrupted by user. Exiting cleanly.")


if __name__ == "__main__":
    main()
