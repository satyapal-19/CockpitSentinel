"""Interactive webcam and video runner for the fatigue monitoring pipeline."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from cockpit_sentinel.alerts.policy import AlertPolicy, load_alert_policy
from cockpit_sentinel.detection import (
    DistractionAnalysis,
    DistractionConfig,
    DistractionDetector,
    load_distraction_config,
)
from cockpit_sentinel.domain import DriverSignals, RiskAssessment, RiskLevel
from cockpit_sentinel.drowsiness import (
    DrowsinessAnalysis,
    DrowsinessConfig,
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

    @property
    def current_stats(self) -> TripStats:
        """Return the current trip statistics snapshot."""
        return TripStats(
            total_seconds=self._total_seconds,
            attentive_seconds=self._attentive_seconds,
            inattentive_seconds=self._inattentive_seconds,
            fps=self._fps,
            total_frames=self._total_frames,
        )


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
        driver_name: str | None = None,
        face_verified: bool = False,
        verification_status: str = "scanning",
        match_confidence: float = 0.0,
    ) -> None:
        self._detector = detector
        self._policy = policy
        self._distraction_detector = distraction_detector
        self.driver_name = driver_name
        self.face_verified = face_verified
        self.verification_status = verification_status
        self.match_confidence = match_confidence

        if time_window_seconds is not None:
            self._time_window_seconds = max(0.0, time_window_seconds)
        else:
            self._time_window_seconds = getattr(policy, "time_window_seconds", 0.0)

        self._temporal_stabilizer = TemporalSignalStabilizer(
            window_seconds=self._time_window_seconds,
            signal_thresholds=signal_thresholds,
        )
        self._trip_tracker = TripTracker()

    @property
    def trip_tracker(self) -> TripTracker:
        """Access the underlying trip tracker."""
        return self._trip_tracker

    def set_driver_verified(self, driver: Any, confidence: float) -> None:
        """Mark the active driver as biometrically verified."""
        self.driver_name = getattr(driver, "name", str(driver))
        self.face_verified = True
        self.verification_status = "verified"
        self.match_confidence = confidence

    def set_driver_unrecognized(self, best_score: float = 0.0) -> None:
        """Mark person in cabin as an unrecognized driver (guest mode)."""
        self.driver_name = None
        self.face_verified = False
        self.verification_status = "unrecognized"
        self.match_confidence = best_score

    def set_driver_mismatch(self, expected_name: str, best_score: float = 0.0) -> None:
        """Mark that an unrecognized person is driving instead of the expected assigned driver."""
        self.driver_name = expected_name
        self.face_verified = False
        self.verification_status = "mismatch"
        self.match_confidence = best_score

    def set_driver_scanning(self) -> None:
        """No face or scanning state."""
        self.face_verified = False
        self.verification_status = "scanning"
        self.match_confidence = 0.0

    def restore_defaults(
        self,
        default_drowsiness_config: Any,
        default_distraction_config: Any,
        default_policy: AlertPolicy,
        default_time_window: float,
    ) -> None:
        """Revert detector and policy back to default safe baseline thresholds."""
        if hasattr(self._detector, "config") and default_drowsiness_config is not None:
            self._detector.config = default_drowsiness_config
        if (
            self._distraction_detector is not None
            and default_distraction_config is not None
            and hasattr(self._distraction_detector, "config")
        ):
            self._distraction_detector.config = default_distraction_config
        self._policy = default_policy
        self._time_window_seconds = default_time_window
        self._temporal_stabilizer = TemporalSignalStabilizer(
            window_seconds=self._time_window_seconds
        )

    def apply_driver_profile(self, driver: Any) -> None:
        """Dynamically adapt detector, policy thresholds, and verified driver status."""
        self.driver_name = getattr(driver, "name", None)
        self.face_verified = True
        self.verification_status = "verified"
        t = getattr(driver, "thresholds", None)
        if not t:
            return

        if hasattr(self._detector, "config"):
            current_cfg = getattr(self._detector, "config")
            min_frames = getattr(current_cfg, "minimum_consecutive_frames", 3)
            self._detector.config = DrowsinessConfig(
                eye_aspect_ratio_threshold=t.eye_aspect_ratio_threshold,
                mouth_aspect_ratio_threshold=t.mouth_aspect_ratio_threshold,
                head_yaw_threshold_degrees=t.head_yaw_threshold_degrees,
                minimum_consecutive_frames=min_frames,
            )

        if self._distraction_detector is not None and hasattr(self._distraction_detector, "config"):
            current_d_cfg = getattr(self._distraction_detector, "config")
            smoking_labels = getattr(
                current_d_cfg, "smoking_labels", ("cigarette", "smoking", "smoke")
            )
            self._distraction_detector.config = DistractionConfig(
                phone_confidence_threshold=t.phone_confidence_threshold,
                smoking_confidence_threshold=t.smoking_confidence_threshold,
                smoking_labels=smoking_labels,
            )

        self._time_window_seconds = t.time_window_seconds
        self._policy = AlertPolicy(
            weights=self._policy.weights,
            caution_score=t.caution_score,
            warning_score=t.warning_score,
            critical_score=t.critical_score,
            time_window_seconds=t.time_window_seconds,
        )
        self._temporal_stabilizer = TemporalSignalStabilizer(
            window_seconds=self._time_window_seconds,
        )

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
            driver_name=self.driver_name,
            face_verified=self.face_verified,
            verification_status=self.verification_status,
            match_confidence=self.match_confidence,
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
    driver_name: str | None = None,
    face_verified: bool = False,
    verification_status: str = "scanning",
    match_confidence: float = 0.0,
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
    header_h = 78 if height < 300 else 88
    cv2.rectangle(image, (0, 0), (width, header_h), (22, 22, 22), thickness=-1)
    cv2.line(image, (0, header_h), (width, header_h), (60, 60, 60), 1)

    # Risk Level & Status
    cv2.putText(
        image,
        f"RISK: {assessment.level.upper()} ({assessment.score})",
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2,
        cv2.LINE_AA,
    )

    status_str = ", ".join(assessment.reasons).upper() if assessment.reasons else "ATTENTIVE"
    status_color = (80, 220, 120) if assessment.level is RiskLevel.SAFE else color
    cv2.putText(
        image,
        f"STATUS: {status_str}",
        (14, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # Line 70: Driver Verification Status Badge
    if verification_status == "verified" and driver_name:
        conf_int = int(round(match_confidence * 100))
        drv_tag = f"DRIVER: {driver_name.upper()} [VERIFIED {conf_int}%]"
        drv_color = (80, 240, 140)  # Bright emerald green
    elif verification_status == "mismatch":
        expected_txt = f" (EXP: {driver_name.upper()})" if driver_name else ""
        drv_tag = f"CABIN ALERT: UNVERIFIED DRIVER{expected_txt}"
        drv_color = (49, 69, 220)  # Alert Red
    elif verification_status == "unrecognized":
        conf_int = int(round(match_confidence * 100))
        conf_txt = f" ({conf_int}% MATCH)" if conf_int > 0 else ""
        drv_tag = f"DRIVER: UNRECOGNIZED PERSON{conf_txt} [GUEST MODE]"
        drv_color = (33, 180, 241)  # Warning Amber
    elif analysis.face_detected:
        drv_tag = "CABIN: SCANNING BIOMETRICS..."
        drv_color = (180, 180, 180)
    else:
        drv_tag = "CABIN: NO DRIVER DETECTED"
        drv_color = (130, 130, 130)

    cv2.putText(
        image,
        drv_tag,
        (14, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        drv_color,
        1,
        cv2.LINE_AA,
    )

    # Right side: Trip Time, Attention %, FPS
    if trip_stats is not None:
        right_x = max(width // 2, width - 230)
        cv2.putText(
            image,
            f"TRIP: {trip_stats.formatted_trip_time}",
            (right_x, 25),
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
            (right_x, 70),
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
    driver_id: str | None = None,
    route_id: str | None = None,
    auto_identify: bool = True,
    face_check_interval: int = 20,
    face_threshold: float = 0.65,
    headless: bool = False,
    cloud_url: str | None = None,
) -> None:
    """Open a webcam, IP camera stream, or video file and monitor driver fatigue and distraction."""

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    db = None
    expected_driver = None
    driver_profile = None
    active_driver_id = None
    route_obj = None
    session_alerts_count = 0
    last_alert_time = 0.0
    trip_id = f"trp-{int(time.time())}"
    trip_start_iso = datetime.now(timezone.utc).isoformat()

    default_drowsiness_config = getattr(detector, "config", None)
    default_distraction_config = (
        getattr(distraction_detector, "config", None) if distraction_detector else None
    )
    default_policy = policy
    default_time_window = time_window_seconds

    try:
        from cockpit_sentinel.database.client import get_db_manager
        from cockpit_sentinel.database.models import AlertIncident, TripSession

        db = get_db_manager()
    except Exception as exc:
        print(f"[DATABASE NOTICE] Operating without database synchronization: {exc}")
        db = None

    if db and driver_id:
        try:
            expected_driver = db.get_driver(driver_id)
        except Exception:
            expected_driver = None

    if db and route_id:
        try:
            route_obj = db.get_route(route_id)
        except Exception:
            route_obj = None

    # Biometric face recognition initialization
    face_identifier = None
    enrolled_drivers = []
    if db and auto_identify:
        try:
            from cockpit_sentinel.recognition import get_face_identifier

            all_drivers = db.list_drivers()
            enrolled_drivers = [
                d
                for d in all_drivers
                if getattr(d, "face_registered", False) and getattr(d, "face_embedding", None)
            ]
            if enrolled_drivers:
                face_identifier = get_face_identifier()
                print(
                    f"[FACE ID] Biometric scanner ready with {len(enrolled_drivers)} enrolled driver profile(s)."
                )
        except Exception as exc:
            print(f"[FACE ID NOTICE] In-cabin biometric identification unavailable: {exc}")

    # Set initial driver state:
    # If auto_identify is True, require camera face verification (do not assume driver_id is verified)
    if not auto_identify and expected_driver:
        # Manual override mode: trust specified driver profile without camera biometric check
        driver_profile = expected_driver
        active_driver_id = expected_driver.id
        initial_status = "verified"
        initial_verified = True
        initial_driver_name = expected_driver.name
    elif expected_driver:
        # Expected driver provided, but biometric verification pending
        initial_status = "scanning"
        initial_verified = False
        initial_driver_name = expected_driver.name
    else:
        # No driver specified, biometric scanning mode
        initial_status = "scanning"
        initial_verified = False
        initial_driver_name = None

    monitor = LiveMonitor(
        detector,
        policy,
        distraction_detector,
        time_window_seconds=time_window_seconds,
        driver_name=initial_driver_name,
        face_verified=initial_verified,
        verification_status=initial_status,
    )

    if not auto_identify and driver_profile:
        monitor.apply_driver_profile(driver_profile)

    frame_counter = 0
    no_face_frames = 0
    consecutive_mismatches = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_counter += 1

            # Biometric cabin face verification (periodic check + quick checks on startup)
            if (
                auto_identify
                and face_identifier is not None
                and enrolled_drivers
                and (frame_counter % face_check_interval == 0 or frame_counter in (3, 6))
            ):
                matched, score = face_identifier.identify_driver(
                    frame, enrolled_drivers, threshold=face_threshold
                )
                if matched is not None:
                    consecutive_mismatches = 0
                    monitor.set_driver_verified(matched, score)
                    if active_driver_id != matched.id:
                        active_driver_id = matched.id
                        driver_profile = matched
                        monitor.apply_driver_profile(matched)
                        print("\n" + "=" * 64)
                        print(" CABIN ACCESS GRANTED: BIOMETRIC FACE VERIFIED!")
                        print(f" Driver: {matched.name} ({matched.license_number})")
                        print(f" Match Confidence: {int(score * 100)}% (threshold >= {int(face_threshold * 100)}%)")
                        print(
                            f" Applied Personalized Thresholds: EAR={matched.thresholds.eye_aspect_ratio_threshold}, "
                            f"MAR={matched.thresholds.mouth_aspect_ratio_threshold}, "
                            f"Yaw={matched.thresholds.head_yaw_threshold_degrees} deg, "
                            f"Window={matched.thresholds.time_window_seconds}s"
                        )
                        print("=" * 64 + "\n")
                else:
                    consecutive_mismatches += 1
                    if score > 0.0 or consecutive_mismatches >= 2:
                        if expected_driver is not None:
                            monitor.set_driver_mismatch(expected_driver.name, score)
                        else:
                            monitor.set_driver_unrecognized(score)
                        if active_driver_id is not None:
                            active_driver_id = None
                            driver_profile = None
                            monitor.restore_defaults(
                                default_drowsiness_config,
                                default_distraction_config,
                                default_policy,
                                default_time_window,
                            )
                            print(
                                f"[FACE ID] Driver face mismatch / unrecognized (score={score:.2f}). "
                                "Reverting to default safe baseline thresholds."
                            )

            processed = monitor.process(frame)

            # Track face presence and reset verified state if seat is empty
            if processed.analysis.face_detected:
                no_face_frames = 0
            else:
                no_face_frames += 1
                if no_face_frames >= 40:
                    if active_driver_id is not None:
                        active_driver_id = None
                        driver_profile = None
                        monitor.restore_defaults(
                            default_drowsiness_config,
                            default_distraction_config,
                            default_policy,
                            default_time_window,
                        )
                    monitor.set_driver_scanning()

            # Audit logging of WARNING/CRITICAL incidents
            now = time.monotonic()
            if (
                db
                and processed.assessment.level in (RiskLevel.WARNING, RiskLevel.CRITICAL)
            ):
                if now - last_alert_time >= 3.5:
                    last_alert_time = now
                    session_alerts_count += 1
                    try:
                        active_sigs = [
                            s
                            for s in (
                                "eyes_closed",
                                "yawning",
                                "looking_away",
                                "phone_detected",
                                "smoking_detected",
                            )
                            if getattr(processed.analysis.signals, s, False)
                            or (
                                processed.distraction
                                and getattr(processed.distraction.signals, s, False)
                            )
                        ]
                        dur = 1.0
                        if processed.active_durations:
                            dur = max(processed.active_durations.values()) or 1.0

                        log_drv_id = (
                            driver_profile.id
                            if driver_profile
                            else (expected_driver.id if expected_driver else "unverified-guest")
                        )
                        log_drv_name = (
                            driver_profile.name
                            if driver_profile
                            else (
                                f"UNVERIFIED (Exp: {expected_driver.name})"
                                if expected_driver
                                else "Guest Driver"
                            )
                        )

                        incident = AlertIncident(
                            trip_id=trip_id,
                            driver_id=log_drv_id,
                            driver_name=log_drv_name,
                            route_name=route_obj.route_name
                            if route_obj
                            else "Active Highway Corridor",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            risk_level=processed.assessment.level.value,
                            risk_score=processed.assessment.score,
                            active_signals=active_sigs,
                            duration_seconds=round(dur, 1),
                            notes=f"Triggered by: {', '.join(processed.assessment.reasons) or 'driver distraction'}",
                        )
                        if db:
                            db.save_alert(incident)

                        if cloud_url:
                            try:
                                import json
                                from urllib.request import Request, urlopen

                                payload = json.dumps(incident.to_dict()).encode("utf-8")
                                req = Request(
                                    f"{cloud_url.rstrip('/')}/api/alerts",
                                    data=payload,
                                    headers={"Content-Type": "application/json"},
                                    method="POST",
                                )
                                with urlopen(req, timeout=2.0):
                                    pass
                            except Exception:
                                pass
                    except Exception:
                        pass

            if not headless:
                cv2.imshow(WINDOW_TITLE, processed.image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                time.sleep(0.001)
                st = monitor.trip_tracker.current_stats
                if st.total_frames % 30 == 0:
                    status_level = processed.assessment.level.value.upper()
                    print(
                        f"\r[LIVE MONITOR] {st.formatted_trip_time} | Attentive: {st.attention_score:.1f}% | "
                        f"Risk: {status_level} | FPS: {st.fps:.1f}   ",
                        end="",
                        flush=True,
                    )
    finally:
        capture.release()
        if not headless:
            cv2.destroyAllWindows()
        if db or cloud_url:
            try:
                stats = monitor.trip_tracker.current_stats
                if stats.total_frames > 0:
                    safety_rating = round(min(5.0, max(1.0, stats.attention_score / 20.0)), 2)
                    save_drv_id = (
                        driver_profile.id
                        if driver_profile
                        else (expected_driver.id if expected_driver else "unverified-guest")
                    )
                    save_drv_name = (
                        driver_profile.name
                        if driver_profile
                        else (
                            f"UNVERIFIED (Exp: {expected_driver.name})"
                            if expected_driver
                            else "Guest Driver"
                        )
                    )
                    trip_session = TripSession(
                        id=trip_id,
                        driver_id=save_drv_id,
                        driver_name=save_drv_name,
                        route_id=route_obj.id if route_obj else None,
                        route_name=route_obj.route_name if route_obj else "Direct Transit",
                        start_time=trip_start_iso,
                        end_time=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=round(stats.total_seconds, 1),
                        attentive_seconds=round(stats.attentive_seconds, 1),
                        inattentive_seconds=round(stats.inattentive_seconds, 1),
                        attention_score=round(stats.attention_score, 1),
                        total_frames=stats.total_frames,
                        average_fps=round(stats.fps, 1),
                        alert_count=session_alerts_count,
                        safety_rating=safety_rating,
                        status="completed",
                    )
                    if db:
                        db.save_trip(trip_session)

                    if cloud_url:
                        try:
                            import json
                            from urllib.request import Request, urlopen

                            payload = json.dumps(trip_session.to_dict()).encode("utf-8")
                            req = Request(
                                f"{cloud_url.rstrip('/')}/api/trips",
                                data=payload,
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            with urlopen(req, timeout=5.0):
                                pass
                            print(f" [CLOUD SYNC] Trip session synchronized with {cloud_url}")
                        except Exception as sync_err:
                            print(f"[CLOUD SYNC NOTICE] Could not upload trip to cloud: {sync_err}")

                    print("\n" + "=" * 62)
                    print(f" [DATABASE] Trip session saved for {save_drv_name}")
                    print(f" Duration: {stats.formatted_trip_time} | Attentive: {stats.attention_score:.1f}%")
                    print(f" Alerts Logged: {session_alerts_count} | Trip Safety Rating: {safety_rating}/5.0")
                    print("=" * 62 + "\n")
            except Exception as err:
                print(f"[DATABASE WARNING] Could not persist trip session: {err}")


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
        "--distraction-model",
        type=Path,
        default=None,
        help="Path to unified YOLO distraction model (.pt, .engine, or openvino dir).",
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
    parser.add_argument(
        "--driver-id",
        type=str,
        default=None,
        help="Driver profile ID to load custom thresholds and record trip telemetry.",
    )
    parser.add_argument(
        "--route-id",
        type=str,
        default=None,
        help="Route corridor ID for trip telemetry tracking.",
    )
    parser.add_argument(
        "--auto-identify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically detect driver's face in cabin and apply their profile (default: True).",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.65,
        help="Biometric face identification cosine similarity threshold (default: 0.65).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a GUI video window (for background services, ECUs, and SSH).",
    )
    parser.add_argument(
        "--cloud-url",
        type=str,
        default=os.environ.get("COCKPIT_CLOUD_URL", None),
        help="Remote cloud dashboard URL (e.g. https://cockpit-sentinel.onrender.com) to push telemetry trips and alerts over HTTPS.",
    )
    args = parser.parse_args()

    model_path = args.model_path or resolve_model_path("face_landmarker.task", args.models_root)
    if args.distraction_model is not None:
        phone_model_path = args.distraction_model
        smoking_model_path = args.distraction_model
    else:
        phone_model_path = args.phone_model_path or resolve_model_path("yolov8n.pt", args.models_root)
        smoking_model_path = args.smoking_model_path or resolve_model_path(
            "yolov8s-worldv2.pt", args.models_root
        )

    model_checks: list[tuple[str, Path]] = [("Face Landmarker", model_path)]
    if phone_model_path == smoking_model_path:
        model_checks.append(("Distraction Detector", phone_model_path))
    else:
        model_checks.append(("Phone Detector", phone_model_path))
        model_checks.append(("Smoking Detector", smoking_model_path))

    missing_models: list[str] = []
    for name, path in model_checks:
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

    time_window_sec = (
        args.time_window
        if args.time_window is not None
        else getattr(alert_policy, "time_window_seconds", 2.0)
    )

    if args.driver_id:
        try:
            from cockpit_sentinel.database.client import get_db_manager

            db = get_db_manager()
            drv = db.get_driver(args.driver_id)
            if drv:
                if not args.auto_identify:
                    # In manual override mode (--no-auto-identify), apply driver custom thresholds immediately
                    t = drv.thresholds
                    drowsiness_config = DrowsinessConfig(
                        eye_aspect_ratio_threshold=t.eye_aspect_ratio_threshold,
                        mouth_aspect_ratio_threshold=t.mouth_aspect_ratio_threshold,
                        head_yaw_threshold_degrees=t.head_yaw_threshold_degrees,
                        minimum_consecutive_frames=drowsiness_config.minimum_consecutive_frames,
                    )
                    distraction_config = DistractionConfig(
                        phone_confidence_threshold=t.phone_confidence_threshold,
                        smoking_confidence_threshold=t.smoking_confidence_threshold,
                        smoking_labels=distraction_config.smoking_labels,
                    )
                    time_window_sec = t.time_window_seconds
                    alert_policy = AlertPolicy(
                        weights=alert_policy.weights,
                        caution_score=t.caution_score,
                        warning_score=t.warning_score,
                        critical_score=t.critical_score,
                        time_window_seconds=t.time_window_seconds,
                    )
                    print("\n" + "=" * 62)
                    print(f" PROFILE LOADED (MANUAL OVERRIDE): {drv.name} ({drv.license_number})")
                    print(f" Status: {drv.status.upper()} | Rating: {drv.rating}★ | Safety Score: {drv.safety_score}%")
                    print(f" Custom EAR: {t.eye_aspect_ratio_threshold} | MAR: {t.mouth_aspect_ratio_threshold} | Yaw: {t.head_yaw_threshold_degrees}°")
                    print(f" Phone Conf: {t.phone_confidence_threshold} | Smoking Conf: {t.smoking_confidence_threshold} | Window: {time_window_sec}s")
                    print("=" * 62 + "\n")
                else:
                    print("\n" + "=" * 62)
                    print(f" EXPECTED DRIVER: {drv.name} ({drv.license_number})")
                    print(f" Status: {drv.status.upper()} | Rating: {drv.rating}★ | Safety Score: {drv.safety_score}%")
                    print(" Biometric cabin verification active — scanning camera face...")
                    print("=" * 62 + "\n")
            else:
                print(f"[WARNING] Driver ID '{args.driver_id}' not found in database. Using defaults.")
        except Exception as exc:
            print(f"[DATABASE NOTICE] Unable to load driver profile from database ({exc}). Using defaults.")

    target_device = resolve_device(args.device)

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
                driver_id=args.driver_id,
                route_id=args.route_id,
                auto_identify=args.auto_identify,
                face_threshold=args.face_threshold,
                headless=args.headless,
                cloud_url=args.cloud_url,
            )
    except KeyboardInterrupt:
        print("\nMonitor interrupted by user. Exiting cleanly.")


if __name__ == "__main__":
    main()
