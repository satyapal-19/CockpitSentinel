"""Tests for the live monitoring frame composition layer."""

import numpy as np

from cockpit_sentinel.alerts.policy import DEFAULT_POLICY
from cockpit_sentinel.detection import DistractionAnalysis
from cockpit_sentinel.domain import DriverSignals, RiskLevel
from cockpit_sentinel.drowsiness import DrowsinessAnalysis
from cockpit_sentinel.pipeline.live_monitor import (
    LiveMonitor,
    TemporalSignalStabilizer,
    TripTracker,
    parse_source,
)


class StubDetector:
    """Return fixed detector output without requiring a camera or model asset."""

    def analyze(self, frame: np.ndarray) -> DrowsinessAnalysis:
        del frame
        return DrowsinessAnalysis(
            signals=DriverSignals(yawning=True),
            face_detected=True,
            eye_aspect_ratio=0.31,
            mouth_aspect_ratio=0.72,
            head_yaw_degrees=4.0,
        )


class StubDistractionDetector:
    def analyze(self, frame: np.ndarray) -> DistractionAnalysis:
        del frame
        return DistractionAnalysis(
            signals=DriverSignals(phone_detected=True), phone_confidence=0.91
        )


def test_live_monitor_composes_detector_and_risk_policy():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    processed = LiveMonitor(StubDetector(), DEFAULT_POLICY).process(frame)

    assert processed.assessment.level is RiskLevel.CAUTION
    assert processed.analysis.signals.yawning
    assert processed.image.shape == frame.shape
    assert np.any(processed.image != frame)


def test_parse_source_handles_webcams_and_video_paths():
    assert parse_source("0") == 0
    assert parse_source("sample-video.mp4") == "sample-video.mp4"


def test_live_monitor_merges_drowsiness_and_distraction_signals():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    processed = LiveMonitor(
        StubDetector(), DEFAULT_POLICY, StubDistractionDetector()
    ).process(frame)

    assert processed.distraction is not None
    assert processed.assessment.level is RiskLevel.CRITICAL
    assert "phone detected" in processed.assessment.reasons


def test_trip_tracker_records_duration_and_stats():
    tracker = TripTracker()
    # Frame 1 at t=100.0s (attentive)
    stats = tracker.update(is_attentive=True, timestamp=100.0)
    assert stats.total_seconds == 0.0
    assert stats.formatted_trip_time == "00:00:00"
    assert stats.attention_score == 100.0

    # Frame 2 at t=165.0s (attentive) -> 65 seconds later
    stats = tracker.update(is_attentive=True, timestamp=165.0)
    assert stats.total_seconds == 65.0
    assert stats.formatted_trip_time == "00:01:05"
    assert stats.attentive_seconds == 65.0
    assert stats.inattentive_seconds == 0.0
    assert stats.attention_score == 100.0

    # Frame 3 at t=195.0s (inattentive) -> 30 seconds of distraction
    stats = tracker.update(is_attentive=False, timestamp=195.0)
    assert stats.total_seconds == 95.0
    assert stats.inattentive_seconds == 30.0
    assert stats.attention_score < 100.0


def test_temporal_signal_stabilizer_requires_sustained_duration():
    stabilizer = TemporalSignalStabilizer(window_seconds=2.0)
    signals = DriverSignals(eyes_closed=True)

    # t = 0.0: first appearance, duration = 0.0s < 2.0s
    stab_signals, durations = stabilizer.update(signals, timestamp=0.0)
    assert not stab_signals.eyes_closed
    assert durations["eyes_closed"] == 0.0

    # t = 1.0: duration = 1.0s < 2.0s (e.g. long blink/brief glance down)
    stab_signals, durations = stabilizer.update(signals, timestamp=1.0)
    assert not stab_signals.eyes_closed
    assert durations["eyes_closed"] == 1.0

    # t = 2.0: duration = 2.0s >= 2.0s -> alert triggered
    stab_signals, durations = stabilizer.update(signals, timestamp=2.0)
    assert stab_signals.eyes_closed
    assert durations["eyes_closed"] == 2.0


def test_temporal_signal_stabilizer_promptly_clears_on_attentive():
    stabilizer = TemporalSignalStabilizer(window_seconds=2.0)
    active_signals = DriverSignals(phone_detected=True)
    safe_signals = DriverSignals()

    # Reach triggered alert state at t=2.5s
    stabilizer.update(active_signals, timestamp=0.0)
    stab_signals, _ = stabilizer.update(active_signals, timestamp=2.5)
    assert stab_signals.phone_detected

    # Driver puts phone away at t=2.6s: MUST clear immediately back to attentive!
    stab_signals, durations = stabilizer.update(safe_signals, timestamp=2.6)
    assert not stab_signals.phone_detected
    assert durations["phone_detected"] == 0.0


class StubDistractionWithBoxes:
    def analyze(self, frame: np.ndarray) -> DistractionAnalysis:
        del frame
        from cockpit_sentinel.detection import DetectionBox

        return DistractionAnalysis(
            signals=DriverSignals(phone_detected=True),
            phone_confidence=0.92,
            detections=(
                DetectionBox(box=(10, 10, 50, 50), label="cell phone", confidence=0.92),
            ),
        )


def test_live_monitor_with_temporal_window_and_dashboard_boxes():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    monitor = LiveMonitor(
        StubDetector(),
        DEFAULT_POLICY,
        StubDistractionWithBoxes(),
        time_window_seconds=2.0,
    )

    # At t=0.0: raw signals exist but duration is 0.0 < 2.0s -> not yet triggered
    frame_0 = monitor.process(frame, timestamp=0.0)
    assert frame_0.assessment.level is RiskLevel.SAFE
    assert frame_0.trip_stats is not None
    assert frame_0.trip_stats.total_seconds == 0.0

    # At t=2.2: sustained phone duration >= 2.0s -> phone alert fires (WARNING)
    frame_alert = monitor.process(frame, timestamp=2.2)
    assert frame_alert.assessment.level is RiskLevel.WARNING
    assert "phone detected" in frame_alert.assessment.reasons
    assert frame_alert.trip_stats.formatted_trip_time == "00:00:02"

    # At t=2.6: yawning also exceeds 2.5s -> escalated to CRITICAL
    frame_critical = monitor.process(frame, timestamp=2.6)
    assert frame_critical.assessment.level is RiskLevel.CRITICAL
    assert "phone detected" in frame_critical.assessment.reasons
    assert "yawning" in frame_critical.assessment.reasons

    # Image rendered with overlay
    assert frame_critical.image.shape == frame.shape
    assert np.any(frame_critical.image != frame)


def test_live_monitor_driver_verification_states_and_overlay():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    monitor = LiveMonitor(
        StubDetector(),
        DEFAULT_POLICY,
        driver_name="Satyapal Gaikwad",
        face_verified=False,
        verification_status="scanning",
    )

    # Initial state: scanning
    assert monitor.verification_status == "scanning"
    assert not monitor.face_verified
    proc = monitor.process(frame)
    assert proc.image is not None

    # Biometric verified state
    from cockpit_sentinel.database.models import DriverProfile, DriverThresholds

    driver = DriverProfile(
        id="drv-test-01",
        name="Satyapal Gaikwad",
        license_number="MH-12-2024",
        thresholds=DriverThresholds(eye_aspect_ratio_threshold=0.25),
    )
    monitor.set_driver_verified(driver, confidence=0.88)
    assert monitor.verification_status == "verified"
    assert monitor.face_verified
    assert monitor.driver_name == "Satyapal Gaikwad"
    assert monitor.match_confidence == 0.88
    proc_verified = monitor.process(frame)
    assert proc_verified.image is not None

    # Mismatch state (different person in cabin when expected driver assigned)
    monitor.set_driver_mismatch(expected_name="Satyapal Gaikwad", best_score=0.18)
    assert monitor.verification_status == "mismatch"
    assert not monitor.face_verified
    assert monitor.driver_name == "Satyapal Gaikwad"
    assert monitor.match_confidence == 0.18
    proc_mismatch = monitor.process(frame)
    assert proc_mismatch.image is not None

    # Unrecognized guest driver state
    monitor.set_driver_unrecognized(best_score=0.10)
    assert monitor.verification_status == "unrecognized"
    assert not monitor.face_verified
    assert monitor.driver_name is None
    proc_unrec = monitor.process(frame)
    assert proc_unrec.image is not None

    # Scanning / empty cabin state
    monitor.set_driver_scanning()
    assert monitor.verification_status == "scanning"
    assert not monitor.face_verified

