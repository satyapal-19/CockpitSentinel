"""Data models for CockpitSentinel driver profiles, routes, trips, and alerts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def get_current_iso_time() -> str:
    """Return ISO 8601 UTC formatted string."""
    return datetime.now(timezone.utc).isoformat()


class DriverThresholds(BaseModel):
    """Personalized detection and alert sensitivity thresholds for a driver."""

    eye_aspect_ratio_threshold: float = Field(
        default=0.22, ge=0.10, le=0.45, description="Eye Aspect Ratio (EAR) threshold"
    )
    mouth_aspect_ratio_threshold: float = Field(
        default=0.60, ge=0.30, le=0.90, description="Mouth Aspect Ratio (MAR) yawn threshold"
    )
    head_yaw_threshold_degrees: float = Field(
        default=30.0, ge=10.0, le=60.0, description="Head turn angle threshold (degrees)"
    )
    phone_confidence_threshold: float = Field(
        default=0.30, ge=0.10, le=0.95, description="Phone detection confidence threshold"
    )
    smoking_confidence_threshold: float = Field(
        default=0.30, ge=0.10, le=0.95, description="Smoking detection confidence threshold"
    )
    time_window_seconds: float = Field(
        default=2.0, ge=0.5, le=10.0, description="Temporal smoothing alert window in seconds"
    )
    caution_score: int = Field(default=2, ge=1, le=10, description="Score threshold for CAUTION")
    warning_score: int = Field(default=4, ge=1, le=10, description="Score threshold for WARNING")
    critical_score: int = Field(default=6, ge=1, le=15, description="Score threshold for CRITICAL")


class DriverProfile(BaseModel):
    """Driver profile with license info, custom thresholds, and driving ratings."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=2, max_length=100)
    license_number: str = Field(..., min_length=3, max_length=50)
    experience_years: int = Field(default=1, ge=0, le=60)
    status: str = Field(default="active", pattern="^(active|resting|inactive)$")
    avatar_url: str = Field(default="")
    phone: str = Field(default="")
    notes: str = Field(default="")
    thresholds: DriverThresholds = Field(default_factory=DriverThresholds)
    rating: float = Field(default=5.0, ge=1.0, le=5.0, description="1 to 5 star rating")
    safety_score: int = Field(default=95, ge=0, le=100, description="0 to 100 Safety Index")
    total_trips: int = Field(default=0, ge=0)
    total_distance_km: float = Field(default=0.0, ge=0.0)
    total_alerts: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=get_current_iso_time)
    updated_at: str = Field(default_factory=get_current_iso_time)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class DriverRoute(BaseModel):
    """Trip route specification and driver assignment."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    route_name: str = Field(..., min_length=2, max_length=150)
    origin: str = Field(..., min_length=2, max_length=100)
    destination: str = Field(..., min_length=2, max_length=100)
    distance_km: float = Field(..., gt=0.0)
    estimated_duration_minutes: int = Field(..., gt=0)
    risk_level: str = Field(
        default="moderate", pattern="^(low|moderate|high|expressway)$"
    )
    speed_limit_kmh: int = Field(default=80, ge=20, le=150)
    assigned_driver_id: str | None = None
    assigned_driver_name: str | None = None
    checkpoints: list[str] = Field(default_factory=list)
    notes: str = Field(default="")
    created_at: str = Field(default_factory=get_current_iso_time)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class TripSession(BaseModel):
    """Log of a completed or ongoing driver monitoring session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    driver_id: str
    driver_name: str
    route_id: str | None = None
    route_name: str | None = None
    start_time: str = Field(default_factory=get_current_iso_time)
    end_time: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    attentive_seconds: float = Field(default=0.0, ge=0.0)
    inattentive_seconds: float = Field(default=0.0, ge=0.0)
    attention_score: float = Field(default=100.0, ge=0.0, le=100.0)
    total_frames: int = Field(default=0, ge=0)
    average_fps: float = Field(default=0.0, ge=0.0)
    alert_count: int = Field(default=0, ge=0)
    safety_rating: float = Field(default=5.0, ge=1.0, le=5.0)
    status: str = Field(default="completed", pattern="^(in_progress|completed|aborted)$")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AlertIncident(BaseModel):
    """Detailed audit record for a single detected distraction or fatigue event."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trip_id: str | None = None
    driver_id: str
    driver_name: str
    route_name: str | None = None
    timestamp: str = Field(default_factory=get_current_iso_time)
    risk_level: str = Field(default="warning", pattern="^(caution|warning|critical)$")
    risk_score: int = Field(default=4, ge=1)
    active_signals: list[str] = Field(default_factory=list)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    notes: str = Field(default="")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
