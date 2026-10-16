"""FastAPI dashboard application for CockpitSentinel driver management and telemetry."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from cockpit_sentinel.database.client import get_db_manager
from cockpit_sentinel.database.models import (
    AlertIncident,
    DriverProfile,
    DriverRoute,
    DriverThresholds,
    TripSession,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"
MONITOR_HTML_PATH = TEMPLATES_DIR / "monitor.html"

app = FastAPI(
    title="CockpitSentinel Fleet & Driver Management Dashboard",
    description="Real-time telemetry, personalized driver thresholds, route tracking, alert audits, and safety ratings.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request Schemas
# =============================================================================


class DriverCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    license_number: str = Field(..., min_length=3, max_length=50)
    experience_years: int = Field(default=1, ge=0, le=60)
    status: str = Field(default="active", pattern="^(active|resting|inactive)$")
    phone: str = Field(default="")
    notes: str = Field(default="")
    thresholds: DriverThresholds = Field(default_factory=DriverThresholds)


class DriverUpdateRequest(BaseModel):
    name: str | None = None
    license_number: str | None = None
    experience_years: int | None = None
    status: str | None = None
    phone: str | None = None
    notes: str | None = None
    thresholds: DriverThresholds | None = None
    rating: float | None = None
    safety_score: int | None = None


class FaceEnrollRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image data string")


class RouteCreateRequest(BaseModel):
    route_name: str = Field(..., min_length=2, max_length=150)
    origin: str = Field(..., min_length=2, max_length=100)
    destination: str = Field(..., min_length=2, max_length=100)
    distance_km: float = Field(..., gt=0.0)
    estimated_duration_minutes: int = Field(..., gt=0)
    risk_level: str = Field(default="moderate", pattern="^(low|moderate|high|expressway)$")
    speed_limit_kmh: int = Field(default=80, ge=20, le=150)
    assigned_driver_id: str | None = None
    assigned_driver_name: str | None = None
    checkpoints: list[str] = Field(default_factory=list)
    notes: str = Field(default="")


class RouteUpdateRequest(BaseModel):
    route_name: str | None = None
    origin: str | None = None
    destination: str | None = None
    distance_km: float | None = None
    estimated_duration_minutes: int | None = None
    risk_level: str | None = None
    speed_limit_kmh: int | None = None
    assigned_driver_id: str | None = None
    assigned_driver_name: str | None = None
    checkpoints: list[str] | None = None
    notes: str | None = None


# =============================================================================
# UI Route
# =============================================================================


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    """Serve the interactive responsive cockpit dashboard."""
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard index.html template not found on disk.",
        )
    content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=content)


@app.get("/monitor", response_class=HTMLResponse)
@app.get("/live", response_class=HTMLResponse)
async def serve_monitor() -> HTMLResponse:
    """Serve the standalone in-cabin live camera monitor HUD."""
    if not MONITOR_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Monitor template monitor.html not found on disk.",
        )
    content = MONITOR_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=content)


# =============================================================================
# REST API: Status & Fleet Summary
# =============================================================================


@app.get("/api/status")
async def get_system_status() -> dict[str, Any]:
    """Get system and database backend status."""
    db = get_db_manager()
    return db.get_backend_info()


@app.get("/api/fleet/summary")
async def get_fleet_summary() -> dict[str, Any]:
    """Get high-level fleet statistics for overview KPI cards."""
    db = get_db_manager()
    return db.get_fleet_summary()


# =============================================================================
# REST API: Drivers CRUD
# =============================================================================


@app.get("/api/drivers", response_model=list[DriverProfile])
async def list_drivers() -> list[DriverProfile]:
    """List all driver profiles."""
    db = get_db_manager()
    return db.list_drivers()


@app.get("/api/drivers/{driver_id}", response_model=DriverProfile)
async def get_driver(driver_id: str) -> DriverProfile:
    """Get a specific driver profile by ID."""
    db = get_db_manager()
    driver = db.get_driver(driver_id)
    if not driver:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Driver not found")
    return driver


@app.post("/api/drivers", response_model=DriverProfile, status_code=status.HTTP_201_CREATED)
async def create_driver(req: DriverCreateRequest) -> DriverProfile:
    """Register a new driver profile."""
    db = get_db_manager()
    driver = DriverProfile(
        name=req.name,
        license_number=req.license_number,
        experience_years=req.experience_years,
        status=req.status,
        phone=req.phone,
        notes=req.notes,
        thresholds=req.thresholds,
    )
    return db.save_driver(driver)


@app.put("/api/drivers/{driver_id}", response_model=DriverProfile)
async def update_driver(driver_id: str, req: DriverUpdateRequest) -> DriverProfile:
    """Update driver details or custom biometric detection thresholds."""
    db = get_db_manager()
    driver = db.get_driver(driver_id)
    if not driver:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Driver not found")

    update_fields = req.model_dump(exclude_unset=True)
    for key, val in update_fields.items():
        if val is not None:
            if key == "thresholds":
                driver.thresholds = DriverThresholds(**val)
            else:
                setattr(driver, key, val)

    return db.save_driver(driver)


@app.delete("/api/drivers/{driver_id}", status_code=status.HTTP_200_OK)
async def delete_driver(driver_id: str) -> dict[str, Any]:
    """Remove a driver profile."""
    db = get_db_manager()
    success = db.delete_driver(driver_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Driver not found")
    return {"deleted": True, "id": driver_id}


@app.post("/api/drivers/{driver_id}/enroll-face")
async def enroll_driver_face(driver_id: str, req: FaceEnrollRequest) -> dict[str, Any]:
    """Enroll or update a driver's facial biometric embedding from base64 image data."""
    db = get_db_manager()
    driver = db.get_driver(driver_id)
    if not driver:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Driver not found")

    try:
        header_split = req.image_base64.split(",")
        raw_b64 = header_split[1] if len(header_split) > 1 else header_split[0]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from provided base64 data")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image format: {exc}"
        )

    try:
        from cockpit_sentinel.recognition import get_face_identifier

        identifier = get_face_identifier()
        updated_driver = identifier.enroll_driver_face(driver, img)
        db.save_driver(updated_driver)
        return {
            "success": True,
            "message": f"Facial biometrics enrolled successfully for {updated_driver.name}",
            "driver_id": updated_driver.id,
            "face_registered": True,
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.post("/api/identify-face")
async def identify_face(req: FaceEnrollRequest) -> dict[str, Any]:
    """Identify the driver in the provided image against registered database profiles."""
    try:
        header_split = req.image_base64.split(",")
        raw_b64 = header_split[1] if len(header_split) > 1 else header_split[0]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from provided base64 data")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image data: {exc}"
        )

    try:
        from cockpit_sentinel.recognition import get_face_identifier

        db = get_db_manager()
        drivers = db.list_drivers()
        identifier = get_face_identifier()

        matched_driver, score = identifier.identify_driver(img, drivers, threshold=0.55)
        if matched_driver:
            return {
                "identified": True,
                "driver": matched_driver.to_dict(),
                "confidence": round(score, 3),
                "confidence_percent": int(round(score * 100)),
            }
        return {
            "identified": False,
            "driver": None,
            "confidence": round(score, 3),
            "message": "No registered driver matched",
        }
    except Exception as exc:
        return {
            "identified": False,
            "driver": None,
            "confidence": 0.0,
            "message": f"Face identification unavailable: {exc}",
        }


# =============================================================================
# REST API: Routes CRUD
# =============================================================================


@app.get("/api/routes", response_model=list[DriverRoute])
async def list_routes() -> list[DriverRoute]:
    """List all routes."""
    db = get_db_manager()
    return db.list_routes()


@app.get("/api/routes/{route_id}", response_model=DriverRoute)
async def get_route(route_id: str) -> DriverRoute:
    """Get a specific route by ID."""
    db = get_db_manager()
    route = db.get_route(route_id)
    if not route:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Route not found")
    return route


@app.post("/api/routes", response_model=DriverRoute, status_code=status.HTTP_201_CREATED)
async def create_route(req: RouteCreateRequest) -> DriverRoute:
    """Create a new route corridor."""
    db = get_db_manager()
    route = DriverRoute(
        route_name=req.route_name,
        origin=req.origin,
        destination=req.destination,
        distance_km=req.distance_km,
        estimated_duration_minutes=req.estimated_duration_minutes,
        risk_level=req.risk_level,
        speed_limit_kmh=req.speed_limit_kmh,
        assigned_driver_id=req.assigned_driver_id,
        assigned_driver_name=req.assigned_driver_name,
        checkpoints=req.checkpoints,
        notes=req.notes,
    )
    return db.save_route(route)


@app.put("/api/routes/{route_id}", response_model=DriverRoute)
async def update_route(route_id: str, req: RouteUpdateRequest) -> DriverRoute:
    """Update route information."""
    db = get_db_manager()
    route = db.get_route(route_id)
    if not route:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Route not found")

    update_fields = req.model_dump(exclude_unset=True)
    for key, val in update_fields.items():
        if val is not None:
            setattr(route, key, val)

    return db.save_route(route)


@app.delete("/api/routes/{route_id}", status_code=status.HTTP_200_OK)
async def delete_route(route_id: str) -> dict[str, Any]:
    """Delete a route."""
    db = get_db_manager()
    success = db.delete_route(route_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Route not found")
    return {"deleted": True, "id": route_id}


# =============================================================================
# REST API: Trips & Telemetry
# =============================================================================


@app.get("/api/trips", response_model=list[TripSession])
async def list_trips(
    driver_id: str | None = Query(default=None, description="Optional driver filter"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[TripSession]:
    """List completed or active trip sessions."""
    db = get_db_manager()
    return db.list_trips(driver_id=driver_id, limit=limit)


@app.post("/api/trips", response_model=TripSession, status_code=status.HTTP_201_CREATED)
async def record_trip(trip: TripSession) -> TripSession:
    """Record a trip session log."""
    db = get_db_manager()
    return db.save_trip(trip)


# =============================================================================
# REST API: Alerts Audit History
# =============================================================================


@app.get("/api/alerts", response_model=list[AlertIncident])
async def list_alerts(
    driver_id: str | None = Query(default=None, description="Filter by driver ID"),
    risk_level: str | None = Query(default=None, description="Filter by risk level"),
    limit: int = Query(default=150, ge=1, le=500),
) -> list[AlertIncident]:
    """List detected distraction and fatigue alerts."""
    db = get_db_manager()
    return db.list_alerts(driver_id=driver_id, risk_level=risk_level, limit=limit)


@app.post("/api/alerts", response_model=AlertIncident, status_code=status.HTTP_201_CREATED)
async def record_alert(alert: AlertIncident) -> AlertIncident:
    """Record an alert event."""
    db = get_db_manager()
    return db.save_alert(alert)


# =============================================================================
# REST API: Leaderboard & Ratings
# =============================================================================


@app.get("/api/leaderboard")
async def get_leaderboard() -> list[dict[str, Any]]:
    """Get ranked driver safety ratings leaderboard."""
    db = get_db_manager()
    return db.get_leaderboard()


# =============================================================================
# REST API: Distraction Detection (Phone & Smoking)
# =============================================================================

_distraction_detector = None


def get_distraction_detector():
    global _distraction_detector
    if _distraction_detector is None:
        candidate_paths = [
            Path("weights/trained/yolo11s_balanced_distraction/best.pt"),
            Path("weights/trained/yolo11s_distraction/best.pt"),
            Path(__file__).resolve().parents[3] / "weights" / "trained" / "yolo11s_balanced_distraction" / "best.pt",
            Path(__file__).resolve().parents[3] / "weights" / "trained" / "yolo11s_distraction" / "best.pt",
        ]
        found_path = None
        for p in candidate_paths:
            if p.exists():
                found_path = p
                break

        if found_path:
            try:
                from cockpit_sentinel.detection.distraction import DistractionConfig, DistractionDetector

                cfg = DistractionConfig(
                    phone_confidence_threshold=0.25,
                    smoking_confidence_threshold=0.25,
                    smoking_labels=("smoke", "cigarette", "smoking", "vape"),
                )
                _distraction_detector = DistractionDetector(found_path, config=cfg, device="cpu")
            except Exception as e:
                print(f"[DISTRACTION DETECTOR] Initialization error: {e}")
                _distraction_detector = None
    return _distraction_detector


class DetectDistractionRequest(BaseModel):
    image_base64: str
    phone_threshold: float = 0.25
    smoking_threshold: float = 0.25


@app.post("/api/monitor/detect-distraction")
async def detect_distraction(req: DetectDistractionRequest) -> dict[str, Any]:
    """Detect mobile phone use and smoking in a submitted camera frame."""
    detector = get_distraction_detector()
    if not detector:
        return {
            "available": False,
            "phone_detected": False,
            "smoking_detected": False,
            "detections": [],
            "message": "YOLO distraction model not available on server",
        }

    try:
        header_split = req.image_base64.split(",")
        raw_b64 = header_split[1] if len(header_split) > 1 else header_split[0]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image data: {exc}"
        )

    try:
        analysis = detector.analyze(img)
        detections = []
        for d in analysis.detections:
            detections.append(
                {
                    "box": [int(x) for x in d.box],
                    "label": str(d.label),
                    "confidence": float(round(d.confidence, 3)),
                }
            )

        phone_detected = bool(analysis.signals.phone_detected)
        smoking_detected = bool(analysis.signals.smoking_detected)

        return {
            "available": True,
            "phone_detected": phone_detected,
            "phone_confidence": round(analysis.phone_confidence, 3) if analysis.phone_confidence else 0.0,
            "smoking_detected": smoking_detected,
            "smoking_confidence": round(analysis.smoking_confidence, 3) if analysis.smoking_confidence else 0.0,
            "detections": detections,
        }
    except Exception as err:
        return {
            "available": True,
            "phone_detected": False,
            "smoking_detected": False,
            "detections": [],
            "error": str(err),
        }


# =============================================================================
# CLI Main Entrypoint
# =============================================================================


def run() -> None:
    """Launch the dashboard with uvicorn."""
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "8000")))
    uvicorn.run("cockpit_sentinel.dashboard.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
