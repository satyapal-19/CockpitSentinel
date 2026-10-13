"""Unit tests for CockpitSentinel dashboard FastAPI application and endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cockpit_sentinel.dashboard.app import app


@pytest.fixture
def client():
    """Test client for FastAPI dashboard."""
    return TestClient(app)


def test_dashboard_ui_served(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "CockpitSentinel" in response.text
    assert "Driver Profiles & Thresholds" in response.text


def test_api_status_endpoint(client: TestClient):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "backend" in data
    assert "connected" in data


def test_api_fleet_summary(client: TestClient):
    response = client.get("/api/fleet/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_drivers" in data
    assert "active_drivers" in data
    assert "total_trips" in data
    assert "average_attention_score" in data


def test_api_drivers_crud(client: TestClient):
    # 1. List drivers
    res_list = client.get("/api/drivers")
    assert res_list.status_code == 200
    initial_count = len(res_list.json())

    # 2. Create driver
    new_driver = {
        "name": "Ajay Shinde",
        "license_number": "DL-MH-12-2024-0019284",
        "experience_years": 4,
        "status": "active",
        "phone": "+91 98221 44556",
        "notes": "Express intercity driver",
        "thresholds": {
            "eye_aspect_ratio_threshold": 0.23,
            "mouth_aspect_ratio_threshold": 0.62,
            "head_yaw_threshold_degrees": 28.0,
            "phone_confidence_threshold": 0.32,
            "smoking_confidence_threshold": 0.30,
            "time_window_seconds": 2.2,
        },
    }
    res_create = client.post("/api/drivers", json=new_driver)
    assert res_create.status_code == 201
    created = res_create.json()
    driver_id = created["id"]
    assert created["name"] == "Ajay Shinde"

    # 3. Get driver
    res_get = client.get(f"/api/drivers/{driver_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == driver_id

    # 4. Update thresholds
    update_data = {
        "thresholds": {
            "eye_aspect_ratio_threshold": 0.25,
            "mouth_aspect_ratio_threshold": 0.58,
            "head_yaw_threshold_degrees": 32.0,
            "phone_confidence_threshold": 0.35,
            "smoking_confidence_threshold": 0.28,
            "time_window_seconds": 2.5,
        }
    }
    res_update = client.put(f"/api/drivers/{driver_id}", json=update_data)
    assert res_update.status_code == 200
    assert (
        res_update.json()["thresholds"]["eye_aspect_ratio_threshold"] == 0.25
    )

    # 5. Delete driver
    res_del = client.delete(f"/api/drivers/{driver_id}")
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] is True


def test_api_routes_crud(client: TestClient):
    new_route = {
        "route_name": "Nashik - Shirdi Corridor",
        "origin": "Nashik",
        "destination": "Shirdi",
        "distance_km": 88.0,
        "estimated_duration_minutes": 105,
        "risk_level": "moderate",
        "speed_limit_kmh": 80,
        "checkpoints": ["Sinnar", "Pangri"],
    }
    res_create = client.post("/api/routes", json=new_route)
    assert res_create.status_code == 201
    route_id = res_create.json()["id"]

    res_get = client.get(f"/api/routes/{route_id}")
    assert res_get.status_code == 200
    assert res_get.json()["route_name"] == "Nashik - Shirdi Corridor"

    res_del = client.delete(f"/api/routes/{route_id}")
    assert res_del.status_code == 200


def test_api_trips_and_alerts_endpoints(client: TestClient):
    res_trips = client.get("/api/trips")
    assert res_trips.status_code == 200
    assert isinstance(res_trips.json(), list)

    res_alerts = client.get("/api/alerts")
    assert res_alerts.status_code == 200
    assert isinstance(res_alerts.json(), list)

    res_leaderboard = client.get("/api/leaderboard")
    assert res_leaderboard.status_code == 200
    assert len(res_leaderboard.json()) >= 1


def test_api_face_enroll_and_identify(client: TestClient):
    import base64
    import cv2
    import numpy as np

    # Encode blank test image to base64
    dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", dummy_img)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    b64_payload = f"data:image/jpeg;base64,{b64_str}"

    # Non-existent driver returns 404
    res_404 = client.post(
        "/api/drivers/drv-unknown-999/enroll-face",
        json={"image_base64": b64_payload},
    )
    assert res_404.status_code == 404

    # Corrupt base64 string returns 400
    res_bad = client.post("/api/identify-face", json={"image_base64": "not_valid_b64!!!"})
    assert res_bad.status_code == 400

    # Valid base64 frame identification check
    res_ident = client.post("/api/identify-face", json={"image_base64": b64_payload})
    assert res_ident.status_code == 200
    data = res_ident.json()
    assert "identified" in data

