"""Unit tests for CockpitSentinel database models and manager."""

from __future__ import annotations

import pytest

from cockpit_sentinel.database.client import DatabaseManager
from cockpit_sentinel.database.models import (
    AlertIncident,
    DriverProfile,
    DriverRoute,
    DriverThresholds,
    TripSession,
)


@pytest.fixture
def isolated_db(tmp_path):
    """Provide a fresh isolated SQLite database instance for testing."""
    test_db_path = tmp_path / "test_sentinel.db"
    return DatabaseManager(
        sqlite_path=test_db_path,
        preferred_backend="sqlite",
    )


def test_database_initialization_and_seeding(isolated_db: DatabaseManager):
    assert isolated_db.active_backend == "sqlite"
    drivers = isolated_db.list_drivers()
    assert len(drivers) >= 4

    routes = isolated_db.list_routes()
    assert len(routes) >= 4

    trips = isolated_db.list_trips()
    assert len(trips) >= 3

    alerts = isolated_db.list_alerts()
    assert len(alerts) >= 4


def test_driver_profile_crud(isolated_db: DatabaseManager):
    profile = DriverProfile(
        id="drv-test-999",
        name="Vikram Rathod",
        license_number="DL-MH-15-2023-1122334",
        experience_years=7,
        status="active",
        phone="+91 99999 88888",
        thresholds=DriverThresholds(
            eye_aspect_ratio_threshold=0.20,
            mouth_aspect_ratio_threshold=0.65,
            head_yaw_threshold_degrees=35.0,
            phone_confidence_threshold=0.35,
            smoking_confidence_threshold=0.30,
            time_window_seconds=2.5,
        ),
    )

    saved = isolated_db.save_driver(profile)
    assert saved.id == "drv-test-999"

    retrieved = isolated_db.get_driver("drv-test-999")
    assert retrieved is not None
    assert retrieved.name == "Vikram Rathod"
    assert retrieved.thresholds.eye_aspect_ratio_threshold == 0.20
    assert retrieved.thresholds.head_yaw_threshold_degrees == 35.0

    # Test update
    retrieved.thresholds.eye_aspect_ratio_threshold = 0.24
    retrieved.rating = 4.9
    isolated_db.save_driver(retrieved)

    updated = isolated_db.get_driver("drv-test-999")
    assert updated.thresholds.eye_aspect_ratio_threshold == 0.24
    assert updated.rating == 4.9

    # Test delete
    assert isolated_db.delete_driver("drv-test-999") is True
    assert isolated_db.get_driver("drv-test-999") is None


def test_route_crud(isolated_db: DatabaseManager):
    route = DriverRoute(
        id="rt-test-101",
        route_name="Nagpur - Pune Expressway",
        origin="Nagpur",
        destination="Pune",
        distance_km=701.0,
        estimated_duration_minutes=540,
        risk_level="expressway",
        speed_limit_kmh=120,
        checkpoints=["Wardha", "Jalna", "Chhatrapati Sambhajinagar", "Ahmednagar"],
    )

    isolated_db.save_route(route)
    retrieved = isolated_db.get_route("rt-test-101")
    assert retrieved is not None
    assert retrieved.route_name == "Nagpur - Pune Expressway"
    assert len(retrieved.checkpoints) == 4

    assert isolated_db.delete_route("rt-test-101") is True
    assert isolated_db.get_route("rt-test-101") is None


def test_trip_and_driver_stats_refresh(isolated_db: DatabaseManager):
    drivers = isolated_db.list_drivers()
    driver = drivers[0]
    initial_trips = driver.total_trips

    trip = TripSession(
        id="trp-test-unique-1",
        driver_id=driver.id,
        driver_name=driver.name,
        route_id="rt-pune-mum-exp",
        route_name="Pune - Mumbai Expressway",
        duration_seconds=7200.0,
        attentive_seconds=7000.0,
        inattentive_seconds=200.0,
        attention_score=97.2,
        alert_count=2,
        safety_rating=4.9,
    )
    isolated_db.save_trip(trip)

    # Verify refreshed stats
    refreshed_driver = isolated_db.get_driver(driver.id)
    assert refreshed_driver is not None
    assert refreshed_driver.total_trips >= initial_trips + 1


def test_alert_incident_logging_and_filtering(isolated_db: DatabaseManager):
    drivers = isolated_db.list_drivers()
    driver = drivers[0]

    alert = AlertIncident(
        id="alt-test-999",
        driver_id=driver.id,
        driver_name=driver.name,
        route_name="Test Expressway",
        risk_level="critical",
        risk_score=6,
        active_signals=["eyes_closed", "phone_detected"],
        duration_seconds=3.2,
        notes="Automated test alert",
    )
    isolated_db.save_alert(alert)

    # Filter by risk level
    critical_alerts = isolated_db.list_alerts(risk_level="critical")
    assert any(a.id == "alt-test-999" for a in critical_alerts)

    # Filter by driver
    driver_alerts = isolated_db.list_alerts(driver_id=driver.id)
    assert any(a.id == "alt-test-999" for a in driver_alerts)


def test_leaderboard_ranking(isolated_db: DatabaseManager):
    leaderboard = isolated_db.get_leaderboard()
    assert len(leaderboard) >= 4
    # Ensure ordered by safety score descending
    scores = [item["safety_score"] for item in leaderboard]
    assert scores == sorted(scores, reverse=True)
    assert leaderboard[0]["badge"] == "Gold"
