"""Database manager providing MongoDB Atlas storage with offline SQLite fallback."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from cockpit_sentinel.database.models import (
    AlertIncident,
    DriverProfile,
    DriverRoute,
    DriverThresholds,
    TripSession,
    get_current_iso_time,
)
from cockpit_sentinel.utils.paths import find_project_root

load_dotenv()
logger = logging.getLogger(__name__)


def _sanitize_mongo_doc(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip MongoDB internal _id field and ensure id is preserved."""
    if not doc:
        return None
    d = dict(doc)
    d.pop("_id", None)
    return d


class DatabaseManager:
    """Resilient database abstraction supporting MongoDB Atlas and SQLite fallback."""

    def __init__(
        self,
        mongodb_uri: str | None = None,
        db_name: str = "CockpitSentinel",
        sqlite_path: Path | str | None = None,
        preferred_backend: str | None = None,
    ) -> None:
        self.root_dir = find_project_root()
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")
        self.db_name = db_name
        self.preferred_backend = preferred_backend or os.getenv("DATABASE_BACKEND", "auto").lower()

        if sqlite_path is None:
            sqlite_dir = self.root_dir / "data"
            sqlite_dir.mkdir(parents=True, exist_ok=True)
            self.sqlite_path = sqlite_dir / "cockpit_sentinel.db"
        else:
            self.sqlite_path = Path(sqlite_path)

        self._active_backend: str = "sqlite"
        self._mongo_client = None
        self._mongo_db = None

        self._initialize_backend()
        self._initialize_sqlite_schema()
        self.seed_default_data_if_empty()

    def _initialize_backend(self) -> None:
        """Connect to preferred backend with automatic fallback."""
        if self.preferred_backend in ("mongodb", "auto") and self.mongodb_uri:
            try:
                from pymongo import MongoClient
                from pymongo.errors import PyMongoError

                logger.info("Attempting connection to MongoDB Atlas...")
                client = MongoClient(
                    self.mongodb_uri,
                    serverSelectionTimeoutMS=3500,
                    connectTimeoutMS=3500,
                )
                # Test connection via ping
                client.admin.command("ping")
                self._mongo_client = client
                self._mongo_db = client[self.db_name]
                self._active_backend = "mongodb"
                logger.info("Connected successfully to MongoDB Atlas (%s)", self.db_name)
                return
            except Exception as exc:
                logger.warning(
                    "MongoDB Atlas unavailable (%s). Falling back to local SQLite at %s",
                    exc,
                    self.sqlite_path,
                )

        self._active_backend = "sqlite"
        logger.info("Operating in local SQLite mode at: %s", self.sqlite_path)

    @property
    def active_backend(self) -> str:
        """Currently active backend: 'mongodb' or 'sqlite'."""
        return self._active_backend

    def get_backend_info(self) -> dict[str, Any]:
        """Return operational status and metadata about the database layer."""
        return {
            "backend": self._active_backend,
            "connected": True,
            "database": self.db_name if self._active_backend == "mongodb" else str(self.sqlite_path),
            "mongodb_configured": bool(self.mongodb_uri),
        }

    # =========================================================================
    # SQLite Setup & Helpers
    # =========================================================================

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_sqlite_schema(self) -> None:
        """Ensure local SQLite tables and indexes exist."""
        with self._get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS drivers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    license_number TEXT NOT NULL,
                    experience_years INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active',
                    avatar_url TEXT DEFAULT '',
                    phone TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    thresholds TEXT NOT NULL,
                    rating REAL DEFAULT 5.0,
                    safety_score INTEGER DEFAULT 95,
                    total_trips INTEGER DEFAULT 0,
                    total_distance_km REAL DEFAULT 0.0,
                    total_alerts INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS routes (
                    id TEXT PRIMARY KEY,
                    route_name TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    distance_km REAL NOT NULL,
                    estimated_duration_minutes INTEGER NOT NULL,
                    risk_level TEXT NOT NULL DEFAULT 'moderate',
                    speed_limit_kmh INTEGER NOT NULL DEFAULT 80,
                    assigned_driver_id TEXT,
                    assigned_driver_name TEXT,
                    checkpoints TEXT DEFAULT '[]',
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trips (
                    id TEXT PRIMARY KEY,
                    driver_id TEXT NOT NULL,
                    driver_name TEXT NOT NULL,
                    route_id TEXT,
                    route_name TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL DEFAULT 0.0,
                    attentive_seconds REAL DEFAULT 0.0,
                    inattentive_seconds REAL DEFAULT 0.0,
                    attention_score REAL DEFAULT 100.0,
                    total_frames INTEGER DEFAULT 0,
                    average_fps REAL DEFAULT 0.0,
                    alert_count INTEGER DEFAULT 0,
                    safety_rating REAL DEFAULT 5.0,
                    status TEXT DEFAULT 'completed'
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    trip_id TEXT,
                    driver_id TEXT NOT NULL,
                    driver_name TEXT NOT NULL,
                    route_name TEXT,
                    timestamp TEXT NOT NULL,
                    risk_level TEXT NOT NULL DEFAULT 'warning',
                    risk_score INTEGER NOT NULL DEFAULT 4,
                    active_signals TEXT DEFAULT '[]',
                    duration_seconds REAL DEFAULT 0.0,
                    notes TEXT DEFAULT ''
                )
                """
            )
            conn.commit()

    # =========================================================================
    # Driver Profiles CRUD
    # =========================================================================

    def list_drivers(self) -> list[DriverProfile]:
        """Retrieve all driver profiles ordered by rating descending."""
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            docs = list(self._mongo_db["drivers"].find().sort("rating", -1))
            return [DriverProfile(**_sanitize_mongo_doc(d)) for d in docs]

        with self._get_sqlite_conn() as conn:
            rows = conn.execute("SELECT * FROM drivers ORDER BY rating DESC").fetchall()
            drivers = []
            for r in rows:
                d = dict(r)
                if isinstance(d.get("thresholds"), str):
                    d["thresholds"] = json.loads(d["thresholds"])
                drivers.append(DriverProfile(**d))
            return drivers

    def get_driver(self, driver_id: str) -> DriverProfile | None:
        """Fetch a single driver profile by ID."""
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            doc = self._mongo_db["drivers"].find_one({"id": driver_id})
            return DriverProfile(**_sanitize_mongo_doc(doc)) if doc else None

        with self._get_sqlite_conn() as conn:
            row = conn.execute("SELECT * FROM drivers WHERE id = ?", (driver_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            if isinstance(d.get("thresholds"), str):
                d["thresholds"] = json.loads(d["thresholds"])
            return DriverProfile(**d)

    def save_driver(self, profile: DriverProfile) -> DriverProfile:
        """Insert or update a driver profile."""
        doc = profile.to_dict()
        doc["updated_at"] = get_current_iso_time()

        if self._active_backend == "mongodb" and self._mongo_db is not None:
            self._mongo_db["drivers"].replace_one({"id": profile.id}, doc, upsert=True)
            return profile

        with self._get_sqlite_conn() as conn:
            thresholds_json = json.dumps(doc["thresholds"])
            conn.execute(
                """
                INSERT INTO drivers (
                    id, name, license_number, experience_years, status,
                    avatar_url, phone, notes, thresholds, rating, safety_score,
                    total_trips, total_distance_km, total_alerts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    license_number=excluded.license_number,
                    experience_years=excluded.experience_years,
                    status=excluded.status,
                    avatar_url=excluded.avatar_url,
                    phone=excluded.phone,
                    notes=excluded.notes,
                    thresholds=excluded.thresholds,
                    rating=excluded.rating,
                    safety_score=excluded.safety_score,
                    total_trips=excluded.total_trips,
                    total_distance_km=excluded.total_distance_km,
                    total_alerts=excluded.total_alerts,
                    updated_at=excluded.updated_at
                """,
                (
                    doc["id"],
                    doc["name"],
                    doc["license_number"],
                    doc["experience_years"],
                    doc["status"],
                    doc["avatar_url"],
                    doc["phone"],
                    doc["notes"],
                    thresholds_json,
                    doc["rating"],
                    doc["safety_score"],
                    doc["total_trips"],
                    doc["total_distance_km"],
                    doc["total_alerts"],
                    doc["created_at"],
                    doc["updated_at"],
                ),
            )
            conn.commit()
        return profile

    def delete_driver(self, driver_id: str) -> bool:
        """Remove a driver profile by ID."""
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            result = self._mongo_db["drivers"].delete_one({"id": driver_id})
            return result.deleted_count > 0

        with self._get_sqlite_conn() as conn:
            cur = conn.execute("DELETE FROM drivers WHERE id = ?", (driver_id,))
            conn.commit()
            return cur.rowcount > 0

    # =========================================================================
    # Routes CRUD
    # =========================================================================

    def list_routes(self) -> list[DriverRoute]:
        """Retrieve all driver routes."""
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            docs = list(self._mongo_db["routes"].find().sort("route_name", 1))
            return [DriverRoute(**_sanitize_mongo_doc(d)) for d in docs]

        with self._get_sqlite_conn() as conn:
            rows = conn.execute("SELECT * FROM routes ORDER BY route_name ASC").fetchall()
            routes = []
            for r in rows:
                d = dict(r)
                if isinstance(d.get("checkpoints"), str):
                    d["checkpoints"] = json.loads(d["checkpoints"])
                routes.append(DriverRoute(**d))
            return routes

    def get_route(self, route_id: str) -> DriverRoute | None:
        """Fetch a specific route by ID."""
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            doc = self._mongo_db["routes"].find_one({"id": route_id})
            return DriverRoute(**_sanitize_mongo_doc(doc)) if doc else None

        with self._get_sqlite_conn() as conn:
            row = conn.execute("SELECT * FROM routes WHERE id = ?", (route_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            if isinstance(d.get("checkpoints"), str):
                d["checkpoints"] = json.loads(d["checkpoints"])
            return DriverRoute(**d)

    def save_route(self, route: DriverRoute) -> DriverRoute:
        """Save or update route details."""
        doc = route.to_dict()
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            self._mongo_db["routes"].replace_one({"id": route.id}, doc, upsert=True)
            return route

        with self._get_sqlite_conn() as conn:
            checkpoints_json = json.dumps(doc.get("checkpoints", []))
            conn.execute(
                """
                INSERT INTO routes (
                    id, route_name, origin, destination, distance_km,
                    estimated_duration_minutes, risk_level, speed_limit_kmh,
                    assigned_driver_id, assigned_driver_name, checkpoints, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    route_name=excluded.route_name,
                    origin=excluded.origin,
                    destination=excluded.destination,
                    distance_km=excluded.distance_km,
                    estimated_duration_minutes=excluded.estimated_duration_minutes,
                    risk_level=excluded.risk_level,
                    speed_limit_kmh=excluded.speed_limit_kmh,
                    assigned_driver_id=excluded.assigned_driver_id,
                    assigned_driver_name=excluded.assigned_driver_name,
                    checkpoints=excluded.checkpoints,
                    notes=excluded.notes
                """,
                (
                    doc["id"],
                    doc["route_name"],
                    doc["origin"],
                    doc["destination"],
                    doc["distance_km"],
                    doc["estimated_duration_minutes"],
                    doc["risk_level"],
                    doc["speed_limit_kmh"],
                    doc.get("assigned_driver_id"),
                    doc.get("assigned_driver_name"),
                    checkpoints_json,
                    doc.get("notes", ""),
                    doc["created_at"],
                ),
            )
            conn.commit()
        return route

    def delete_route(self, route_id: str) -> bool:
        """Delete route by ID."""
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            res = self._mongo_db["routes"].delete_one({"id": route_id})
            return res.deleted_count > 0

        with self._get_sqlite_conn() as conn:
            cur = conn.execute("DELETE FROM routes WHERE id = ?", (route_id,))
            conn.commit()
            return cur.rowcount > 0

    # =========================================================================
    # Trip Sessions CRUD
    # =========================================================================

    def list_trips(self, driver_id: str | None = None, limit: int = 100) -> list[TripSession]:
        """Fetch trip logs with optional driver filter."""
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            filter_query = {"driver_id": driver_id} if driver_id else {}
            docs = (
                self._mongo_db["trips"]
                .find(filter_query)
                .sort("start_time", -1)
                .limit(limit)
            )
            return [TripSession(**_sanitize_mongo_doc(d)) for d in docs]

        with self._get_sqlite_conn() as conn:
            if driver_id:
                rows = conn.execute(
                    "SELECT * FROM trips WHERE driver_id = ? ORDER BY start_time DESC LIMIT ?",
                    (driver_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trips ORDER BY start_time DESC LIMIT ?", (limit,)
                ).fetchall()
            return [TripSession(**dict(r)) for r in rows]

    def save_trip(self, trip: TripSession) -> TripSession:
        """Save a completed or active trip session and update driver counters."""
        doc = trip.to_dict()
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            self._mongo_db["trips"].replace_one({"id": trip.id}, doc, upsert=True)
        else:
            with self._get_sqlite_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO trips (
                        id, driver_id, driver_name, route_id, route_name,
                        start_time, end_time, duration_seconds, attentive_seconds,
                        inattentive_seconds, attention_score, total_frames,
                        average_fps, alert_count, safety_rating, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        end_time=excluded.end_time,
                        duration_seconds=excluded.duration_seconds,
                        attentive_seconds=excluded.attentive_seconds,
                        inattentive_seconds=excluded.inattentive_seconds,
                        attention_score=excluded.attention_score,
                        total_frames=excluded.total_frames,
                        average_fps=excluded.average_fps,
                        alert_count=excluded.alert_count,
                        safety_rating=excluded.safety_rating,
                        status=excluded.status
                    """,
                    (
                        doc["id"],
                        doc["driver_id"],
                        doc["driver_name"],
                        doc.get("route_id"),
                        doc.get("route_name"),
                        doc["start_time"],
                        doc.get("end_time"),
                        doc["duration_seconds"],
                        doc["attentive_seconds"],
                        doc["inattentive_seconds"],
                        doc["attention_score"],
                        doc["total_frames"],
                        doc["average_fps"],
                        doc["alert_count"],
                        doc["safety_rating"],
                        doc["status"],
                    ),
                )
                conn.commit()

        # Update driver stats
        self._refresh_driver_statistics(trip.driver_id)
        return trip

    # =========================================================================
    # Alerts CRUD
    # =========================================================================

    def list_alerts(
        self,
        driver_id: str | None = None,
        risk_level: str | None = None,
        limit: int = 150,
    ) -> list[AlertIncident]:
        """Fetch alert incidents audit logs with optional filters."""
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            filter_query: dict[str, Any] = {}
            if driver_id:
                filter_query["driver_id"] = driver_id
            if risk_level:
                filter_query["risk_level"] = risk_level
            docs = (
                self._mongo_db["alerts"]
                .find(filter_query)
                .sort("timestamp", -1)
                .limit(limit)
            )
            return [AlertIncident(**_sanitize_mongo_doc(d)) for d in docs]

        with self._get_sqlite_conn() as conn:
            query = "SELECT * FROM alerts WHERE 1=1"
            params: list[Any] = []
            if driver_id:
                query += " AND driver_id = ?"
                params.append(driver_id)
            if risk_level:
                query += " AND risk_level = ?"
                params.append(risk_level)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, tuple(params)).fetchall()
            alerts = []
            for r in rows:
                d = dict(r)
                if isinstance(d.get("active_signals"), str):
                    d["active_signals"] = json.loads(d["active_signals"])
                alerts.append(AlertIncident(**d))
            return alerts

    def save_alert(self, alert: AlertIncident) -> AlertIncident:
        """Log a detected fatigue/distraction incident."""
        doc = alert.to_dict()
        if self._active_backend == "mongodb" and self._mongo_db is not None:
            self._mongo_db["alerts"].replace_one({"id": alert.id}, doc, upsert=True)
        else:
            with self._get_sqlite_conn() as conn:
                signals_json = json.dumps(doc.get("active_signals", []))
                conn.execute(
                    """
                    INSERT INTO alerts (
                        id, trip_id, driver_id, driver_name, route_name,
                        timestamp, risk_level, risk_score, active_signals,
                        duration_seconds, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        risk_level=excluded.risk_level,
                        risk_score=excluded.risk_score,
                        active_signals=excluded.active_signals,
                        duration_seconds=excluded.duration_seconds,
                        notes=excluded.notes
                    """,
                    (
                        doc["id"],
                        doc.get("trip_id"),
                        doc["driver_id"],
                        doc["driver_name"],
                        doc.get("route_name"),
                        doc["timestamp"],
                        doc["risk_level"],
                        doc["risk_score"],
                        signals_json,
                        doc["duration_seconds"],
                        doc.get("notes", ""),
                    ),
                )
                conn.commit()

        # Increment total alerts on driver
        driver = self.get_driver(alert.driver_id)
        if driver:
            driver.total_alerts += 1
            self.save_driver(driver)
        return alert

    # =========================================================================
    # Aggregates & Leaderboard
    # =========================================================================

    def _refresh_driver_statistics(self, driver_id: str) -> None:
        """Recalculate total trips, attention, safety rating, and alerts for a driver."""
        driver = self.get_driver(driver_id)
        if not driver:
            return

        trips = self.list_trips(driver_id=driver_id, limit=500)
        if not trips:
            return

        driver.total_trips = len(trips)
        avg_attention = sum(t.attention_score for t in trips) / len(trips)
        avg_safety = sum(t.safety_rating for t in trips) / len(trips)

        # Safety score out of 100 based on attention and ratings
        driver.safety_score = int(round(max(0, min(100, avg_attention))))
        driver.rating = round(max(1.0, min(5.0, avg_safety)), 2)
        self.save_driver(driver)

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Calculate driver safety leaderboard and ranking."""
        drivers = self.list_drivers()
        # Sort by safety score descending, then rating descending, then trips
        drivers_sorted = sorted(
            drivers,
            key=lambda d: (d.safety_score, d.rating, d.total_trips),
            reverse=True,
        )

        leaderboard = []
        for rank, d in enumerate(drivers_sorted, start=1):
            badge = "Gold" if rank == 1 else "Silver" if rank == 2 else "Bronze" if rank == 3 else "Standard"
            leaderboard.append(
                {
                    "rank": rank,
                    "id": d.id,
                    "name": d.name,
                    "license_number": d.license_number,
                    "status": d.status,
                    "rating": d.rating,
                    "safety_score": d.safety_score,
                    "total_trips": d.total_trips,
                    "total_distance_km": d.total_distance_km,
                    "total_alerts": d.total_alerts,
                    "badge": badge,
                }
            )
        return leaderboard

    def get_fleet_summary(self) -> dict[str, Any]:
        """Fleet overview KPIs for dashboard cards."""
        drivers = self.list_drivers()
        routes = self.list_routes()
        trips = self.list_trips(limit=1000)
        alerts = self.list_alerts(limit=1000)

        total_drivers = len(drivers)
        active_drivers = sum(1 for d in drivers if d.status == "active")
        total_trips = len(trips)
        total_distance = sum(d.total_distance_km for d in drivers)
        avg_attention = (
            sum(t.attention_score for t in trips) / total_trips if total_trips > 0 else 96.5
        )
        avg_fleet_rating = (
            sum(d.rating for d in drivers) / total_drivers if total_drivers > 0 else 4.8
        )

        return {
            "total_drivers": total_drivers,
            "active_drivers": active_drivers,
            "total_routes": len(routes),
            "total_trips": total_trips,
            "total_alerts": len(alerts),
            "total_distance_km": round(total_distance, 1),
            "average_attention_score": round(avg_attention, 1),
            "average_fleet_rating": round(avg_fleet_rating, 2),
            "backend_info": self.get_backend_info(),
        }

    # =========================================================================
    # Seeding Realistic Default Fleet Data
    # =========================================================================

    def seed_default_data_if_empty(self) -> None:
        """Seed realistic drivers, routes, trips, and alerts if database is unpopulated."""
        existing = self.list_drivers()
        if existing:
            return  # Already seeded

        logger.info("Database is empty. Seeding initial fleet data...")

        # 1. Driver Profiles
        driver_satyapal = DriverProfile(
            id="drv-satyapal-001",
            name="Satyapal Gaikwad",
            license_number="DL-MH-13-2021-0089201",
            experience_years=5,
            status="active",
            phone="+91 98230 11425",
            notes="Lead Fleet Specialist. Certified night & expressway driver.",
            thresholds=DriverThresholds(
                eye_aspect_ratio_threshold=0.22,
                mouth_aspect_ratio_threshold=0.60,
                head_yaw_threshold_degrees=30.0,
                phone_confidence_threshold=0.30,
                smoking_confidence_threshold=0.30,
                time_window_seconds=2.0,
            ),
            rating=4.95,
            safety_score=98,
            total_trips=142,
            total_distance_km=14280.5,
            total_alerts=12,
        )

        driver_darshan = DriverProfile(
            id="drv-darshan-002",
            name="Darshan Patil",
            license_number="DL-MH-12-2022-0041289",
            experience_years=4,
            status="active",
            phone="+91 97654 22310",
            notes="Senior Long-Haul Logistics Driver.",
            thresholds=DriverThresholds(
                eye_aspect_ratio_threshold=0.21,
                mouth_aspect_ratio_threshold=0.62,
                head_yaw_threshold_degrees=28.0,
                phone_confidence_threshold=0.32,
                smoking_confidence_threshold=0.30,
                time_window_seconds=1.8,
            ),
            rating=4.82,
            safety_score=94,
            total_trips=98,
            total_distance_km=9850.0,
            total_alerts=19,
        )

        driver_yashodhan = DriverProfile(
            id="drv-yashodhan-003",
            name="Yashodhan Jadhav",
            license_number="DL-MH-14-2020-0078124",
            experience_years=3,
            status="active",
            phone="+91 94220 88714",
            notes="Commercial Highway Pilot. High attentiveness record.",
            thresholds=DriverThresholds(
                eye_aspect_ratio_threshold=0.23,
                mouth_aspect_ratio_threshold=0.58,
                head_yaw_threshold_degrees=32.0,
                phone_confidence_threshold=0.30,
                smoking_confidence_threshold=0.28,
                time_window_seconds=2.2,
            ),
            rating=4.71,
            safety_score=91,
            total_trips=76,
            total_distance_km=7120.0,
            total_alerts=27,
        )

        driver_priya = DriverProfile(
            id="drv-priya-004",
            name="Priya Sharma",
            license_number="DL-MH-01-2019-0091483",
            experience_years=6,
            status="resting",
            phone="+91 98901 33418",
            notes="Express logistics & hazardous cargo certified.",
            thresholds=DriverThresholds(
                eye_aspect_ratio_threshold=0.22,
                mouth_aspect_ratio_threshold=0.60,
                head_yaw_threshold_degrees=30.0,
                phone_confidence_threshold=0.30,
                smoking_confidence_threshold=0.30,
                time_window_seconds=2.0,
            ),
            rating=4.88,
            safety_score=96,
            total_trips=115,
            total_distance_km=11900.0,
            total_alerts=15,
        )

        for d in [driver_satyapal, driver_darshan, driver_yashodhan, driver_priya]:
            self.save_driver(d)

        # 2. Driver Routes
        route_exp = DriverRoute(
            id="rt-pune-mum-exp",
            route_name="Pune - Mumbai Expressway (Yashwantrao Chavan)",
            origin="Pune (Bavdhan Toll)",
            destination="Navi Mumbai (Kalamboli Circle)",
            distance_km=94.5,
            estimated_duration_minutes=110,
            risk_level="expressway",
            speed_limit_kmh=100,
            assigned_driver_id="drv-satyapal-001",
            assigned_driver_name="Satyapal Gaikwad",
            checkpoints=["Urse Toll Plaza", "Lonavala Ghat", "Khandala Bypass", "Khalapur Toll"],
            notes="High-speed corridor. Crucial monitoring during night & dawn hours.",
        )

        route_nh48 = DriverRoute(
            id="rt-pune-bengaluru-nh48",
            route_name="Pune - Bengaluru Commercial Corridor (NH-48)",
            origin="Pune (Katraj)",
            destination="Kolhapur / Belagavi border",
            distance_km=240.0,
            estimated_duration_minutes=260,
            risk_level="high",
            speed_limit_kmh=80,
            assigned_driver_id="drv-darshan-002",
            assigned_driver_name="Darshan Patil",
            checkpoints=["Shirwal", "Satara", "Karad", "Kolhapur Toll"],
            notes="Heavy freight traffic with frequent lane changes. Monitored for fatigue.",
        )

        route_nashik = DriverRoute(
            id="rt-mum-nashik-nh160",
            route_name="Mumbai - Nashik Expressway (NH-160)",
            origin="Thane (Majiwada)",
            destination="Nashik (Dwarka Circle)",
            distance_km=165.0,
            estimated_duration_minutes=195,
            risk_level="moderate",
            speed_limit_kmh=80,
            assigned_driver_id="drv-yashodhan-003",
            assigned_driver_name="Yashodhan Jadhav",
            checkpoints=["Bhiwandi Bypass", "Asangaon", "Kasara Ghat", "Igatpuri"],
            notes="Challenging ghat incline section; sharp curves require strict head yaw vigilance.",
        )

        route_urban = DriverRoute(
            id="rt-pune-ring-road",
            route_name="Pune Metropolitan Ring Route",
            origin="Hinjawadi IT Park Ph-3",
            destination="Hadapsar Magarpatta City",
            distance_km=42.0,
            estimated_duration_minutes=65,
            risk_level="low",
            speed_limit_kmh=60,
            assigned_driver_id=None,
            assigned_driver_name=None,
            checkpoints=["Wakad Bridge", "Aundh", "Shivajinagar", "Camp", "Hadapsar"],
            notes="Intense urban traffic; higher distraction risk from mobile phone usage.",
        )

        for r in [route_exp, route_nh48, route_nashik, route_urban]:
            self.save_route(r)

        # 3. Trip Sessions
        trip_1 = TripSession(
            id="trp-20261013-001",
            driver_id="drv-satyapal-001",
            driver_name="Satyapal Gaikwad",
            route_id="rt-pune-mum-exp",
            route_name="Pune - Mumbai Expressway (Yashwantrao Chavan)",
            start_time="2026-10-13T14:30:00Z",
            end_time="2026-10-13T16:22:15Z",
            duration_seconds=6735.0,
            attentive_seconds=6600.0,
            inattentive_seconds=135.0,
            attention_score=98.0,
            total_frames=168300,
            average_fps=25.0,
            alert_count=1,
            safety_rating=4.9,
            status="completed",
        )

        trip_2 = TripSession(
            id="trp-20261013-002",
            driver_id="drv-darshan-002",
            driver_name="Darshan Patil",
            route_id="rt-pune-bengaluru-nh48",
            route_name="Pune - Bengaluru Commercial Corridor (NH-48)",
            start_time="2026-10-13T08:15:00Z",
            end_time="2026-10-13T12:35:40Z",
            duration_seconds=15640.0,
            attentive_seconds=14700.0,
            inattentive_seconds=940.0,
            attention_score=94.0,
            total_frames=391000,
            average_fps=25.0,
            alert_count=3,
            safety_rating=4.8,
            status="completed",
        )

        trip_3 = TripSession(
            id="trp-20261013-003",
            driver_id="drv-yashodhan-003",
            driver_name="Yashodhan Jadhav",
            route_id="rt-mum-nashik-nh160",
            route_name="Mumbai - Nashik Expressway (NH-160)",
            start_time="2026-10-12T19:00:00Z",
            end_time="2026-10-12T22:15:30Z",
            duration_seconds=11730.0,
            attentive_seconds=10675.0,
            inattentive_seconds=1055.0,
            attention_score=91.0,
            total_frames=293250,
            average_fps=25.0,
            alert_count=5,
            safety_rating=4.6,
            status="completed",
        )

        for t in [trip_1, trip_2, trip_3]:
            self.save_trip(t)

        # 4. Alert History
        alert_1 = AlertIncident(
            id="alt-20261013-101",
            trip_id="trp-20261013-001",
            driver_id="drv-satyapal-001",
            driver_name="Satyapal Gaikwad",
            route_name="Pune - Mumbai Expressway (Yashwantrao Chavan)",
            timestamp="2026-10-13T15:42:10Z",
            risk_level="caution",
            risk_score=2,
            active_signals=["yawning"],
            duration_seconds=2.4,
            notes="Brief yawn detected during Lonavala Ghat ascent. Driver recovered quickly.",
        )

        alert_2 = AlertIncident(
            id="alt-20261013-102",
            trip_id="trp-20261013-002",
            driver_id="drv-darshan-002",
            driver_name="Darshan Patil",
            route_name="Pune - Bengaluru Commercial Corridor (NH-48)",
            timestamp="2026-10-13T10:18:45Z",
            risk_level="warning",
            risk_score=4,
            active_signals=["phone_detected"],
            duration_seconds=3.1,
            notes="Mobile phone glance detected at Karad highway junction.",
        )

        alert_3 = AlertIncident(
            id="alt-20261013-103",
            trip_id="trp-20261013-002",
            driver_id="drv-darshan-002",
            driver_name="Darshan Patil",
            route_name="Pune - Bengaluru Commercial Corridor (NH-48)",
            timestamp="2026-10-13T11:45:12Z",
            risk_level="caution",
            risk_score=2,
            active_signals=["looking_away"],
            duration_seconds=2.6,
            notes="Head yaw deviated past 30 degrees for over 2 seconds.",
        )

        alert_4 = AlertIncident(
            id="alt-20261013-104",
            trip_id="trp-20261013-003",
            driver_id="drv-yashodhan-003",
            driver_name="Yashodhan Jadhav",
            route_name="Mumbai - Nashik Expressway (NH-160)",
            timestamp="2026-10-12T20:55:04Z",
            risk_level="critical",
            risk_score=7,
            active_signals=["eyes_closed", "yawning"],
            duration_seconds=4.2,
            notes="Prolonged eye closure (microsleep indicator) + yawning near Kasara Ghat.",
        )

        for a in [alert_1, alert_2, alert_3, alert_4]:
            self.save_alert(a)

        logger.info("Successfully seeded fleet records.")


# Singleton instance helper
_db_manager_instance: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Return or initialize global DatabaseManager singleton."""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
    return _db_manager_instance
