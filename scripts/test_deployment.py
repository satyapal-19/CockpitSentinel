"""End-to-end verification and testing tool for CockpitSentinel deployments.

Tests API health, database connectivity, driver sync, trip telemetry recording,
and alert incident audit feeds against any local, tunneled, or remote cloud URL.

Usage:
    # Test local instance
    python scripts/test_deployment.py

    # Test deployed cloud instance (Render, Railway, or AWS)
    python scripts/test_deployment.py --url https://cockpit-sentinel.onrender.com
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json


def print_banner(url: str) -> None:
    print("\n" + "=" * 65)
    print("      COCKPITSENTINEL CLOUD DEPLOYMENT VERIFICATION TEST")
    print("=" * 65)
    print(f" Target Endpoint : {url}")
    print(f" Timestamp       : {datetime.now(timezone.utc).isoformat()}")
    print("=" * 65 + "\n")


def make_request(
    url: str,
    method: str = "GET",
    data: dict | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict | list | str, float]:
    """Execute an HTTP request and measure latency."""
    headers = {"Content-Type": "application/json", "User-Agent": "CockpitSentinel-DeployTest/1.0"}
    encoded_data = json.dumps(data).encode("utf-8") if data is not None else None
    req = Request(url, data=encoded_data, headers=headers, method=method)

    t0 = time.monotonic()
    try:
        with urlopen(req, timeout=timeout) as resp:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            status_code = resp.status
            body_bytes = resp.read()
            try:
                body = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                body = body_bytes.decode("utf-8", errors="replace")
            return status_code, body, elapsed_ms
    except HTTPError as e:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        err_body = e.read().decode("utf-8", errors="replace")
        return e.code, err_body, elapsed_ms
    except URLError as e:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return 0, str(e.reason), elapsed_ms


def run_tests(base_url: str) -> bool:
    base = base_url.rstrip("/")
    all_passed = True

    # -------------------------------------------------------------------------
    # TEST 1: Root UI & Dashboard Template
    # -------------------------------------------------------------------------
    print("[TEST 1/6] Checking Dashboard HTML Web UI (/)...", end=" ")
    code, body, lat = make_request(f"{base}/")
    if code == 200 and ("CockpitSentinel" in str(body) or "<!DOCTYPE html>" in str(body)):
        print(f"PASS (200 OK, {lat:.1f}ms)")
    else:
        print(f"FAIL (Code {code}, {body})")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 2: Healthcheck & Database Backend Info (/api/status)
    # -------------------------------------------------------------------------
    print("[TEST 2/6] Checking System Status & Database (/api/status)...", end=" ")
    code, body, lat = make_request(f"{base}/api/status")
    if code == 200 and isinstance(body, dict):
        backend = body.get("backend", "unknown")
        status_str = body.get("status", "unknown")
        print(f"PASS (200 OK, {lat:.1f}ms)")
        print(f"          -> DB Backend: {backend.upper()} | Status: {status_str}")
    else:
        print(f"FAIL (Code {code}, {body})")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 3: Fleet KPI Summary (/api/fleet/summary)
    # -------------------------------------------------------------------------
    print("[TEST 3/6] Fetching Fleet KPI Statistics (/api/fleet/summary)...", end=" ")
    code, body, lat = make_request(f"{base}/api/fleet/summary")
    if code == 200 and isinstance(body, dict):
        drivers = body.get("total_drivers", 0)
        trips = body.get("total_trips", 0)
        attention = body.get("avg_attention_score", 0.0)
        print(f"PASS (200 OK, {lat:.1f}ms)")
        print(f"          -> Drivers: {drivers} | Logged Trips: {trips} | Avg Attention: {attention:.1f}%")
    else:
        print(f"FAIL (Code {code}, {body})")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 4: Telemetry Trip Session Ingestion (POST /api/trips)
    # -------------------------------------------------------------------------
    print("[TEST 4/6] Posting Simulated Telemetry Trip (POST /api/trips)...", end=" ")
    test_trip = {
        "id": f"test-trip-{int(time.time())}",
        "driver_id": "test-pilot-01",
        "driver_name": "Cloud Deployment Test Pilot",
        "route_id": "test-corridor",
        "route_name": "Verification Highway Loop",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": 120.0,
        "attentive_seconds": 112.5,
        "inattentive_seconds": 7.5,
        "attention_score": 93.8,
        "total_frames": 3600,
        "average_fps": 30.0,
        "alert_count": 1,
        "safety_rating": 4.69,
        "status": "completed",
    }
    code, body, lat = make_request(f"{base}/api/trips", method="POST", data=test_trip)
    if code in (200, 201):
        print(f"PASS ({code} Created, {lat:.1f}ms)")
        print("          -> Telemetry trip successfully logged into cloud database!")
    else:
        print(f"FAIL (Code {code}, {body})")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 5: Alert Incident Ingestion (POST /api/alerts)
    # -------------------------------------------------------------------------
    print("[TEST 5/6] Posting Distraction Alert Incident (POST /api/alerts)...", end=" ")
    test_alert = {
        "trip_id": test_trip["id"],
        "driver_id": "test-pilot-01",
        "driver_name": "Cloud Deployment Test Pilot",
        "route_name": "Verification Highway Loop",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "risk_level": "warning",
        "risk_score": 5,
        "active_signals": ["phone_detected", "looking_away"],
        "duration_seconds": 2.4,
        "notes": "Verified by automated cloud deployment test suite.",
    }
    code, body, lat = make_request(f"{base}/api/alerts", method="POST", data=test_alert)
    if code in (200, 201):
        print(f"PASS ({code} Created, {lat:.1f}ms)")
        print("          -> Alert incident successfully recorded in audit feed!")
    else:
        print(f"FAIL (Code {code}, {body})")
        all_passed = False

    # -------------------------------------------------------------------------
    # TEST 6: Safety Leaderboard Verification (/api/leaderboard)
    # -------------------------------------------------------------------------
    print("[TEST 6/6] Checking Driver Safety Leaderboard (/api/leaderboard)...", end=" ")
    code, body, lat = make_request(f"{base}/api/leaderboard")
    if code == 200 and isinstance(body, list):
        print(f"PASS (200 OK, {lat:.1f}ms)")
        print(f"          -> Found {len(body)} ranked driver records in leaderboard.")
    else:
        print(f"FAIL (Code {code}, {body})")
        all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print("      ALL DEPLOYMENT VERIFICATION TESTS PASSED (6/6)")
        print(f" Dashboard is fully operational at: {base}")
    else:
        print("      SOME TESTS FAILED - Check server logs above.")
    print("=" * 65 + "\n")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and verify a CockpitSentinel deployment.")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Target base URL to test (default: http://localhost:8000).",
    )
    args = parser.parse_args()

    print_banner(args.url)
    success = run_tests(args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
