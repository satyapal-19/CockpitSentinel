@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe.
    pause
    exit /b 1
)

set "TARGET_PATH="
if "%1"=="-Monitor" set "TARGET_PATH=/monitor"
if "%1"=="--monitor" set "TARGET_PATH=/monitor"
if "%1"=="monitor" set "TARGET_PATH=/monitor"

echo =============================================================
echo   CockpitSentinel Fleet & Driver Management Dashboard
echo =============================================================
echo   URL  : http://localhost:8000%TARGET_PATH%
echo   Stop : Press Ctrl+C in this terminal
echo =============================================================

start "" "http://localhost:8000%TARGET_PATH%"
".venv\Scripts\python.exe" -m uvicorn cockpit_sentinel.dashboard:app --host 0.0.0.0 --port 8000 --reload
