# =============================================================================
# CockpitSentinel — Driver Telemetry & Fleet Management Dashboard Runner
# Launches the FastAPI server and opens the dashboard in your default browser.
#
# Usage:
#     .\scripts\run_dashboard.ps1
#     .\scripts\run_dashboard.ps1 -Port 8080
# =============================================================================

param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$Monitor
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPy   = Join-Path $repoRoot ".venv\Scripts\python.exe"
$envFile  = Join-Path $repoRoot ".env"

if (-not (Test-Path -LiteralPath $venvPy)) {
    Write-Error "Virtual environment not found at $venvPy. Activate or set up .venv first."
    exit 1
}

# If DASHBOARD_PORT is defined in .env and Port parameter wasn't explicitly changed from default 8000
if ($Port -eq 8000 -and (Test-Path -LiteralPath $envFile)) {
    $envLines = Get-Content $envFile
    foreach ($line in $envLines) {
        if ($line -match "^\s*DASHBOARD_PORT\s*=\s*(\d+)") {
            $Port = [int]$matches[1]
            break
        }
    }
}

$targetPath = if ($Monitor) { "/monitor" } else { "" }
$dashboardUrl = "http://localhost:$Port$targetPath"

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  CockpitSentinel - Driver Telemetry & Management Dashboard " -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  Dashboard URL : $dashboardUrl" -ForegroundColor Green
Write-Host "  Documentation : $dashboardUrl/docs" -ForegroundColor Yellow
Write-Host "  Database Sync : MongoDB Atlas + Local SQLite fallback" -ForegroundColor Yellow
Write-Host "  Features      : Profile creation, biometric threshold calibration," -ForegroundColor Yellow
Write-Host "                  route management, trip telemetry, alert audit feed," -ForegroundColor Yellow
Write-Host "                  and driver safety leaderboard ranking." -ForegroundColor Yellow
Write-Host "  Press Ctrl+C to stop the dashboard server." -ForegroundColor Gray
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

if (-not $NoBrowser) {
    # Launch browser after a brief pause so uvicorn is ready
    Start-Job -ScriptBlock {
        param($url)
        Start-Sleep -Seconds 2
        Start-Process $url
    } -ArgumentList $dashboardUrl | Out-Null
}

$env:PYTHONPATH = "$repoRoot\src"
& $venvPy -m uvicorn cockpit_sentinel.dashboard.app:app --host 0.0.0.0 --port $Port --reload
