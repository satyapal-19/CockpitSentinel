# =============================================================================
# CockpitSentinel — 1-Click Live Monitor Launcher
# Automatically activates virtual environment and launches the driver monitor.
#
# Usage (copy-paste directly in terminal):
#     .\scripts\run_live_monitor.ps1         # Launches OpenCV In-Cabin Live Monitor
#     .\scripts\run_live_monitor.ps1 -Web    # Launches Web HUD Live Monitor on Chrome
# =============================================================================

param(
    [switch]$Web,
    [int]$CameraIndex = 0,
    [string]$DriverId = "",
    [string]$RouteId = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$activatePs1 = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"

if (Test-Path -LiteralPath $activatePs1) {
    & $activatePs1
} else {
    Write-Error "Virtual environment not found at $activatePs1. Run .\scripts\setup_environment.ps1 first."
    exit 1
}

if ($Web) {
    Write-Host "[CockpitSentinel] Launching Universal Web Live Monitor (Chrome HUD)..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "run_dashboard.ps1") -Monitor
} else {
    Write-Host "[CockpitSentinel] Launching In-Cabin Edge Live Monitor (OpenCV GUI)..." -ForegroundColor Cyan
    $extraArgs = @("-CameraIndex", $CameraIndex)
    if ($DriverId) { $extraArgs += @("-DriverId", $DriverId) }
    if ($RouteId)  { $extraArgs += @("-RouteId", $RouteId) }
    & (Join-Path $PSScriptRoot "run_webcam_monitor.ps1") @extraArgs
}
