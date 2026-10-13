# =============================================================================
# CockpitSentinel — Live Webcam Monitor
# Runs real-time drowsiness and distraction (phone/smoking) monitoring.
# Run from the project root in PowerShell:
#     .\scripts\run_webcam_monitor.ps1
#
# Optional parameters:
#     .\scripts\run_webcam_monitor.ps1 -CameraIndex 1
#     .\scripts\run_webcam_monitor.ps1 -UsePyTorch
# =============================================================================

param(
    [int]$CameraIndex = 0,
    [switch]$UsePyTorch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot  = Split-Path -Parent $PSScriptRoot
$venvPy    = Join-Path $repoRoot ".venv\Scripts\python.exe"
$monScript = Join-Path $repoRoot "scripts\run_drowsiness_monitor.py"

# --- Checks ---
if (-not (Test-Path -LiteralPath $venvPy)) {
    Write-Error "Virtual environment not found. Activate it first: .\.venv\Scripts\Activate.ps1"
    exit 1
}

if (-not (Test-Path -LiteralPath $monScript)) {
    Write-Error "Monitor script not found: $monScript"
    exit 1
}

# Auto-detect trained unified YOLO11s distraction model
$openvinoModel = Join-Path $repoRoot "weights\trained\yolo11s_distraction\best_openvino_model"
$pytorchModel  = Join-Path $repoRoot "weights\trained\yolo11s_distraction\best.pt"

$distractionArgs = @()
$modelInfo = "Default pretrained models"

if (-not $UsePyTorch -and (Test-Path -LiteralPath $openvinoModel)) {
    $distractionArgs = @("--distraction-model", $openvinoModel)
    $modelInfo = "Trained YOLO11s (OpenVINO Intel CPU accelerated)"
} elseif (Test-Path -LiteralPath $pytorchModel) {
    $distractionArgs = @("--distraction-model", $pytorchModel)
    $modelInfo = "Trained YOLO11s (PyTorch)"
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  CockpitSentinel - Driver Monitoring System " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Camera  : Index $CameraIndex (built-in / USB webcam)" -ForegroundColor Yellow
Write-Host "  Models  : $modelInfo" -ForegroundColor Green
Write-Host "  Signals : EAR  (eye closure)" -ForegroundColor Yellow
Write-Host "            MAR  (yawning)" -ForegroundColor Yellow
Write-Host "            Yaw  (head posture / looking away)" -ForegroundColor Yellow
Write-Host "            Phone & Smoking (YOLO11s detection)" -ForegroundColor Yellow
Write-Host "  Levels  : SAFE -> CAUTION -> WARNING -> CRITICAL" -ForegroundColor Yellow
Write-Host "  Stop    : Press Q or Esc in the video window" -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting monitor..." -ForegroundColor Gray
Write-Host ""

& $venvPy $monScript --source $CameraIndex @distractionArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Monitor exited with error code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[done] Webcam monitor stopped." -ForegroundColor Green
