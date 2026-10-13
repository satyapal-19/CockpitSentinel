# =============================================================================
# CockpitSentinel — Live Webcam Monitor
# Runs real-time drowsiness and distraction (phone/smoking) monitoring.
# Run from the project root in PowerShell:
#     .\scripts\run_webcam_monitor.ps1
#
# Optional parameters:
#     .\scripts\run_webcam_monitor.ps1 -CameraIndex 1
#     .\scripts\run_webcam_monitor.ps1 -Unified   # Use only the custom fine-tuned model
# =============================================================================

param(
    [int]$CameraIndex = 0,
    [switch]$Unified
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

# 1. Best Phone Model: YOLO11 COCO (specialized in detecting the physical cell phone object)
$phoneModel = Join-Path $repoRoot "yolo11n.pt"
if (-not (Test-Path -LiteralPath $phoneModel)) {
    $phoneModel = "D:\CockpitSentinel\models\pretrained\yolov8n.pt"
}

# 2. Best Smoking Model: Fine-tuned YOLO11s (specialized in cigarette & smoke detection)
$smokingModel = Join-Path $repoRoot "weights\trained\yolo11s_distraction\best_openvino_model"
if (-not (Test-Path -LiteralPath $smokingModel)) {
    $smokingModel = Join-Path $repoRoot "weights\trained\yolo11s_distraction\best.pt"
}
if (-not (Test-Path -LiteralPath $smokingModel)) {
    $smokingModel = "D:\CockpitSentinel\models\pretrained\yolov8s-worldv2.pt"
}

if ($Unified) {
    # Run strictly the unified fine-tuned model for both
    $distractionArgs = @("--distraction-model", $smokingModel)
    $modelInfo = "Unified YOLO11s Distraction Model ($smokingModel)"
} else {
    # Best-in-class hybrid: Dedicated COCO Phone Detector + Fine-tuned Smoking Detector
    $distractionArgs = @(
        "--phone-model-path", $phoneModel,
        "--smoking-model-path", $smokingModel
    )
    $modelInfo = "YOLO11 Phone Detector + Fine-tuned YOLO11s Smoking Detector"
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
Write-Host "            Phone Detection (YOLO11 'cell phone')" -ForegroundColor Yellow
Write-Host "            Smoking Detection (Fine-tuned 'smoke')" -ForegroundColor Yellow
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
