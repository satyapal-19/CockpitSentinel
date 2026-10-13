# =============================================================================
# CockpitSentinel — Viva Demo Script
# Runs the drowsiness & distraction monitor on a ~1 minute YawDD sample video.
# Run this from the project root in PowerShell:
#     .\scripts\run_viva_demo.ps1
# =============================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot  = Split-Path -Parent $PSScriptRoot
$venvPy    = Join-Path $repoRoot ".venv\Scripts\python.exe"
$monScript = Join-Path $repoRoot "scripts\run_drowsiness_monitor.py"

# Sample video — ~65 seconds (YawDD Dash Male subject 12, with glasses)
$sampleVideo = "D:\CockpitSentinel\data\external\yawdd\Dash\Male\12-MaleGlasses.avi"

# --- Checks ---
if (-not (Test-Path -LiteralPath $venvPy)) {
    Write-Error "Virtual environment not found. Run: .\.venv\Scripts\Activate.ps1 first."
    exit 1
}
if (-not (Test-Path -LiteralPath $sampleVideo)) {
    Write-Error "Sample video not found at: $sampleVideo"
    Write-Host "Make sure datasets are extracted. See docs/datasets.md"
    exit 1
}

# Auto-detect trained unified YOLO11s distraction model
$openvinoModel = Join-Path $repoRoot "weights\trained\yolo11s_distraction\best_openvino_model"
$pytorchModel  = Join-Path $repoRoot "weights\trained\yolo11s_distraction\best.pt"

$distractionArgs = @()
if (Test-Path -LiteralPath $openvinoModel) {
    $distractionArgs = @("--distraction-model", $openvinoModel)
} elseif (Test-Path -LiteralPath $pytorchModel) {
    $distractionArgs = @("--distraction-model", $pytorchModel)
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  CockpitSentinel — Viva Demo" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Video : $sampleVideo" -ForegroundColor Yellow
Write-Host "  Length: ~65 seconds (YawDD, Male with Glasses)" -ForegroundColor Yellow
Write-Host "  Press Q or Esc in the video window to stop." -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

& $venvPy $monScript --source $sampleVideo @distractionArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Monitor exited with error code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[done] Demo complete." -ForegroundColor Green
