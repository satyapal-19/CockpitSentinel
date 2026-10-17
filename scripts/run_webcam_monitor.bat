@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe.
    pause
    exit /b 1
)

set "PHONE_MODEL=weights\pretrained\yolo11n.pt"
if not exist "%PHONE_MODEL%" set "PHONE_MODEL=yolo11n.pt"

set "SMOKE_MODEL=weights\trained\yolo11s_balanced_distraction\best.onnx"
if not exist "%SMOKE_MODEL%" set "SMOKE_MODEL=weights\trained\yolo11s_distraction\best_openvino_model"
if not exist "%SMOKE_MODEL%" set "SMOKE_MODEL=weights\trained\yolo11s_distraction\best.pt"

echo =============================================================
echo   CockpitSentinel - In-Cabin Driver Monitoring System
echo =============================================================
echo   Camera  : Index 0 (built-in / USB webcam)
echo   Models  : YOLO11 Phone + Fine-tuned Distraction
echo   Signals : EAR, MAR, Head Yaw, Mobile Phone, Smoking
echo   Stop    : Click video window and press 'Q' or 'Esc'
echo =============================================================
echo Starting monitor...

".venv\Scripts\python.exe" scripts\run_drowsiness_monitor.py --source 0 --phone-model-path "%PHONE_MODEL%" --smoking-model-path "%SMOKE_MODEL%" %*
