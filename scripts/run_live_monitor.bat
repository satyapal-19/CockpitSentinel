@echo off
setlocal
cd /d "%~dp0.."

if "%1"=="-Web" goto web
if "%1"=="--web" goto web
if "%1"=="web" goto web

call "%~dp0run_webcam_monitor.bat" %*
goto end

:web
echo [CockpitSentinel] Launching Universal Web Live Monitor (Chrome HUD)...
call "%~dp0run_dashboard.bat" -Monitor

:end
