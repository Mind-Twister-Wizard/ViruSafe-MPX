@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ViruSafe-MPX v1.0.4 - Base Training Pipeline

echo ==============================================================
echo ViruSafe-MPX v1.0.4 - Resumable base training and evaluation
echo ==============================================================
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%CD%\bootstrap.ps1"
if errorlevel 1 goto :error
"%CD%\.venv\Scripts\python.exe" -u "%CD%\.tools\pipeline_launcher.py"
if errorlevel 1 goto :error
if exist "%CD%\outputs\report\index.html" start "" "%CD%\outputs\report\index.html"
echo.
echo Base stage completed. Results: outputs\
pause
exit /b 0
:error
echo.
echo Base stage stopped. Existing checkpoints/state are preserved.
echo Review logs\pipeline.log and rerun this BAT to resume.
pause
exit /b 1
