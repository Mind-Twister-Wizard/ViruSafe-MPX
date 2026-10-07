@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ViruSafe-MPX v1.1 - Full One-Click Research Pipeline

echo ======================================================================
echo ViruSafe-MPX v1.1 - FULL RESUMABLE PIPELINE

echo Base training/evaluation + reliability enhancement + external validation
echo ======================================================================
echo.
echo Safe to rerun after shutdown. Completed datasets, epochs, folds and caches

echo are reused automatically.
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%CD%\bootstrap.ps1"
if errorlevel 1 goto :error

"%CD%\.venv\Scripts\python.exe" -u "%CD%\.tools\pipeline_launcher.py"
if errorlevel 1 goto :error

"%CD%\.venv\Scripts\python.exe" -c "import joblib; print('v1.1 dependencies: PASS')" >nul 2>&1
if errorlevel 1 (
  "%CD%\.venv\Scripts\python.exe" -m pip install -r "%CD%\requirements_v1.1.txt"
  if errorlevel 1 goto :error
)

"%CD%\.venv\Scripts\python.exe" -u "%CD%\run_v11.py"
if errorlevel 1 goto :error

if exist "%CD%\outputs_v1.1\report\index.html" start "" "%CD%\outputs_v1.1\report\index.html"
echo.
echo ======================================================================
echo ViruSafe-MPX v1.1 COMPLETED SUCCESSFULLY

echo Final report: outputs_v1.1\report\index.html

echo Final tables: outputs_v1.1\tables\

echo Final figures: outputs_v1.1\figures\

echo ======================================================================
pause
exit /b 0

:error
echo.
echo ======================================================================
echo ViruSafe-MPX stopped before completion.
echo Nothing already completed was deleted. Re-run this same BAT to resume.
echo Check logs\pipeline.log and logs\v1.1_pipeline.log.
echo ======================================================================
pause
exit /b 1
