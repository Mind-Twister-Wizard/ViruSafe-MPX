@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ViruSafe-MPX v1.1 - Reliability Enhancement

echo ==============================================================
echo ViruSafe-MPX v1.1 - Reliability Enhancement + External Validation
echo ==============================================================
if not exist ".venv\Scripts\python.exe" goto :missing
if not exist "outputs\predictions\fold_1.npz" goto :base
".venv\Scripts\python.exe" -c "import torch, sklearn, scipy, pandas, matplotlib, PIL, yaml, joblib; print('Requirements: PASS'); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements_v1.1.txt
  if errorlevel 1 goto :error
)
".venv\Scripts\python.exe" -u "run_v11.py"
if errorlevel 1 goto :error
if exist "outputs_v1.1\report\index.html" start "" "outputs_v1.1\report\index.html"
echo v1.1 completed successfully.
pause
exit /b 0
:missing
echo ERROR: .venv was not found. Run RUN_VIRUSAFE_MPX.bat or RUN_BASE_v1.0.4.bat first.
pause
exit /b 2
:base
echo ERROR: Base predictions are missing. Complete v1.0.4 first.
pause
exit /b 3
:error
echo v1.1 stopped. Existing caches are preserved; rerun to resume.
pause
exit /b 1
