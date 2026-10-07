$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$LogFile = Join-Path $Logs "bootstrap.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

try {
    Log "Bootstrap started."

    $Tools = Join-Path $Root ".tools"
    $UvDir = Join-Path $Tools "uv"
    $UvExe = Join-Path $UvDir "uv.exe"
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null

    if (-not (Test-Path $UvExe)) {
        Log "Downloading self-contained uv package manager..."
        $UvZip = Join-Path $Tools "uv.zip"
        $UvUrl = "https://github.com/astral-sh/uv/releases/download/0.12.20/uv-x86_64-pc-windows-msvc.zip"
        Invoke-WebRequest -Uri $UvUrl -OutFile $UvZip -UseBasicParsing
        Expand-Archive -Path $UvZip -DestinationPath $UvDir -Force
        Remove-Item $UvZip -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path $UvExe)) {
            $candidate = Get-ChildItem -Path $UvDir -Recurse -Filter "uv.exe" | Select-Object -First 1
            if ($null -eq $candidate) { throw "uv.exe was not found after extraction." }
            Copy-Item $candidate.FullName $UvExe -Force
        }
    }

    Log "Ensuring Python 3.11 is available locally..."
    & $UvExe python install 3.11
    if ($LASTEXITCODE -ne 0) { throw "uv could not install Python 3.11." }

    $Venv = Join-Path $Root ".venv"
    $PyExe = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path $PyExe)) {
        Log "Creating project virtual environment..."
        & $UvExe venv $Venv --python 3.11
        if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
    }

    $InstallMarker = Join-Path $Venv ".virusafe_deps_v3"
    if (-not (Test-Path $InstallMarker)) {
        Log "Installing CUDA-enabled PyTorch (works on RTX GPUs and falls back to CPU if needed)..."
        & $UvExe pip install --python $PyExe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
        if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }

        Log "Installing research dependencies..."
        & $UvExe pip install --python $PyExe -r (Join-Path $Root "requirements.txt")
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
        New-Item -ItemType File -Force -Path $InstallMarker | Out-Null
    } else {
        Log "Dependencies already installed; skipping reinstall."
    }

    $env:PYTHONUTF8 = "1"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    $env:TORCH_HOME = Join-Path $Root ".cache\torch"
    $env:HF_HOME = Join-Path $Root ".cache\huggingface"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
    New-Item -ItemType Directory -Force -Path $env:TORCH_HOME | Out-Null
    New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

    Log "Launching resumable ViruSafe-MPX pipeline..."

    # IMPORTANT: Windows PowerShell 5.1 converts stderr from native programs
    # into PowerShell ErrorRecord objects. With $ErrorActionPreference=Stop,
    # a harmless library warning (for example, Hugging Face's unauthenticated
    # download warning) can terminate the whole launcher even while Python is
    # healthy. Temporarily use Continue only for the Python process, then rely
    # on Python's real exit code to decide whether the pipeline failed.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $PyExe (Join-Path $Root "run_pipeline.py") --resume 2>&1 | Tee-Object -FilePath (Join-Path $Logs "pipeline.log") -Append
        $rc = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($rc -ne 0) { throw "Pipeline exited with code $rc. See logs\pipeline_python.log for the Python traceback." }

    $Report = Join-Path $Root "outputs\report\index.html"
    if (Test-Path $Report) {
        Log "Opening final research report."
        Start-Process $Report
    }

    Log "Bootstrap finished successfully."
    exit 0
}
catch {
    Log ("ERROR: " + $_.Exception.Message)
    Log "Completed checkpoints and downloads are preserved. Re-run RUN_VIRUSAFE_MPX.bat to resume."
    exit 1
}
