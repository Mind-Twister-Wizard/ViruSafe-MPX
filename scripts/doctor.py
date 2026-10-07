from __future__ import annotations

import platform
from pathlib import Path

print("ViruSafe-MPX repository doctor")
print("Python:", platform.python_version())
print("Platform:", platform.platform())
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch check failed:", exc)
for name in ["config.yaml", "config_v1.1.yaml", "run_pipeline.py", "run_v11.py"]:
    print(f"{name}:", "OK" if Path(name).exists() else "MISSING")
