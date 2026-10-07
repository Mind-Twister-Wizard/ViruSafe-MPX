# Troubleshooting

### Kaggle download fails
Place your Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json` and rerun the same BAT.

### Laptop shuts down
Rerun `RUN_VIRUSAFE_MPX.bat`. The pipeline is resumable.

### CUDA is unavailable
Run `.venv\Scripts\python.exe scripts\doctor.py`. The pipeline can fall back to CPU, but training will be much slower.

### GPU out of memory
The base pipeline contains batch-size recovery logic. Close other GPU-heavy applications and rerun; completed checkpoints are preserved.

### Hugging Face unauthenticated warning
The robust launcher merges/suppresses non-fatal hub warnings so they do not terminate Windows PowerShell. A token is not required merely because this warning appears.

### v1.1 says base predictions are missing
Complete the base stage first. v1.1 expects `outputs/predictions/fold_1.npz` through fold 5.
