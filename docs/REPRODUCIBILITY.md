# Reproducibility and resumability

## One-click execution on Windows

Double-click `RUN_VIRUSAFE_MPX.bat`. The launcher:

1. downloads a self-contained `uv` package manager;
2. installs Python 3.11 locally if required;
3. creates `.venv`;
4. installs CUDA-enabled PyTorch and research dependencies;
5. downloads MSLD v2.0 and DermNet;
6. builds grouped five-fold manifests;
7. trains five EfficientNetV2-S models;
8. evaluates calibration, conformal prediction, OOD, robustness and Grad-CAM;
9. runs the v1.1 frozen-feature reliability extension and MSID external validation;
10. produces final figures, tables and HTML reports.

## Resume after shutdown

The base pipeline saves `last.pt` after each completed epoch and keeps stage state under `state/`. v1.1 writes separate caches and `state/v1.1_state.json`. Re-run the same BAT after a shutdown: completed datasets, folds, epochs, predictions and v1.1 caches are reused.

## Important separation

- `outputs/` = base experiment.
- `outputs_v1.1/` = reliability enhancement + external validation.
- v1.1 **does not retrain the CNN**; it reuses the five trained base checkpoints.

## Repository results

The `results/` directory contains frozen manuscript-ready aggregate outputs only. It intentionally excludes datasets, checkpoints and path-bearing prediction manifests.
