# Changelog

## v1.1.1 — 2026-10-14
- Documentation-only alignment with the professor-reviewed IEEE ESCI manuscript.
- Reframed MSID as cross-dataset evaluation under domain shift rather than definitive clinical external validation.
- Added grouped fold accounting, calibration/conformal reuse disclosure, cluster-aware inference, duplicate audit, OOD candidate comparison, robustness and Grad-CAM variability artifacts.
- Added traceability to audited executable commit 06ca6f6.
- No model code, protocol lock, trained checkpoints, or frozen predictions changed in this release.


## v1.1.0 — 2026-10-07
- Frozen-feature reliability head with grouped inner-CV selection.
- Validation-selected probability fusion and recalibration.
- Mondrian class-conditional conformal prediction.
- Feature-space OOD validation/test category separation.
- Duplicate-filtered external MSID validation.
- Paired cluster bootstrap + exact McNemar analysis.
- Test-only Grad-CAM and deletion-faithfulness analysis.

## v1.0.4
- Completed five-fold base pipeline with robustness occlusion writeability hotfix.

## v1.0.3
- Hardened Windows launcher to prevent harmless stderr warnings from terminating execution.
