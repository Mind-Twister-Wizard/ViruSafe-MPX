# ViruSafe-MPX — Experimental Design

## Research question
Can a vision model for six-way differential screening of Mpox and clinically similar
viral exanthems be made more reliable by explicitly combining probability calibration,
open-set/OOD detection, selective abstention, conformal prediction, robustness testing,
and visual explanation?

## Primary in-distribution dataset
**Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)**
Kaggle: `joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20`

The pipeline uses the authors' **Original Images/FOLDS** only to inventory the unique
original images, then automatically rebuilds a **patient-grouped stratified five-fold
protocol** from the filename-derived patient IDs. This provides an explicit leakage
check and keeps each patient entirely within one split of a fold. The supplied
pre-augmented images are not used; online augmentation is training-only.

## Open-set/OOD dataset
**DermNet**
Kaggle: `shubhamgoel27/dermnet`

DermNet is never used to train the six-class disease classifier. A deterministic,
class-balanced sample of clearly non-overlapping dermatology categories is used only as
an unknown/OOD evaluation set. Categories whose names indicate viral infections/warts/
molluscum/pox-like overlap are excluded from OOD sampling.

## Core model
- Backbone: `tf_efficientnetv2_s.in21k_ft_in1k` from timm
- Transfer learning when pretrained weights are reachable
- Weighted cross-entropy with label smoothing
- AdamW + cosine learning-rate decay
- Mixed precision on CUDA
- Test-time horizontal-flip averaging for final inference
- Fixed seed and deterministic fold definitions

## Reliability components
1. **Temperature scaling** fitted on each fold's validation logits.
2. **Open-set scoring** with maximum softmax probability, negative entropy, and energy.
3. **Validation-derived OOD operating point** targeting 95% known-case acceptance.
4. **Selective prediction** with risk-coverage curves and fixed coverage operating points.
5. **Split conformal prediction** targeting 90% marginal coverage.
6. **Grad-CAM** examples for high-confidence correct, uncertain, and erroneous predictions.
7. **Robustness stress tests**: brightness reduction, blur, JPEG compression, central
   occlusion, and Gaussian noise at three severity levels.
8. **Staged post-hoc ablation** quantifying the contribution of calibration, OOD gating,
   abstention, and conformal prediction.

## Resumability
- Dependency installation is idempotent.
- Dataset ZIP files are retained. The public download path attempts HTTP Range resume.
- A `last.pt` checkpoint is saved after every completed epoch for every fold.
- `best.pt` is retained separately.
- Validation/test/OOD logits are cached as compressed NPZ files.
- Robustness conditions are appended to CSV one condition at a time.
- The pipeline state is written atomically to `state/pipeline_state.json`.
- Re-running `RUN_VIRUSAFE_MPX.bat` continues from the most recent completed work.

## Manuscript-ready outputs
The pipeline automatically creates:
- five-fold and pooled diagnostic metrics;
- sensitivity, specificity, F1, AUROC and AUPRC per class;
- normalized confusion matrix;
- ROC and precision-recall curves;
- calibration ECE, Brier score and reliability diagram;
- bootstrap 95% CIs for Macro-F1 and accuracy;
- risk-coverage curves and selective-accuracy operating points;
- conformal empirical coverage, set size and singleton rate;
- OOD AUROC/AUPR/FPR@95TPR and OOD rejection at a validation-selected threshold;
- robustness table/figure;
- Grad-CAM examples;
- ablation table/figure;
- framework architecture diagram;
- experimental workflow diagram;
- risk-adaptive decision-flow diagram;
- all tables in CSV and one Excel workbook;
- a self-contained HTML results report;
- `MANUSCRIPT_READY_RESULTS.md` with automatically populated numerical results.

## Interpretation boundary
This is a retrospective research framework, not a clinical device. A strong chapter
should explicitly distinguish technical validation from clinical validation and discuss
prospective multi-centre testing, subgroup/fairness analysis when metadata permit,
clinical workflow integration, and regulatory oversight.


## Professor-review clarification (v1.1.1 documentation)

The final conference manuscript narrows the interpretation in four important ways:

1. **Validation reuse:** outer validation was reused for fusion selection, temperature fitting and class-conditional conformal quantiles in the saved v1.1 run. Outer-test labels were untouched, but conformal coverage is reported as empirical rather than as a formal finite-sample guarantee.
2. **Cross-dataset language:** MSID is described as cross-dataset evaluation under domain shift because duplicate filtering cannot establish patient-level independence from incomplete public metadata.
3. **Cluster-aware inference:** uncertainty for primary pooled improvements is summarized with patient/group-cluster bootstrap intervals and a patient-group sign-flip permutation analysis.
4. **Clinical boundary:** ViruSafe-MPX is a retrospective research screening/referral prototype; prospective multicentre validation, subgroup analysis, image-quality screening, prespecified thresholds and clinician-led confirmation are required before clinical use.

Machine-readable evidence is stored under results/revision_audit/.
