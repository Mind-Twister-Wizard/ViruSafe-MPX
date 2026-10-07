# ViruSafe-MPX v1.1 — Manuscript-Ready Results

## Internal patient-grouped five-fold evaluation
- Baseline accuracy: **0.7351**
- v1.1 accuracy: **0.8066**
- Baseline Macro-F1: **0.7292**
- v1.1 Macro-F1: **0.7896**
- Baseline Mpox recall: **0.5563**
- v1.1 Mpox recall: **0.7465**
- Baseline ECE: **0.0444**
- v1.1 ECE: **0.0365**
- Mondrian empirical coverage: **0.9205** (target 0.90)

## Held-out-category OOD evaluation
- Selected method: **fusion__knn_cosine_k10__mahalanobis_pca64**
- Mean AUROC: **0.7623**
- Mean FPR@95TPR: **0.7523**
- Mean OOD rejection at validation-derived threshold: **0.2712**

## External MSID validation after duplicate filtering
- Images retained: **653**
- Strict six-class accuracy: **0.7136**
- Restricted shared-four-class accuracy: **0.7642**
- Restricted shared-four-class Macro-F1: **0.7162**

## Statistical comparison
- McNemar exact p-value (baseline vs v1.1): **1.13493e-09**

## Interpretation guardrail
These results are intended for research reporting. External validation and uncertainty/OOD analyses do not establish clinical diagnostic readiness.
