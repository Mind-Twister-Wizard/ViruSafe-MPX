# ViruSafe-MPX v1.1 — Professor-Reviewed Manuscript Results

## Internal patient/group-aware five-fold evaluation
- Baseline accuracy: **0.7351**
- v1.1 accuracy: **0.8066**
- Baseline Macro-F1: **0.7292**
- v1.1 Macro-F1: **0.7896**
- Baseline Mpox recall: **0.5563**
- v1.1 Mpox recall: **0.7465**
- Baseline ECE (temperature-scaled): **0.0444**
- v1.1 ECE: **0.0365**
- v1.1 NLL: **0.6713**
- Mondrian empirical coverage after fallback: **0.9205** at nominal 0.90

## Held-out-category OOD evaluation
- Selected method: **fusion__knn_cosine_k10__mahalanobis_pca64**
- Mean AUROC: **0.7623**
- Mean FPR@95TPR: **0.7523**
- Mean OOD rejection at the validation-derived ID-95% operating point: **0.2712**

OOD is interpreted as a warning signal rather than an autonomous rejection gate.

## Cross-dataset MSID evaluation after duplicate filtering
- Discovered files: **770**
- Cross-dataset exact SHA-256 excluded: **0**
- Cross-dataset pHash near matches excluded: **115**
- Within-MSID exact duplicates excluded: **2**
- Final retained images: **653**
- Strict six-class accuracy: **0.7136**
- Restricted shared-four-class accuracy: **0.7642**
- Restricted shared-four-class Macro-F1: **0.7162**
- Mpox F1: **0.8354**

Because patient identities/provenance cannot be fully established from public repository metadata, this is described as cross-dataset evaluation under domain shift rather than definitive clinical external validation.

## Cluster-aware paired inference
- Mean accuracy delta: **+0.0712**, 95% CI **+0.0433 to +0.0974**
- Mean Macro-F1 delta: **+0.0605**, 95% CI **+0.0330 to +0.0868**
- Patient-group sign-flip permutation p-value: **5×10⁻⁶** (200,000 replicates)

## Conformal qualification
The saved v1.1 run reused the same outer-validation observations for fusion selection, temperature fitting and class-conditional conformal quantiles. Reported coverage is therefore **empirical** and is not presented as a formal finite-sample guarantee.

## Interpretation guardrail
These results support further study of a research screening and referral workflow. They do not establish clinical diagnostic validity or support autonomous diagnosis/rejection.
