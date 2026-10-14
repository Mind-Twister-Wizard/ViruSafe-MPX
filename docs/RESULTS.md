# Results

These are frozen aggregate results from ViruSafe-MPX v1.1, with additional audit tables used for the professor-reviewed IEEE ESCI 2027 revision.

## Internal grouped out-of-fold results

| Metric | Baseline | ViruSafe-MPX v1.1 | Cluster-bootstrap 95% CI for v1.1 |
|---|---:|---:|---:|
| Accuracy | 73.51% | **80.66%** | **76.81–84.26%** |
| Macro-F1 | 72.92% | **78.96%** | **74.65–82.73%** |
| Mpox recall | 55.63% | **74.65%** | **67.66–81.40%** |
| ECE (15 bins) ↓ | 4.44% | **3.65%** | **3.14–7.36%** |
| Brier ↓ | 0.380 | **0.298** | **0.247–0.348** |
| NLL ↓ | 0.848 | **0.671** | **0.556–0.792** |
| Macro-AUROC | 94.25% | **95.50%** | — |

Patient/group-cluster bootstrap inference gives a mean accuracy change of **+7.12 pp** (95% CI **+4.33 to +9.74 pp**) and a Macro-F1 change of **+6.05 pp** (95% CI **+3.30 to +8.68 pp**). A patient-group sign-flip permutation check gives **p = 5×10⁻⁶** using 200,000 replicates.

## Selective prediction

At ~70% realized coverage, v1.1 retains **89.98%** accuracy, **88.97%** Macro-F1 and **83.75%** retained Mpox recall. At ~90% coverage, these are **84.71%**, **83.22%** and **79.67%**.

## Calibration and conformal audit

ECE uses 15 equal-width confidence bins. The saved v1.1 run reused the same outer-validation observations for fusion selection, temperature fitting and Mondrian conformal quantiles. Therefore the reported **92.05%** overall coverage is an **empirical coverage measurement**, not a formal finite-sample guarantee. Mean set size is **1.93**; one empty set occurred before top-1 fallback.

## Open-set / OOD audit

OOD method choice used eight DermNet validation categories; performance was reported on 14 disjoint held-out categories. The validation-selected fusion of cosine k-NN (k=10) and PCA-64 Mahalanobis achieved **0.7623 ± 0.0576** AUROC. FPR@95TPR was **75.23%**, and mean OOD rejection at the validation-derived ID-95% operating point was **27.12%**. OOD is therefore treated as a warning signal, not an autonomous rejection gate.

## Cross-dataset MSID evaluation

MSID was not used for model or threshold tuning. Label mapping was Monkeypox/Mpox→Mpox, Chickenpox→Chickenpox, Measles→Measles, Normal/Healthy→Healthy.

From 770 discovered files, 0 exact cross-dataset SHA-256 matches, 115 pHash near matches and 2 within-MSID exact duplicates were removed; **653** images remained. The pHash threshold was Hamming distance ≤4.

Restricted four-class accuracy was **76.42%**, Macro-F1 **71.62%**, and Mpox F1 **83.54%**. Because patient identities/provenance cannot be fully established from public metadata, this is **cross-dataset evaluation under domain shift**, not definitive clinical external validation.

## Robustness and explanation audit

The saved fold-1 backbone reached Macro-F1 **36.69%** under severe Gaussian noise and **57.61% / 57.13%** under severe blur / JPEG. For 12 held-out Grad-CAM cases, the 30% masking mean confidence drop was **0.249** for CAM-guided masking versus **0.223** for a single-draw random control; intervals were wide and overlapping.

Machine-readable audit tables are under results/revision_audit/.
