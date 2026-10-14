# Reproducibility and resumability

## Audited code state

The professor-reviewed IEEE ESCI manuscript identifies repository commit **06ca6f6** as the audited executable state. Later documentation-only commits clarify reporting, while the model code, trained checkpoints, saved predictions and locked v1.1 protocol underlying the paper remain traceable to that frozen commit.

## Environment used for the completed run

- Windows 11
- Python **3.11.16**
- PyTorch **2.6.0+cu124**
- timm **1.0.30**
- scikit-learn **1.8.0**
- CUDA **12.4**
- NVIDIA GeForce RTX 3050 Ti Laptop GPU, **4 GB**
- base seed **42**; protocol seed **314159**; bootstrap seed **20261007**

## Grouped outer evaluation

MSLD contains 755 original images and 524 recoverable filename-derived patient/groups. Group-aware splitting keeps each group within one partition. Every outer test fold contains 151 images; each image contributes once to pooled OOF results.

| Fold | Train images/groups | Validation images/groups | Test images/groups |
|---:|---:|---:|---:|
| 1 | 483 / 336 | 121 / 82 | 151 / 106 |
| 2 | 483 / 336 | 121 / 83 | 151 / 105 |
| 3 | 483 / 338 | 121 / 82 | 151 / 104 |
| 4 | 483 / 336 | 121 / 84 | 151 / 104 |
| 5 | 483 / 334 | 121 / 85 | 151 / 105 |

## Base model and v1.1 head

Base: EfficientNetV2-S, 224×224, batch 8, AdamW LR 1.5×10⁻⁴, weight decay 1×10⁻⁴, label smoothing 0.05, cosine schedule, AMP, gradient clip 1.0, max 40 epochs, minimum 8, patience 8.

v1.1 freezes the CNN. StandardScaler and multinomial logistic regression are fit on training embeddings only; C and class weighting are chosen by three-fold grouped CV. Fusion alpha is chosen on outer validation. Selected alpha values were **0.25, 0.50, 0.50, 0.25, 0.75**; selected C values were **3, 1, 3, 3, 1**.

## Calibration and conformal qualification

Temperature scaling minimizes validation multiclass NLL. Post-fusion temperatures were **1.003, 0.896, 1.074, 1.045, 1.140**. ECE uses 15 equal-width bins; Brier is mean summed squared error to the one-hot target.

The saved v1.1 run reused outer validation for fusion selection, temperature fitting and class-conditional conformal quantiles. Outer-test labels were never used for tuning, but conformal coverage is reported as **empirical**, not as a formal finite-sample guarantee. A future confirmatory run should separate these roles or use cross-fitting.

## OOD and cross-dataset safeguards

Eight DermNet categories are used for OOD method selection and 14 disjoint categories for held-out OOD reporting.

MSID is excluded from all model/threshold tuning. Exact SHA-256 and 64-bit DCT pHash filtering are applied against MSLD; the pHash near-match threshold is Hamming distance ≤4. Duplicate filtering reduces obvious overlap but cannot prove patient-level independence.

## Statistical inference

Primary uncertainty uses **2,000 patient/group-cluster bootstrap replicates**. Paired model change is additionally checked using **200,000 patient-group sign-flip permutations**. Image-level exact McNemar is a secondary paired check.

## One-click execution and resume

RUN_VIRUSAFE_MPX.bat prepares Python 3.11, the local environment, datasets, grouped base training/evaluation and the v1.1 reliability/cross-dataset stage. Base last.pt checkpoints and state files allow restart after interruption; v1.1 caches and state/v1.1_state.json are also reused.

See docs/MANUSCRIPT_REVISION_AUDIT.md and results/revision_audit/ for the revision audit.
