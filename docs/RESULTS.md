# Results

These files are frozen copies of the manuscript-ready v1.1 experiment outputs.

## Headline internal results

| Metric | Baseline | ViruSafe-MPX v1.1 |
|---|---:|---:|
| Accuracy | 73.51% | **80.66%** |
| Balanced accuracy | 76.83% | **79.45%** |
| Macro-F1 | 72.92% | **78.96%** |
| Macro-AUROC | 94.25% | **95.50%** |
| Mpox recall | 55.63% | **74.65%** |
| ECE ↓ | 4.44% | **3.65%** |
| Brier score ↓ | 0.380 | **0.298** |

![Internal results](../results/figures/internal_performance_confusion.png)

## Reliability

- Selective accuracy at ~70% coverage: **89.98%**.
- Selective accuracy at ~90% coverage: **84.71%**.
- Mondrian conformal coverage: **92.05%** at nominal 90%.
- Mean conformal set size: **1.93** labels.

![Selective calibration](../results/figures/selective_calibration.png)

## Open-set / OOD

The validation-selected fusion of cosine k-NN and PCA-64 Mahalanobis features achieved **0.762 ± 0.058** held-out-category OOD AUROC. The result is useful as a warning signal, not as an autonomous rejection gate.

![Conformal and OOD](../results/figures/conformal_ood.png)

## External MSID validation

After exact/perceptual duplicate filtering against MSLD, **653** external images remained. Restricted shared-four-class accuracy was **76.42%**, Macro-F1 **71.62%**, and external Mpox F1 **83.54%**.

![External validation](../results/figures/external_msid_validation.png)

## Statistical evidence

- Paired group-cluster bootstrap accuracy gain: **+7.12 pp**, 95% CI **+4.33 to +9.74 pp**.
- Macro-F1 gain: **+6.05 pp**, 95% CI **+3.30 to +8.68 pp**.
- McNemar exact test: **p = 1.13 × 10⁻⁹**.

Raw tables are under [`results/tables/`](../results/tables/).
