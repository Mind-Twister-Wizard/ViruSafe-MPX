<p align="center">
  <img src="assets/hero_banner.png" width="100%" alt="ViruSafe-MPX banner"/>
</p>

<h1 align="center">🛡️ ViruSafe-MPX</h1>
<p align="center"><b>Reliability-aware image-based Mpox screening with calibrated uncertainty, selective prediction, OOD warnings and cross-dataset evaluation</b></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2ea44f.svg"/></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Windows-one--click-0078D6?logo=windows&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-6f42c1"/>
</p>

<p align="center">
  <a href="#-results-at-a-glance">Results</a> •
  <a href="#-one-click-reproduction">One-click reproduction</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="docs/RESULTS.md">Detailed results</a> •
  <a href="docs/REPRODUCIBILITY.md">Reproducibility</a> •
  <a href="CITATION.cff">Cite</a>
</p>

---

## 🔬 What is ViruSafe-MPX?

**ViruSafe-MPX** is a research framework for six-class differential screening of **Mpox, chickenpox, measles, cowpox, hand-foot-and-mouth disease (HFMD), and healthy skin**. Instead of optimizing only headline accuracy, it explicitly models **uncertainty, abstention, conformal prediction, out-of-distribution (OOD) behavior, explainability, robustness, and cross-dataset evaluation under domain shift**.

The repository reproduces the code path used for the associated IEEE ESCI 2027 manuscript. The professor-reviewed manuscript frames the system as a retrospective research screening and referral prototype, not as a clinically validated diagnostic device. The v1.1 enhancement reuses five trained EfficientNetV2-S checkpoints and adds a **frozen-feature reliability head** without retraining the CNN.

> [!IMPORTANT]
> This is a **retrospective research prototype**, not a clinical diagnostic device. Outputs must not replace clinician assessment or confirmatory PCR testing.

<p align="center">
  <img src="assets/results_showcase.gif" width="94%" alt="Animated results showcase"/>
</p>

## ✨ Highlights

- 🧠 **EfficientNetV2-S** six-class visual backbone.
- 🧩 **1,280-D frozen-feature reliability head** with grouped inner-CV hyperparameter selection.
- 🎚️ Validation-selected **CNN + reliability-head fusion** and temperature recalibration.
- 🛑 **Selective prediction** / abstention for uncertain cases.
- 🎯 **Mondrian class-conditional conformal prediction** for set-valued uncertainty.
- 🌐 Feature-space **OOD detection** with cosine k-NN + PCA-Mahalanobis fusion.
- 🔍 Test-only **Grad-CAM** plus deletion-faithfulness analysis.
- 🧪 Controlled corruption tests for blur, brightness, JPEG, occlusion and Gaussian noise.
- 🌍 **Duplicate-filtered cross-dataset evaluation** on MSID, with explicit provenance limitations.
- 📊 Paired group-cluster bootstrap confidence intervals and exact **McNemar** significance testing.
- ♻️ **Fully resumable execution**: datasets, epochs, folds, predictions and v1.1 caches survive interruption.

## 📈 Results at a glance

<table>
<tr><th>Metric</th><th>Baseline</th><th>ViruSafe-MPX v1.1</th><th>Change</th></tr>
<tr><td>Accuracy</td><td>73.51%</td><td><b>80.66%</b></td><td>+7.15 pp</td></tr>
<tr><td>Macro-F1</td><td>72.92%</td><td><b>78.96%</b></td><td>+6.03 pp</td></tr>
<tr><td>Macro-AUROC</td><td>94.25%</td><td><b>95.50%</b></td><td>+1.25 pp</td></tr>
<tr><td>Mpox recall</td><td>55.63%</td><td><b>74.65%</b></td><td>+19.01 pp</td></tr>
<tr><td>ECE ↓</td><td>4.44%</td><td><b>3.65%</b></td><td>−0.79 pp</td></tr>
<tr><td>Brier ↓</td><td>0.380</td><td><b>0.298</b></td><td>−0.082</td></tr>
</table>

Additional safety-oriented results:

| Evaluation | Result |
|---|---:|
| Selective accuracy @ ~70% coverage | **89.98%** |
| Selective accuracy @ ~90% coverage | **84.71%** |
| Mondrian conformal coverage (nominal 90%) | **92.05%** |
| Held-out-category OOD AUROC | **0.762 ± 0.058** |
| OOD FPR@95TPR | **75.23%** |
| OOD rejection at the validation-derived ID-95% operating point | **27.12%** |
| Cross-dataset MSID restricted 4-class accuracy | **76.42%** |
| External Mpox F1 | **83.54%** |
| McNemar exact p-value | **1.13 × 10⁻⁹** |

<p align="center"><img src="results/figures/internal_performance_confusion.png" width="94%" alt="Internal performance and confusion matrix"/></p>

## 🩺 Visual disease space

The framework studies six visually overlapping categories. The montage below is included for **research orientation only** and must not be interpreted as a clinical diagnostic atlas. Source-image rights remain with the original image/dataset owners.

<p align="center"><img src="assets/clinical_reference_montage.png" width="88%" alt="Representative lesion categories: Mpox, Chickenpox, Measles, Cowpox, HFMD and Healthy"/></p>

## 🧠 Architecture

<p align="center"><img src="results/figures/architecture.png" width="95%" alt="ViruSafe-MPX architecture"/></p>

The prediction path is deliberately **bounded**: calibrated class probabilities are accompanied by selective prediction, conformal sets, OOD warning signals and a mandatory referral path for ambiguous cases. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

<p align="center"><img src="assets/pipeline_walkthrough.gif" width="92%" alt="Animated workflow walkthrough"/></p>

## 🚀 One-click reproduction

### Windows — recommended

1. Download/clone this repository.
2. Optional but recommended for Kaggle reliability: place `kaggle.json` in `%USERPROFILE%\.kaggle\kaggle.json`.
3. Double-click:

```text
RUN_VIRUSAFE_MPX.bat
```

The launcher automatically creates a local Python 3.11 environment, installs CUDA-enabled PyTorch, downloads the required datasets, trains/evaluates the five folds, and then runs the v1.1 reliability/cross-dataset evaluation stage.

> [!TIP]
> If the laptop shuts down or execution is interrupted, **run the same BAT again**. Completed work is preserved and reused.

### Stage-specific execution

```text
RUN_BASE_v1.0.4.bat        # base 5-fold training + evaluation
RUN_RELIABILITY_v1.1.bat   # v1.1 enhancement; reuses base checkpoints
```

Environment diagnostic:

```bash
.venv\Scripts\python.exe scripts\doctor.py
```

## 📁 Repository map

```text
ViruSafe-MPX/
├── RUN_VIRUSAFE_MPX.bat          # full one-click resumable execution
├── RUN_BASE_v1.0.4.bat           # base stage only
├── RUN_RELIABILITY_v1.1.bat      # v1.1 stage only
├── run_pipeline.py                # base orchestrator
├── run_v11.py                     # reliability/external-validation orchestrator
├── config.yaml                    # base protocol
├── config_v1.1.yaml               # locked v1.1 protocol
├── src/                           # base implementation
├── src_v11/                       # v1.1 reliability implementation
├── results/
│   ├── figures/                   # manuscript-ready figures
│   ├── tables/                    # frozen aggregate result tables
│   └── summaries/                 # JSON/Markdown summaries + protocol lock
├── assets/                        # GitHub hero + animated walkthroughs
├── docs/                          # architecture, results, licenses, reproducibility
├── paper/                         # conference manuscript / citation note
└── .github/                       # CI + issue/PR templates
```

## 🗂️ Data protocol

| Dataset | Role | Leakage guardrail |
|---|---|---|
| **MSLD v2.0** | six-class development and grouped out-of-fold testing | patient/group-aware splits + duplicate audit |
| **DermNet** | OOD method selection and held-out-category test | disjoint category subsets; never used to train classifier |
| **MSID** | cross-dataset evaluation only | exact SHA-256 + perceptual-hash filtering vs MSLD; patient-level independence cannot be proven from repository metadata |

Dataset images are **not redistributed** in this repository. See [`docs/DATASETS_AND_LICENSES.md`](docs/DATASETS_AND_LICENSES.md).

## 🛡️ Reliability, not just accuracy

<p align="center"><img src="results/figures/selective_calibration.png" width="92%" alt="Selective prediction and calibration"/></p>

At approximately 70% retained coverage, selective accuracy reaches **89.98%**. Mondrian conformal prediction obtains **92.05%** empirical overall coverage at a nominal 90% target. The OOD component improves separation of unfamiliar dermatology images, but its operating-point performance is intentionally reported as a **warning signal rather than an autonomous gate**.

<p align="center"><img src="results/figures/conformal_ood.png" width="92%" alt="Conformal and OOD results"/></p>

## 🌍 Cross-dataset evaluation on MSID

MSID is excluded from all training, model selection and threshold tuning. Of **770** discovered files, **115** cross-dataset near matches (64-bit pHash, Hamming distance ≤4) and **2** within-MSID exact SHA-256 duplicates were removed; **653** images remained. Because public-image provenance and patient identifiers are incomplete, this is reported as **cross-dataset evaluation under domain shift**, not definitive clinical external validation.

<p align="center"><img src="results/figures/external_msid_validation.png" width="92%" alt="External MSID validation"/></p>

## 🔍 Explainability and robustness

<p align="center"><img src="assets/gradcam_montage.png" width="78%" alt="Held-out Grad-CAM montage"/></p>

Grad-CAM visualizations are restricted to held-out test examples. Deletion faithfulness compares confidence changes after CAM-guided masking with equally sized random masks. Robustness tests expose sensitivity to acquisition degradation, especially severe Gaussian noise.

<p align="center"><img src="results/figures/robustness_stress_test.png" width="90%" alt="Robustness stress test"/></p>

## 📊 Frozen manuscript-ready outputs

The repository ships aggregate outputs from the completed v1.1 experiment so readers can inspect the reported numbers without rerunning multi-hour training:

- [`results/tables/`](results/tables/) — CSV + Excel result tables.
- [`results/summaries/results_summary_v1.1.json`](results/summaries/results_summary_v1.1.json) — machine-readable headline results.
- [`results/summaries/MANUSCRIPT_READY_RESULTS_v1.1.md`](results/summaries/MANUSCRIPT_READY_RESULTS_v1.1.md) — manuscript-oriented summary.
- [`results/summaries/PROTOCOL_LOCK_v1.1.json`](results/summaries/PROTOCOL_LOCK_v1.1.json) — pre-specified v1.1 selection rules.
- [`docs/results_report.html`](docs/results_report.html) — generated HTML report.

Datasets, CNN checkpoints, caches and local-path prediction manifests are intentionally excluded from Git.

## 🧪 Reproducibility guardrails

- Outer test labels are **not used** to fit the v1.1 feature-head hyperparameters or choose the CNN/reliability-head blend.
- The same outer-validation observations were reused for fusion selection, temperature fitting and Mondrian quantiles in the saved v1.1 run; therefore conformal coverage is reported as **empirical coverage**, not as a formal finite-sample guarantee.
- OOD methods are selected on **different DermNet categories** from the held-out OOD test categories.
- MSID is **cross-dataset-only** and duplicate-filtered against MSLD before scoring; incomplete public metadata prevent proof of patient-level independence.
- v1.1 does **not retrain** the CNN; it reuses the five independently trained base checkpoints.
- A protocol-lock artifact records candidate grids and selection rules.

## 🔎 Manuscript revision audit & traceability

The professor-reviewed ESCI revision adds explicit audit material for grouped fold accounting, calibration/conformal reuse, cluster-aware inference, OOD candidate selection, MSID duplicate filtering, cross-dataset uncertainty, robustness and Grad-CAM variability.

See docs/MANUSCRIPT_REVISION_AUDIT.md and results/revision_audit/.

> [!NOTE]
> The executable research code and locked protocol used by the revised manuscript remain traceable to audited commit **06ca6f6**. This documentation update does **not** alter the trained models, saved predictions, or experimental protocol.

## 📝 Paper

**ViruSafe-MPX: Reliability-Aware Image-Based Screening for Mpox Using Calibrated Uncertainty, Selective Prediction, and Cross-Dataset Evaluation**  
Prepared for **IEEE ESCI 2027**.

See [`paper/README.md`](paper/README.md). 

## 👥 Authors

| Author | Role | ORCID |
|---|---|---|
| **Alok Raj*** | First & corresponding author | [0009-0003-7870-7065](https://orcid.org/0009-0003-7870-7065) |
| **Jasmine Kaur** | Second author | — |
| **Pankaj Vaidya** | Third author, Professor | [0000-0003-1304-630X](https://orcid.org/0000-0003-1304-630X) |
| **Anurag Rana** | Fourth author, Professor | [0000-0003-0247-8908](https://orcid.org/0000-0003-0247-8908) |

**Affiliation:** Yogananda School of AI, Computers and Data Sciences, Shoolini University, Solan, Himachal Pradesh, India.

## 📚 Citation

GitHub will automatically expose the repository citation from [`CITATION.cff`](CITATION.cff). Until the conference paper receives its final bibliographic record, cite the software release as:

```bibtex
@software{raj_virusafe_mpx_2026,
  author  = {Raj, Alok and Kaur, Jasmine and Vaidya, Pankaj and Rana, Anurag},
  title   = {ViruSafe-MPX: Reliability-Aware Image-Based Screening for Mpox Using Calibrated Uncertainty, Selective Prediction, and Cross-Dataset Evaluation},
  year    = {2026},
  version = {1.1.0},
  url     = {https://github.com/Mind-Twister-Wizard/ViruSafe-MPX}
}
```

## ✅ Repository self-check

Before publishing a release, run:

```bash
python scripts/verify_repo.py
python scripts/print_results.py
```

On Windows, after the local environment exists:

```text
.venv\Scripts\python.exe scripts\doctor.py
```

The GitHub Actions workflow in `.github/workflows/ci.yml` also performs lightweight syntax/repository checks without downloading medical datasets.

<details>
<summary><b>What is intentionally not stored in Git?</b></summary>

- Kaggle/DermNet source datasets
- Personal Kaggle credentials or API tokens
- Multi-gigabyte training checkpoints and temporary caches
- Patient/local-machine absolute paths
- Any source image whose redistribution rights are unclear

The one-click runner recreates the local research environment and downloads permitted source datasets from their original providers.
</details>

## ⚖️ License and responsible use

The **software** is released under the [MIT License](LICENSE). Dataset images and third-party visual assets remain subject to their original licenses and rights. This project is intended for **research and screening-method evaluation only** and makes no claim of clinical diagnostic readiness.

---

<p align="center"><b>Reliable AI should know when to predict — and when to defer.</b></p>
