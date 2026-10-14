# Manuscript revision audit

This document records the methodological clarifications added after faculty review of the IEEE ESCI 2027 manuscript.

## Revised paper framing

**Current title:**  
**ViruSafe-MPX: Reliability-Aware Image-Based Screening for Mpox Using Calibrated Uncertainty, Selective Prediction, and Cross-Dataset Evaluation**

The manuscript now frames ViruSafe-MPX as a **retrospective research screening and referral framework**. It does not claim clinical diagnostic validity, autonomous diagnosis, or reliable autonomous rejection of unfamiliar dermatological cases.

The contribution is the **integration and systematic evaluation** of established components rather than the invention of a new classifier or uncertainty algorithm.

## Review concern → repository evidence

| Review concern | Clarification / artifact |
|---|---|
| Group/fold accounting | fold_split_audit.csv records train/validation/test images, patient/groups and class counts for all five folds. |
| Validation reuse | temperature_nll_audit.csv documents that outer validation was reused for fusion selection, temperature fitting and conformal quantiles; conformal coverage is therefore labeled empirical. |
| Statistical sampling unit | cluster_bootstrap_primary_metrics.csv and paired_cluster_inference.csv use patient/group clusters; a 200,000-replicate sign-flip check supplements the image-level McNemar result. |
| External-data mapping | MSID mapping is Monkeypox/Mpox→Mpox, Chickenpox→Chickenpox, Measles→Measles, Normal/Healthy→Healthy. |
| Duplicate audit | external_duplicate_audit_summary.csv: 770 discovered, 0 cross-dataset exact SHA matches, 115 pHash near matches removed, 2 within-MSID exact duplicates removed, 653 retained; pHash threshold ≤4 bits. |
| External per-class uncertainty | external_per_class_with_image_bootstrap_ci.csv provides class support and image-bootstrap intervals; patient-level CIs are not claimed because MSID patient IDs are unavailable. |
| Calibration definition | pooled_reliability_metrics_audit.csv and temperature_nll_audit.csv record NLL, Brier and 15-bin ECE. |
| Selective prediction | selective_prediction_full_audit.csv records target/realized coverage, referral fraction, accuracy, Macro-F1, retained Mpox recall and thresholds. |
| Conformal details | conformal_calibration_sample_sizes.csv, conformal_coverage_audit.csv and conformal_set_size_distribution.csv record per-class sample sizes, coverage, set sizes and fallback behavior. |
| OOD selection/test separation | ood_candidate_method_selection.csv gives validation candidates; ood_all_candidates_heldout_test.csv and the category summary report held-out behavior. |
| Component contribution | component_comparison.csv shows that closed-set gains enter with the frozen-feature head/fusion; conformal/OOD layers add uncertainty outputs rather than argmax accuracy. |
| Robustness detail | robustness_all_levels.csv reports every corruption family/severity for the saved fold-1 backbone. |
| Explainability variability | gradcam_faithfulness_variability.csv reports mean, SD and bootstrap intervals; the random control used one draw per image and is described as exploratory. |

## Important interpretation limits

1. MSLD is small, web-sourced and only partially supported by recoverable patient metadata.
2. Within-MSLD perceptual near-duplicate grouping was not performed; exact SHA/group separation was enforced.
3. The same outer validation observations had multiple post-training roles in v1.1; empirical conformal coverage is therefore not a formal coverage guarantee.
4. MSID duplicate filtering reduces obvious overlap but cannot prove patient-level independence.
5. OOD FPR@95TPR remains high (**75.23%**), so OOD is a warning signal rather than an autonomous gate.
6. Grad-CAM deletion analysis is exploratory; intervals overlap and the random masking control used one draw per image.
7. Clinical use would require prospective multicentre data, prespecified operating thresholds, subgroup analysis, image-quality screening and clinician-led confirmatory testing.

## Code freeze

The revised manuscript identifies **commit 06ca6f6** as the audited executable code state. This documentation update does not change the model code or frozen experimental outputs.
