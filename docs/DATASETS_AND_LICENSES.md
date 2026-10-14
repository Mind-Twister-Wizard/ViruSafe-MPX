# Datasets and licensing

The repository **does not redistribute dataset image files**. The pipeline downloads them from their original/public hosting locations at execution time.

| Dataset | Purpose | Identifier |
|---|---|---|
| MSLD v2.0 | 6-class development + patient/group-aware OOF testing | Kaggle: joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20 |
| DermNet | OOD method selection + disjoint held-out-category testing | Kaggle: shubhamgoel27/dermnet |
| MSID | Cross-dataset evaluation only | Kaggle: sujaykapadnis/monkeypox-skin-images-dataset-msid |

## MSLD split safeguards

The completed run used 755 original images and 524 recoverable filename-derived patient/groups. Patient/group IDs were kept within a single partition for each fold, and exact byte-identical SHA-256 images were constrained to one partition. Perceptual near-duplicate grouping was **not** applied within MSLD, so visually near-identical variants cannot be completely excluded.

## MSID cross-dataset duplicate audit

Label mapping: Monkeypox/Mpox→Mpox, Chickenpox→Chickenpox, Measles→Measles, Normal/Healthy→Healthy.

From 770 discovered files:
- 0 exact cross-dataset SHA-256 matches were removed;
- 115 near matches to MSLD were removed with a 64-bit DCT perceptual hash and Hamming distance ≤4;
- 2 additional within-MSID exact SHA-256 duplicates were removed;
- 653 images remained.

This filtering reduces obvious image overlap but **does not establish patient-level independence** because public repository metadata are incomplete. Accordingly, the paper describes this as **cross-dataset evaluation under domain shift**, not definitive clinical external validation.

See results/revision_audit/external_duplicate_audit_summary.csv.

Please verify current source licenses and image-reproduction permissions before redistributing photographs or derived overlays.
