from __future__ import annotations

import json
import math
import os
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.special import softmax
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from .data import (
    IMG_EXTS,
    audit_msld,
    build_fold_loaders,
    build_manifest_fold_loaders,
    create_patient_grouped_manifests,
    ensure_kaggle_dataset,
    make_ood_loader,
    make_transforms,
    sample_dermnet,
)
from .metrics import (
    aurc,
    bootstrap_ci,
    bootstrap_group_ci,
    classification_metrics,
    conformal_metrics,
    conformal_qhat,
    expected_calibration_error,
    fit_temperature,
    multiclass_brier,
    ood_metrics,
    ood_scores,
    probabilities,
    reliability_bins,
    risk_coverage_curve,
    selective_at_coverage,
)
from .modeling import infer_with_features, load_best_model, train_fold
from .plots import (
    plot_ablation,
    plot_architecture_diagram,
    plot_class_distribution,
    plot_conformal_sizes,
    plot_confusion_matrix,
    plot_decision_flow,
    plot_experimental_flow,
    plot_fold_metrics,
    plot_multiclass_pr,
    plot_multiclass_roc,
    plot_ood_roc,
    plot_per_class_f1,
    plot_reliability,
    plot_risk_coverage,
    plot_robustness,
    plot_training_history,
)
from .utils import PipelineState, atomic_json_dump, disk_free_gb, save_tables_excel, system_info


def _npz_save(path: Path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp, path)


def _npz_load(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _stage(state: PipelineState, name: str, logger):
    class Ctx:
        def __enter__(self):
            logger.info("=== Stage: %s ===", name)
            state.mark(name, "running")
            return self
        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                state.mark(name, "done")
                logger.info("=== Completed: %s ===", name)
            else:
                state.mark(name, "failed", {"error": str(exc)})
                logger.error("Stage %s failed: %s", name, exc)
            return False
    return Ctx()


def download_datasets(cfg, paths, state, logger):
    if state.is_done("datasets"):
        logger.info("Datasets stage already complete.")
        return
    with _stage(state, "datasets", logger):
        free = disk_free_gb(paths["root"])
        logger.info("Free disk space: %.2f GB", free)
        if free < 6:
            raise RuntimeError("At least ~6 GB free disk space is required before dataset extraction; 12 GB is recommended.")
        kcfg = cfg["kaggle"]
        ensure_kaggle_dataset(kcfg["msld_slug"], paths["data"] / "downloads", paths["data"] / "msld", logger)
        if bool(kcfg.get("download_dermnet", True)):
            ensure_kaggle_dataset(kcfg["dermnet_slug"], paths["data"] / "downloads", paths["data"] / "dermnet", logger)


def data_audit(cfg, paths, state, logger):
    out_json = paths["outputs"] / "audit" / "msld_audit.json"
    generated_marker = paths["data"] / "generated_patient_folds" / ".generated_ok"
    if state.is_done("data_audit") and out_json.exists() and generated_marker.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    with _stage(state, "data_audit", logger):
        audit = audit_msld(paths["data"] / "msld", cfg["training"]["folds"])
        generated = create_patient_grouped_manifests(
            paths["data"] / "msld",
            paths["data"] / "generated_patient_folds",
            cfg["training"]["folds"],
            int(cfg["project"]["seed"]),
            logger=logger,
        )
        audit["generated_patient_grouped_protocol"] = generated
        atomic_json_dump(audit, out_json)
        pd.DataFrame([{"class": k, "count": v} for k, v in generated.get("class_counts", {}).items()]).to_csv(
            paths["outputs"] / "tables" / "dataset_class_distribution.csv", index=False
        )
        split_rows = []
        for f, info in generated.get("folds", {}).items():
            for split in ("train", "val", "test"):
                split_rows.append({
                    "fold": int(f), "split": split,
                    "images": info.get(f"{split}_images"), "patients": info.get(f"{split}_patients"),
                    **{f"class_{k}": v for k, v in info.get(f"{split}_class_counts", {}).items()},
                })
        pd.DataFrame(split_rows).to_csv(paths["outputs"] / "tables" / "patient_grouped_split_summary.csv", index=False)
        if audit.get("patient_overlap_detected", False):
            logger.warning(
                "The supplied fold layout showed possible patient/image overlap. ViruSafe-MPX will NOT use it; "
                "training will use the automatically generated patient-grouped manifests instead."
            )
        logger.info("MSLD audit and leakage-resistant patient-grouped manifests saved.")
        return audit


def run_folds(cfg, paths, state, device, logger):
    tcfg = cfg["training"]
    model_name = tcfg["model_name"]
    class_names: List[str] | None = None
    class_map_path = paths["outputs"] / "class_mapping.json"
    if class_map_path.exists():
        mapping = json.loads(class_map_path.read_text(encoding="utf-8"))
        class_names = [c for c, _ in sorted(mapping.items(), key=lambda kv: kv[1])]

    for fold in tcfg["folds"]:
        stage = f"fold_{fold}"
        pred_path = paths["outputs"] / "predictions" / f"fold_{fold}.npz"
        best_path = paths["outputs"] / "checkpoints" / f"fold_{fold}" / "best.pt"
        if state.is_done(stage) and pred_path.exists() and best_path.exists():
            logger.info("Fold %d already complete; skipping.", fold)
            continue

        with _stage(state, stage, logger):
            effective_batch = int(tcfg["batch_size"])
            while True:
                train_loader, val_loader, test_loader, mapping, split_dirs = build_manifest_fold_loaders(
                    paths["data"] / "generated_patient_folds",
                    int(fold),
                    int(tcfg["image_size"]),
                    effective_batch,
                    int(tcfg["num_workers"]),
                )
                this_names = [c for c, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
                if class_names is None:
                    class_names = this_names
                    atomic_json_dump(mapping, class_map_path)
                elif this_names != class_names:
                    raise RuntimeError(f"Class mapping mismatch at fold {fold}: {this_names} vs {class_names}")

                logger.info("Fold %d sizes: train=%d val=%d test=%d | batch=%d", fold, len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset), effective_batch)
                try:
                    model, history, best_path = train_fold(
                        int(fold), train_loader, val_loader, model_name, len(class_names), cfg, device,
                        paths["outputs"] / "checkpoints", paths["outputs"] / "tables", logger
                    )
                    break
                except RuntimeError as e:
                    if device.type == "cuda" and "out of memory" in str(e).lower() and effective_batch > 1:
                        new_batch = max(1, effective_batch // 2)
                        logger.warning("CUDA out-of-memory at batch %d. Automatically retrying fold %d with batch %d; checkpoints are preserved.", effective_batch, fold, new_batch)
                        effective_batch = new_batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise
            val = infer_with_features(model, val_loader, device, amp=bool(tcfg.get("use_amp", True)))
            test = infer_with_features(model, test_loader, device, amp=bool(tcfg.get("use_amp", True)))
            temperature = fit_temperature(val["logits"], val["y"]) if cfg["calibration"].get("enabled", True) else 1.0
            cal_val_probs = probabilities(val["logits"], temperature)
            qhat = conformal_qhat(cal_val_probs, val["y"], float(cfg["conformal"]["alpha"]))
            _npz_save(
                pred_path,
                val_logits=val["logits"], val_y=val["y"], val_paths=val["paths"],
                test_logits=test["logits"], test_y=test["y"], test_paths=test["paths"],
                test_features=test["features"], temperature=np.asarray([temperature]), qhat=np.asarray([qhat]),
            )
            logger.info("Fold %d cached. Temperature=%.4f qhat=%.4f", fold, temperature, qhat)

    if class_names is None:
        raise RuntimeError("No fold completed and no class mapping available.")
    return class_names


def aggregate_closed_set(cfg, paths, state, class_names, logger) -> Dict[str, Any]:
    summary_path = paths["outputs"] / "closed_set_summary.json"
    if state.is_done("aggregate_closed_set") and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    with _stage(state, "aggregate_closed_set", logger):
        fold_rows = []
        histories = []
        raw_probs_all, cal_probs_all, y_all, paths_all = [], [], [], []
        conformal_sizes_all, conformal_covered_all = [], []
        fold_cache = {}
        raw_ece_by_fold, cal_ece_by_fold = [], []
        for fold in cfg["training"]["folds"]:
            z = _npz_load(paths["outputs"] / "predictions" / f"fold_{fold}.npz")
            fold_cache[int(fold)] = z
            temp = float(z["temperature"][0])
            qhat = float(z["qhat"][0])
            rawp = probabilities(z["test_logits"], 1.0)
            calp = probabilities(z["test_logits"], temp)
            m, _, _ = classification_metrics(z["test_y"], calp, class_names)
            m.update({"fold": int(fold), "temperature": temp, "conformal_qhat": qhat})
            raw_ece = expected_calibration_error(rawp, z["test_y"])
            cal_ece = expected_calibration_error(calp, z["test_y"])
            m["ece_before_calibration"] = raw_ece
            m["ece_after_calibration"] = cal_ece
            fold_rows.append(m)
            raw_ece_by_fold.append(raw_ece)
            cal_ece_by_fold.append(cal_ece)
            raw_probs_all.append(rawp)
            cal_probs_all.append(calp)
            y_all.append(z["test_y"])
            paths_all.append(z["test_paths"])
            threshold = 1.0 - qhat
            sets = calp >= threshold
            empty = sets.sum(1) == 0
            if empty.any():
                sets[empty, calp[empty].argmax(1)] = True
            conformal_sizes_all.append(sets.sum(1))
            conformal_covered_all.append(sets[np.arange(len(z["test_y"])), z["test_y"]])
            hp = paths["outputs"] / "tables" / f"training_history_fold_{fold}.csv"
            if hp.exists():
                histories.append(pd.read_csv(hp))

        y = np.concatenate(y_all)
        rawp = np.concatenate(raw_probs_all)
        calp = np.concatenate(cal_probs_all)
        test_paths = np.concatenate(paths_all)
        conformal_sizes = np.concatenate(conformal_sizes_all)
        conformal_covered = np.concatenate(conformal_covered_all)
        fold_df = pd.DataFrame(fold_rows).sort_values("fold")
        metrics, per_class, cm = classification_metrics(y, calp, class_names)
        raw_metrics, _, _ = classification_metrics(y, rawp, class_names)
        patient_groups = np.asarray(["_".join(Path(str(p)).stem.split("_")[:2]) for p in test_paths], dtype=object)
        ci_f1 = bootstrap_group_ci(y, calp, patient_groups, "macro_f1", n_boot=1000, seed=int(cfg["project"]["seed"]))
        ci_acc = bootstrap_group_ci(y, calp, patient_groups, "accuracy", n_boot=1000, seed=int(cfg["project"]["seed"]) + 1)
        rc_raw = risk_coverage_curve(rawp, y)
        rc_cal = risk_coverage_curve(calp, y)
        selective = selective_at_coverage(calp, y, cfg["selective_prediction"]["target_coverages"])
        conf_metrics = {
            "coverage": float(conformal_covered.mean()),
            "mean_set_size": float(conformal_sizes.mean()),
            "singleton_rate": float((conformal_sizes == 1).mean()),
            "target_coverage": 1.0 - float(cfg["conformal"]["alpha"]),
        }
        metrics["aurc"] = aurc(rc_cal)
        raw_metrics["aurc"] = aurc(rc_raw)

        # Tables
        tab_dir = paths["outputs"] / "tables"
        fold_df.to_csv(tab_dir / "fold_metrics.csv", index=False)
        per_class.to_csv(tab_dir / "per_class_metrics.csv", index=False)
        selective.to_csv(tab_dir / "selective_prediction.csv", index=False)
        rc_cal.to_csv(tab_dir / "risk_coverage_calibrated.csv", index=False)
        pd.DataFrame([metrics]).to_csv(tab_dir / "pooled_metrics.csv", index=False)
        pd.DataFrame([raw_metrics]).to_csv(tab_dir / "pooled_metrics_uncalibrated.csv", index=False)
        pd.DataFrame([
            {"metric": "Macro-F1", **ci_f1},
            {"metric": "Accuracy", **ci_acc},
        ]).to_csv(tab_dir / "patient_cluster_bootstrap_ci.csv", index=False)
        pd.DataFrame([conf_metrics]).to_csv(tab_dir / "conformal_metrics.csv", index=False)
        pred_df = pd.DataFrame({
            "path": test_paths,
            "true_index": y,
            "true_class": [class_names[i] for i in y],
            "pred_index": calp.argmax(1),
            "pred_class": [class_names[i] for i in calp.argmax(1)],
            "confidence": calp.max(1),
            "conformal_set_size": conformal_sizes,
            "patient_group": patient_groups,
        })
        pred_df.to_csv(tab_dir / "pooled_oof_predictions.csv", index=False)
        if histories:
            history_df = pd.concat(histories, ignore_index=True)
            history_df.to_csv(tab_dir / "all_training_history.csv", index=False)
        else:
            history_df = pd.DataFrame()

        # Figures
        fig_dir = paths["outputs"] / "figures"
        counts = pd.DataFrame({"class": class_names, "count": np.bincount(y, minlength=len(class_names))})
        plot_class_distribution(counts, fig_dir / "01_class_distribution")
        if len(history_df):
            plot_training_history(history_df, fig_dir / "02_training_convergence")
        plot_fold_metrics(fold_df, fig_dir / "03_fold_performance")
        plot_confusion_matrix(cm, class_names, fig_dir / "04_confusion_matrix_normalized", normalize=True)
        plot_multiclass_roc(y, calp, class_names, fig_dir / "05_multiclass_roc")
        plot_multiclass_pr(y, calp, class_names, fig_dir / "06_multiclass_precision_recall")
        plot_reliability(reliability_bins(rawp, y), reliability_bins(calp, y), fig_dir / "07_reliability_diagram")
        plot_risk_coverage({"Uncalibrated": rc_raw, "Calibrated": rc_cal}, fig_dir / "08_risk_coverage")
        plot_conformal_sizes(conformal_sizes, fig_dir / "09_conformal_set_sizes")
        plot_per_class_f1(per_class, fig_dir / "10_per_class_f1")
        plot_architecture_diagram(fig_dir / "11_framework_architecture")
        plot_experimental_flow(fig_dir / "12_experimental_flow")
        plot_decision_flow(fig_dir / "13_decision_logic")

        summary = {
            "pooled_calibrated": metrics,
            "pooled_uncalibrated": raw_metrics,
            "bootstrap_macro_f1": ci_f1,
            "bootstrap_accuracy": ci_acc,
            "conformal": conf_metrics,
            "mean_temperature": float(fold_df["temperature"].mean()),
            "mean_ece_before": float(np.mean(raw_ece_by_fold)),
            "mean_ece_after": float(np.mean(cal_ece_by_fold)),
            "n_oof": int(len(y)),
        }
        atomic_json_dump(summary, summary_path)
        return summary


def run_ood(cfg, paths, state, device, class_names, logger) -> Dict[str, Any]:
    summary_path = paths["outputs"] / "ood_summary.json"
    if state.is_done("ood") and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    with _stage(state, "ood", logger):
        ocfg = cfg["ood"]
        samples = sample_dermnet(
            paths["data"] / "dermnet", int(ocfg["max_images_total"]), int(ocfg["max_images_per_class"]), int(cfg["project"]["seed"])
        )
        manifest = pd.DataFrame({"path": [str(p) for p, _ in samples], "category": [c for _, c in samples]})
        manifest.to_csv(paths["outputs"] / "tables" / "dermnet_ood_manifest.csv", index=False)
        ood_loader = make_ood_loader(samples, int(cfg["training"]["image_size"]), int(ocfg["batch_size"]), int(cfg["training"]["num_workers"]))

        fold_rows = []
        gate_rows = []
        # To make a pooled visualization, temperature-scale each fold before concatenation.
        pooled_id_scaled, pooled_ood_scaled = [], []
        for fold in cfg["training"]["folds"]:
            cache_path = paths["outputs"] / "predictions" / f"ood_fold_{fold}.npz"
            if cache_path.exists():
                oz = _npz_load(cache_path)
            else:
                best_path = paths["outputs"] / "checkpoints" / f"fold_{fold}" / "best.pt"
                model = load_best_model(best_path, cfg["training"]["model_name"], len(class_names), device)
                pred = infer_with_features(model, ood_loader, device, amp=bool(cfg["training"].get("use_amp", True)))
                _npz_save(cache_path, logits=pred["logits"], paths=pred["paths"], categories=pred["extra"])
                oz = _npz_load(cache_path)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            iz = _npz_load(paths["outputs"] / "predictions" / f"fold_{fold}.npz")
            temp = float(iz["temperature"][0])
            df, curves = ood_metrics(iz["test_logits"], oz["logits"], temp)
            df["fold"] = int(fold)
            fold_rows.append(df)

            # Operating point is selected from validation ID only: accept the requested fraction of known cases.
            val_scores = ood_scores(iz["val_logits"], temp)
            test_scores = ood_scores(iz["test_logits"], temp)
            unknown_scores = ood_scores(oz["logits"], temp)
            accept_target = float(ocfg.get("id_acceptance_target", 0.95))
            for score_name in val_scores:
                threshold = float(np.quantile(val_scores[score_name], max(0.0, 1.0 - accept_target)))
                gate_rows.append({
                    "fold": int(fold),
                    "score": score_name,
                    "threshold_from_validation": threshold,
                    "target_id_acceptance": accept_target,
                    "test_id_acceptance": float((test_scores[score_name] >= threshold).mean()),
                    "ood_rejection_rate": float((unknown_scores[score_name] < threshold).mean()),
                })

            pooled_id_scaled.append(iz["test_logits"] / temp)
            pooled_ood_scaled.append(oz["logits"] / temp)

        ood_fold_df = pd.concat(fold_rows, ignore_index=True)
        ood_summary_df = ood_fold_df.groupby("score", as_index=False).agg(
            ood_auroc_mean=("ood_auroc", "mean"),
            ood_auroc_std=("ood_auroc", "std"),
            ood_aupr_mean=("ood_aupr_id_positive", "mean"),
            fpr95_mean=("fpr_at_95_tpr", "mean"),
        )
        ood_fold_df.to_csv(paths["outputs"] / "tables" / "ood_metrics_by_fold.csv", index=False)
        ood_summary_df.to_csv(paths["outputs"] / "tables" / "ood_metrics_summary.csv", index=False)
        gate_df = pd.DataFrame(gate_rows)
        gate_df.to_csv(paths["outputs"] / "tables" / "ood_gate_operating_points.csv", index=False)

        pooled_df, pooled_curves = ood_metrics(np.concatenate(pooled_id_scaled), np.concatenate(pooled_ood_scaled), temperature=1.0)
        pooled_df.to_csv(paths["outputs"] / "tables" / "ood_metrics_pooled.csv", index=False)
        plot_ood_roc(pooled_curves, paths["outputs"] / "figures" / "14_ood_roc")
        best_row = ood_summary_df.sort_values("ood_auroc_mean", ascending=False).iloc[0]
        best_gate = gate_df[gate_df["score"] == str(best_row["score"])]
        summary = {
            "n_ood_images_unique": int(len(samples)),
            "best_score": str(best_row["score"]),
            "best_ood_auroc_mean": float(best_row["ood_auroc_mean"]),
            "best_ood_auroc_std": float(best_row["ood_auroc_std"]) if not pd.isna(best_row["ood_auroc_std"]) else 0.0,
            "fpr95_mean_for_best": float(best_row["fpr95_mean"]),
            "mean_test_id_acceptance_at_validation_threshold": float(best_gate["test_id_acceptance"].mean()) if len(best_gate) else float("nan"),
            "mean_ood_rejection_at_validation_threshold": float(best_gate["ood_rejection_rate"].mean()) if len(best_gate) else float("nan"),
        }
        atomic_json_dump(summary, summary_path)
        return summary


class CorruptPathDataset(Dataset):
    def __init__(self, paths: np.ndarray, labels: np.ndarray, image_size: int, corruption: str, severity: int):
        self.paths = [Path(str(p)) for p in paths]
        self.labels = labels.astype(int)
        self.image_size = image_size
        self.corruption = corruption
        self.severity = severity
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

    def __len__(self): return len(self.paths)

    def _corrupt(self, im: Image.Image) -> Image.Image:
        s = self.severity
        if self.corruption == "brightness":
            factor = [0.75, 0.55, 0.35][s-1]
            return ImageEnhance.Brightness(im).enhance(factor)
        if self.corruption == "blur":
            return im.filter(ImageFilter.GaussianBlur(radius=[1.0, 2.0, 3.5][s-1]))
        if self.corruption == "jpeg":
            import io
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=[55, 35, 20][s-1])
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        if self.corruption == "occlusion":
            # PIL-backed NumPy views may be read-only.  The robustness test
            # writes an occlusion mask in-place, so force a writable copy.
            arr = np.array(im, copy=True)
            h, w = arr.shape[:2]
            frac = [0.12, 0.20, 0.28][s-1]
            oh, ow = int(h*frac), int(w*frac)
            y0, x0 = max(0,(h-oh)//2), max(0,(w-ow)//2)
            fill = np.median(arr.reshape(-1, 3), axis=0).astype(arr.dtype)
            arr[y0:y0+oh, x0:x0+ow] = fill
            return Image.fromarray(arr)
        if self.corruption == "gaussian_noise":
            arr = np.asarray(im).astype(np.float32) / 255.0
            rng = np.random.default_rng(1000 + s)
            arr = np.clip(arr + rng.normal(0, [0.03,0.06,0.10][s-1], arr.shape), 0, 1)
            return Image.fromarray((arr*255).astype(np.uint8))
        return im

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = self._corrupt(im.convert("RGB"))
            x = self.to_tensor(im)
        return x, int(self.labels[idx]), str(p)


def run_robustness(cfg, paths, state, device, class_names, logger):
    if not cfg["robustness"].get("enabled", True):
        return {}
    out_csv = paths["outputs"] / "tables" / "robustness_results.csv"
    if state.is_done("robustness") and out_csv.exists():
        return pd.read_csv(out_csv).to_dict(orient="records")

    with _stage(state, "robustness", logger):
        fold = int(cfg["robustness"]["fold"])
        z = _npz_load(paths["outputs"] / "predictions" / f"fold_{fold}.npz")
        best_path = paths["outputs"] / "checkpoints" / f"fold_{fold}" / "best.pt"
        model = load_best_model(best_path, cfg["training"]["model_name"], len(class_names), device)
        existing = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
        rows = existing.to_dict(orient="records") if len(existing) else []
        done = {(r["corruption"], int(r["severity"])) for r in rows}
        for corruption in cfg["robustness"]["corruptions"]:
            for severity in cfg["robustness"]["severities"]:
                key = (corruption, int(severity))
                if key in done:
                    continue
                ds = CorruptPathDataset(z["test_paths"], z["test_y"], int(cfg["training"]["image_size"]), corruption, int(severity))
                loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
                                    num_workers=int(cfg["training"]["num_workers"]), pin_memory=torch.cuda.is_available())
                pred = infer_with_features(model, loader, device, amp=bool(cfg["training"].get("use_amp", True)))
                probs = probabilities(pred["logits"], float(z["temperature"][0]))
                m, _, _ = classification_metrics(pred["y"], probs, class_names)
                rows.append({"corruption": corruption, "severity": int(severity), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "ece": m["ece"]})
                pd.DataFrame(rows).to_csv(out_csv, index=False)
        df = pd.DataFrame(rows).sort_values(["corruption", "severity"])
        plot_robustness(df, paths["outputs"] / "figures" / "15_robustness")
        return rows


def run_explainability(cfg, paths, state, device, class_names, logger):
    if not cfg["explainability"].get("enabled", True):
        return []
    manifest_path = paths["outputs"] / "tables" / "gradcam_manifest.csv"
    if state.is_done("explainability") and manifest_path.exists():
        return pd.read_csv(manifest_path).to_dict(orient="records")

    with _stage(state, "explainability", logger):
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except Exception as e:
            logger.warning("Grad-CAM package unavailable: %s", e)
            manifest_path.write_text("status,error\nskipped,Grad-CAM import failed\n", encoding="utf-8")
            return []

        fold = int(cfg["explainability"]["fold"])
        z = _npz_load(paths["outputs"] / "predictions" / f"fold_{fold}.npz")
        temp = float(z["temperature"][0])
        probs = probabilities(z["test_logits"], temp)
        pred = probs.argmax(1)
        conf = probs.max(1)
        correct_idx = np.where(pred == z["test_y"])[0]
        error_idx = np.where(pred != z["test_y"])[0]
        n = int(cfg["explainability"].get("examples_per_type", 6))
        high_correct = correct_idx[np.argsort(-conf[correct_idx])[:n]] if len(correct_idx) else np.array([], dtype=int)
        uncertain = np.argsort(conf)[:n]
        errors = error_idx[np.argsort(-conf[error_idx])[:n]] if len(error_idx) else np.array([], dtype=int)
        selections = [(int(i), "high_confidence_correct") for i in high_correct]
        selections += [(int(i), "uncertain") for i in uncertain]
        selections += [(int(i), "error") for i in errors]
        # Deduplicate while keeping first category.
        seen, unique = set(), []
        for x in selections:
            if x[0] not in seen:
                unique.append(x); seen.add(x[0])

        model = load_best_model(paths["outputs"] / "checkpoints" / f"fold_{fold}" / "best.pt", cfg["training"]["model_name"], len(class_names), device)
        target_layer = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        if target_layer is None:
            logger.warning("No Conv2d target layer found; Grad-CAM skipped.")
            manifest_path.write_text("status,error\nskipped,No Conv2d layer found\n", encoding="utf-8")
            return []
        _, eval_tf = make_transforms(int(cfg["training"]["image_size"]))
        cam = GradCAM(model=model, target_layers=[target_layer])
        out_dir = paths["outputs"] / "explanations"
        rows = []
        for rank, (idx, kind) in enumerate(unique, start=1):
            p = Path(str(z["test_paths"][idx]))
            try:
                with Image.open(p) as im:
                    rgb = im.convert("RGB").resize((int(cfg["training"]["image_size"]), int(cfg["training"]["image_size"])))
                    rgb_float = np.asarray(rgb).astype(np.float32) / 255.0
                    tensor = eval_tf(rgb).unsqueeze(0).to(device)
                target_class = int(pred[idx])
                grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_class)])[0]
                overlay = show_cam_on_image(rgb_float, grayscale, use_rgb=True)
                out = out_dir / f"gradcam_{rank:02d}_{kind}_{p.stem}.png"
                Image.fromarray(overlay).save(out)
                rows.append({
                    "kind": kind, "source": str(p), "output": str(out.relative_to(paths["root"])),
                    "true_class": class_names[int(z["test_y"][idx])], "pred_class": class_names[target_class], "confidence": float(conf[idx])
                })
            except Exception as e:
                logger.warning("Grad-CAM failed for %s: %s", p, e)
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return rows


def run_ablation(cfg, paths, state, logger, closed_summary, ood_summary):
    out_csv = paths["outputs"] / "tables" / "ablation_study.csv"
    if state.is_done("ablation") and out_csv.exists():
        return pd.read_csv(out_csv).to_dict(orient="records")
    with _stage(state, "ablation", logger):
        raw = closed_summary["pooled_uncalibrated"]
        cal = closed_summary["pooled_calibrated"]
        sel = pd.read_csv(paths["outputs"] / "tables" / "selective_prediction.csv")
        primary_cov = float(cfg["selective_prediction"]["primary_coverage"])
        sr = sel.iloc[int(np.argmin(np.abs(sel["target_coverage"].values - primary_cov)))]
        conf = closed_summary["conformal"]
        ood_auc = float(ood_summary.get("best_ood_auroc_mean", np.nan))

        def composite(f1, ece, aurc_val, ood=np.nan, selacc=np.nan, confcov=np.nan):
            vals = [float(f1), max(0.0, 1.0-float(ece)), max(0.0, 1.0-min(float(aurc_val), 1.0))]
            if not np.isnan(ood): vals.append(float(ood))
            if not np.isnan(selacc): vals.append(float(selacc))
            if not np.isnan(confcov): vals.append(1.0 - abs(float(confcov) - (1.0-float(cfg["conformal"]["alpha"]))))
            return float(np.mean(vals))

        rows = [
            {
                "variant": "Baseline classifier",
                "macro_f1": raw["macro_f1"], "ece": raw["ece"], "aurc": raw["aurc"],
                "ood_auroc": np.nan, "selective_accuracy_at_90": np.nan, "conformal_coverage": np.nan,
                "composite_reliability_score": composite(raw["macro_f1"], raw["ece"], raw["aurc"]),
            },
            {
                "variant": "+ Temperature calibration",
                "macro_f1": cal["macro_f1"], "ece": cal["ece"], "aurc": cal["aurc"],
                "ood_auroc": np.nan, "selective_accuracy_at_90": np.nan, "conformal_coverage": np.nan,
                "composite_reliability_score": composite(cal["macro_f1"], cal["ece"], cal["aurc"]),
            },
            {
                "variant": "+ Open-set OOD gate",
                "macro_f1": cal["macro_f1"], "ece": cal["ece"], "aurc": cal["aurc"],
                "ood_auroc": ood_auc, "selective_accuracy_at_90": np.nan, "conformal_coverage": np.nan,
                "composite_reliability_score": composite(cal["macro_f1"], cal["ece"], cal["aurc"], ood=ood_auc),
            },
            {
                "variant": "+ Selective abstention",
                "macro_f1": cal["macro_f1"], "ece": cal["ece"], "aurc": cal["aurc"],
                "ood_auroc": ood_auc, "selective_accuracy_at_90": float(sr["selective_accuracy"]), "conformal_coverage": np.nan,
                "composite_reliability_score": composite(cal["macro_f1"], cal["ece"], cal["aurc"], ood=ood_auc, selacc=float(sr["selective_accuracy"])),
            },
            {
                "variant": "Full ViruSafe-MPX (+ conformal)",
                "macro_f1": cal["macro_f1"], "ece": cal["ece"], "aurc": cal["aurc"],
                "ood_auroc": ood_auc, "selective_accuracy_at_90": float(sr["selective_accuracy"]), "conformal_coverage": conf["coverage"],
                "composite_reliability_score": composite(cal["macro_f1"], cal["ece"], cal["aurc"], ood=ood_auc, selacc=float(sr["selective_accuracy"]), confcov=conf["coverage"]),
            },
        ]
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        plot_ablation(df, paths["outputs"] / "figures" / "16_ablation_study")
        return rows


def build_report(cfg, paths, state, device, class_names, closed_summary, ood_summary, logger):
    report_path = paths["outputs"] / "report" / "index.html"
    if state.is_done("report") and report_path.exists():
        return
    with _stage(state, "report", logger):
        tables_dir = paths["outputs"] / "tables"
        figs_dir = paths["outputs"] / "figures"
        # Gather tables for workbook.
        tables = {}
        for csv in sorted(tables_dir.glob("*.csv")):
            try:
                tables[csv.stem] = pd.read_csv(csv)
            except Exception:
                pass
        save_tables_excel(tables, tables_dir / "ViruSafe_MPX_All_Tables.xlsx")

        sysinfo = system_info(device)
        atomic_json_dump(sysinfo, paths["outputs"] / "system_info.json")
        final_summary = {
            "project": cfg["project"]["name"],
            "closed_set": closed_summary,
            "ood": ood_summary,
            "system": sysinfo,
            "classes": class_names,
        }
        atomic_json_dump(final_summary, paths["outputs"] / "results_summary.json")

        m = closed_summary["pooled_calibrated"]
        raw = closed_summary["pooled_uncalibrated"]
        conf = closed_summary["conformal"]
        f1ci = closed_summary["bootstrap_macro_f1"]
        manuscript = "# ViruSafe-MPX — Manuscript-Ready Results\n\n"
        manuscript += f"**Pooled five-fold OOF performance:** accuracy {m['accuracy']:.4f}, balanced accuracy {m['balanced_accuracy']:.4f}, Macro-F1 {m['macro_f1']:.4f} (patient-cluster bootstrap 95% CI {f1ci['ci_low']:.4f}–{f1ci['ci_high']:.4f}), macro AUROC {m.get('macro_auroc_ovr', float('nan')):.4f}.\n\n"
        manuscript += f"**Calibration:** ECE changed from {raw['ece']:.4f} before temperature scaling to {m['ece']:.4f} after calibration; calibrated Brier score {m['brier']:.4f}.\n\n"
        manuscript += f"**Conformal prediction:** empirical coverage {conf['coverage']:.4f} for a target of {conf['target_coverage']:.2f}, mean prediction-set size {conf['mean_set_size']:.3f}, singleton rate {conf['singleton_rate']:.4f}.\n\n"
        manuscript += f"**Open-set/OOD evaluation:** best mean OOD AUROC {ood_summary.get('best_ood_auroc_mean', float('nan')):.4f} using {ood_summary.get('best_score','N/A')}, evaluated on {ood_summary.get('n_ood_images_unique',0)} sampled unseen DermNet images.\n\n"
        manuscript += "The outputs are research results from public datasets and are not evidence of clinical readiness. Prospective multi-centre validation is required before patient-care use.\n"
        (paths["outputs"] / "MANUSCRIPT_READY_RESULTS.md").write_text(manuscript, encoding="utf-8")

        # HTML report uses relative links from outputs/report/ to outputs/figures.
        figure_files = sorted(figs_dir.glob("*.png"))
        figures_html = "\n".join(
            f'<section><h3>{p.stem.replace("_"," ").title()}</h3><img src="../figures/{p.name}" alt="{p.stem}"></section>'
            for p in figure_files
        )
        key_table = pd.DataFrame([m]).round(4).to_html(index=False, border=0, classes="data")
        per_class = tables.get("per_class_metrics", pd.DataFrame()).round(4).to_html(index=False, border=0, classes="data")
        ablation = tables.get("ablation_study", pd.DataFrame()).round(4).to_html(index=False, border=0, classes="data")
        oodtab = tables.get("ood_metrics_summary", pd.DataFrame()).round(4).to_html(index=False, border=0, classes="data")
        explanation_files = sorted((paths["outputs"] / "explanations").glob("*.png"))
        explanations_html = "\n".join(
            f'<figure><img src="../explanations/{p.name}" alt="{p.stem}"><figcaption>{p.stem.replace("_"," ")}</figcaption></figure>'
            for p in explanation_files[:18]
        ) or "<p>No Grad-CAM images were generated.</p>"
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ViruSafe-MPX Research Report</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;line-height:1.5;color:#222}}
h1,h2,h3{{line-height:1.2}} .note{{padding:14px;border:1px solid #999;border-radius:8px}}
img{{max-width:100%;height:auto;border:1px solid #ddd;margin:10px 0 25px}}
table.data{{border-collapse:collapse;width:100%;font-size:13px;overflow:auto;display:block}}
table.data th,table.data td{{border:1px solid #ddd;padding:6px 8px;text-align:right}} table.data th:first-child,table.data td:first-child{{text-align:left}}
code{{background:#f3f3f3;padding:2px 5px}} section{{margin:28px 0}}
</style></head><body>
<h1>ViruSafe-MPX</h1>
<p><strong>An Open-Set, Uncertainty-Calibrated and Explainable AI Framework for Reliable Differential Diagnosis of Mpox and Clinically Similar Viral Exanthems</strong></p>
<div class="note"><strong>Research-use warning:</strong> This is a retrospective public-dataset research pipeline, not a clinically validated diagnostic device.</div>
<h2>Key pooled five-fold results</h2>{key_table}
<p>Macro-F1 patient-cluster bootstrap 95% CI: {f1ci['ci_low']:.4f}–{f1ci['ci_high']:.4f}. Calibration ECE: {raw['ece']:.4f} → {m['ece']:.4f}. Conformal coverage: {conf['coverage']:.4f} (target {conf['target_coverage']:.2f}).</p>
<h2>Per-class metrics</h2>{per_class}
<h2>Open-set/OOD results</h2>{oodtab}
<h2>Ablation study</h2>{ablation}
<h2>Publication-ready figures</h2>{figures_html}
<h2>Explainability examples</h2>{explanations_html}
<h2>Reproducibility</h2><p>System information, JSON summaries, raw CSV tables, Excel workbook, cached predictions, model checkpoints and training histories are saved under <code>outputs/</code>.</p>
</body></html>"""
        report_path.write_text(html, encoding="utf-8")
        logger.info("Final HTML report written to %s", report_path)


def run_pipeline(cfg, paths, state, device, logger):
    download_datasets(cfg, paths, state, logger)
    data_audit(cfg, paths, state, logger)
    class_names = run_folds(cfg, paths, state, device, logger)
    closed = aggregate_closed_set(cfg, paths, state, class_names, logger)
    ood = run_ood(cfg, paths, state, device, class_names, logger)
    run_robustness(cfg, paths, state, device, class_names, logger)
    run_explainability(cfg, paths, state, device, class_names, logger)
    run_ablation(cfg, paths, state, logger, closed, ood)
    build_report(cfg, paths, state, device, class_names, closed, ood, logger)
    return {"closed": closed, "ood": ood}
