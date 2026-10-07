from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(counts: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts["class"], counts["count"])
    ax.set_ylabel("Images")
    ax.set_title("MSLD v2.0 Class Distribution")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, out)


def plot_training_history(history: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for fold, g in history.groupby("fold"):
        ax.plot(g["epoch"], g["val_macro_f1"], marker="o", markersize=2, linewidth=1.3, label=f"Fold {fold}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cross-Validation Training Convergence")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, out)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out: Path, normalize: bool = True):
    arr = cm.astype(float)
    if normalize:
        arr = arr / np.maximum(arr.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Pooled Out-of-Fold Confusion Matrix" + (" (Row Normalized)" if normalize else ""))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt = f"{arr[i,j]:.2f}" if normalize else str(int(arr[i,j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    _save(fig, out)


def plot_reliability(raw_bins: pd.DataFrame, cal_bins: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    if len(raw_bins):
        ax.plot(raw_bins["confidence"], raw_bins["accuracy"], marker="o", label="Before calibration")
    if len(cal_bins):
        ax.plot(cal_bins["confidence"], cal_bins["accuracy"], marker="s", label="Temperature scaled")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability Diagram")
    ax.legend()
    _save(fig, out)


def plot_risk_coverage(curves: Dict[str, pd.DataFrame], out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, df in curves.items():
        ax.plot(df["coverage"], df["risk"], label=name)
    ax.set_xlabel("Coverage (fraction automatically accepted)")
    ax.set_ylabel("Selective risk (1 - accuracy)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title("Risk-Coverage Trade-off")
    ax.legend()
    _save(fig, out)


def plot_ood_roc(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (ID accepted)")
    ax.set_title("Open-Set / OOD Detection ROC")
    ax.legend(fontsize=8)
    _save(fig, out)


def plot_fold_metrics(fold_df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fold_df))
    w = 0.25
    ax.bar(x - w, fold_df["accuracy"], width=w, label="Accuracy")
    ax.bar(x, fold_df["balanced_accuracy"], width=w, label="Balanced accuracy")
    ax.bar(x + w, fold_df["macro_f1"], width=w, label="Macro-F1")
    ax.set_xticks(x, [f"Fold {int(v)}" for v in fold_df["fold"]])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Five-Fold Patient-Split Performance")
    ax.legend()
    _save(fig, out)


def plot_per_class_f1(per_class: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(per_class["class"], per_class["f1"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-Class Differential Diagnosis Performance")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, out)


def plot_multiclass_roc(y: np.ndarray, probs: np.ndarray, class_names: List[str], out: Path):
    yb = label_binarize(y, classes=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(class_names):
        if len(np.unique(yb[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(fontsize=8)
    _save(fig, out)


def plot_multiclass_pr(y: np.ndarray, probs: np.ndarray, class_names: List[str], out: Path):
    yb = label_binarize(y, classes=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(class_names):
        if yb[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(yb[:, i], probs[:, i])
        ax.plot(recall, precision, label=c)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("One-vs-Rest Precision-Recall Curves")
    ax.legend(fontsize=8)
    _save(fig, out)


def plot_conformal_sizes(sizes: np.ndarray, out: Path):
    values, counts = np.unique(sizes, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(values.astype(str), counts)
    ax.set_xlabel("Prediction-set size")
    ax.set_ylabel("Test images")
    ax.set_title("Conformal Prediction-Set Sizes")
    _save(fig, out)


def plot_robustness(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for corruption, g in df.groupby("corruption"):
        ax.plot(g["severity"], g["macro_f1"], marker="o", label=corruption)
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Robustness Under Image Degradation")
    ax.legend(fontsize=8)
    _save(fig, out)


def plot_ablation(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    metric = "composite_reliability_score" if "composite_reliability_score" in df else "macro_f1"
    ax.bar(df["variant"], df[metric])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Ablation Study")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, out)


def plot_architecture_diagram(out: Path):
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    labels = [
        "Skin-lesion\nimage",
        "Quality +\nnormalization",
        "EfficientNetV2-S\nvisual encoder",
        "6-class\ndifferential logits",
        "Temperature\ncalibration",
        "OOD + uncertainty\ngating",
        "Conformal\nprediction set",
        "DIAGNOSE /\nREFER",
    ]
    xs = np.linspace(0.05, 0.95, len(labels))
    for i, (x, lab) in enumerate(zip(xs, labels)):
        ax.text(x, 0.52, lab, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.55", fc="none"), fontsize=9)
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i+1]-0.055, 0.52), xytext=(x+0.055, 0.52),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.set_title("ViruSafe-MPX: Reliability-Aware Diagnostic Decision Pipeline", fontsize=13)
    _save(fig, out)


def plot_experimental_flow(out: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")
    steps = [
        (0.5, 0.91, "MSLD v2.0: patient-level five-fold splits"),
        (0.5, 0.77, "Train EfficientNetV2-S on each fold\n+ checkpoint every epoch"),
        (0.5, 0.63, "Validation logits → temperature scaling\n+ conformal calibration"),
        (0.5, 0.49, "Held-out fold test → closed-set metrics\n+ risk-coverage analysis"),
        (0.5, 0.35, "DermNet unseen diseases → OOD evaluation"),
        (0.5, 0.21, "Robustness + Grad-CAM + ablation"),
        (0.5, 0.07, "Pooled manuscript tables, figures and HTML report"),
    ]
    for i, (x, y, txt) in enumerate(steps):
        ax.text(x, y, txt, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.65", fc="none"), fontsize=10)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, steps[i+1][1] + 0.05), xytext=(0.5, y - 0.05),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.set_title("Experimental Design and Leakage-Aware Evaluation Flow", fontsize=13)
    _save(fig, out)


def plot_decision_flow(out: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")
    nodes = {
        "Prediction": (0.15, 0.75),
        "OOD?": (0.42, 0.75),
        "High uncertainty?": (0.68, 0.75),
        "Conformal set\nsize > 1?": (0.68, 0.42),
        "REFER": (0.90, 0.58),
        "DIAGNOSE": (0.68, 0.13),
    }
    for name, (x, y) in nodes.items():
        ax.text(x, y, name, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.6", fc="none"), fontsize=10)
    def arrow(a, b, label=""):
        x1,y1=nodes[a]; x2,y2=nodes[b]
        ax.annotate(label, xy=(x2,y2), xytext=(x1,y1), xycoords=ax.transAxes, textcoords=ax.transAxes,
                    ha="center", va="center", arrowprops=dict(arrowstyle="->", lw=1.3))
    arrow("Prediction", "OOD?")
    arrow("OOD?", "REFER", "Yes")
    arrow("OOD?", "High uncertainty?", "No")
    arrow("High uncertainty?", "REFER", "Yes")
    arrow("High uncertainty?", "Conformal set\nsize > 1?", "No")
    arrow("Conformal set\nsize > 1?", "REFER", "Yes")
    arrow("Conformal set\nsize > 1?", "DIAGNOSE", "No")
    ax.set_title("Risk-Adaptive Decision Logic", fontsize=13)
    _save(fig, out)
