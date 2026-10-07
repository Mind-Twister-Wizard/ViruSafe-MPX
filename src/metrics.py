from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(temp: float) -> float:
        p = softmax(logits / max(temp, 1e-4), axis=1)
        return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
    res = minimize_scalar(nll, bounds=(0.25, 10.0), method="bounded", options={"xatol": 1e-4, "maxiter": 200})
    return float(res.x if res.success else 1.0)


def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(logits / max(float(temperature), 1e-6), axis=1)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def multiclass_brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def classification_metrics(y: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(class_names)), zero_division=0
    )
    per_class = pd.DataFrame({
        "class": class_names,
        "precision": precision,
        "recall_sensitivity": recall,
        "f1": f1,
        "support": support,
    })
    cm = confusion_matrix(y, pred, labels=np.arange(len(class_names)))
    # Specificity one-vs-rest
    specs = []
    for i in range(len(class_names)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specs.append(tn / max(tn + fp, 1))
    per_class["specificity"] = specs

    try:
        ybin = label_binarize(y, classes=np.arange(len(class_names)))
        aurocs = []
        auprcs = []
        for i in range(len(class_names)):
            aurocs.append(roc_auc_score(ybin[:, i], probs[:, i]) if len(np.unique(ybin[:, i])) > 1 else np.nan)
            auprcs.append(average_precision_score(ybin[:, i], probs[:, i]) if ybin[:, i].sum() else np.nan)
        per_class["auroc_ovr"] = aurocs
        per_class["auprc_ovr"] = auprcs
    except Exception:
        per_class["auroc_ovr"] = np.nan
        per_class["auprc_ovr"] = np.nan

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_sensitivity": float(np.mean(recall)),
        "macro_specificity": float(np.mean(specs)),
        "ece": expected_calibration_error(probs, y),
        "brier": multiclass_brier(probs, y),
        "n": int(len(y)),
    }
    try:
        metrics["macro_auroc_ovr"] = float(roc_auc_score(label_binarize(y, classes=np.arange(len(class_names))), probs, average="macro", multi_class="ovr"))
    except Exception:
        metrics["macro_auroc_ovr"] = float("nan")
    return metrics, per_class, cm


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    rows = []
    bins = np.linspace(0, 1, n_bins + 1)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if mask.any():
            rows.append({"bin_low": lo, "bin_high": hi, "confidence": conf[mask].mean(), "accuracy": correct[mask].mean(), "count": int(mask.sum())})
    return pd.DataFrame(rows)


def risk_coverage_curve(probs: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    order = np.argsort(-conf)
    correct = correct[order]
    conf = conf[order]
    cum_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    coverage = np.arange(1, len(correct) + 1) / len(correct)
    risk = 1.0 - cum_acc
    return pd.DataFrame({"coverage": coverage, "risk": risk, "accuracy": cum_acc, "threshold_confidence": conf})


def aurc(rc: pd.DataFrame) -> float:
    return float(np.trapezoid(rc["risk"].values, rc["coverage"].values))


def selective_at_coverage(probs: np.ndarray, y: np.ndarray, coverages: Iterable[float]) -> pd.DataFrame:
    rc = risk_coverage_curve(probs, y)
    rows = []
    for target in coverages:
        idx = int(np.argmin(np.abs(rc["coverage"].values - float(target))))
        row = rc.iloc[idx]
        rows.append({
            "target_coverage": float(target),
            "actual_coverage": float(row["coverage"]),
            "selective_accuracy": float(row["accuracy"]),
            "selective_risk": float(row["risk"]),
            "confidence_threshold": float(row["threshold_confidence"]),
        })
    return pd.DataFrame(rows)


def conformal_qhat(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    scores = 1.0 - cal_probs[np.arange(len(cal_y)), cal_y]
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    try:
        return float(np.quantile(scores, q_level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, q_level, interpolation="higher"))


def conformal_metrics(probs: np.ndarray, y: np.ndarray, qhat: float) -> Tuple[Dict[str, float], np.ndarray]:
    threshold = 1.0 - qhat
    sets = probs >= threshold
    # Guarantee nonempty set for numerical corner cases.
    empty = sets.sum(1) == 0
    if empty.any():
        sets[empty, probs[empty].argmax(1)] = True
    covered = sets[np.arange(len(y)), y]
    sizes = sets.sum(1)
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "qhat": float(qhat),
    }, sizes


def ood_scores(logits: np.ndarray, temperature: float = 1.0) -> Dict[str, np.ndarray]:
    scaled = logits / max(float(temperature), 1e-6)
    probs = softmax(scaled, axis=1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)
    return {
        "MSP": probs.max(1),
        "NegEntropy": -entropy,
        "EnergyID": logsumexp(scaled, axis=1),
    }


def _fpr_at_tpr(y_binary: np.ndarray, score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_binary, score)
    idx = np.where(tpr >= target_tpr)[0]
    return float(fpr[idx[0]]) if len(idx) else 1.0


def ood_metrics(id_logits: np.ndarray, ood_logits: np.ndarray, temperature: float = 1.0) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    ids = ood_scores(id_logits, temperature)
    ods = ood_scores(ood_logits, temperature)
    rows = []
    curves = {}
    for name in ids:
        score = np.concatenate([ids[name], ods[name]])
        labels = np.concatenate([np.ones(len(ids[name])), np.zeros(len(ods[name]))])
        auroc = roc_auc_score(labels, score)
        aupr = average_precision_score(labels, score)
        fpr95 = _fpr_at_tpr(labels, score, 0.95)
        fpr, tpr, _ = roc_curve(labels, score)
        rows.append({"score": name, "ood_auroc": auroc, "ood_aupr_id_positive": aupr, "fpr_at_95_tpr": fpr95})
        curves[name] = (fpr, tpr)
    return pd.DataFrame(rows), curves


def bootstrap_ci(y: np.ndarray, probs: np.ndarray, metric: str = "macro_f1", n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    pred = probs.argmax(1)
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if metric == "macro_f1":
            vals.append(f1_score(y[idx], pred[idx], average="macro", zero_division=0))
        elif metric == "accuracy":
            vals.append(accuracy_score(y[idx], pred[idx]))
    vals = np.asarray(vals)
    return {"mean": float(vals.mean()), "ci_low": float(np.quantile(vals, 0.025)), "ci_high": float(np.quantile(vals, 0.975))}


def bootstrap_group_ci(y: np.ndarray, probs: np.ndarray, groups: np.ndarray, metric: str = "macro_f1", n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    """Cluster bootstrap by patient/group to respect within-patient image dependence."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique}
    pred = probs.argmax(1)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        if metric == "macro_f1":
            vals.append(f1_score(y[idx], pred[idx], average="macro", zero_division=0))
        elif metric == "accuracy":
            vals.append(accuracy_score(y[idx], pred[idx]))
    vals = np.asarray(vals, dtype=float)
    return {"mean": float(vals.mean()), "ci_low": float(np.quantile(vals, 0.025)), "ci_high": float(np.quantile(vals, 0.975)), "bootstrap_unit": "patient"}
