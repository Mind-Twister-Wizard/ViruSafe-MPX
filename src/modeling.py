from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def create_model(model_name: str, num_classes: int, pretrained: bool, logger=None) -> nn.Module:
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as e:
        if pretrained:
            if logger:
                logger.warning("Pretrained weight download/model initialization failed (%s). Retrying without pretrained weights.", e)
            return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        raise


def class_weights_from_targets(targets: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(targets), minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts * num_classes, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _save_ckpt_atomic(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device: torch.device, amp: bool = True) -> Dict[str, Any]:
    model.eval()
    logits_all, y_all, paths_all = [], [], []
    for batch in tqdm(loader, desc="Inference", leave=False):
        x, y, paths = batch[:3]
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            logits = model(x)
        logits_all.append(logits.detach().float().cpu().numpy())
        y_all.append(np.asarray(y))
        paths_all.extend(list(paths))
    logits = np.concatenate(logits_all, axis=0) if logits_all else np.empty((0, 0), dtype=np.float32)
    y = np.concatenate(y_all, axis=0) if y_all else np.empty((0,), dtype=int)
    pred = logits.argmax(1) if len(logits) else np.empty((0,), dtype=int)
    return {
        "logits": logits,
        "y": y,
        "paths": np.asarray(paths_all, dtype=object),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(y) else float("nan"),
        "macro_f1": _macro_f1(y, pred) if len(y) else float("nan"),
    }


def train_fold(
    fold: int,
    train_loader,
    val_loader,
    model_name: str,
    num_classes: int,
    cfg: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    history_dir: Path,
    logger,
) -> Tuple[nn.Module, pd.DataFrame, Path]:
    tcfg = cfg["training"]
    fold_dir = checkpoint_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    last_path = fold_dir / "last.pt"
    best_path = fold_dir / "best.pt"
    history_path = history_dir / f"training_history_fold_{fold}.csv"
    complete_path = fold_dir / "training_complete.json"

    model = create_model(model_name, num_classes, bool(tcfg.get("pretrained", True)), logger=logger).to(device)

    if complete_path.exists() and best_path.exists():
        logger.info("Fold %d: training already marked complete; loading best checkpoint.", fold)
        best = torch.load(best_path, map_location=device)
        model.load_state_dict(best["model"])
        history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        return model, history_df, best_path

    weights = class_weights_from_targets(train_loader.dataset.targets, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(tcfg.get("label_smoothing", 0.0)))
    optimizer = AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(tcfg["max_epochs"]), eta_min=float(tcfg["lr"]) * 0.03)
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(tcfg.get("use_amp", True)) and device.type == "cuda"))

    start_epoch, best_metric, bad_epochs = 0, -1.0, 0
    history: List[Dict[str, Any]] = []

    if last_path.exists():
        logger.info("Fold %d: resuming from %s", fold, last_path)
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_metric", -1))
        bad_epochs = int(ckpt.get("bad_epochs", 0))
        history = ckpt.get("history", [])

    max_epochs = int(tcfg["max_epochs"])
    patience = int(tcfg["early_stopping_patience"])
    min_epochs = int(tcfg.get("min_epochs", 1))
    accumulation = max(1, int(tcfg.get("accumulation_steps", 1)))
    clip = float(tcfg.get("grad_clip_norm", 0.0))
    amp = bool(tcfg.get("use_amp", True))

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        y_true, y_pred = [], []
        pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{max_epochs}", leave=False)
        for step, (x, y, _paths) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y) / accumulation
            scaler.scale(loss).backward()
            if ((step + 1) % accumulation == 0) or (step + 1 == len(train_loader)):
                if clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.item()) * accumulation * len(y)
            pred = logits.detach().argmax(1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item()*accumulation:.4f}")

        scheduler.step()
        train_loss /= max(len(train_loader.dataset), 1)
        train_f1 = _macro_f1(y_true, y_pred)
        val = evaluate_loader(model, val_loader, device, amp=amp)
        row = {
            "fold": fold,
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        monitor = float(val["macro_f1"])
        improved = monitor > best_metric + 1e-5
        if improved:
            best_metric = monitor
            bad_epochs = 0
            best_obj = {"model": model.state_dict(), "fold": fold, "epoch": epoch, "best_metric": best_metric, "model_name": model_name, "num_classes": num_classes}
            _save_ckpt_atomic(best_obj, best_path)
        else:
            bad_epochs += 1

        last_obj = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else {},
            "epoch": epoch,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
            "history": history,
            "fold": fold,
            "model_name": model_name,
            "num_classes": num_classes,
        }
        _save_ckpt_atomic(last_obj, last_path)
        logger.info(
            "Fold %d epoch %d: train F1 %.4f | val F1 %.4f | val acc %.4f | best %.4f",
            fold, epoch + 1, train_f1, val["macro_f1"], val["accuracy"], best_metric,
        )
        if (epoch + 1) >= min_epochs and bad_epochs >= patience:
            logger.info("Fold %d: early stopping after %d epochs.", fold, epoch + 1)
            break

    if not best_path.exists():
        raise RuntimeError(f"Fold {fold}: no best checkpoint was produced.")
    complete_path.write_text(
        f'{{"fold": {fold}, "best_metric": {best_metric:.10f}, "status": "complete"}}',
        encoding="utf-8"
    )
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"])
    return model, pd.DataFrame(history), best_path


def load_best_model(best_path: Path, model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    model = create_model(model_name, num_classes, pretrained=False).to(device)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def infer_with_features(model: nn.Module, loader, device: torch.device, amp: bool = True, tta: bool = True) -> Dict[str, Any]:
    model.eval()
    logits_all, feat_all, y_all, paths_all = [], [], [], []
    extra_all = []
    for batch in tqdm(loader, desc="Inference+features", leave=False):
        x, y, paths = batch[:3]
        extra = batch[3] if len(batch) > 3 else None
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            logits = model(x)
            if tta:
                logits_flip = model(torch.flip(x, dims=[3]))
                logits = 0.5 * (logits + logits_flip)
            feat = model.forward_features(x)
            if feat.ndim == 4:
                feat = feat.mean(dim=(-2, -1))
            elif feat.ndim == 3:
                feat = feat.mean(dim=1)
            elif feat.ndim > 2:
                feat = feat.flatten(1)
        logits_all.append(logits.detach().float().cpu().numpy())
        feat_all.append(feat.detach().float().cpu().numpy())
        y_all.append(np.asarray(y))
        paths_all.extend(list(paths))
        if extra is not None:
            extra_all.extend(list(extra))
    return {
        "logits": np.concatenate(logits_all, axis=0),
        "features": np.concatenate(feat_all, axis=0),
        "y": np.concatenate(y_all, axis=0),
        "paths": np.asarray(paths_all, dtype=object),
        "extra": np.asarray(extra_all, dtype=object) if extra_all else np.asarray([], dtype=object),
    }
