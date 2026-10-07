from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import psutil
import torch
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(root: Path, cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths = {
        "root": root,
        "data": root / cfg["paths"]["data_dir"],
        "outputs": root / cfg["paths"]["output_dir"],
        "state": root / cfg["paths"]["state_dir"],
        "logs": root / cfg["paths"]["logs_dir"],
    }
    subdirs = [
        paths["data"] / "downloads",
        paths["data"] / "msld",
        paths["data"] / "dermnet",
        paths["outputs"] / "checkpoints",
        paths["outputs"] / "predictions",
        paths["outputs"] / "figures",
        paths["outputs"] / "tables",
        paths["outputs"] / "explanations",
        paths["outputs"] / "report",
        paths["outputs"] / "audit",
    ]
    for p in list(paths.values()) + subdirs:
        p.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("virusafe")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(x)}")


@dataclass
class PipelineState:
    path: Path

    def load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {"stages": {}}
        return {"stages": {}}

    def is_done(self, stage: str) -> bool:
        return self.load().get("stages", {}).get(stage, {}).get("status") == "done"

    def mark(self, stage: str, status: str, extra: Dict[str, Any] | None = None) -> None:
        state = self.load()
        state.setdefault("stages", {})[stage] = {
            "status": status,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(extra or {}),
        }
        atomic_json_dump(state, self.path)


def system_info(device: torch.device) -> Dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "cuda_runtime": torch.version.cuda,
                "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            }
        )
    return info


def count_files(root: Path, suffixes: Iterable[str]) -> int:
    suffixes = tuple(s.lower() for s in suffixes)
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def save_tables_excel(tables: Dict[str, pd.DataFrame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for name, df in tables.items():
            safe = name[:31].replace("/", "_").replace("\\", "_")
            df.to_excel(writer, sheet_name=safe, index=False)
            ws = writer.sheets[safe]
            if len(df.columns):
                for i, col in enumerate(df.columns):
                    max_len = max([len(str(col))] + [len(str(v)) for v in df[col].head(500).tolist()])
                    ws.set_column(i, i, min(max(max_len + 2, 10), 40))


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)
