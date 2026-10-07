from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            return bad is None
    except Exception:
        return False


def _download_with_resume(url: str, dest: Path, logger) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    existing = dest.stat().st_size if dest.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        logger.info("Resuming download %s from %.2f MB", dest.name, existing / 1024**2)
    else:
        logger.info("Downloading %s", dest.name)

    with requests.get(url, stream=True, timeout=(30, 300), headers=headers, allow_redirects=True) as r:
        if existing and r.status_code == 200:
            logger.info("Server ignored Range request; restarting %s", dest.name)
            mode = "wb"
            existing = 0
        r.raise_for_status()
        total = r.headers.get("content-length")
        total = int(total) + existing if total and mode == "ab" else (int(total) if total else None)
        downloaded = existing
        with open(dest, mode) as f:
            last_pct = -1
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct // 5 != last_pct // 5:
                        logger.info("%s: %d%% (%.2f/%.2f GB)", dest.name, pct, downloaded/1024**3, total/1024**3)
                        last_pct = pct


def _download_via_kaggle(slug: str, out_dir: Path, final_zip: Path, logger) -> None:
    logger.info("Trying authenticated Kaggle API for %s", slug)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(out_dir), quiet=False, unzip=False)
        expected = out_dir / f"{slug.split('/')[-1]}.zip"
        dataset_name = slug.split('/')[-1].lower()
        candidates = [expected] + sorted(
            [p for p in out_dir.glob("*.zip") if dataset_name in p.stem.lower()],
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        src = next((p for p in candidates if p.exists() and _is_valid_zip(p)), None)
        if src is None:
            raise RuntimeError("Kaggle API returned no valid ZIP file.")
        if src.resolve() != final_zip.resolve():
            shutil.move(str(src), str(final_zip))
    except Exception as e:
        raise RuntimeError(
            f"Automatic Kaggle download failed for {slug}. If Kaggle blocks anonymous downloads, "
            "place kaggle.json in %USERPROFILE%\\.kaggle\\kaggle.json and rerun the same BAT. "
            f"Original error: {e}"
        ) from e


def ensure_kaggle_dataset(slug: str, download_dir: Path, extract_dir: Path, logger) -> Path:
    dataset_name = slug.split("/")[-1]
    zip_path = download_dir / f"{dataset_name}.zip"
    marker = extract_dir / ".extracted_ok"

    if marker.exists() and any(extract_dir.iterdir()):
        logger.info("Dataset already extracted: %s", slug)
        return extract_dir

    if not _is_valid_zip(zip_path):
        url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
        try:
            _download_with_resume(url, zip_path, logger)
        except Exception as e:
            logger.warning("Public Kaggle endpoint failed: %s", e)
        if not _is_valid_zip(zip_path):
            if zip_path.exists():
                bad = zip_path.with_suffix(".invalid")
                try:
                    zip_path.replace(bad)
                except Exception:
                    zip_path.unlink(missing_ok=True)
            _download_via_kaggle(slug, download_dir, zip_path, logger)

    if not _is_valid_zip(zip_path):
        raise RuntimeError(f"Downloaded ZIP is invalid: {zip_path}")

    logger.info("Extracting %s...", zip_path.name)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    return extract_dir


def _lower_map(root: Path) -> Dict[str, List[Path]]:
    m = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_dir():
            m[p.name.lower()].append(p)
    return m


def find_msld_fold_dirs(msld_root: Path, fold: int) -> Dict[str, Path]:
    # Search robustly for .../Original Images/FOLDS/foldX/{train,val/test}
    candidates = [p for p in msld_root.rglob(f"fold{fold}") if p.is_dir()]
    candidates += [p for p in msld_root.rglob(f"Fold{fold}") if p.is_dir()]
    seen = set()
    for fold_dir in candidates:
        if fold_dir.resolve() in seen:
            continue
        seen.add(fold_dir.resolve())
        names = {c.name.lower(): c for c in fold_dir.iterdir() if c.is_dir()}
        train = names.get("train") or names.get("training")
        val = names.get("valid") or names.get("validation") or names.get("val")
        test = names.get("test") or names.get("testing")
        if train and val and test:
            return {"train": train, "val": val, "test": test}
    raise FileNotFoundError(
        f"Could not locate train/validation/test directories for fold {fold} under {msld_root}."
    )


def canonical_class_name(name: str) -> str:
    n = name.lower().replace("_", " ").replace("-", " ").strip()
    aliases = {
        "monkeypox": "Mpox",
        "monkey pox": "Mpox",
        "mpox": "Mpox",
        "chickenpox": "Chickenpox",
        "chicken pox": "Chickenpox",
        "measles": "Measles",
        "cowpox": "Cowpox",
        "cow pox": "Cowpox",
        "hand foot mouth disease": "HFMD",
        "hand foot and mouth disease": "HFMD",
        "hand,foot and mouth disease": "HFMD",
        "hfmd": "HFMD",
        "healthy": "Healthy",
    }
    return aliases.get(n, name.strip())


class CanonicalImageFolder(datasets.ImageFolder):
    def __init__(self, root: str | Path, transform=None, class_to_idx: Dict[str, int] | None = None):
        super().__init__(str(root), transform=transform)
        original_classes = self.classes[:]
        canon = [canonical_class_name(x) for x in original_classes]
        if class_to_idx is None:
            unique = sorted(set(canon))
            # Stable clinical order if available.
            preferred = ["Mpox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
            unique = [x for x in preferred if x in unique] + [x for x in unique if x not in preferred]
            class_to_idx = {c: i for i, c in enumerate(unique)}
        old_idx_to_new = {old_idx: class_to_idx[canonical_class_name(old_name)] for old_idx, old_name in enumerate(original_classes)}
        self.samples = [(p, old_idx_to_new[y]) for p, y in self.samples]
        self.imgs = self.samples
        self.targets = [y for _, y in self.samples]
        self.class_to_idx = class_to_idx
        self.classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path = self.samples[index][0]
        return image, target, path


def make_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03),
        transforms.RandomPerspective(distortion_scale=0.10, p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, eval_tf


def build_fold_loaders(msld_root: Path, fold: int, image_size: int, batch_size: int, num_workers: int):
    dirs = find_msld_fold_dirs(msld_root, fold)
    train_tf, eval_tf = make_transforms(image_size)
    tmp = CanonicalImageFolder(dirs["train"], transform=train_tf)
    mapping = tmp.class_to_idx
    train_ds = tmp
    val_ds = CanonicalImageFolder(dirs["val"], transform=eval_tf, class_to_idx=mapping)
    test_ds = CanonicalImageFolder(dirs["test"], transform=eval_tf, class_to_idx=mapping)
    pin = torch.cuda.is_available()
    loader_args = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_args)
    return train_loader, val_loader, test_loader, mapping, dirs


def build_eval_dataset(path: Path, image_size: int, class_to_idx: Dict[str, int]):
    _, eval_tf = make_transforms(image_size)
    return CanonicalImageFolder(path, transform=eval_tf, class_to_idx=class_to_idx)


def _patient_key_from_name(path: str | Path) -> str | None:
    parts = Path(path).stem.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}"
    return None


def audit_msld(msld_root: Path, folds: List[int]) -> Dict[str, Any]:
    audit: Dict[str, Any] = {"folds": {}, "patient_overlap_detected": False}
    for f in folds:
        dirs = find_msld_fold_dirs(msld_root, f)
        fa: Dict[str, Any] = {}
        patient_sets: Dict[str, set] = {}
        basename_sets: Dict[str, set] = {}
        for split, p in dirs.items():
            ds = CanonicalImageFolder(p, transform=None)
            counts = Counter(ds.classes[y] for y in ds.targets)
            basenames = [Path(x).name for x, _ in ds.samples]
            patients = {k for k in (_patient_key_from_name(x) for x, _ in ds.samples) if k is not None}
            patient_sets[split] = patients
            basename_sets[split] = set(basenames)
            fa[split] = {
                "n": len(ds),
                "class_counts": dict(counts),
                "unique_basenames": len(set(basenames)),
                "parsed_unique_patients": len(patients),
            }
        overlaps = {}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            p_overlap = sorted(patient_sets[a] & patient_sets[b])
            i_overlap = sorted(basename_sets[a] & basename_sets[b])
            overlaps[f"{a}_vs_{b}"] = {
                "patient_overlap_count": len(p_overlap),
                "image_basename_overlap_count": len(i_overlap),
                "patient_overlap_examples": p_overlap[:10],
                "image_overlap_examples": i_overlap[:10],
            }
            if p_overlap or i_overlap:
                audit["patient_overlap_detected"] = True
        fa["leakage_check"] = overlaps
        audit["folds"][str(f)] = fa
    return audit



class OODDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, str]], image_size: int):
        self.samples = samples
        _, self.transform = make_transforms(image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, category = self.samples[idx]
        with Image.open(path) as im:
            img = self.transform(im.convert("RGB"))
        return img, -1, str(path), category


def sample_dermnet(dermnet_root: Path, max_total: int, max_per_class: int, seed: int) -> List[Tuple[Path, str]]:
    # Exclude DermNet categories that may contain viral exanthems overlapping the six known MSLD classes.
    # OOD evaluation should use clearly unseen dermatological conditions rather than hidden in-distribution cases.
    exclude_terms = ("viral", "wart", "mollusc", "chicken", "measles", "pox", "hand foot", "mouth disease")
    files_by_class: Dict[str, List[Path]] = defaultdict(list)
    for p in dermnet_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            category = p.parent.name
            if any(term in category.lower() for term in exclude_terms):
                continue
            files_by_class[category].append(p)
    if not files_by_class:
        raise FileNotFoundError(f"No eligible DermNet OOD images found under {dermnet_root}")
    rng = random.Random(seed)
    selected: List[Tuple[Path, str]] = []
    for cls in sorted(files_by_class):
        arr = files_by_class[cls][:]
        rng.shuffle(arr)
        for p in arr[:max_per_class]:
            selected.append((p, cls))
    rng.shuffle(selected)
    return selected[:max_total]


def make_ood_loader(samples: List[Tuple[Path, str]], image_size: int, batch_size: int, num_workers: int):
    ds = OODDataset(samples, image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0))



def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _duplicate_aware_group_ids(df):
    """Keep same-patient images and byte-identical duplicates in one split group."""
    n = len(df)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    first_by_patient = {}
    for i, patient in enumerate(df["patient_id"].astype(str)):
        if patient in first_by_patient:
            union(i, first_by_patient[patient])
        else:
            first_by_patient[patient] = i

    first_by_hash = {}
    exact_duplicate_links = 0
    for i, path in enumerate(df["path"].astype(str)):
        try:
            digest = _sha256_file(Path(path))
        except Exception:
            continue
        if digest in first_by_hash:
            if union(i, first_by_hash[digest]):
                exact_duplicate_links += 1
        else:
            first_by_hash[digest] = i

    roots = [find(i) for i in range(n)]
    unique_roots = {r: k for k, r in enumerate(sorted(set(roots)))}
    group_ids = [f"G{unique_roots[r]:04d}" for r in roots]
    return group_ids, exact_duplicate_links, len(unique_roots)


def create_patient_grouped_manifests(msld_root: Path, out_root: Path, folds: List[int], seed: int, logger=None) -> Dict[str, Any]:
    """Create leakage-resistant patient-grouped folds from the unique original images.

    The official fold-1 train/validation/test union is used only as an inventory of the
    755 original images. No augmented image is included. Images are then re-partitioned
    with StratifiedGroupKFold using filename-derived patient identifiers.
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    marker = out_root / ".generated_ok"
    audit_path = out_root / "generated_fold_audit.json"
    if marker.exists() and audit_path.exists():
        return json.loads(audit_path.read_text(encoding="utf-8"))

    rows = {}
    # Union all official original-image fold directories, then deduplicate by basename.
    # This is more robust than assuming any one supplied fold contains the complete inventory.
    for source_fold in folds:
        dirs = find_msld_fold_dirs(msld_root, int(source_fold))
        for split_root in dirs.values():
            for p in split_root.rglob("*"):
                if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
                    continue
                cls = canonical_class_name(p.parent.name)
                patient = _patient_key_from_name(p)
                if patient is None:
                    patient = p.stem
                key = p.name.lower()
                current = {"path": str(p.resolve()), "class": cls, "patient_id": patient, "basename": p.name}
                if key in rows and rows[key]["class"] != cls:
                    raise RuntimeError(f"Conflicting class labels for image basename {p.name}")
                rows.setdefault(key, current)

    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        raise RuntimeError("Could not build a unique MSLD original-image inventory.")
    preferred = ["Mpox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
    present = list(df["class"].unique())
    class_names = [c for c in preferred if c in present] + sorted(c for c in present if c not in preferred)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    df["label_index"] = df["class"].map(class_to_idx).astype(int)
    df = df.sort_values(["class", "patient_id", "basename"]).reset_index(drop=True)
    group_ids, exact_duplicate_links, effective_group_count = _duplicate_aware_group_ids(df)
    df["group_id"] = group_ids

    n_splits = len(folds)
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = np.zeros((len(df), 1), dtype=np.float32)
    y = df["label_index"].to_numpy()
    groups = df["group_id"].astype(str).to_numpy()

    out_root.mkdir(parents=True, exist_ok=True)
    audit: Dict[str, Any] = {
        "inventory_images": int(len(df)),
        "inventory_patients": int(df["patient_id"].nunique()),
        "effective_duplicate_aware_groups": int(effective_group_count),
        "exact_duplicate_links_grouped": int(exact_duplicate_links),
        "class_counts": df["class"].value_counts().to_dict(),
        "class_to_idx": class_to_idx,
        "folds": {},
    }

    for fold_number, (dev_idx, test_idx) in zip(folds, outer.split(X, y, groups)):
        dev = df.iloc[dev_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        inner_n = min(5, max(2, int(dev["patient_id"].nunique() // 10)))
        inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True, random_state=seed + int(fold_number))
        Xd = np.zeros((len(dev), 1), dtype=np.float32)
        yd = dev["label_index"].to_numpy()
        gd = dev["group_id"].astype(str).to_numpy()
        train_local, val_local = next(inner.split(Xd, yd, gd))
        train = dev.iloc[train_local].reset_index(drop=True)
        val = dev.iloc[val_local].reset_index(drop=True)

        fold_dir = out_root / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        cols = ["path", "class", "label_index", "patient_id", "group_id", "basename"]
        train[cols].to_csv(fold_dir / "train.csv", index=False)
        val[cols].to_csv(fold_dir / "val.csv", index=False)
        test[cols].to_csv(fold_dir / "test.csv", index=False)

        ps = {"train": set(train["patient_id"]), "val": set(val["patient_id"]), "test": set(test["patient_id"])}
        gs = {"train": set(train["group_id"]), "val": set(val["group_id"]), "test": set(test["group_id"])}
        overlap = {
            "train_val_patients": len(ps["train"] & ps["val"]),
            "train_test_patients": len(ps["train"] & ps["test"]),
            "val_test_patients": len(ps["val"] & ps["test"]),
            "train_val_duplicate_groups": len(gs["train"] & gs["val"]),
            "train_test_duplicate_groups": len(gs["train"] & gs["test"]),
            "val_test_duplicate_groups": len(gs["val"] & gs["test"]),
        }
        if any(overlap.values()):
            raise RuntimeError(f"Generated duplicate-aware patient-group fold {fold_number} contains leakage: {overlap}")
        audit["folds"][str(fold_number)] = {
            "train_images": int(len(train)), "val_images": int(len(val)), "test_images": int(len(test)),
            "train_patients": int(train["patient_id"].nunique()), "val_patients": int(val["patient_id"].nunique()),
            "test_patients": int(test["patient_id"].nunique()), "patient_overlap": overlap,
            "train_class_counts": train["class"].value_counts().to_dict(),
            "val_class_counts": val["class"].value_counts().to_dict(),
            "test_class_counts": test["class"].value_counts().to_dict(),
        }

    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    marker.write_text("ok", encoding="utf-8")
    if logger:
        logger.info("Generated patient-grouped five-fold manifests: %d images, %d patients", len(df), df["patient_id"].nunique())
    return audit


class ManifestImageDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.targets = self.df["label_index"].astype(int).tolist()
        self.samples = list(zip(self.df["path"].astype(str).tolist(), self.targets))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = Path(str(row["path"]))
        with Image.open(p) as im:
            image = im.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["label_index"]), str(p)


def build_manifest_fold_loaders(manifest_root: Path, fold: int, image_size: int, batch_size: int, num_workers: int):
    import pandas as pd
    fold_dir = manifest_root / f"fold_{fold}"
    train_csv, val_csv, test_csv = fold_dir / "train.csv", fold_dir / "val.csv", fold_dir / "test.csv"
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(f"Generated manifest files missing for fold {fold}")
    train_tf, eval_tf = make_transforms(image_size)
    train_ds = ManifestImageDataset(train_csv, train_tf)
    val_ds = ManifestImageDataset(val_csv, eval_tf)
    test_ds = ManifestImageDataset(test_csv, eval_tf)
    map_df = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)], ignore_index=True)
    mapping = {r["class"]: int(r["label_index"]) for _, r in map_df[["class", "label_index"]].drop_duplicates().iterrows()}
    pin = torch.cuda.is_available()
    loader_args = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_args),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_args),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_args),
        mapping,
        {"train": train_csv, "val": val_csv, "test": test_csv},
    )
