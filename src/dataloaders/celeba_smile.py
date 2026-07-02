"""Dataloaders for smile classification on CelebA/CelebA-HQ style datasets.
   Link : https://www.kaggle.com/datasets/j53t3r/celebahq for HQ Annotation
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.configs.train_smile_schema import (
    SmileDataloaderConfig,
    SmileDatasetConfig,
    SmileModelConfig,
)


Sample = Tuple[str, int]


class SmileImageDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform=None, return_index: bool = False):
        self.samples = list(samples)
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label_t = torch.tensor(float(label), dtype=torch.float32)
        if self.return_index:
            return image, label_t, index
        return image, label_t


@dataclass
class SmileDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_counts: Dict[str, int]
    pos_weight: float
    # OOD certification samples: train partition attr=1 (excluded from training,
    # not used for model selection to truly unseen OOD). None when no ood_exclude_attribute.
    certify_ood_samples: Optional[List] = None


def _to_label(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "smile", "smiling"}:
        return 1
    if text in {"0", "-1", "false", "no", "n", "non-smile", "not_smiling"}:
        return 0
    try:
        num = float(text)
        return 1 if num > 0 else 0
    except ValueError as exc:
        raise ValueError(f"Cannot parse smile label from value: {value}") from exc


def _resolve_image_path(image_dir: Path, file_name: str, extension: str) -> Path:
    candidate = image_dir / file_name
    if candidate.exists():
        return candidate

    if extension and not candidate.suffix:
        candidate_ext = image_dir / f"{file_name}{extension}"
        if candidate_ext.exists():
            return candidate_ext

    raise FileNotFoundError(f"Image not found for sample: {file_name} in {image_dir}")


def _read_celeba_annotations(annotation_path: Path) -> Dict[str, int]:
    lines = [line.strip() for line in annotation_path.read_text().splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Invalid CelebA attribute file: {annotation_path}")

    attr_names = lines[1].split()
    if "Smiling" not in attr_names:
        raise ValueError(f"Smiling attribute not found in {annotation_path}")
    smile_idx = attr_names.index("Smiling")

    labels: Dict[str, int] = {}
    for row in lines[2:]:
        parts = row.split()
        if len(parts) < 2 + smile_idx:
            continue
        file_name = parts[0]
        labels[file_name] = _to_label(parts[1 + smile_idx])
    return labels


def _read_csv_annotations(annotation_path: Path, image_column: str, label_column: str) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with annotation_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get(image_column)
            label = row.get(label_column)
            if not file_name or label is None:
                continue
            labels[file_name] = _to_label(label)
    return labels


def _read_partition_file(partition_path: Path) -> Dict[str, int]:
    partitions: Dict[str, int] = {}
    for line in partition_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name, split_id = line.split()
        partitions[name] = int(split_id)
    return partitions


def _balanced_subset(samples: List[Sample], n_per_class: int, seed: int) -> List[Sample]:
    """Return at most n_per_class smile + n_per_class non-smile, shuffled."""
    rng = random.Random(seed)
    pos = [s for s in samples if s[1] == 1]
    neg = [s for s in samples if s[1] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    chosen = pos[:n_per_class] + neg[:n_per_class]
    rng.shuffle(chosen)
    return chosen


def _split_by_ratio(samples: List[Sample], train_ratio: float, val_ratio: float, seed: int):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def _build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return train_transform, eval_transform


def build_dataloader_from_samples(
    samples: List[Sample],
    image_size: int,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap a pre-split (path, label) list in a DataLoader with Resize+ToTensor."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = SmileImageDataset(samples, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def build_smile_dataloaders(
    dataset_cfg: SmileDatasetConfig,
    loader_cfg: SmileDataloaderConfig,
    model_cfg: SmileModelConfig,
) -> SmileDataBundle:
    root_dir = Path(dataset_cfg.root_dir)
    image_dir = root_dir / dataset_cfg.image_dir
    annotation_path = root_dir / dataset_cfg.annotation_file

    if dataset_cfg.annotation_format == "celeba":
        labels = _read_celeba_annotations(annotation_path)
    elif dataset_cfg.annotation_format == "csv":
        labels = _read_csv_annotations(annotation_path, dataset_cfg.image_column, dataset_cfg.label_column)
    else:
        raise ValueError(f"Unsupported annotation format: {dataset_cfg.annotation_format}")

    all_samples: List[Sample] = []
    for file_name, label in labels.items():
        try:
            image_path = _resolve_image_path(image_dir, file_name, dataset_cfg.file_extension)
        except FileNotFoundError:
            continue
        all_samples.append((str(image_path), label))

    if not all_samples:
        raise RuntimeError("No valid samples found. Check paths and annotation settings.")

    partition_map: Optional[Dict[str, int]] = None
    if dataset_cfg.partition_file:
        partition_path = root_dir / dataset_cfg.partition_file
        if partition_path.exists():
            partition_map = _read_partition_file(partition_path)

    if partition_map is not None:
        train_samples: List[Sample] = []
        val_samples: List[Sample] = []
        test_samples: List[Sample] = []
        for image_path, label in all_samples:
            file_name = Path(image_path).name
            split_id = partition_map.get(file_name)
            if split_id == 0:
                train_samples.append((image_path, label))
            elif split_id == 1:
                val_samples.append((image_path, label))
            elif split_id == 2:
                test_samples.append((image_path, label))
        if not train_samples or not val_samples or not test_samples:
            train_samples, val_samples, test_samples = _split_by_ratio(
                all_samples,
                train_ratio=dataset_cfg.train_ratio,
                val_ratio=dataset_cfg.val_ratio,
                seed=dataset_cfg.split_seed,
            )
    else:
        train_samples, val_samples, test_samples = _split_by_ratio(
            all_samples,
            train_ratio=dataset_cfg.train_ratio,
            val_ratio=dataset_cfg.val_ratio,
            seed=dataset_cfg.split_seed,
        )

    # OOD exclusion with no-leakage setup:
    #
    #   train:           attr=0, train partition  : classifier trains on non-OOD only
    #   val:             attr=1, val partition    : early stopping on OOD
    #   test:            attr=1, test partition   : classifier test_acc (held-out, not used in training/selection)
    #   certify_ood:     attr=1, TRAIN partition  : certification (excluded from training AND from val/test)
    #                    : truly unseen: model never trained on them, not used for model selection
    certify_ood_samples: Optional[List] = None
    ood_attr = getattr(dataset_cfg, "ood_exclude_attribute", None)
    if ood_attr:
        attr_path = root_dir / dataset_cfg.annotation_file
        attr_lines = [l.strip() for l in attr_path.read_text().splitlines() if l.strip()]
        attr_names = attr_lines[1].split()
        if ood_attr not in attr_names:
            raise ValueError(f"ood_exclude_attribute '{ood_attr}' not found in {attr_path}. "
                             f"Available: {attr_names}")
        attr_idx = attr_names.index(ood_attr)
        attr_map: Dict[str, int] = {}
        for row in attr_lines[2:]:
            parts = row.split()
            attr_map[parts[0]] = int(parts[1 + attr_idx])

        def _is_ood(sample: Sample) -> bool:
            return attr_map.get(Path(sample[0]).name, -1) == 1

        # Collect train-partition attr=1 BEFORE removing them for  certification pool samples (unseen OOD)
        certify_ood_samples = [s for s in train_samples if _is_ood(s)]

        # Train: attr=0 only
        train_samples = [s for s in train_samples if not _is_ood(s)]
        # Val:   attr=1 only (early stopping on OOD)
        val_samples   = [s for s in val_samples   if _is_ood(s)]
        # Test:  attr=1 only (classifier test_acc evaluation)
        test_samples  = [s for s in test_samples  if _is_ood(s)]

        # ── Balanced equal-size subsets ───────────────────────────────
        # ood_train_subset_per_class: N smile + N non-smile for training (attr=0)
        # ood_test_subset_per_class:  N smile + N non-smile for testing  (attr=1)
        # Set in config to make all OOD experiments comparable.
        train_npc = getattr(dataset_cfg, "ood_train_subset_per_class", None)
        test_npc  = getattr(dataset_cfg, "ood_test_subset_per_class",  None)
        if train_npc is not None and train_npc > 0:
            train_samples = _balanced_subset(train_samples, train_npc, dataset_cfg.split_seed)
        if test_npc is not None and test_npc > 0:
            val_samples  = _balanced_subset(val_samples,  test_npc, dataset_cfg.split_seed)
            test_samples = _balanced_subset(test_samples, test_npc, dataset_cfg.split_seed)

        print(f"OOD setup '{ood_attr}':  "
              f"train={len(train_samples)} (attr=0)  "
              f"val={len(val_samples)} (attr=1)  "
              f"test={len(test_samples)} (attr=1, test partition)  "
              f"certify_ood={len(certify_ood_samples)} (attr=1, train partition : unseen)"
              + (f"  [balanced: {train_npc}/class train, {test_npc}/class test]"
                 if train_npc or test_npc else ""))

    train_transform, eval_transform = _build_transforms(model_cfg.input_size)
    train_dataset = SmileImageDataset(train_samples, transform=train_transform)
    val_dataset = SmileImageDataset(val_samples, transform=eval_transform)
    test_dataset = SmileImageDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=loader_cfg.shuffle_train,
        num_workers=dataset_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )

    positives = sum(label for _, label in train_samples)
    negatives = len(train_samples) - positives
    pos_weight = (negatives / max(1, positives)) if positives > 0 else 1.0

    class_counts = {
        "train_positive": int(positives),
        "train_negative": int(negatives),
        "train_total": len(train_samples),
        "val_total": len(val_samples),
        "test_total": len(test_samples),
    }

    return SmileDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_counts=class_counts,
        pos_weight=float(pos_weight),
        certify_ood_samples=certify_ood_samples,
    )