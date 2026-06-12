"""CelebAMask-HQ segmentation dataloader.

Dataset layout:
    root_dir/
    ├── CelebA-HQ-img/          ← .jpg images (512×512), named 0.jpg … 29999.jpg
    └── CelebAMask-HQ-mask-anno/
        ├── 0/                   ← mask parts for images 0-1999
        │   ├── 00000_skin.png
        │   ├── 00000_l_brow.png
        │   ...
        ├── 1/                   ← images 2000-3999
        ...
        └── 14/                  ← images 28000-29999

Mask convention: each part is stored as a separate PNG; we merge them into
a single (H, W) int64 tensor with class indices 0-18 (0 = background).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Class name → index mapping (matches BiSeNet training code)
SEG_PART_NAMES = [
    "skin", "l_brow", "r_brow", "l_eye", "r_eye",
    "eye_g", "l_ear", "r_ear", "ear_r", "nose", "mouth",
    "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat",
]
PART_TO_IDX: Dict[str, int] = {name: i + 1 for i, name in enumerate(SEG_PART_NAMES)}
N_CLASSES = 19   # 0=background + 18 parts
CLASS_NAMES = ["background"] + SEG_PART_NAMES


Sample = Tuple[str, int]   # (image_path, image_id)


def _get_mask_folder(image_id: int) -> int:
    """CelebAMask-HQ stores masks in subfolders of 2000 images each."""
    return image_id // 2000


def _load_mask(mask_dir: Path, image_id: int, image_size: int) -> torch.Tensor:
    """Load and merge all part masks for one image into a (H, W) class tensor."""
    folder = _get_mask_folder(image_id)
    mask = np.zeros((image_size, image_size), dtype=np.int64)
    prefix = f"{image_id:05d}_"
    part_dir = mask_dir / str(folder)
    for part_name, class_idx in PART_TO_IDX.items():
        part_path = part_dir / f"{prefix}{part_name}.png"
        if part_path.exists():
            part_mask = np.array(Image.open(part_path).convert("L").resize(
                (image_size, image_size), Image.NEAREST))
            mask[part_mask > 0] = class_idx
    return torch.from_numpy(mask).long()


class CelebAHQSegDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],   # (image_path, image_id)
        mask_dir: Path,
        image_size: int = 512,
        img_transform=None,
        return_index: bool = False,
    ):
        self.samples = samples
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.img_transform = img_transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, image_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            img_t = self.img_transform(img)
        else:
            img_t = transforms.ToTensor()(img)
        mask = _load_mask(self.mask_dir, image_id, self.image_size)
        if self.return_index:
            return img_t, mask, idx
        return img_t, mask


def build_seg_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# Official face-parsing.PyTorch split (matches pretrained checkpoint training)
OFFICIAL_TRAIN_IDS = range(0, 28000)       # 0–27999  (28000 images)
OFFICIAL_VAL_IDS   = range(28000, 28600)   # 28000–28599 (600 images, ~2%)
OFFICIAL_TEST_IDS  = range(28000, 30000)   # 28000–29999 (2000 images)


def build_seg_dataloaders(cfg) -> Dict:
    """Build train/val/test dataloaders for CelebAMask-HQ segmentation.

    Split strategy (controlled by cfg.dataset.use_official_split):
      True  (default) — use official face-parsing.PyTorch split:
                         train=0–27999, test=28000–29999.
                         Ensures certification metrics match published results.
      False           — use ratio split (train_ratio/val_ratio) for custom experiments.

    Returns dict with keys: train_loader, val_loader, test_loader,
    train_samples, val_samples, test_samples.
    """
    root = Path(cfg.dataset.root_dir)
    image_dir = root / cfg.dataset.image_dir
    mask_dir  = root / cfg.dataset.mask_dir
    image_size = cfg.dataset.image_size

    # Collect all available image IDs
    all_ids = sorted(int(p.stem) for p in image_dir.glob(f"*{cfg.dataset.file_extension}"))
    if not all_ids:
        raise RuntimeError(f"No images found in {image_dir}")
    all_ids_set = set(all_ids)

    use_official = getattr(cfg.dataset, "use_official_split", True)

    if use_official:
        # Official split — filter to IDs that actually exist on disk
        train_ids = [i for i in OFFICIAL_TRAIN_IDS if i in all_ids_set]
        val_ids   = [i for i in OFFICIAL_VAL_IDS   if i in all_ids_set]
        test_ids  = [i for i in OFFICIAL_TEST_IDS  if i in all_ids_set]
        print(f"Official split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    else:
        # Ratio split
        rng = random.Random(cfg.dataset.split_seed)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * cfg.dataset.train_ratio)
        n_val   = int(n * cfg.dataset.val_ratio)
        train_ids = shuffled[:n_train]
        val_ids   = shuffled[n_train:n_train + n_val]
        test_ids  = shuffled[n_train + n_val:]
        print(f"Ratio split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Subset for fast testing (applied after split, always from test)
    if cfg.dataset.subset_size and cfg.dataset.subset_size < len(test_ids):
        test_ids = test_ids[:cfg.dataset.subset_size]
        print(f"Test subset: {len(test_ids)} images")

    def ids_to_samples(ids):
        return [(str(image_dir / f"{i}{cfg.dataset.file_extension}"), i) for i in ids]

    train_samples = ids_to_samples(train_ids)
    val_samples   = ids_to_samples(val_ids)
    test_samples  = ids_to_samples(test_ids)

    tf = build_seg_transform(image_size)
    train_ds = CelebAHQSegDataset(train_samples, mask_dir, image_size, tf)
    val_ds   = CelebAHQSegDataset(val_samples,   mask_dir, image_size, tf)
    test_ds  = CelebAHQSegDataset(test_samples,  mask_dir, image_size, tf)

    nw = cfg.dataset.num_workers
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=nw, pin_memory=True)

    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "train_samples": train_samples,
        "val_samples":   val_samples,
        "test_samples":  test_samples,
        "mask_dir":      mask_dir,
        "image_size":    image_size,
    }
