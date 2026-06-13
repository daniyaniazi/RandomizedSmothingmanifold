"""Build pixel-space kNN index for CelebAMask-HQ segmentation certification.

Indexes the TRAIN split (official: IDs 0-27999) in raw [0,1] pixel space.
Mirrors src/experiments/indexing/celeba_images.py but uses the seg dataloader
and seg certify config.

Output:
    output/segmentation/celebahq/index/pixel/{backend}/{metric}/index.ann

Usage:
    python -m src.experiments.indexing.celebahq_seg_images \\
        --config src/configs/experiments/certify_celebahq_seg_manifold.yaml

    # Override metric
    python -m src.experiments.indexing.celebahq_seg_images \\
        --config src/configs/experiments/certify_celebahq_seg_manifold.yaml \\
        --metric angular
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.certify_seg_io import load_seg_certify_config
from src.dataloaders.celebahq_seg import build_seg_dataloaders, build_seg_transform
from src.indexing.image_index import build_or_load_pixel_index_streaming
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class _ImgOnlyDataset(Dataset):
    """Wraps (image_path, image_id) samples, returns dict for image_index pipeline."""
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, _ = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        return {"image": self.transform(img), "label": 0, "path": img_path}


def main():
    parser = argparse.ArgumentParser(
        description="Build pixel-space kNN index for CelebAMask-HQ segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.experiments.indexing.celebahq_seg_images \\
      --config src/configs/experiments/certify_celebahq_seg_manifold.yaml

  python -m src.experiments.indexing.celebahq_seg_images \\
      --config src/configs/experiments/certify_celebahq_seg_manifold.yaml \\
      --metric angular
        """
    )
    parser.add_argument("--config",  required=True,
                        help="Segmentation certify config YAML")
    parser.add_argument("--metric",  default=None, choices=["euclidean", "angular"],
                        help="Override index metric from config")
    parser.add_argument("--backend", default="annoy", choices=["annoy"],
                        help="Index backend (default: annoy)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild even if index already exists")
    args = parser.parse_args()

    cfg = load_seg_certify_config(args.config)
    if args.metric:
        cfg.index.metric = args.metric
    if args.backend:
        cfg.index.backend = args.backend

    _log(f"Config:   {args.config}")
    _log(f"Dataset:  {cfg.dataset.name}")
    _log(f"Backend:  {cfg.index.backend}  |  Metric: {cfg.index.metric}")

    # ── Build train samples using seg dataloader (official split) ─────────────
    data = build_seg_dataloaders(cfg)
    train_samples = data["train_samples"]
    image_size    = data["image_size"]
    _log(f"Train samples: {len(train_samples):,}  (IDs {train_samples[0][1]}–{train_samples[-1][1]})")

    # ── Output path — same formula as SegPaths ─────────────────────────────────
    dataset_tag     = cfg.dataset.name.lower().replace("-", "").replace("_", "")
    base_dir        = Path(cfg.output.output_dir) / "segmentation" / dataset_tag
    pixel_index_dir = base_dir / "index" / "pixel" / cfg.index.backend / cfg.index.metric
    pixel_index_dir.mkdir(parents=True, exist_ok=True)
    index_path = pixel_index_dir / "index.ann"

    if index_path.exists() and not args.rebuild:
        _log(f"Index already exists: {index_path}  (use --rebuild to force)")
        # Verify item count
        from src.indexing.base import load_index
        dim = 3 * image_size * image_size
        idx = load_index(dim=dim, index_path=str(index_path),
                         backend=cfg.index.backend, metric=cfg.index.metric)
        _log(f"Loaded existing index: {idx.index.get_n_items():,} vectors  dim={dim}")
        return

    # ── Build index ────────────────────────────────────────────────────────────
    tf     = build_seg_transform(image_size)
    ds     = _ImgOnlyDataset(train_samples, tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        num_workers=cfg.dataset.num_workers, pin_memory=False)

    dim = 3 * image_size * image_size
    _log(f"Building pixel index — STREAMING mode ({len(train_samples):,} images, dim={dim:,}) …")
    # Streaming: adds vectors one batch at a time → avoids loading all 28k×512×512×3 into RAM

    artifacts = build_or_load_pixel_index_streaming(
        out_dir    = pixel_index_dir,
        dataloader = loader,
        dim        = dim,
        image_key  = "image",
        metric     = cfg.index.metric,
        n_trees    = cfg.index.n_trees,
        metadata   = {
            "dataset":    cfg.dataset.name,
            "split":      "train_official",
            "n_items":    len(train_samples),
            "image_size": image_size,
            "id_range":   f"{train_samples[0][1]}–{train_samples[-1][1]}",
        },
        log_fn = _log,
    )

    fnames = [str(p) for p, _ in train_samples]
    artifacts.index.filenames = fnames
    n = artifacts.metadata.get("n_vectors", "?") if artifacts.metadata else "?"
    _log(f"Index saved: {index_path}  ({n} vectors  dim={dim})")


if __name__ == "__main__":
    main()
