"""Training entry point for ConvVAE on CelebA / CelebA-HQ.

Usage:
    python -m src.experiments.training.vae.main --config src/configs/training/vae_celeba.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.configs.train_vae_io import load_vae_training_config, save_vae_resolved_config
from src.configs.train_vae_schema import VaeTrainingConfig
from src.models.VAE.model import ConvVAE
from src.models.VAE.trainer import save_checkpoint, train_vae


class FlatImageDataset(Dataset):
    """Dataset for folders containing image files directly (no class subfolders needed)."""

    def __init__(self, image_paths: list[Path], transform=None, in_channels: int = 3):
        self.image_paths = image_paths
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        mode = "RGB" if self.in_channels == 3 else "L"
        img = Image.open(path).convert(mode)
        if self.transform is not None:
            img = self.transform(img)
        # trainer expects (x, _)
        return img, 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_cfg(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def _build_transform(image_size: int, in_channels: int):
    mean = [0.5] * in_channels
    std = [0.5] * in_channels
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _scan_images(image_root: Path, file_extension: str | None = None) -> list[Path]:
    if not image_root.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_root}")

    if file_extension:
        ext = file_extension.lower().lstrip(".")
        exts = {f".{ext}"}
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    paths = [p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    paths.sort()
    if not paths:
        raise RuntimeError(f"No image files found in {image_root} with extensions {sorted(exts)}")
    return paths


def _init_wandb(cfg: VaeTrainingConfig):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("W&B enabled in config but package not installed. Continuing without W&B.")
        return None

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name or cfg.experiment_name,
        config={"experiment": cfg.experiment_name},
    )
    return run


def _save_history(history: list[dict], out_dir: Path) -> None:
    (out_dir / "training_summary.json").write_text(json.dumps(history, indent=2))
    if history:
        with (out_dir / "training_history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)


def train(cfg: VaeTrainingConfig) -> None:
    set_seed(cfg.train.seed)
    device = device_from_cfg(cfg.train.device)

    out_dir = Path(cfg.output_dir) / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_vae_resolved_config(cfg, out_dir / "resolved_config.yaml")

    ckpt_dir = Path(cfg.checkpoint.base_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    image_root = Path(cfg.dataset.root_dir) / cfg.dataset.image_dir
    image_paths = _scan_images(image_root, cfg.dataset.file_extension)
    print(f"Found {len(image_paths)} images in {image_root}")

    dataset = FlatImageDataset(
        image_paths=image_paths,
        transform=_build_transform(cfg.model.image_size, cfg.model.in_channels),
        in_channels=cfg.model.in_channels,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle_train,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

    model = ConvVAE(
        in_channels=cfg.model.in_channels,
        image_size=cfg.model.image_size,
        latent_dim=cfg.model.latent_dim,
    ).to(device)
    print(f"Model params: {model.num_params():,}")

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    wandb_run = _init_wandb(cfg)

    history = train_vae(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        device=device,
        epochs=cfg.train.epochs,
        beta=cfg.train.beta,
    )

    # Save final and best-style checkpoint names for easy downstream loading.
    final_path = ckpt_dir / "final.pt"
    save_checkpoint(model, final_path, extra={"history": history, "config": cfg.experiment_name})

    best_path = ckpt_dir / "best.pt"
    save_checkpoint(model, best_path, extra={"history": history, "config": cfg.experiment_name})

    _save_history(history, out_dir)

    if wandb_run is not None:
        for row in history:
            wandb_run.log({f"train/{k}": v for k, v in row.items() if k != "epoch"} | {"epoch": row["epoch"]})
        wandb_run.finish()

    print(f"Training complete. Artifacts in: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvVAE")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/training/vae_celeba.yaml",
        help="Path to VAE training YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_vae_training_config(args.config)
    train(cfg)
