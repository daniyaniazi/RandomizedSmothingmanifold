"""Training entry point for ResNet smile classifier on CelebA / CelebA-HQ.

Usage:
    python -m src.experiments.training.resnet_smile.main \\
        --config src/configs/training/smile_resnet_celeba.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.configs.train_smile_io import load_smile_training_config, save_smile_resolved_config
from src.configs.train_smile_schema import SmileTrainingConfig
from src.dataloaders.celeba_smile import build_smile_dataloaders
from src.models.resnet import build_resnet_classifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).float()
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion, device, train: bool, log_every: int):
    model.train(train)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for step, (images, labels) in enumerate(tqdm(loader, leave=False)):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze(-1)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += compute_accuracy(logits, labels)
            n_batches += 1

            if train and (step + 1) % log_every == 0:
                print(f"  step {step+1:4d}  loss={total_loss/n_batches:.4f}  acc={total_acc/n_batches:.4f}")

    return total_loss / n_batches, total_acc / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: SmileTrainingConfig) -> None:
    set_seed(cfg.train.seed)
    device = device_from_cfg(cfg.train.device)

    output_dir = Path(cfg.output_dir)
    ckpt_dir = Path(cfg.checkpoint.base_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    save_smile_resolved_config(cfg, output_dir / "resolved_config.yaml")

    # Data
    data = build_smile_dataloaders(
        dataset_cfg=cfg.dataset,
        loader_cfg=cfg.dataloader,
        model_cfg=cfg.model,
    )
    print(f"Train: {len(data.train_loader)} batches | Val: {len(data.val_loader)} batches")
    print(f"Positive-weight: {data.pos_weight:.4f}")

    # Model
    model = build_resnet_classifier(
        name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
        num_classes=1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(data.pos_weight, device=device)
    )
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # W&B (optional)
    wandb_run = None
    if cfg.wandb.enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.run_name or cfg.experiment_name,
                config={"experiment": cfg.experiment_name},
            )
        except ImportError:
            print("WARNING: wandb not installed, skipping logging.")

    best_val_acc = 0.0
    saved_ckpts: list[Path] = []

    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\n── Epoch {epoch}/{cfg.train.epochs} ──────────────")

        train_loss, train_acc = run_epoch(
            model, data.train_loader, optimizer, criterion, device,
            train=True, log_every=cfg.logging.log_every_n_steps,
        )
        val_loss, val_acc = run_epoch(
            model, data.val_loader, optimizer, criterion, device,
            train=False, log_every=999,
        )

        print(f"  Train loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val   loss={val_loss:.4f}  acc={val_acc:.4f}")

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc,
                           "val/loss": val_loss, "val/acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = ckpt_dir / "best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc}, best_path)
            print(f"  ✓ Best model saved → {best_path}")

        if epoch % cfg.train.save_every_n_epochs == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)
            saved_ckpts.append(ckpt_path)
            # Keep only last k checkpoints
            while len(saved_ckpts) > cfg.checkpoint.keep_last_k:
                old = saved_ckpts.pop(0)
                if old.exists():
                    old.unlink()

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet smile classifier")
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    args = parser.parse_args()

    cfg = load_smile_training_config(args.config)
    print(f"Experiment : {cfg.experiment_name}")
    print(f"Dataset    : {cfg.dataset.name}")
    print(f"Model      : {cfg.model.name}  pretrained={cfg.model.pretrained}")
    print(f"Epochs     : {cfg.train.epochs}  lr={cfg.train.lr}")

    train(cfg)


if __name__ == "__main__":
    main()
