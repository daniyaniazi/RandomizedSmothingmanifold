"""Training entry point for ResNet smile classifier on CelebA / CelebA-HQ.

Supports optional smoothing augmentation (isotropic or manifold) so that
the trained classifier matches the noise distribution used at certification
time — following the Cohen et al. (2019) randomized smoothing training scheme.

Usage:
    # Plain training (no augmentation):
    python -m src.experiments.training.resnet_smile.main \\
        --config src/configs/training/smile_resnet_celeba.yaml

    # With isotropic augmentation at sigma=0.25:
    python -m src.experiments.training.resnet_smile.main \\
        --config src/configs/training/smile_resnet_celeba.yaml \\
        --sigma 0.25 --aug_mode isotropic

    # With manifold augmentation (requires pre-built index):
    python -m src.experiments.training.resnet_smile.main \\
        --config src/configs/training/smile_resnet_celeba.yaml \\
        --sigma 0.25 --aug_mode manifold \\
        --index_path output/smile_classification/celeba/index/pixel/annoy/euclidean/index.ann
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.configs.train_smile_io import load_smile_training_config, save_smile_resolved_config
from src.configs.train_smile_schema import SmileTrainingConfig
from src.dataloaders.celeba_smile import build_smile_dataloaders
from src.models.resnet import build_resnet_classifier
from src.smoothing.isotropic import IsotropicSmoother
from src.smoothing.manifold import ManifoldSmoother


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


def build_smoother(cfg: SmileTrainingConfig) -> Optional[IsotropicSmoother | ManifoldSmoother]:
    """Build the training-time smoother (None if augmentation is disabled)."""
    aug = cfg.smoothing_aug
    if not aug.enabled:
        return None

    if aug.mode == "isotropic":
        print(f"Smoothing aug: ISOTROPIC  σ={aug.sigma}")
        return IsotropicSmoother(sigma=aug.sigma)

    elif aug.mode == "manifold":
        from src.indexing.base import load_index
        if aug.index_path is None:
            raise ValueError(
                "smoothing_aug.mode='manifold' requires smoothing_aug.index_path to be set. "
                "Build the pixel index first (certify run builds it automatically)."
            )
        index_path = Path(aug.index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Pixel index not found: {index_path}")
        # Infer dimension from config
        dim = cfg.model.input_size * cfg.model.input_size * 3
        index = load_index(dim=dim, index_path=str(index_path), backend="annoy")
        print(f"Smoothing aug: MANIFOLD  σ={aug.sigma}  knn_k={aug.knn_k}  index={index_path}")
        return ManifoldSmoother(sigma=aug.sigma, index=index, knn_k=aug.knn_k, eps_eig=aug.eps_eig)

    else:
        raise ValueError(f"Unknown smoothing_aug.mode: '{aug.mode}'. Choose 'isotropic' or 'manifold'.")


def apply_smoother_to_batch(
    images: torch.Tensor,
    smoother: IsotropicSmoother | ManifoldSmoother,
) -> torch.Tensor:
    """Apply smoother to each image in a batch (CPU, numpy round-trip).

    One noise sample per image per training step — this is the standard
    Gaussian data augmentation from Cohen et al. (2019).
    """
    B, C, H, W = images.shape
    noisy = []
    imgs_np = images.cpu().numpy()  # (B, C, H, W)
    for i in range(B):
        flat = imgs_np[i].flatten().astype("float32")
        noisy_flat = smoother.sample(flat)
        noisy.append(noisy_flat.reshape(C, H, W))
    noisy_np = np.stack(noisy, axis=0)  # (B, C, H, W)
    return torch.from_numpy(noisy_np).to(images.device)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(
    model, loader, optimizer, criterion, device, train: bool, log_every: int,
    smoother: Optional[IsotropicSmoother | ManifoldSmoother] = None,
):
    model.train(train)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for step, (images, labels) in enumerate(tqdm(loader, leave=False)):
            images = images.to(device)
            labels = labels.to(device)

            # ── Smoothing augmentation (train only) ───────────────────────
            # Apply one noise sample per image, matching the certification
            # distribution.  Eval always runs on clean images.
            if train and smoother is not None:
                images = apply_smoother_to_batch(images.cpu(), smoother).to(device)

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

    # ── Smoother (optional training-time augmentation) ────────────────────
    smoother = build_smoother(cfg)
    if smoother is not None:
        print(f"Training WITH smoothing augmentation: mode={cfg.smoothing_aug.mode}  σ={cfg.smoothing_aug.sigma}")
    else:
        print("Training WITHOUT smoothing augmentation (clean images only).")

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
    history: list[dict] = []
    start_epoch = 1

    # ── Resume from checkpoint ──
    resume_path = ckpt_dir / "latest.pt"
    if resume_path.exists():
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        print(f"\n── Epoch {epoch}/{cfg.train.epochs} ──────────────")

        train_loss, train_acc = run_epoch(
            model, data.train_loader, optimizer, criterion, device,
            train=True, log_every=cfg.logging.log_every_n_steps,
            smoother=smoother,
        )
        val_loss, val_acc = run_epoch(
            model, data.val_loader, optimizer, criterion, device,
            train=False, log_every=999,
            smoother=None,  # always eval on clean images
        )

        print(f"  Train loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val   loss={val_loss:.4f}  acc={val_acc:.4f}")

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)

        # Save training history JSON after every epoch
        history_path = output_dir / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2))

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

        # Always save latest.pt for resume (includes optimizer state)
        latest_path = ckpt_dir / "latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
        }, latest_path)

    # Final evaluation on the test set
    print("\n── Test set evaluation ──────────────")
    test_loss, test_acc = run_epoch(
        model, data.test_loader, optimizer, criterion, device,
        train=False, log_every=999,
        smoother=None,
    )
    print(f"  Test  loss={test_loss:.4f}  acc={test_acc:.4f}")
    history.append({"epoch": "test", "test_loss": test_loss, "test_acc": test_acc})
    history_path.write_text(json.dumps(history, indent=2))

    if wandb_run is not None:
        wandb_run.log({"test/loss": test_loss, "test/acc": test_acc})

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}  |  Test acc: {test_acc:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet smile classifier")
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    # CLI overrides for sigma sweep (override smoothing_aug fields without editing the YAML)
    parser.add_argument("--sigma", type=float, default=None,
                        help="Override smoothing_aug.sigma (also enables augmentation)")
    parser.add_argument("--aug_mode", type=str, default=None,
                        choices=["isotropic", "manifold"],
                        help="Override smoothing_aug.mode")
    parser.add_argument("--index_path", type=str, default=None,
                        help="Override smoothing_aug.index_path (manifold mode)")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Override checkpoint.base_dir (used by sweep script)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output_dir")
    args = parser.parse_args()

    cfg = load_smile_training_config(args.config)

    # Apply CLI overrides
    if args.sigma is not None:
        cfg.smoothing_aug.sigma = args.sigma
        cfg.smoothing_aug.enabled = True
    if args.aug_mode is not None:
        cfg.smoothing_aug.mode = args.aug_mode
        cfg.smoothing_aug.enabled = True
    if args.index_path is not None:
        cfg.smoothing_aug.index_path = args.index_path
    if args.ckpt_dir is not None:
        cfg.checkpoint.base_dir = args.ckpt_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    print(f"Experiment : {cfg.experiment_name}")
    print(f"Dataset    : {cfg.dataset.name}")
    print(f"Model      : {cfg.model.name}  pretrained={cfg.model.pretrained}")
    print(f"Epochs     : {cfg.train.epochs}  lr={cfg.train.lr}")
    if cfg.smoothing_aug.enabled:
        print(f"Aug        : mode={cfg.smoothing_aug.mode}  σ={cfg.smoothing_aug.sigma}")

    train(cfg)


if __name__ == "__main__":
    main()
