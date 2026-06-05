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


# ---------------------------------------------------------------------------
# GPU noise functions (no CPU round-trip)
# ---------------------------------------------------------------------------

def apply_isotropic_noise_gpu(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add N(0, σ²I) noise — fully on GPU, zero CPU involvement."""
    return images + torch.randn_like(images) * sigma


def apply_manifold_noise_gpu(
    images: torch.Tensor,
    indices: torch.Tensor,
    gpu_cache: dict,
    sigma: float,
) -> torch.Tensor:
    """Add manifold noise (whiten → Gaussian → unwhiten) — fully on GPU.

    Replicates ManifoldSmoother._sample_from_pca() in batched torch form:
        w  = (x - mean) @ evecs / sqrt(evals)          # whiten
        w' = w + N(0, (σ/√λ_max)² I)                  # add noise
        x' = (w' * sqrt(evals)) @ evecs.T + mean       # unwhiten

    Args:
        images:    (B, C, H, W) float32 on GPU
        indices:   (B,) long tensor — dataset sample indices for cache lookup
        gpu_cache: dict with GPU tensors:
                     'mean'  (N, D)
                     'evals' (N, K)
                     'evecs' (N, D, K)
        sigma:     noise scale
    Returns:
        (B, C, H, W) noisy images on GPU
    """
    B, C, H, W = images.shape
    D = C * H * W

    x     = images.view(B, D)                                   # (B, D)
    mean  = gpu_cache['mean'][indices]                           # (B, D)
    evecs = gpu_cache['evecs'][indices]                          # (B, D, K)
    evals = gpu_cache['evals'][indices]                          # (B, K)

    # Whiten: w = (x - mean) @ evecs / sqrt(evals)
    x_c = (x - mean).unsqueeze(1)                               # (B, 1, D)
    w   = torch.bmm(x_c, evecs).squeeze(1)                      # (B, K)
    w   = w / torch.sqrt(evals.clamp(min=1e-12))                # (B, K)

    # Noise scale: alpha = sigma / sqrt(lambda_max)
    lambda_max = evals[:, 0].clamp(min=1e-12)                   # (B,)
    alpha      = sigma / torch.sqrt(lambda_max)                  # (B,)
    noise      = torch.randn_like(w) * alpha.unsqueeze(1)       # (B, K)
    w_noisy    = w + noise                                       # (B, K)

    # Unwhiten: x' = (w_noisy * sqrt(evals)) @ evecs.T + mean
    w_scaled = w_noisy * torch.sqrt(evals.clamp(min=1e-12))     # (B, K)
    x_noisy  = torch.bmm(
        w_scaled.unsqueeze(1), evecs.transpose(1, 2)
    ).squeeze(1) + mean                                          # (B, D)

    return x_noisy.view(B, C, H, W)


def build_gpu_pca_tensors_from_disk(
    dataset,
    npz_path: str,
    device: torch.device,
) -> dict:
    """Load pre-computed PCA .npz and build GPU tensors aligned to dataset order.

    Tensors are indexed by dataset sample index so each training batch can do
    a single index_select — no string lookups at training time.

    Returns:
        dict with GPU tensors  mean (N,D), evals (N,K), evecs (N,D,K)
    """
    print(f"Loading PCA cache from disk: {npz_path}")
    raw = np.load(npz_path)

    npz_stems: set = {k[:-5] for k in raw.files if k.endswith("_mean")}
    first_stem = next(iter(npz_stems))
    D = int(raw[f"{first_stem}_mean"].shape[0])
    K = int(raw[f"{first_stem}_evals"].shape[0])

    N = len(dataset.samples)
    means  = np.zeros((N, D), dtype=np.float32)
    evals_ = np.zeros((N, K), dtype=np.float32)
    evecs_ = np.zeros((N, D, K), dtype=np.float32)

    missing = 0
    for i, (img_path, _) in enumerate(dataset.samples):
        stem = Path(img_path).stem
        if stem in npz_stems:
            means[i]  = raw[f"{stem}_mean"]
            evals_[i] = raw[f"{stem}_evals"][:K]
            evecs_[i] = raw[f"{stem}_evecs"][:, :K]
        else:
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing}/{N} images not in PCA cache — those get zero noise direction.")
    print(f"  PCA tensors: N={N}, D={D}, K={K} → moving to {device}")
    return {
        'mean':  torch.from_numpy(means).to(device),
        'evals': torch.from_numpy(evals_).to(device),
        'evecs': torch.from_numpy(evecs_).to(device),
    }


def precompute_gpu_pca_tensors(
    smoother: ManifoldSmoother,
    dataset,
    device: torch.device,
) -> dict:
    """Compute kNN+SVD once for every training image, return GPU tensors.

    Called when no disk cache exists.  After this, every training step samples
    noise with batched GPU ops only.

    Returns:
        dict with GPU tensors  mean (N,D), evals (N,K), evecs (N,D,K)
    """
    N = len(dataset.samples)
    # Probe first image to get D and K
    img0, _ = dataset[0]
    flat0 = img0.numpy().flatten().astype("float32")
    cached0 = smoother.compute_pca(flat0)
    D = int(cached0.pca.mean.shape[0])
    K = int(cached0.pca.evals.shape[0])

    print(f"Pre-computing PCA for {N} images (D={D}, K={K}) — runs once before training ...")
    means  = np.zeros((N, D), dtype=np.float32)
    evals_ = np.zeros((N, K), dtype=np.float32)
    evecs_ = np.zeros((N, D, K), dtype=np.float32)

    means[0]  = cached0.pca.mean
    evals_[0] = cached0.pca.evals[:K]
    evecs_[0] = cached0.pca.evecs[:, :K]

    for i in tqdm(range(1, N), desc="PCA precompute", leave=True):
        img, _ = dataset[i]
        flat = img.numpy().flatten().astype("float32")
        cached = smoother.compute_pca(flat)
        means[i]  = cached.pca.mean
        evals_[i] = cached.pca.evals[:K]
        evecs_[i] = cached.pca.evecs[:, :K]

    print(f"  Done. Moving PCA tensors to {device} ...")
    return {
        'mean':  torch.from_numpy(means).to(device),
        'evals': torch.from_numpy(evals_).to(device),
        'evecs': torch.from_numpy(evecs_).to(device),
    }


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(
    model, loader, optimizer, criterion, device, train: bool, log_every: int,
    aug_mode: Optional[str] = None,
    sigma: float = 0.0,
    gpu_cache: Optional[dict] = None,
):
    model.train(train)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for step, batch in enumerate(tqdm(loader, leave=False)):
            # Batch may be (images, labels) or (images, labels, indices)
            if len(batch) == 3:
                images, labels, indices = batch
                indices = indices.to(device)
            else:
                images, labels = batch
                indices = None
            images = images.to(device)
            labels = labels.to(device)

            # ── Smoothing augmentation (train only) ───────────────────────
            # One fresh noise sample per image per step — fully on GPU.
            # Eval always uses clean images.
            if train and aug_mode is not None:
                if aug_mode == "manifold" and gpu_cache is not None:
                    images = apply_manifold_noise_gpu(images, indices, gpu_cache, sigma)
                else:
                    images = apply_isotropic_noise_gpu(images, sigma)

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
    # Noise is sampled fresh every step during training (Cohen et al. 2019).
    aug = cfg.smoothing_aug

    smoother = build_smoother(cfg)
    if smoother is not None:
        print(f"Training WITH smoothing augmentation: mode={aug.mode}  σ={aug.sigma}")
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

    # ── Build GPU PCA tensors for manifold noise (once, before epoch loop) ──
    # kNN+SVD is computed once here.  Every training step then samples fresh
    # noise with cheap batched GPU ops (whiten → Gaussian → unwhiten).
    gpu_cache: Optional[dict] = None
    if aug.enabled and aug.mode == "manifold":
        assert isinstance(smoother, ManifoldSmoother)
        if aug.pca_cache_path is not None:
            if not Path(aug.pca_cache_path).exists():
                raise FileNotFoundError(f"pca_cache_path not found: {aug.pca_cache_path}")
            gpu_cache = smoother.build_gpu_cache_from_npz(
                data.train_loader.dataset, aug.pca_cache_path, device
            )
        elif aug.use_pca_cache:
            gpu_cache = smoother.build_gpu_cache(
                data.train_loader.dataset, device
            )
        else:
            print("PCA cache disabled (use_pca_cache=False) — kNN+SVD will run per step (slow).")
        # Enable integer-index return so batches can index directly into GPU tensors
        data.train_loader.dataset.return_index = True

    # Determine effective aug mode (None = no augmentation)
    aug_mode_active: Optional[str] = aug.mode if aug.enabled else None

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
    history_path = output_dir / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

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
            aug_mode=aug_mode_active,
            sigma=aug.sigma,
            gpu_cache=gpu_cache,
        )
        val_loss, val_acc = run_epoch(
            model, data.val_loader, optimizer, criterion, device,
            train=False, log_every=999,
        )

        print(f"  Train loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val   loss={val_loss:.4f}  acc={val_acc:.4f}")

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)

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
