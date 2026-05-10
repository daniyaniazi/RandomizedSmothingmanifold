"""CelebA / CelebA-HQ Certification Experiment.

Run randomized smoothing certification on smile classifiers.
Supports pixel-space and latent-space (VAE) smoothing.

Output Directory Structure:
    output/smile_classification/{celeba|celebahq}/
    ├── index/
    │   ├── pixel/annoy/euclidean/
    │   └── latent/annoy/euclidean/
    └── certify/
        ├── pixel_manifold/sigma_0_50/
        └── latent_manifold/sigma_0_50/
            ├── metrics.json
            ├── results.csv
            └── visualizations/

Key Design:
    - Index is built from TRAIN split
    - Certification runs on TEST split

Usage:
    python -m src.experiments.certify.celeba --config CONFIG

Examples:
    # Latent space manifold smoothing
    python -m src.experiments.certify.celeba \\
        --config src/configs/experiments/certify_celeba_latent_128.yaml

    # Pixel space manifold smoothing
    python -m src.experiments.certify.celeba \\
        --config src/configs/experiments/certify_celeba_pixel_128.yaml

    # CelebA-HQ
    python -m src.experiments.certify.celeba \\
        --config src/configs/experiments/certify_celebahq_latent_128.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.configs.certify_celeba_io import load_certify_config, save_certify_config
from src.configs.certify_celeba_schema import CertifyConfig
from src.configs.train_smile_schema import SmileDataloaderConfig, SmileDatasetConfig, SmileModelConfig
from src.certify.randomized import certify_token_from_counts, TokenCertificate
from src.dataloaders.celeba_smile import build_smile_dataloaders
from src.indexing.base import load_index, NeighborIndex
from src.indexing.image_index import build_or_load_image_index, ImageIndexArtifacts
from src.models.resnet import build_resnet_classifier
from src.models.VAE import ConvVAE, load_checkpoint as load_vae_checkpoint
from src.smoothing.isotropic import IsotropicSmoother
from src.smoothing.manifold import ManifoldSmoother


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint/Resume Support
# ─────────────────────────────────────────────────────────────────────────────

PARTIAL_STATE_FILE = "results.partial.json"


def _get_partial_state_path(experiment_dir: Path) -> Path:
    """Get path to partial state checkpoint file."""
    return experiment_dir / PARTIAL_STATE_FILE


def _save_partial_state(
    experiment_dir: Path,
    next_idx: int,
    results: List[Dict],
    total_correct: int,
    total_certified: int,
    total_abstained: int,
    radii: List[float],
    num_test_samples: int,
) -> None:
    """Save partial certification state to disk for resumption."""
    state = {
        "next_idx": next_idx,
        "results": results,
        "total_correct": total_correct,
        "total_certified": total_certified,
        "total_abstained": total_abstained,
        "radii": radii,
        "num_test_samples": num_test_samples,
        "timestamp": datetime.now().isoformat(),
    }
    
    path = _get_partial_state_path(experiment_dir)
    path.write_text(json.dumps(state, indent=2))
    _log(f"Checkpoint saved: {path} (processed {next_idx}/{num_test_samples} samples)")


def _load_partial_state(experiment_dir: Path, num_test_samples: int) -> Dict | None:
    """Load partial certification state from disk if available."""
    path = _get_partial_state_path(experiment_dir)
    
    if not path.exists():
        return None
    
    try:
        state = json.loads(path.read_text())
        
        # Validate state consistency
        if state.get("num_test_samples") != num_test_samples:
            _log(f"Warning: Partial state has {state.get('num_test_samples')} samples, "
                 f"but current run has {num_test_samples}. Ignoring checkpoint.")
            return None
        
        _log(f"Found checkpoint: {path}")
        _log(f"  - Processed: {state['next_idx']}/{num_test_samples} samples")
        _log(f"  - Timestamp: {state.get('timestamp', 'unknown')}")
        return state
    except (json.JSONDecodeError, KeyError) as e:
        _log(f"Warning: Failed to load partial state: {e}. Starting fresh.")
        return None


def _remove_partial_state(experiment_dir: Path) -> None:
    """Remove partial state file after successful completion."""
    path = _get_partial_state_path(experiment_dir)
    if path.exists():
        path.unlink()
        _log(f"Removed checkpoint file: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Directory Structure
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CertifyPaths:
    """Organized output directory structure."""
    base_dir: Path
    dataset_dir: Path
    index_dir: Path
    certify_dir: Path
    pixel_index_dir: Path
    latent_index_dir: Path
    experiment_dir: Path
    
    @classmethod
    def from_config(cls, cfg: CertifyConfig) -> "CertifyPaths":
        """Build directory structure from config."""
        dataset_name = cfg.dataset.name.lower().replace("-", "").replace("_", "")
        base_dir = Path(cfg.output.output_dir) / "smile_classification" / dataset_name
        
        # Index: index/{pixel|latent}/{backend}/{metric}/
        index_base = base_dir / "index"
        pixel_index_dir = index_base / "pixel" / cfg.index.backend / cfg.index.metric
        latent_index_dir = index_base / "latent" / cfg.index.backend / cfg.index.metric
        
        # Certify: certify/{mode}/sigma_{sigma}/
        sigma_tag = f"sigma_{cfg.smoothing.sigma:.2f}".replace(".", "_")
        mode_tag = f"{cfg.smoothing.mode}_{'manifold' if cfg.smoothing.use_manifold else 'isotropic'}"
        experiment_dir = base_dir / "certify" / mode_tag / sigma_tag
        
        return cls(
            base_dir=base_dir,
            dataset_dir=base_dir / "dataset",
            index_dir=index_base,
            certify_dir=base_dir / "certify",
            pixel_index_dir=pixel_index_dir,
            latent_index_dir=latent_index_dir,
            experiment_dir=experiment_dir,
        )
    
    def ensure_dirs(self):
        for d in [self.dataset_dir, self.pixel_index_dir, self.latent_index_dir, self.experiment_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset - Uses existing dataloader
# ─────────────────────────────────────────────────────────────────────────────


CELEBA_MEAN = [0.5, 0.5, 0.5]
CELEBA_STD = [0.5, 0.5, 0.5]


def get_train_test_samples(cfg: CertifyConfig) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Get train and test samples using existing dataloader logic."""
    dataset_cfg = SmileDatasetConfig(
        name=cfg.dataset.name,
        root_dir=cfg.dataset.root_dir,
        image_dir=cfg.dataset.image_dir,
        annotation_file=cfg.dataset.annotation_file,
        annotation_format=cfg.dataset.annotation_format,
        image_column="image",
        label_column="smile",
        file_extension=cfg.dataset.file_extension,
        num_workers=cfg.dataset.num_workers,
        train_ratio=0.8,
        val_ratio=0.1,
        split_seed=cfg.seed,
    )
    
    loader_cfg = SmileDataloaderConfig(batch_size=64, shuffle_train=False, pin_memory=True)
    model_cfg = SmileModelConfig(name=cfg.model.name, pretrained=False, dropout=0.0, input_size=cfg.model.input_size)
    
    bundle = build_smile_dataloaders(dataset_cfg, loader_cfg, model_cfg)
    train_samples = list(bundle.train_loader.dataset.samples)
    test_samples = list(bundle.test_loader.dataset.samples)
    
    _log(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
    return train_samples, test_samples


# ─────────────────────────────────────────────────────────────────────────────
# Index Building (uses generic utilities from src/indexing/image_index)
# ─────────────────────────────────────────────────────────────────────────────


def _create_image_dataloader(
    samples: List[Tuple[str, int]],
    image_size: int,
    batch_size: int = 32,
) -> torch.utils.data.DataLoader:
    """Create a simple dataloader from (path, label) samples."""
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleImageDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
            return {"image": img_tensor, "label": label, "path": path}
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
    ])
    
    dataset = SimpleImageDataset(samples, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def load_or_build_pixel_index(
    train_samples: List[Tuple[str, int]],
    image_size: int,
    index_dir: Path,
    n_trees: int = 50,
    force_rebuild: bool = False,
) -> NeighborIndex:
    """Build or load pixel-space index using generic utilities."""
    index_path = index_dir / "index.ann"
    
    # Check if index exists
    if index_path.exists() and not force_rebuild:
        dim = 3 * image_size * image_size
        _log(f"Loading existing pixel index: {index_path}")
        return load_index(dim=dim, index_path=str(index_path), backend="annoy")
    
    # Build using generic utilities
    index_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Building pixel index from {len(train_samples)} samples...")
    
    dataloader = _create_image_dataloader(train_samples, image_size)
    
    artifacts = build_or_load_image_index(
        out_dir=index_dir,
        dataloader=dataloader,
        space="pixel",
        backend="annoy",
        metric="euclidean",
        index_path=str(index_path),
        n_trees=n_trees,
        rebuild=force_rebuild,
        metadata={"image_size": image_size, "split": "train", "num_items": len(train_samples)},
    )
    
    _log(f"Pixel index saved: {index_path} ({len(train_samples)} items)")
    return artifacts.index


def load_or_build_latent_index(
    train_samples: List[Tuple[str, int]],
    vae: ConvVAE,
    index_dir: Path,
    n_trees: int = 50,
    device: torch.device = torch.device("cuda"),
    force_rebuild: bool = False,
) -> NeighborIndex:
    """Build or load latent-space index using generic utilities."""
    index_path = index_dir / "index.ann"
    
    # Check if index exists
    if index_path.exists() and not force_rebuild:
        _log(f"Loading existing latent index: {index_path}")
        return load_index(dim=vae.latent_dim, index_path=str(index_path), backend="annoy")
    
    # Build using generic utilities
    index_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Building latent index from {len(train_samples)} samples...")
    
    dataloader = _create_image_dataloader(train_samples, vae.image_size)
    
    # Create encoder function
    vae.eval()
    def vae_encoder(images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            images = images.to(device)
            if images.shape[-1] != vae.image_size:
                images = torch.nn.functional.interpolate(
                    images, size=vae.image_size, mode="bilinear", align_corners=False
                )
            mu, _ = vae.encode(images)
            return mu.cpu()
    
    artifacts = build_or_load_image_index(
        out_dir=index_dir,
        dataloader=dataloader,
        space="latent",
        encoder=vae_encoder,
        backend="annoy",
        metric="euclidean",
        index_path=str(index_path),
        n_trees=n_trees,
        rebuild=force_rebuild,
        metadata={"latent_dim": vae.latent_dim, "split": "train", "num_items": len(train_samples)},
    )
    
    _log(f"Latent index saved: {index_path} ({len(train_samples)} items)")
    return artifacts.index


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing Helpers (use generic smoothers from src/smoothing)
# ─────────────────────────────────────────────────────────────────────────────


def create_pixel_smoother(
    cfg: CertifyConfig,
    index: Optional[NeighborIndex] = None,
) -> IsotropicSmoother | ManifoldSmoother:
    """Create pixel-space smoother based on config."""
    if cfg.smoothing.use_manifold and index is not None:
        return ManifoldSmoother(
            sigma=cfg.smoothing.sigma,
            index=index,
            knn_k=cfg.smoothing.knn_k,
            eps_eig=cfg.smoothing.eps_eig,
        )
    return IsotropicSmoother(sigma=cfg.smoothing.sigma)


def create_latent_smoother(
    cfg: CertifyConfig,
    index: Optional[NeighborIndex] = None,
) -> IsotropicSmoother | ManifoldSmoother:
    """Create latent-space smoother based on config."""
    if cfg.smoothing.use_manifold and index is not None:
        return ManifoldSmoother(
            sigma=cfg.smoothing.sigma,
            index=index,
            knn_k=cfg.smoothing.knn_k,
            eps_eig=cfg.smoothing.eps_eig,
        )
    return IsotropicSmoother(sigma=cfg.smoothing.sigma)


def sample_pixel(
    img_tensor: torch.Tensor,
    smoother: IsotropicSmoother | ManifoldSmoother,
) -> torch.Tensor:
    """Sample noisy pixel-space image using smoother."""
    flat = img_tensor.numpy().flatten().astype(np.float32)
    noisy_flat = smoother.sample(flat)
    return torch.from_numpy(noisy_flat.reshape(img_tensor.shape)).float()


def sample_latent(
    img_tensor: torch.Tensor,
    vae: ConvVAE,
    smoother: IsotropicSmoother | ManifoldSmoother,
    device: torch.device,
) -> torch.Tensor:
    """Sample noisy latent-space image using VAE + smoother."""
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        mu, _ = vae.encode(x)
        z = mu.squeeze(0).cpu().numpy().astype(np.float32)
    
    z_noised = smoother.sample(z)
    
    with torch.no_grad():
        z_t = torch.from_numpy(z_noised[None, :]).to(device=device, dtype=torch.float32)
        x_hat = vae.decode(z_t)
    return x_hat.squeeze(0).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────


def _tensor_to_pil(tensor: torch.Tensor, mean: List[float] = None, std: List[float] = None) -> Image.Image:
    """Convert normalized tensor (C,H,W) to PIL Image."""
    if mean is None:
        mean = CELEBA_MEAN
    if std is None:
        std = CELEBA_STD
    
    # Denormalize
    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    
    # Convert to PIL
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def _get_nn_images(
    index: NeighborIndex,
    query_vec: np.ndarray,
    n: int,
    img_shape: tuple,
    vae: ConvVAE | None,
    device: torch.device,
    is_latent: bool,
) -> List[torch.Tensor]:
    """Retrieve decoded neighbor images from the kNN index."""
    imgs = []
    if index is not None and hasattr(index.index, "get_nns_by_vector"):
        nn_ids = index.index.get_nns_by_vector(query_vec.tolist(), n + 1)
        for nn_id in nn_ids[:n]:
            nn_vec = np.array(index.index.get_item_vector(nn_id), dtype=np.float32)
            if is_latent and vae is not None:
                with torch.no_grad():
                    z_t = torch.from_numpy(nn_vec[None, :]).to(device=device, dtype=torch.float32)
                    nn_img = vae.decode(z_t).squeeze(0).cpu()
                imgs.append(nn_img)
            else:
                imgs.append(torch.from_numpy(nn_vec.reshape(img_shape)).float())
    return imgs


def save_sample_visualization(
    viz_dir: Path,
    sample_idx: int,
    img_tensor: torch.Tensor,
    label: int,
    pred: int,
    radius: float,
    abstained: bool,
    index: NeighborIndex,
    vae: ConvVAE | None,
    cfg,
    device: torch.device,
    pixel_smoother: IsotropicSmoother | ManifoldSmoother | None = None,
    latent_smoother: IsotropicSmoother | ManifoldSmoother | None = None,
    n_noisy_samples: int = 5,
) -> None:
    """Save visualization grid for a single sample.

    Layout varies by mode:

    **Pixel Isotropic** (mode=pixel, use_manifold=False) — 2 rows:
        Row 1: Original (+ empty)
        Row 2: Gaussian noise samples in pixel space

    **Latent Isotropic** (mode=latent, use_manifold=False) — 3 rows:
        Row 1: Original (+ empty)
        Row 2: Isotropic noise in latent space (decoded)
        Row 3: Same samples shown as if isotropic noise were in pixel space

    **Pixel Manifold** (mode=pixel, use_manifold=True) — 4 rows (Jonas-style):
        Row 1: Original | PCA Reconstruction
        Row 2: Samples with noise in whitened space
        Row 3: Samples with Gaussian noise in original pixel space
        Row 4: k-NN Neighbors

    **Latent Manifold** (mode=latent, use_manifold=True) — 6 rows:
        Row 1: Original | PCA Reconstruction (Latent)
        Row 2: Latent samples with noise in whitened space (decoded)
        Row 3: Pixel samples with noise in whitened space (for comparison)
        Row 4: Latent samples with Gaussian noise (decoded)
        Row 5: Pixel samples with Gaussian noise (for comparison)
        Row 6: k-NN Neighbors
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        _log("matplotlib not available, skipping visualization")
        return

    viz_dir.mkdir(parents=True, exist_ok=True)

    is_latent = cfg.smoothing.mode == "latent" and vae is not None
    is_manifold = cfg.smoothing.use_manifold

    # Isotropic smoothers (pixel and latent) for comparison rows
    iso_pixel = IsotropicSmoother(sigma=cfg.smoothing.sigma)
    iso_latent = IsotropicSmoother(sigma=cfg.smoothing.sigma) if is_latent else None

    # Query vector for neighbor lookup
    if is_latent:
        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            mu, _ = vae.encode(x)
            query_vec = mu.squeeze(0).cpu().numpy().astype(np.float32)
    else:
        query_vec = img_tensor.numpy().flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # Helper: draw a row of noisy samples
    # ------------------------------------------------------------------
    def _draw_sample_row(gs, row, title, sample_fn):
        for i in range(n_noisy_samples):
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(_tensor_to_pil(sample_fn()))
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=10)

    # ------------------------------------------------------------------
    # PIXEL ISOTROPIC  (no PCA, no whitening, no neighbors)
    # ------------------------------------------------------------------
    if not is_latent and not is_manifold:
        n_rows = 2
        fig = plt.figure(figsize=(3 * n_noisy_samples, 3 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_noisy_samples, figure=fig, hspace=0.3, wspace=0.1)

        # Row 0: Original
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(_tensor_to_pil(img_tensor))
        ax.set_title("Original", fontsize=10)
        ax.axis("off")
        for i in range(1, n_noisy_samples):
            fig.add_subplot(gs[0, i]).axis("off")

        # Row 1: Gaussian noise in pixel space
        _draw_sample_row(gs, 1, "Samples with Gaussian noise in pixel space",
                         lambda: sample_pixel(img_tensor, iso_pixel))

    # ------------------------------------------------------------------
    # LATENT ISOTROPIC  (no PCA, no whitening, no neighbors)
    # ------------------------------------------------------------------
    elif is_latent and not is_manifold:
        n_rows = 3
        fig = plt.figure(figsize=(3 * n_noisy_samples, 3 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_noisy_samples, figure=fig, hspace=0.3, wspace=0.1)

        # Row 0: Original
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(_tensor_to_pil(img_tensor))
        ax.set_title("Original", fontsize=10)
        ax.axis("off")
        for i in range(1, n_noisy_samples):
            fig.add_subplot(gs[0, i]).axis("off")

        # Row 1: Isotropic noise in latent space (decoded)
        _draw_sample_row(gs, 1, "Samples with isotropic noise in latent space",
                         lambda: sample_latent(img_tensor, vae, iso_latent, device))

        # Row 2: Isotropic noise in pixel space (for visual comparison)
        _draw_sample_row(gs, 2, "Samples with isotropic noise in pixel space",
                         lambda: sample_pixel(img_tensor, iso_pixel))

    # ------------------------------------------------------------------
    # PIXEL MANIFOLD  (Jonas-style: PCA whitened + Gaussian + neighbors)
    # ------------------------------------------------------------------
    elif not is_latent and is_manifold:
        manifold_sm = pixel_smoother if pixel_smoother is not None else iso_pixel
        n_rows = 4
        fig = plt.figure(figsize=(3 * n_noisy_samples, 3 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_noisy_samples, figure=fig, hspace=0.3, wspace=0.1)

        # Row 0: Original + PCA Reconstruction
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(_tensor_to_pil(img_tensor))
        ax.set_title("Original", fontsize=10)
        ax.axis("off")

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(_tensor_to_pil(img_tensor))
        ax.set_title("PCA Reconstruction", fontsize=10)
        ax.axis("off")
        for i in range(2, n_noisy_samples):
            fig.add_subplot(gs[0, i]).axis("off")

        # Row 1: Whitened space noise (manifold smoother)
        _draw_sample_row(gs, 1, "Samples with noise in whitened space",
                         lambda: sample_pixel(img_tensor, manifold_sm))

        # Row 2: Gaussian noise in original pixel space
        _draw_sample_row(gs, 2, "Samples with noise in original space",
                         lambda: sample_pixel(img_tensor, iso_pixel))

        # Row 3: Neighbors
        nn_imgs = _get_nn_images(index, query_vec, n_noisy_samples,
                                 img_tensor.shape, vae, device, is_latent=False)
        for i in range(n_noisy_samples):
            ax = fig.add_subplot(gs[3, i])
            if i < len(nn_imgs):
                ax.imshow(_tensor_to_pil(nn_imgs[i]))
            ax.axis("off")
            if i == 0:
                ax.set_title("Neighbors", fontsize=10)

    # ------------------------------------------------------------------
    # LATENT MANIFOLD  (Jonas-style in latent + pixel comparison rows)
    # ------------------------------------------------------------------
    else:  # is_latent and is_manifold
        manifold_sm_latent = latent_smoother if latent_smoother is not None else iso_latent
        manifold_sm_pixel = pixel_smoother if pixel_smoother is not None else iso_pixel

        n_rows = 6
        fig = plt.figure(figsize=(3 * n_noisy_samples, 3 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_noisy_samples, figure=fig, hspace=0.3, wspace=0.1)

        # Row 0: Original + PCA Reconstruction (Latent)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(_tensor_to_pil(img_tensor))
        ax.set_title("Original", fontsize=10)
        ax.axis("off")

        ax = fig.add_subplot(gs[0, 1])
        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            mu, _ = vae.encode(x)
            x_recon = vae.decode(mu).squeeze(0).cpu()
        ax.imshow(_tensor_to_pil(x_recon))
        ax.set_title("PCA Reconstruction (Latent)", fontsize=10)
        ax.axis("off")
        for i in range(2, n_noisy_samples):
            fig.add_subplot(gs[0, i]).axis("off")

        # Row 1: Latent whitened space noise (manifold smoother, decoded)
        _draw_sample_row(gs, 1, "Samples with noise in whitened latent space",
                         lambda: sample_latent(img_tensor, vae, manifold_sm_latent, device))

        # Row 2: Pixel whitened space noise (for comparison)
        _draw_sample_row(gs, 2, "Samples with noise in whitened pixel space",
                         lambda: sample_pixel(img_tensor, manifold_sm_pixel))

        # Row 3: Latent Gaussian noise (decoded)
        _draw_sample_row(gs, 3, "Samples with Gaussian noise in latent space",
                         lambda: sample_latent(img_tensor, vae, iso_latent, device))

        # Row 4: Pixel Gaussian noise (for comparison)
        _draw_sample_row(gs, 4, "Samples with Gaussian noise in pixel space",
                         lambda: sample_pixel(img_tensor, iso_pixel))

        # Row 5: Neighbors
        nn_imgs = _get_nn_images(index, query_vec, n_noisy_samples,
                                 img_tensor.shape, vae, device, is_latent=True)
        for i in range(n_noisy_samples):
            ax = fig.add_subplot(gs[5, i])
            if i < len(nn_imgs):
                ax.imshow(_tensor_to_pil(nn_imgs[i]))
            ax.axis("off")
            if i == 0:
                ax.set_title("Neighbors", fontsize=10)

    # ------------------------------------------------------------------
    # Title with certification result
    # ------------------------------------------------------------------
    cert_status = "ABSTAIN" if abstained else ("CORRECT" if pred == label else "WRONG")
    pred_label = "smile" if pred == 1 else "no smile"
    true_label = "smile" if label == 1 else "no smile"
    smoothing_type = "Manifold" if cfg.smoothing.use_manifold else "Isotropic"
    fig.suptitle(
        f"Sample {sample_idx} | {cfg.smoothing.mode.capitalize()} {smoothing_type} | "
        f"σ={cfg.smoothing.sigma} | True: {true_label} | Pred: {pred_label} | "
        f"Radius: {radius:.4f} | {cert_status}",
        fontsize=11, fontweight="bold",
    )

    # Save
    fig_path = viz_dir / f"sample_{sample_idx:04d}.png"
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Certification
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def certify_single_sample(
    classifier: torch.nn.Module,
    sample_fn: Callable[[], torch.Tensor],
    n_samples: int,
    classifier_transform: transforms.Compose,
    device: torch.device,
    alpha_conf: float = 0.001,
    sigma: float = 0.25,
) -> TokenCertificate:
    classifier.eval()
    class_counts = np.zeros(2, dtype=np.int64)
    
    for _ in range(n_samples):
        noised_img = sample_fn()
        noised_clipped = noised_img.clamp(-1, 1) * 0.5 + 0.5
        noised_pil = transforms.ToPILImage()(noised_clipped)
        x = classifier_transform(noised_pil).unsqueeze(0).to(device)
        
        logit = classifier(x).squeeze()
        pred = 1 if torch.sigmoid(logit).item() > 0.5 else 0
        class_counts[pred] += 1
    
    return certify_token_from_counts(
        class_counts,
        alpha_noise=sigma,
        alpha_conf=alpha_conf,
        abstain_label=-1,
    )


def run_certification(cfg: CertifyConfig) -> Dict:
    """Run full certification pipeline."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")
    
    paths = CertifyPaths.from_config(cfg)
    paths.ensure_dirs()
    
    _log(f"Output structure:")
    _log(f"  Base:    {paths.base_dir}")
    _log(f"  Index:   {paths.index_dir}")
    _log(f"  Certify: {paths.experiment_dir}")
    
    save_certify_config(cfg, paths.experiment_dir / "config.yaml")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load train/test samples using existing dataloader
    # ─────────────────────────────────────────────────────────────────────────
    train_samples, test_samples = get_train_test_samples(cfg)
    
    if cfg.dataset.subset_size and cfg.dataset.subset_size < len(test_samples):
        random.shuffle(test_samples)
        test_samples = test_samples[:cfg.dataset.subset_size]
        test_samples.sort(key=lambda x: x[0])
        _log(f"Using subset of {len(test_samples)} test samples")
    
    dataset_info = {
        "dataset": cfg.dataset.name,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "seed": cfg.seed,
    }
    (paths.dataset_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load classifier
    # ─────────────────────────────────────────────────────────────────────────
    classifier = build_resnet_classifier(
        name=cfg.model.name,
        pretrained=False,
        dropout=cfg.model.dropout,  # must match training config
        num_classes=cfg.model.num_classes,
    ).to(device)
    
    ckpt = torch.load(cfg.model.checkpoint_path, map_location=device)
    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        classifier.load_state_dict(ckpt["model_state_dict"])
    elif "model_state" in ckpt:
        classifier.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and "conv1.weight" not in ckpt:
        # Try to find state dict in common keys
        for key in ["state_dict", "model"]:
            if key in ckpt:
                classifier.load_state_dict(ckpt[key])
                break
        else:
            raise ValueError(f"Cannot find model weights in checkpoint. Keys: {list(ckpt.keys())}")
    else:
        classifier.load_state_dict(ckpt)
    classifier.eval()
    _log(f"Classifier: {cfg.model.checkpoint_path}")
    
    classifier_transform = transforms.Compose([
        transforms.Resize((cfg.model.input_size, cfg.model.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load VAE (if latent mode)
    # ─────────────────────────────────────────────────────────────────────────
    vae = None
    if cfg.vae.enabled and cfg.smoothing.mode in ("latent", "both"):
        vae = ConvVAE(
            in_channels=cfg.vae.in_channels,
            image_size=cfg.vae.image_size,
            latent_dim=cfg.vae.latent_dim,
        ).to(device)
        load_vae_checkpoint(vae, cfg.vae.checkpoint_path, device)
        vae.eval()
        _log(f"VAE: {cfg.vae.checkpoint_path}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build or load indices (from TRAIN set)
    # ─────────────────────────────────────────────────────────────────────────
    pixel_index = None
    latent_index = None
    pixel_size = cfg.vae.image_size if cfg.vae.enabled else 128
    
    # Build pixel index for pixel mode OR for latent manifold mode (needed for comparison viz rows)
    if cfg.smoothing.use_manifold:
        pixel_index = load_or_build_pixel_index(
            train_samples, pixel_size, paths.pixel_index_dir, cfg.index.n_trees
        )
    
    if cfg.smoothing.mode in ("latent", "both") and cfg.smoothing.use_manifold and vae is not None:
        latent_index = load_or_build_latent_index(
            train_samples, vae, paths.latent_index_dir, cfg.index.n_trees, device
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Transform for smoothing
    # ─────────────────────────────────────────────────────────────────────────
    smooth_size = vae.image_size if vae is not None else pixel_size
    smooth_transform = transforms.Compose([
        transforms.Resize((smooth_size, smooth_size)),
        transforms.ToTensor(),
        transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
    ])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Create smoothers (using generic classes from src/smoothing)
    # ─────────────────────────────────────────────────────────────────────────
    pixel_smoother = create_pixel_smoother(cfg, pixel_index)
    latent_smoother = create_latent_smoother(cfg, latent_index)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Run certification on TEST set (with checkpoint support)
    # ─────────────────────────────────────────────────────────────────────────
    _log(f"Certifying {len(test_samples)} TEST samples")
    _log(f"Mode: {cfg.smoothing.mode}, Manifold: {cfg.smoothing.use_manifold}, Sigma: {cfg.smoothing.sigma}")
    
    # Initialize state (or resume from checkpoint)
    results: List[Dict] = []
    total_correct = 0
    total_certified = 0
    total_abstained = 0
    radii: List[float] = []
    start_idx = 0
    
    # Check for existing checkpoint if resume is enabled
    if cfg.checkpoint.resume:
        partial_state = _load_partial_state(paths.experiment_dir, len(test_samples))
        if partial_state is not None:
            start_idx = partial_state["next_idx"]
            results = partial_state["results"]
            total_correct = partial_state["total_correct"]
            total_certified = partial_state["total_certified"]
            total_abstained = partial_state["total_abstained"]
            radii = partial_state["radii"]
            _log(f"Resuming from sample {start_idx}/{len(test_samples)}")
    
    if start_idx > 0:
        _log(f"Skipping first {start_idx} already-processed samples")
    
    # Certification loop with checkpointing
    checkpoint_every = cfg.checkpoint.checkpoint_every if cfg.checkpoint.enabled else 0
    
    for idx in tqdm(range(start_idx, len(test_samples)), desc="Certifying (test)", initial=start_idx, total=len(test_samples)):
        img_path, label = test_samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = smooth_transform(img)
        
        # Create sample function using generic smoothers
        if cfg.smoothing.mode == "pixel":
            sample_fn = lambda t=img_tensor, s=pixel_smoother: sample_pixel(t, s)
        elif cfg.smoothing.mode == "latent" and vae is not None:
            sample_fn = lambda t=img_tensor, s=latent_smoother, v=vae, d=device: sample_latent(t, v, s, d)
        else:
            # Default to pixel isotropic
            sample_fn = lambda t=img_tensor, s=pixel_smoother: sample_pixel(t, s)
        
        cert = certify_single_sample(
            classifier=classifier,
            sample_fn=sample_fn,
            n_samples=cfg.smoothing.n_samples,
            classifier_transform=classifier_transform,
            device=device,
            alpha_conf=cfg.alpha_conf,
            sigma=cfg.smoothing.sigma,
        )
        
        result = {
            "idx": idx,
            "image_path": img_path,
            "label": label,
            "pred": cert.pred,
            "radius": cert.radius,
            "abstained": cert.abstained,
            "p_a_lower": cert.p_a_lower,
            "p_b_upper": cert.p_b_upper,
            "correct": (cert.pred == label) if not cert.abstained else False,
            "certified_correct": (cert.pred == label and not cert.abstained),
        }
        results.append(result)
        
        if cert.abstained:
            total_abstained += 1
        else:
            total_certified += 1
            radii.append(cert.radius)
            if cert.pred == label:
                total_correct += 1
        
        # Save visualization for selected samples
        if cfg.output.save_visualizations and idx < cfg.output.num_viz_samples:
            viz_dir = paths.experiment_dir / "visualizations"
            current_index = latent_index if cfg.smoothing.mode == "latent" else pixel_index
            save_sample_visualization(
                viz_dir=viz_dir,
                sample_idx=idx,
                img_tensor=img_tensor,
                label=label,
                pred=cert.pred,
                radius=cert.radius,
                abstained=cert.abstained,
                index=current_index,
                vae=vae,
                cfg=cfg,
                device=device,
                pixel_smoother=pixel_smoother,
                latent_smoother=latent_smoother,
            )
        
        # Save checkpoint periodically
        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
            _save_partial_state(
                experiment_dir=paths.experiment_dir,
                next_idx=idx + 1,
                results=results,
                total_correct=total_correct,
                total_certified=total_certified,
                total_abstained=total_abstained,
                radii=radii,
                num_test_samples=len(test_samples),
            )
    
    # Remove partial state after successful completion
    _remove_partial_state(paths.experiment_dir)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute metrics
    # ─────────────────────────────────────────────────────────────────────────
    total = len(test_samples)
    metrics = {
        "experiment": cfg.experiment_name,
        "dataset": cfg.dataset.name,
        "smoothing_mode": cfg.smoothing.mode,
        "use_manifold": cfg.smoothing.use_manifold,
        "sigma": cfg.smoothing.sigma,
        "n_samples": cfg.smoothing.n_samples,
        "knn_k": cfg.smoothing.knn_k,
        "index_split": "train",
        "certify_split": "test",
        "total_test_samples": total,
        "certified_samples": total_certified,
        "abstained_samples": total_abstained,
        "certified_correct": total_correct,
        "certified_accuracy": total_correct / total if total > 0 else 0.0,
        "abstain_rate": total_abstained / total if total > 0 else 0.0,
        "mean_radius": float(np.mean(radii)) if radii else 0.0,
        "median_radius": float(np.median(radii)) if radii else 0.0,
        "max_radius": float(np.max(radii)) if radii else 0.0,
        "std_radius": float(np.std(radii)) if radii else 0.0,
    }
    
    # Class-wise accuracy
    for cls in [0, 1]:
        cls_name = "smile" if cls == 1 else "no_smile"
        cls_results = [r for r in results if r["label"] == cls]
        cls_correct = sum(1 for r in cls_results if r["certified_correct"])
        cls_radii = [r["radius"] for r in cls_results if not r["abstained"]]
        
        metrics[f"class_{cls_name}_total"] = len(cls_results)
        metrics[f"class_{cls_name}_correct"] = cls_correct
        metrics[f"class_{cls_name}_accuracy"] = cls_correct / len(cls_results) if cls_results else 0.0
        metrics[f"class_{cls_name}_mean_radius"] = float(np.mean(cls_radii)) if cls_radii else 0.0
    
    # Print results
    _log("=" * 70)
    _log(f"CERTIFICATION RESULTS: {cfg.experiment_name}")
    _log("=" * 70)
    _log(f"Dataset:            {cfg.dataset.name}")
    _log(f"Index split:        TRAIN ({len(train_samples)} samples)")
    _log(f"Certify split:      TEST ({total} samples)")
    _log(f"Smoothing:          {cfg.smoothing.mode} ({'manifold' if cfg.smoothing.use_manifold else 'isotropic'})")
    _log(f"Sigma:              {cfg.smoothing.sigma}")
    _log(f"Certified accuracy: {100*metrics['certified_accuracy']:.2f}%")
    _log(f"Abstain rate:       {100*metrics['abstain_rate']:.2f}%")
    _log(f"Mean radius:        {metrics['mean_radius']:.4f}")
    _log(f"Class No-Smile:     {metrics['class_no_smile_correct']}/{metrics['class_no_smile_total']} = {100*metrics['class_no_smile_accuracy']:.2f}%")
    _log(f"Class Smile:        {metrics['class_smile_correct']}/{metrics['class_smile_total']} = {100*metrics['class_smile_accuracy']:.2f}%")
    _log("=" * 70)
    
    # Save results
    if cfg.output.save_results:
        (paths.experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        _log(f"Metrics: {paths.experiment_dir / 'metrics.json'}")
        
        if cfg.output.save_per_sample:
            csv_path = paths.experiment_dir / "results.csv"
            with open(csv_path, "w", newline="") as f:
                fieldnames = ["idx", "image_path", "label", "pred", "radius", "abstained", "correct", "certified_correct"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow({k: r[k] for k in fieldnames})
            _log(f"Results: {csv_path}")
        
        # Save summary visualization
        if cfg.output.save_visualizations:
            _save_summary_visualization(paths.experiment_dir, results, metrics, cfg)
    
    return {"metrics": metrics, "results": results, "paths": paths}


def _save_summary_visualization(out_dir: Path, results: List[Dict], metrics: Dict, cfg) -> None:
    """Save aggregate visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _log("matplotlib not available, skipping summary visualization")
        return
    
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    radii = [r["radius"] for r in results if not r["abstained"]]
    correct_radii = [r["radius"] for r in results if r["certified_correct"]]
    wrong_radii = [r["radius"] for r in results if not r["abstained"] and not r["correct"]]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Radius distribution
    ax = axes[0, 0]
    ax.hist(radii, bins=30, alpha=0.7, label=f"All certified (n={len(radii)})", color="blue")
    ax.axvline(np.mean(radii), color="blue", linestyle="--", label=f"Mean: {np.mean(radii):.4f}")
    ax.set_xlabel("Certified Radius")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Certified Radii")
    ax.legend()
    
    # 2. Correct vs Wrong radius comparison
    ax = axes[0, 1]
    if correct_radii and wrong_radii:
        ax.hist(correct_radii, bins=20, alpha=0.6, label=f"Correct (n={len(correct_radii)})", color="green")
        ax.hist(wrong_radii, bins=20, alpha=0.6, label=f"Wrong (n={len(wrong_radii)})", color="red")
        ax.set_xlabel("Certified Radius")
        ax.set_ylabel("Count")
        ax.set_title("Radius: Correct vs Wrong Predictions")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No wrong predictions", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Radius: Correct vs Wrong")
    
    # 3. Class-wise accuracy bar chart
    ax = axes[1, 0]
    classes = ["No Smile", "Smile"]
    accuracies = [metrics["class_no_smile_accuracy"] * 100, metrics["class_smile_accuracy"] * 100]
    counts = [metrics["class_no_smile_total"], metrics["class_smile_total"]]
    bars = ax.bar(classes, accuracies, color=["coral", "lightgreen"])
    ax.set_ylabel("Certified Accuracy (%)")
    ax.set_title("Class-wise Certified Accuracy")
    ax.set_ylim(0, 100)
    for bar, acc, cnt in zip(bars, accuracies, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f"{acc:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=9)
    
    # 4. Summary text
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = f"""
    Experiment: {cfg.experiment_name}
    Dataset: {cfg.dataset.name}
    
    Smoothing: {cfg.smoothing.mode} ({'manifold' if cfg.smoothing.use_manifold else 'isotropic'})
    Sigma: {cfg.smoothing.sigma}
    K-NN: {cfg.smoothing.knn_k}
    MC Samples: {cfg.smoothing.n_samples}
    
    Total Test Samples: {metrics['total_test_samples']}
    Certified: {metrics['certified_samples']}
    Abstained: {metrics['abstained_samples']} ({metrics['abstain_rate']*100:.1f}%)
    
    Certified Accuracy: {metrics['certified_accuracy']*100:.2f}%
    Mean Radius: {metrics['mean_radius']:.4f}
    Median Radius: {metrics['median_radius']:.4f}
    Max Radius: {metrics['max_radius']:.4f}
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    fig_path = viz_dir / "summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Summary visualization: {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="CelebA Certification Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_certify_config(args.config)
    run_certification(cfg)
