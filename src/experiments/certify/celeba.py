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
        --config src/configs/experiments/certify_celeba_pixel.yaml

    # CelebA-HQ
    python -m src.experiments.certify.celeba \\
        --config src/configs/experiments/certify_celebahq_latent.yaml
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.configs.certify_celeba_io import load_certify_config, save_certify_config
from src.configs.certify_celeba_schema import CertifyConfig
from src.configs.train_smile_schema import SmileDataloaderConfig, SmileDatasetConfig, SmileModelConfig
from src.certify.randomized import certify_token_from_counts_two_stage_paper, TokenCertificate
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
# Checkpoint path resolution
# ─────────────────────────────────────────────────────────────────────────────


def resolve_classifier_checkpoint(cfg) -> str:
    """Return the path to the classifier checkpoint.

    If cfg.model.use_smoothed_classifier is True the path is auto-built as:
        {base_dir}/{mode}/{dataset}/smile_resnet_{dataset}_sigma_{s}/best.pt

    where
        mode    = "manifold" or "isotropic"  (from cfg.smoothing.use_manifold)
        dataset = cfg.dataset.name lowercased / cleaned  (e.g. "celeba", "celebahq")
        s       = cfg.smoothing.sigma formatted as "0_25"

    Otherwise cfg.model.checkpoint_path is returned unchanged.
    """
    if not cfg.model.use_smoothed_classifier:
        return cfg.model.checkpoint_path

    mode = "manifold" if cfg.smoothing.use_manifold else "isotropic"
    dataset_tag = cfg.dataset.name.lower().replace("-", "").replace("_", "")
    sigma_tag = f"sigma_{cfg.smoothing.sigma:.2f}".replace(".", "_")
    model_name = cfg.model.name  # e.g. "resnet18"

    ckpt = (
        Path(cfg.model.smoothed_classifier_base_dir)
        / mode
        / dataset_tag
        / f"smile_{model_name}_{dataset_tag}_{sigma_tag}"
        / "best.pt"
    )
    _log(f"Auto-resolved classifier checkpoint: {ckpt}")
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Smoothed classifier not found: {ckpt}\n"
            f"Train it first with:\n"
            f"  python -m src.experiments.training.resnet_smile.train_smooth_sweep \\\n"
            f"      --config <training_cfg> --modes {mode} "
            f"--sigmas {cfg.smoothing.sigma} "
            f"--base_ckpt_dir {cfg.model.smoothed_classifier_base_dir}"
        )
    return str(ckpt)
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
        
        # Certify output path:
        #   standard : certify/{mode}/sigma_{s}/
        #   OOD      : certify_ood/{attr}/{mode}/sigma_{s}/
        sigma_tag = f"sigma_{cfg.smoothing.sigma:.2f}".replace(".", "_")
        mode_tag = f"{cfg.smoothing.mode}_{'manifold' if cfg.smoothing.use_manifold else 'isotropic'}"
        ood_attr = getattr(cfg.dataset, "ood_attribute", None)
        if ood_attr:
            experiment_dir = base_dir / "certify_ood" / ood_attr.lower() / mode_tag / sigma_tag
        else:
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
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio,
        split_seed=cfg.seed,
    )
    
    loader_cfg = SmileDataloaderConfig(batch_size=64, shuffle_train=False, pin_memory=True)
    model_cfg = SmileModelConfig(name=cfg.model.name, pretrained=False, dropout=0.0, input_size=cfg.model.input_size)
    
    bundle = build_smile_dataloaders(dataset_cfg, loader_cfg, model_cfg)
    train_samples = list(bundle.train_loader.dataset.samples)
    test_samples = list(bundle.test_loader.dataset.samples)
    
    _log(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
    return train_samples, test_samples


def get_ood_test_samples(
    cfg: CertifyConfig,
    test_samples: List[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    """Filter test_samples to an OOD attribute subset.

    If cfg.dataset.ood_attribute is set, keeps only test images where that
    attribute == 1.  If cfg.dataset.ood_balanced is True (default), also
    balances to exactly subset_size/2 smile + subset_size/2 non-smile.

    Returns the filtered (and possibly subsampled) list.
    Falls back to standard subset_size random sampling when ood_attribute is null.
    """
    ood_attr = getattr(cfg.dataset, "ood_attribute", None)
    subset_size = getattr(cfg.dataset, "subset_size", None)

    if not ood_attr:
        # Standard: random subset of test split
        if subset_size and subset_size < len(test_samples):
            import random
            rng = random.Random(cfg.seed)
            test_samples = rng.sample(test_samples, subset_size)
        return test_samples

    # Parse the full CelebA attribute file to get attr values per filename
    import random
    attr_path = Path(cfg.dataset.root_dir) / cfg.dataset.annotation_file
    lines = [l.strip() for l in attr_path.read_text().splitlines() if l.strip()]
    attr_names = lines[1].split()
    if ood_attr not in attr_names:
        raise ValueError(f"OOD attribute '{ood_attr}' not found in {attr_path}. "
                         f"Available: {attr_names}")
    attr_idx = attr_names.index(ood_attr)

    # Build filename -> attr_value map
    attr_map: dict = {}
    for row in lines[2:]:
        parts = row.split()
        fname = parts[0]
        attr_map[fname] = int(parts[1 + attr_idx])  # -1 or 1

    # Filter test samples where attr == 1
    filtered = [
        (path, label) for path, label in test_samples
        if attr_map.get(Path(path).name, -1) == 1
    ]
    _log(f"OOD filter '{ood_attr}': {len(filtered)}/{len(test_samples)} test images match")

    balanced = getattr(cfg.dataset, "ood_balanced", True)
    ood_seed = getattr(cfg.dataset, "ood_seed", cfg.seed)
    n_per_class = (subset_size // 2) if subset_size else None

    if balanced and n_per_class:
        smile    = [s for s in filtered if s[1] == 1]
        nonsmile = [s for s in filtered if s[1] == 0]
        if len(smile) < n_per_class or len(nonsmile) < n_per_class:
            _log(f"WARNING: not enough images for balanced OOD subset "
                 f"(smile={len(smile)}, non-smile={len(nonsmile)}, need {n_per_class} each). "
                 f"Using all available.")
            filtered = smile + nonsmile
        else:
            rng = random.Random(ood_seed)
            filtered = rng.sample(smile, n_per_class) + rng.sample(nonsmile, n_per_class)
        _log(f"OOD balanced subset: {len(filtered)} images "
             f"({sum(s[1] for s in filtered)} smile, {sum(1-s[1] for s in filtered)} non-smile)")
    elif subset_size and len(filtered) > subset_size:
        rng = random.Random(ood_seed)
        filtered = rng.sample(filtered, subset_size)

    return filtered


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
    metric: str = "euclidean",
) -> NeighborIndex:
    """Build or load pixel-space index using generic utilities."""
    index_path = index_dir / "index.ann"
    lock_path = index_dir / "index.lock"
    
    # Filenames list in the same order as vectors were inserted into the index.
    # Attached to the NeighborIndex so OOD lookup code can resolve neighbour IDs → filenames.
    _fnames = [str(p) for p, _ in train_samples]

    # Check if index already exists (fast path, no lock needed)
    if index_path.exists() and not force_rebuild:
        dim = 3 * image_size * image_size
        _log(f"Loading existing pixel index: {index_path}")
        _idx = load_index(dim=dim, index_path=str(index_path), backend="annoy", metric=metric)
        _idx.filenames = _fnames
        return _idx
    
    # Use file lock to prevent parallel jobs from building simultaneously
    index_dir.mkdir(parents=True, exist_ok=True)
    import filelock
    lock = filelock.FileLock(str(lock_path), timeout=7200)  # 2h timeout
    with lock:
        # Re-check after acquiring lock (another job may have built it)
        if index_path.exists() and not force_rebuild:
            dim = 3 * image_size * image_size
            _log(f"Loading existing pixel index (built by another job): {index_path}")
            _idx = load_index(dim=dim, index_path=str(index_path), backend="annoy", metric=metric)
            _idx.filenames = _fnames
            return _idx
        
        _log(f"Building pixel index from {len(train_samples)} samples (metric={metric})...")
        
        dataloader = _create_image_dataloader(train_samples, image_size)
        
        artifacts = build_or_load_image_index(
            out_dir=index_dir,
            dataloader=dataloader,
            space="pixel",
            backend="annoy",
            metric=metric,
            index_path=str(index_path),
            n_trees=n_trees,
            rebuild=force_rebuild,
            metadata={"image_size": image_size, "split": "train", "num_items": len(train_samples)},
        )
        
        _log(f"Pixel index saved: {index_path} ({len(train_samples)} items)")
        artifacts.index.filenames = _fnames
        return artifacts.index


def load_or_build_latent_index(
    train_samples: List[Tuple[str, int]],
    vae: ConvVAE,
    index_dir: Path,
    n_trees: int = 50,
    device: torch.device = torch.device("cuda"),
    force_rebuild: bool = False,
    metric: str = "angular",
) -> NeighborIndex:
    """Build or load latent-space index using generic utilities."""
    index_path = index_dir / "index.ann"
    lock_path = index_dir / "index.lock"

    # Filenames list in insertion order — attached to the index for OOD neighbour lookup.
    _fnames = [str(p) for p, _ in train_samples]
    
    # Check if index already exists (fast path, no lock needed)
    if index_path.exists() and not force_rebuild:
        _log(f"Loading existing latent index: {index_path}")
        _idx = load_index(dim=vae.latent_dim, index_path=str(index_path), backend="annoy", metric=metric)
        _idx.filenames = _fnames
        return _idx
    
    # Use file lock to prevent parallel jobs from building simultaneously
    index_dir.mkdir(parents=True, exist_ok=True)
    import filelock
    lock = filelock.FileLock(str(lock_path), timeout=7200)  # 2h timeout
    with lock:
        # Re-check after acquiring lock
        if index_path.exists() and not force_rebuild:
            _log(f"Loading existing latent index (built by another job): {index_path}")
            _idx = load_index(dim=vae.latent_dim, index_path=str(index_path), backend="annoy", metric=metric)
            _idx.filenames = _fnames
            return _idx
        
        _log(f"Building latent index from {len(train_samples)} samples (metric={metric})...")
        
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
            metric=metric,
            index_path=str(index_path),
            n_trees=n_trees,
            rebuild=force_rebuild,
            metadata={"latent_dim": vae.latent_dim, "split": "train", "num_items": len(train_samples)},
        )
        
        _log(f"Latent index saved: {index_path} ({len(train_samples)} items)")
        artifacts.index.filenames = _fnames
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


def make_pixel_sample_fn(
    img_tensor: torch.Tensor,
    smoother: IsotropicSmoother | ManifoldSmoother,
) -> Callable[[], torch.Tensor]:
    """Create a sample function with cached PCA (avoids recomputing kNN+PCA per sample).
    
    For ManifoldSmoother: computes PCA once, returns closure that samples from cached PCA.
    For IsotropicSmoother: just wraps sample() (already fast, no PCA).
    """
    flat = img_tensor.numpy().flatten().astype(np.float32)
    shape = img_tensor.shape
    
    if isinstance(smoother, ManifoldSmoother):
        cached = smoother.compute_pca(flat)
        def _sample():
            noisy_flat = smoother.sample_from_cached(cached)
            return torch.from_numpy(noisy_flat.reshape(shape)).float()
        return _sample
    else:
        def _sample():
            noisy_flat = smoother.sample(flat)
            return torch.from_numpy(noisy_flat.reshape(shape)).float()
        return _sample


def sample_latent(
    img_tensor: torch.Tensor,
    vae: ConvVAE,
    smoother: IsotropicSmoother | ManifoldSmoother,
    device: torch.device,
) -> torch.Tensor:
    """Sample noisy latent-space image using VAE + smoother."""
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        # Resize to VAE's expected image_size if needed (classifier may use different resolution)
        if x.shape[-1] != vae.image_size or x.shape[-2] != vae.image_size:
            x = F.interpolate(x, size=vae.image_size, mode="bilinear", align_corners=False)
        mu, _ = vae.encode(x)
        z = mu.squeeze(0).cpu().numpy().astype(np.float32)
    
    z_noised = smoother.sample(z)
    
    with torch.no_grad():
        z_t = torch.from_numpy(z_noised[None, :]).to(device=device, dtype=torch.float32)
        x_hat = vae.decode(z_t)
    return x_hat.squeeze(0).cpu()


def make_latent_sample_fn(
    img_tensor: torch.Tensor,
    vae: ConvVAE,
    smoother: IsotropicSmoother | ManifoldSmoother,
    device: torch.device,
) -> Callable[[], torch.Tensor]:
    """Create a sample function with cached PCA for latent space."""
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        if x.shape[-1] != vae.image_size or x.shape[-2] != vae.image_size:
            x = F.interpolate(x, size=vae.image_size, mode="bilinear", align_corners=False)
        mu, _ = vae.encode(x)
        z = mu.squeeze(0).cpu().numpy().astype(np.float32)
    
    if isinstance(smoother, ManifoldSmoother):
        cached = smoother.compute_pca(z)
        def _sample():
            z_noised = smoother.sample_from_cached(cached)
            with torch.no_grad():
                z_t = torch.from_numpy(z_noised[None, :]).to(device=device, dtype=torch.float32)
                x_hat = vae.decode(z_t)
            return x_hat.squeeze(0).cpu()
        return _sample
    else:
        def _sample():
            z_noised = smoother.sample(z)
            with torch.no_grad():
                z_t = torch.from_numpy(z_noised[None, :]).to(device=device, dtype=torch.float32)
                x_hat = vae.decode(z_t)
            return x_hat.squeeze(0).cpu()
        return _sample


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
    ood_attr_map: Optional[dict] = None,  # filename -> 1/0, for OOD neighbour highlighting
) -> None:
    """Save visualization grid for a single sample.

    Layout varies by mode:

    **Pixel Isotropic** (mode=pixel, use_manifold=False) — 2 rows:
        Row 0: Original (+ empty)
        Row 1: Gaussian noise samples in pixel space

    **Latent Isotropic** (mode=latent, use_manifold=False) — 3 rows:
        Row 0: Original (+ empty)
        Row 1: Isotropic noise in latent space (decoded)
        Row 2: Isotropic noise in pixel space

    **Pixel Manifold** (mode=pixel, use_manifold=True) — 5 rows (notebook-style):
        Row 0: Original | PCA Reconstruction
        Row 1: Manifold noise  α = σ/√λ_max  (correct/final — used for certification)
        Row 2: Manifold noise  σ = scale_weight  (unscaled, reference comparison)
        Row 3: Isotropic pixel noise
        Row 4: k-NN Neighbours

    **Latent Manifold** (mode=latent, use_manifold=True) — 7 rows:
        Row 0: Original | PCA Reconstruction (Latent)
        Row 1: Latent manifold noise  α = σ/√λ_max  (correct/final — used for certification)
        Row 2: Latent manifold noise  σ = scale_weight  (unscaled, reference)
        Row 3: Pixel manifold noise  α = σ/√λ_max  (decoded via pixel smoother)
        Row 4: Latent Gaussian noise (decoded)
        Row 5: Pixel Gaussian noise (for comparison)
        Row 6: k-NN Neighbours
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

    sigma = cfg.smoothing.sigma

    # Isotropic smoothers (pixel and latent) for comparison rows
    iso_pixel = IsotropicSmoother(sigma=sigma)
    iso_latent = IsotropicSmoother(sigma=sigma) if is_latent else None

    # Query vector for neighbor lookup
    if is_latent:
        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            if x.shape[-1] != vae.image_size or x.shape[-2] != vae.image_size:
                x = F.interpolate(x, size=vae.image_size, mode="bilinear", align_corners=False)
            mu, _ = vae.encode(x)
            query_vec = mu.squeeze(0).cpu().numpy().astype(np.float32)
    else:
        query_vec = img_tensor.numpy().flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # Helper: add a bold row-ID label to the left of axis [row, 0]
    # ------------------------------------------------------------------
    def _add_row_label(axes_grid, row, label_text):
        ax0 = axes_grid[row, 0]
        ax0.text(
            -0.18, 0.5, label_text,
            transform=ax0.transAxes,
            fontsize=8, fontweight="bold",
            va="center", ha="right",
            rotation=0, clip_on=False,
        )

    # ------------------------------------------------------------------
    # Helper: draw a row of noisy samples
    # ------------------------------------------------------------------
    def _draw_sample_row(axes_grid, row, title, sample_fn):
        for i in range(n_noisy_samples):
            ax = axes_grid[row, i]
            ax.imshow(_tensor_to_pil(sample_fn()))
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=9)

    # ------------------------------------------------------------------
    # PIXEL ISOTROPIC
    # ------------------------------------------------------------------
    if not is_latent and not is_manifold:
        row_labels = ["Row 0\nOriginal", "Row 1\nIsotropic\npixel noise σ"]
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_noisy_samples,
                                 figsize=(3 * n_noisy_samples, 3.2 * n_rows))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")

        for r, lbl in enumerate(row_labels):
            _add_row_label(axes, r, lbl)

        axes[0, 0].imshow(_tensor_to_pil(img_tensor))
        axes[0, 0].set_title("Original", fontsize=9)

        for i in range(n_noisy_samples):
            axes[1, i].imshow(_tensor_to_pil(sample_pixel(img_tensor, iso_pixel)))
            if i == 0:
                axes[1, i].set_title(f"Isotropic pixel noise  σ={sigma}", fontsize=9)

        # ── Isotropic geometry figure: A/B/C + noisy MC cloud ─────────────
        try:
            from matplotlib.patches import Circle as _Circle
            import matplotlib.pyplot as _plt_geom
            from sklearn.decomposition import PCA as _PCA

            # PCA fit on MC noisy samples (iso geometry: no KNN, just noise cloud around anchor)
            _flat = query_vec

            # Generate N Monte Carlo noisy samples in pixel space
            _N_mc = cfg.smoothing.n_samples
            _mc_flat = np.stack([
                _flat + np.random.randn(*_flat.shape).astype(np.float32) * sigma
                for _ in range(_N_mc)
            ])  # (N, D)

            # Fit PCA on the MC cloud itself (centred on anchor)
            _pca2 = _PCA(n_components=2).fit(_mc_flat)
            _anchor_2d = _pca2.transform(_flat.reshape(1, -1))[0]
            _mc_2d = _pca2.transform(_mc_flat)

            _pc1_std = float(np.std(_mc_2d[:, 0]))
            _pc2_std = float(np.std(_mc_2d[:, 1]))
            _zoom_mid = 3 * sigma
            _zoom_tight = sigma
            _zoom_specs = [
                (None,        f"[A] Full cloud  σ/std={sigma/(_pc1_std+1e-12):.3f}"),
                (_zoom_mid,   f"[B] Mid-zoom ±3σ={_zoom_mid:.4f}"),
                (_zoom_tight, f"[C] Tight ±σ={sigma:.4f}"),
            ]

            # ── OOD label per ISO MC sample via nearest pixel-index neighbour ──
            # Each MC sample is assigned the OOD status of its nearest training neighbour.
            # Green = lands in OOD territory (nearest neighbour has ood_attr=1)
            # Teal  = lands in non-OOD territory (nearest neighbour has ood_attr=0)
            _iso_mc_is_ood = np.zeros(len(_mc_flat), dtype=bool)
            if (ood_attr_map is not None and index is not None
                    and hasattr(index, "index") and hasattr(index.index, "get_nns_by_vector")
                    and hasattr(index, "filenames")):
                for _mi, _ms in enumerate(_mc_flat):
                    _nid = index.index.get_nns_by_vector(_ms.tolist(), 1, include_distances=False)
                    if _nid:
                        _fname = Path(index.filenames[_nid[0]]).name
                        _iso_mc_is_ood[_mi] = bool(ood_attr_map.get(_fname, 0))

            _gfig, _gaxes = _plt_geom.subplots(1, 3, figsize=(15, 5), facecolor="white")
            _mc_pad = 0.05 * max(float(np.ptp(_mc_2d[:, 0])), float(np.ptp(_mc_2d[:, 1])), 1e-6)
            # Total counts over all N_mc samples (independent of zoom) — shown in legend
            _n_iso_ood_total     = int(_iso_mc_is_ood.sum())
            _n_iso_not_ood_total = int((~_iso_mc_is_ood).sum())

            for _gax, (_zr, _gtitle) in zip(_gaxes, _zoom_specs):
                if _zr is None:
                    _mask_mc = np.ones(len(_mc_2d), dtype=bool)
                else:
                    _mask_mc = (
                        (np.abs(_mc_2d[:, 0] - _anchor_2d[0]) <= _zr) &
                        (np.abs(_mc_2d[:, 1] - _anchor_2d[1]) <= _zr)
                    )

                # Iso MC noisy samples — coloured by OOD territory of nearest index neighbour
                _iso_vis_ood     = _mask_mc & _iso_mc_is_ood
                _iso_vis_not_ood = _mask_mc & ~_iso_mc_is_ood
                if not _iso_vis_ood.any() and not _iso_vis_not_ood.any():
                    # Fallback: no OOD map available — plain blue
                    _gax.scatter(_mc_2d[_mask_mc, 0], _mc_2d[_mask_mc, 1],
                                 s=6, alpha=0.40, color="#4c78a8", marker="o", linewidths=0,
                                 zorder=3, label=f"Iso MC samples (n={_mask_mc.sum()})")
                else:
                    # Always plot both — legend shows TOTAL count across all N_mc, not just in-zoom
                    _gax.scatter(_mc_2d[_iso_vis_ood, 0], _mc_2d[_iso_vis_ood, 1],
                                 s=6, alpha=0.50, color="#2ca02c", marker="o", linewidths=0,
                                 zorder=3, label=f"MC in OOD territory ({_n_iso_ood_total}/{_N_mc})")
                    _gax.scatter(_mc_2d[_iso_vis_not_ood, 0], _mc_2d[_iso_vis_not_ood, 1],
                                 s=6, alpha=0.50, color="#17becf", marker="o", linewidths=0,
                                 zorder=3, label=f"MC in non-OOD territory ({_n_iso_not_ood_total}/{_N_mc})")

                # iso circle
                _gax.add_patch(_plt_geom.Circle(
                    (_anchor_2d[0], _anchor_2d[1]), sigma,
                    fill=False, edgecolor="tab:blue", linewidth=2,
                    linestyle=(0, (4, 2)), alpha=0.9, zorder=4, label=f"σ-circle r={sigma}",
                ))
                _gax.scatter(_anchor_2d[0], _anchor_2d[1], s=160, marker="*",
                             c="gold", edgecolors="black", linewidths=0.8, zorder=5, label="Anchor")
                if _zr is None:
                    _gax.set_xlim(float(np.min(_mc_2d[:, 0])) - _mc_pad, float(np.max(_mc_2d[:, 0])) + _mc_pad)
                    _gax.set_ylim(float(np.min(_mc_2d[:, 1])) - _mc_pad, float(np.max(_mc_2d[:, 1])) + _mc_pad)
                else:
                    _gax.set_xlim(_anchor_2d[0] - _zr, _anchor_2d[0] + _zr)
                    _gax.set_ylim(_anchor_2d[1] - _zr, _anchor_2d[1] + _zr)
                _gax.set_aspect("equal")
                _gax.set_title(_gtitle, fontsize=9)
                _gax.set_xlabel(f"PC1 (MC std={_pc1_std:.4f})")
                _gax.set_ylabel(f"PC2 (MC std={_pc2_std:.4f})")
                _gax.grid(alpha=0.25)
                _gax.legend(fontsize=7, loc="upper right")

                _ood_tag = f"  |  OOD: {getattr(cfg.dataset, 'ood_attribute', None)}=1" if ood_attr_map is not None and getattr(cfg.dataset, 'ood_attribute', None) else ""
                _gfig.suptitle(
                    f"Isotropic: Circle Geometry (idx={sample_idx})\n"
                    f"σ={sigma}  |  MC samples={_N_mc}  |  PC1 std={_pc1_std:.4f}  |  PC2 std={_pc2_std:.4f}{_ood_tag}",
                    fontsize=11,
                )
                _plt_geom.tight_layout()
                _geom_iso_path = viz_dir / f"sample_{sample_idx:04d}_geometry_iso.png"
                _gfig.savefig(_geom_iso_path, dpi=150, bbox_inches="tight")
                _plt_geom.close(_gfig)
        except Exception as _geom_err_iso:
            _log(f"Isotropic geometry figure skipped for sample {sample_idx}: {_geom_err_iso}")

    # ------------------------------------------------------------------
    # LATENT ISOTROPIC
    # ------------------------------------------------------------------
    elif is_latent and not is_manifold:
        row_labels = [
            "Row 0\nOriginal",
            "Row 1\nIsotropic\nlatent noise σ",
            "Row 2\nIsotropic\npixel noise σ",
        ]
        n_rows = 3
        fig, axes = plt.subplots(n_rows, n_noisy_samples,
                                 figsize=(3 * n_noisy_samples, 3.2 * n_rows))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")

        for r, lbl in enumerate(row_labels):
            _add_row_label(axes, r, lbl)

        axes[0, 0].imshow(_tensor_to_pil(img_tensor))
        axes[0, 0].set_title("Original", fontsize=9)

        for i in range(n_noisy_samples):
            axes[1, i].imshow(_tensor_to_pil(sample_latent(img_tensor, vae, iso_latent, device)))
            axes[2, i].imshow(_tensor_to_pil(sample_pixel(img_tensor, iso_pixel)))
            if i == 0:
                axes[1, i].set_title(f"Isotropic latent noise  σ={sigma}", fontsize=9)
                axes[2, i].set_title(f"Isotropic pixel noise  σ={sigma}", fontsize=9)

    # ------------------------------------------------------------------
    # PIXEL MANIFOLD  — 5-row notebook-style grid
    # ------------------------------------------------------------------
    elif not is_latent and is_manifold:
        manifold_sm = pixel_smoother if pixel_smoother is not None else iso_pixel

        # Compute alpha = σ/√λ_max for display — the smoother already uses this internally
        alpha_display = sigma  # fallback
        if isinstance(manifold_sm, ManifoldSmoother):
            from src.smoothing.pca import whiten, unwhiten
            _cached_pca = manifold_sm.compute_pca(query_vec)
            lambda_max = float(_cached_pca.pca.evals[0])
            alpha_display = sigma / np.sqrt(max(lambda_max, 1e-12))

        row_labels = [
            "Row 0\nOriginal &\nPCA Recon",
            f"Row 1\nManifold noise\nα=σ/√λ_max={alpha_display:.4f}\n(certified)",
            f"Row 2\nManifold noise\nσ={sigma} unscaled\n(ref)",
            f"Row 3\nIsotropic\npixel noise σ={sigma}",
            "Row 4\nNeighbours",
        ]
        n_rows = 5
        fig, axes = plt.subplots(n_rows, n_noisy_samples,
                                 figsize=(3 * n_noisy_samples, 3.5 * n_rows))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")

        for r, lbl in enumerate(row_labels):
            _add_row_label(axes, r, lbl)

        # Row 0: Original + PCA Reconstruction
        axes[0, 0].imshow(_tensor_to_pil(img_tensor))
        axes[0, 0].set_title("Original", fontsize=9)

        if isinstance(manifold_sm, ManifoldSmoother):
            w = whiten(query_vec, _cached_pca.pca)
            recon_vec = unwhiten(w, _cached_pca.pca)
            recon_tensor = torch.from_numpy(recon_vec.reshape(img_tensor.shape)).float()
            axes[0, 1].imshow(_tensor_to_pil(recon_tensor))
        else:
            axes[0, 1].imshow(_tensor_to_pil(img_tensor))
        axes[0, 1].set_title("PCA Reconstruction", fontsize=9)

        # Row 1: Manifold noise α = σ/√λ_max  (correct final — used for certification)
        if isinstance(manifold_sm, ManifoldSmoother):
            def _sample_scaled_pixel():
                w_anchor = whiten(query_vec, _cached_pca.pca)
                noise = np.random.normal(0.0, alpha_display, size=len(w_anchor)).astype(np.float32)
                noisy_flat = unwhiten(w_anchor + noise, _cached_pca.pca)
                return torch.from_numpy(noisy_flat.reshape(img_tensor.shape)).float()

            for i in range(n_noisy_samples):
                axes[1, i].imshow(_tensor_to_pil(_sample_scaled_pixel()))
                if i == 0:
                    axes[1, i].set_title(f"Manifold noise  α=σ/√λ_max={alpha_display:.4f}  (certified)", fontsize=9)
        else:
            for i in range(n_noisy_samples):
                axes[1, i].imshow(_tensor_to_pil(sample_pixel(img_tensor, manifold_sm)))
                if i == 0:
                    axes[1, i].set_title(f"Manifold noise  α=σ/√λ_max={alpha_display:.4f}  (certified)", fontsize=9)

        # Row 2: Manifold noise σ unscaled (reference — add noise with raw σ in whitened space)
        if isinstance(manifold_sm, ManifoldSmoother):
            def _sample_unscaled_pixel():
                w_anchor = whiten(query_vec, _cached_pca.pca)
                noise = np.random.normal(0.0, sigma, size=len(w_anchor)).astype(np.float32)
                noisy_flat = unwhiten(w_anchor + noise, _cached_pca.pca)
                return torch.from_numpy(noisy_flat.reshape(img_tensor.shape)).float()
            for i in range(n_noisy_samples):
                axes[2, i].imshow(_tensor_to_pil(_sample_unscaled_pixel()))
                if i == 0:
                    axes[2, i].set_title(f"Manifold noise  σ={sigma} unscaled  (ref)", fontsize=9)
        else:
            for i in range(n_noisy_samples):
                axes[2, i].imshow(_tensor_to_pil(sample_pixel(img_tensor, iso_pixel)))
                if i == 0:
                    axes[2, i].set_title(f"Isotropic pixel noise  σ={sigma}  (ref)", fontsize=9)

        # Row 3: Isotropic pixel noise
        for i in range(n_noisy_samples):
            axes[3, i].imshow(_tensor_to_pil(sample_pixel(img_tensor, iso_pixel)))
            if i == 0:
                axes[3, i].set_title(f"Isotropic pixel noise  σ={sigma}", fontsize=9)

        # Row 4: Neighbours
        nn_imgs = _get_nn_images(index, query_vec, n_noisy_samples,
                                 img_tensor.shape, vae, device, is_latent=False)
        for i in range(n_noisy_samples):
            if i < len(nn_imgs):
                axes[4, i].imshow(_tensor_to_pil(nn_imgs[i]))
            if i == 0:
                axes[4, i].set_title("Neighbours", fontsize=9)

        # ── Circle/Ellipse geometry figure (like notebook cell 7) ─────────
        if isinstance(manifold_sm, ManifoldSmoother):
            try:
                from matplotlib.patches import Ellipse as _Ellipse
                import matplotlib.pyplot as _plt_geom
                from sklearn.decomposition import PCA as _PCA

                _pca_obj = _cached_pca.pca
                _nbrs = _cached_pca.neighbors  # (k, D) centred
                _ev = np.asarray(_pca_obj.evals, dtype=np.float64)
                _Vt = _pca_obj.evecs.T  # (n_comp, D)
                _ev_norm = np.maximum(_ev, 1e-12) / float(_ev.max())
                _lambda_max = float(_ev[0])
                _sqrt_lambda_max = float(np.sqrt(_lambda_max))
                _anchor_2d = (query_vec - np.asarray(_pca_obj.mean, dtype=np.float64)) @ _Vt[:2].T
                _neigh_2d = _nbrs.astype(np.float64) @ _Vt[:2].T

                _a1 = float(sigma * np.sqrt(_ev_norm[0]))
                _a2 = float(sigma * np.sqrt(_ev_norm[1]))
                _mid_idx = len(_ev_norm) // 2
                _a_mid  = float(sigma * np.sqrt(_ev_norm[_mid_idx]))  # median component
                _a_last = float(sigma * np.sqrt(_ev_norm[-1]))         # true last component
                _pc1_std = float(np.std(_neigh_2d[:, 0]))
                _pc1_range = float(np.max(_neigh_2d[:, 0]) - np.min(_neigh_2d[:, 0]))
                _zoom_mid = 0.10 * _pc1_range
                _zoom_tight = _a1

                _zoom_labels = [
                    (None, f"[A] Full cloud  σ/std={sigma / (_pc1_std + 1e-12):.4f}", _a1, _a2),
                    (_zoom_mid, f"[B] Mid-zoom  ±{_zoom_mid:.2f}", _a1, _a2),
                    (_zoom_tight, f"[C] Tight ±σ={_zoom_tight:.4f}  a2={_a2:.4f}", _a1, _a2),
                    (_zoom_tight, f"[D] Mid PC (k={_mid_idx})  a_mid={_a_mid:.4f}\na_mid/a1={_a_mid/_a1:.4f}", _a1, _a_mid),
                    (_zoom_tight, f"[E] Last PC (k={len(_ev_norm)-1})  a_last={_a_last:.6f}\na_last/a1={_a_last/_a1:.6f}", _a1, _a_last),
                ]

                # ── Generate Monte Carlo noisy samples in manifold space → project to PCA-2D
                _N_mc = cfg.smoothing.n_samples
                _mc_samples_2d = []
                for _ in range(_N_mc):
                    _ns = manifold_sm.sample_from_cached(_cached_pca)
                    _ns_centred = _ns - np.asarray(_pca_obj.mean, dtype=np.float64)
                    _mc_samples_2d.append(_ns_centred @ _Vt[:2].T)
                _mc_2d = np.array(_mc_samples_2d)  # (N_mc, 2)

                # ── OOD flag per neighbour (look up by filename in index) ──
                _nn_has_attr = np.zeros(len(_neigh_2d), dtype=bool)
                if ood_attr_map is not None and index is not None and hasattr(index, "filenames"):
                    _nn_ids_ood = index.index.get_nns_by_vector(
                        query_vec.astype(np.float32),
                        len(_neigh_2d), include_distances=False,
                    ) if hasattr(index.index, "get_nns_by_vector") else []
                    for _ni, _nid in enumerate(_nn_ids_ood[:len(_neigh_2d)]):
                        _fname = Path(index.filenames[_nid]).name if hasattr(index, "filenames") else ""
                        _nn_has_attr[_ni] = bool(ood_attr_map.get(_fname, 0))

                # ── Per-MC-sample OOD label: nearest neighbour in 2D PCA space ──
                # Each MC noisy sample is assigned the OOD status of its closest KNN neighbour.
                # Green  = lands in OOD territory   (nearest KNN has ood_attr=1)
                # Teal   = lands in non-OOD territory (nearest KNN has ood_attr=0)
                if ood_attr_map is not None and len(_neigh_2d) > 0:
                    _mc_nn_dists = np.linalg.norm(
                        _mc_2d[:, None, :] - _neigh_2d[None, :, :], axis=-1
                    )  # (N_mc, k)
                    _mc_nn_idx  = np.argmin(_mc_nn_dists, axis=1)  # (N_mc,)
                    _mc_is_ood  = _nn_has_attr[_mc_nn_idx]          # (N_mc,) bool
                else:
                    _mc_is_ood = np.zeros(len(_mc_2d), dtype=bool)

                _gfig, _gaxes = _plt_geom.subplots(1, 5, figsize=(22, 5.5), facecolor="white")
                _x_all = np.concatenate([_neigh_2d[:, 0], [_anchor_2d[0]]])
                _y_all = np.concatenate([_neigh_2d[:, 1], [_anchor_2d[1]]])
                _pad = 0.05 * max(float(np.max(_x_all) - np.min(_x_all)),
                                  float(np.max(_y_all) - np.min(_y_all)), 1e-6)
                # Total counts over all N_mc samples (independent of zoom) — shown in legend
                _n_ood_mc_total     = int(_mc_is_ood.sum())
                _n_notood_mc_total  = int((~_mc_is_ood).sum())

                for _gax, (_zr, _gtitle, _ea1, _ea2) in zip(_gaxes, _zoom_labels):
                    if _zr is None:
                        _mask = np.ones(len(_neigh_2d), dtype=bool)
                        _mask_mc = np.ones(len(_mc_2d), dtype=bool)
                    else:
                        _mask = (
                            (_neigh_2d[:, 0] >= _anchor_2d[0] - _zr) & (_neigh_2d[:, 0] <= _anchor_2d[0] + _zr) &
                            (_neigh_2d[:, 1] >= _anchor_2d[1] - _zr) & (_neigh_2d[:, 1] <= _anchor_2d[1] + _zr)
                        )
                        _mask_mc = (
                            (_mc_2d[:, 0] >= _anchor_2d[0] - _zr) & (_mc_2d[:, 0] <= _anchor_2d[0] + _zr) &
                            (_mc_2d[:, 1] >= _anchor_2d[1] - _zr) & (_mc_2d[:, 1] <= _anchor_2d[1] + _zr)
                        )
                    # Neighbours: grey=has OOD attr (safe), orange=missing OOD attr (risky)
                    _has_attr = _mask & _nn_has_attr
                    _no_attr  = _mask & ~_nn_has_attr
                    _gax.scatter(_neigh_2d[_has_attr, 0], _neigh_2d[_has_attr, 1],
                                 s=5, alpha=0.22, color="#aaaaaa", linewidths=0, zorder=1,
                                 label="KNN neighbours (OOD attr=1)")
                    if _no_attr.any():
                        _gax.scatter(_neigh_2d[_no_attr, 0], _neigh_2d[_no_attr, 1],
                                     s=10, alpha=0.35, color="#ffaa00", linewidths=0, zorder=2,
                                     label="KNN (OOD attr=0, risky)")
                    # Manifold MC noisy samples — coloured by OOD territory of nearest KNN neighbour
                    _mc_vis_ood     = _mask_mc & _mc_is_ood
                    _mc_vis_not_ood = _mask_mc & ~_mc_is_ood
                    # Always plot both groups — legend shows TOTAL count across all N_mc, not just in-zoom
                    _gax.scatter(_mc_2d[_mc_vis_ood, 0], _mc_2d[_mc_vis_ood, 1],
                                 s=8, alpha=0.50, color="#2ca02c", marker="o", linewidths=0,
                                 zorder=3, label=f"MC in OOD territory ({_n_ood_mc_total}/{_N_mc})")
                    _gax.scatter(_mc_2d[_mc_vis_not_ood, 0], _mc_2d[_mc_vis_not_ood, 1],
                                 s=8, alpha=0.50, color="#17becf", marker="o", linewidths=0,
                                 zorder=3, label=f"MC in non-OOD territory ({_n_notood_mc_total}/{_N_mc})")
                    _gax.add_patch(_plt_geom.Circle(
                        (_anchor_2d[0], _anchor_2d[1]), sigma,
                        fill=False, edgecolor="tab:blue", linewidth=2.5,
                        linestyle=(0, (4, 2)), alpha=0.95, zorder=3,
                        label=f"Iso circle r=σ={sigma}",
                    ))
                    _gax.add_patch(_Ellipse(
                        (_anchor_2d[0], _anchor_2d[1]),
                        width=2.0 * _ea1, height=2.0 * _ea2,
                        fill=False, edgecolor="tab:orange", linewidth=2.5,
                        linestyle="solid", alpha=0.95, zorder=3,
                        label=f"Mani ellipse a1={_ea1:.4f} a2={_ea2:.4f}",
                    ))
                    _gax.scatter(_anchor_2d[0], _anchor_2d[1], s=160, marker="*",
                                 c="black", edgecolors="white", linewidths=1.0, zorder=5, label="Anchor")
                    if _zr is None:
                        _gax.set_xlim(float(np.min(_x_all)) - _pad, float(np.max(_x_all)) + _pad)
                        _gax.set_ylim(float(np.min(_y_all)) - _pad, float(np.max(_y_all)) + _pad)
                    else:
                        _gax.set_xlim(_anchor_2d[0] - _zr, _anchor_2d[0] + _zr)
                        _gax.set_ylim(_anchor_2d[1] - _zr, _anchor_2d[1] + _zr)
                    _gax.set_aspect("equal")
                    _gax.set_title(_gtitle, fontsize=8.5)
                    _gax.set_xlabel(f"PC1 (std={_pc1_std:.2f})")
                    _gax.set_ylabel("PC2")
                    _gax.grid(alpha=0.25)
                    _gax.legend(fontsize=7, loc="upper right")

                _ood_tag_mani = f"  |  OOD: {getattr(cfg.dataset, 'ood_attribute', None)}=1" if ood_attr_map is not None and getattr(cfg.dataset, 'ood_attribute', None) else ""
                _gfig.suptitle(
                    f"Manifold: Circle + Ellipse Geometry (idx={sample_idx})  "
                    f"| λ_max={_lambda_max:.4f}  √λ_max={_sqrt_lambda_max:.4f}  "
                    f"|  α=σ/√λ_max={alpha_display:.4f}  (α/σ={alpha_display/sigma:.4f})\n"
                    f"σ={sigma}  |  K=500  |  PC1 std={_pc1_std:.3f}{_ood_tag_mani}",
                    fontsize=11, y=1.02,
                )
                _plt_geom.tight_layout()
                _geom_path = viz_dir / f"sample_{sample_idx:04d}_geometry.png"
                _gfig.savefig(_geom_path, dpi=150, bbox_inches="tight")
                _plt_geom.close(_gfig)
            except Exception as _geom_err:
                _log(f"Geometry figure skipped for sample {sample_idx}: {_geom_err}")

    # ------------------------------------------------------------------
    # LATENT MANIFOLD  — 7-row notebook-style grid
    # ------------------------------------------------------------------
    else:  # is_latent and is_manifold
        manifold_sm_latent = latent_smoother if latent_smoother is not None else iso_latent
        manifold_sm_pixel = pixel_smoother if pixel_smoother is not None else iso_pixel

        # Compute alpha for display
        alpha_display = sigma
        _cached_latent_pca = None
        if isinstance(manifold_sm_latent, ManifoldSmoother):
            from src.smoothing.pca import whiten, unwhiten
            _cached_latent_pca = manifold_sm_latent.compute_pca(query_vec)
            lambda_max_lat = float(_cached_latent_pca.pca.evals[0])
            alpha_display = sigma / np.sqrt(max(lambda_max_lat, 1e-12))

        row_labels = [
            "Row 0\nOriginal &\nPCA Recon (Latent)",
            f"Row 1\nLatent manifold\nα=σ/√λ_max={alpha_display:.4f}\n(certified)",
            f"Row 2\nLatent manifold\nσ={sigma} unscaled\n(ref)",
            "Row 3\nPixel manifold\nα=σ/√λ_max (decoded)",
            f"Row 4\nLatent iso\nnoise σ={sigma}",
            f"Row 5\nPixel iso\nnoise σ={sigma}",
            "Row 6\nNeighbours",
        ]
        n_rows = 7
        fig, axes = plt.subplots(n_rows, n_noisy_samples,
                                 figsize=(3 * n_noisy_samples, 3.5 * n_rows))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")

        for r, lbl in enumerate(row_labels):
            _add_row_label(axes, r, lbl)

        # Row 0: Original + PCA Reconstruction (Latent)
        axes[0, 0].imshow(_tensor_to_pil(img_tensor))
        axes[0, 0].set_title("Original", fontsize=9)

        if isinstance(manifold_sm_latent, ManifoldSmoother) and _cached_latent_pca is not None:
            w_lat = whiten(query_vec, _cached_latent_pca.pca)
            recon_z = unwhiten(w_lat, _cached_latent_pca.pca)
            with torch.no_grad():
                z_t = torch.from_numpy(recon_z[None, :]).to(device=device, dtype=torch.float32)
                x_recon = vae.decode(z_t).squeeze(0).cpu()
        else:
            with torch.no_grad():
                x = img_tensor.unsqueeze(0).to(device)
                if x.shape[-1] != vae.image_size or x.shape[-2] != vae.image_size:
                    x = F.interpolate(x, size=vae.image_size, mode="bilinear", align_corners=False)
                mu, _ = vae.encode(x)
                x_recon = vae.decode(mu).squeeze(0).cpu()
        axes[0, 1].imshow(_tensor_to_pil(x_recon))
        axes[0, 1].set_title("PCA Recon (Latent)", fontsize=9)

        # Row 1: Latent manifold noise α = σ/√λ_max  (certified samples)
        _latent_sample_fn = make_latent_sample_fn(img_tensor, vae, manifold_sm_latent, device)
        for i in range(n_noisy_samples):
            axes[1, i].imshow(_tensor_to_pil(_latent_sample_fn()))
            if i == 0:
                axes[1, i].set_title(f"Latent manifold  α={alpha_display:.4f}  (certified)", fontsize=9)

        # Row 2: Latent manifold noise σ unscaled (reference)
        if isinstance(manifold_sm_latent, ManifoldSmoother) and _cached_latent_pca is not None:
            def _sample_latent_unscaled():
                w_anch = whiten(query_vec, _cached_latent_pca.pca)
                noise = np.random.normal(0.0, sigma, size=len(w_anch)).astype(np.float32)
                z_noised = unwhiten(w_anch + noise, _cached_latent_pca.pca)
                with torch.no_grad():
                    z_t = torch.from_numpy(z_noised[None, :]).to(device=device, dtype=torch.float32)
                    return vae.decode(z_t).squeeze(0).cpu()
            for i in range(n_noisy_samples):
                axes[2, i].imshow(_tensor_to_pil(_sample_latent_unscaled()))
                if i == 0:
                    axes[2, i].set_title(f"Latent manifold  σ={sigma} unscaled  (ref)", fontsize=9)
        else:
            for i in range(n_noisy_samples):
                axes[2, i].imshow(_tensor_to_pil(sample_latent(img_tensor, vae, iso_latent, device)))

        # Row 3: Pixel manifold noise α = σ/√λ_max  (decoded)
        for i in range(n_noisy_samples):
            axes[3, i].imshow(_tensor_to_pil(sample_pixel(img_tensor, manifold_sm_pixel)))
            if i == 0:
                axes[3, i].set_title("Pixel manifold  α=σ/√λ_max  (decoded)", fontsize=9)

        # Row 4: Latent Gaussian noise
        for i in range(n_noisy_samples):
            axes[4, i].imshow(_tensor_to_pil(sample_latent(img_tensor, vae, iso_latent, device)))
            if i == 0:
                axes[4, i].set_title(f"Latent iso noise  σ={sigma}", fontsize=9)

        # Row 5: Pixel Gaussian noise
        for i in range(n_noisy_samples):
            axes[5, i].imshow(_tensor_to_pil(sample_pixel(img_tensor, iso_pixel)))
            if i == 0:
                axes[5, i].set_title(f"Pixel iso noise  σ={sigma}", fontsize=9)

        # Row 6: Neighbours
        nn_imgs = _get_nn_images(index, query_vec, n_noisy_samples,
                                 img_tensor.shape, vae, device, is_latent=True)
        for i in range(n_noisy_samples):
            if i < len(nn_imgs):
                axes[6, i].imshow(_tensor_to_pil(nn_imgs[i]))
            if i == 0:
                axes[6, i].set_title("Neighbours", fontsize=9)

        # ── Latent Manifold geometry figure ───────────────────────────────
        if isinstance(manifold_sm_latent, ManifoldSmoother) and _cached_latent_pca is not None:
            try:
                from matplotlib.patches import Ellipse as _Ellipse
                import matplotlib.pyplot as _plt_geom

                _pca_obj = _cached_latent_pca.pca
                _nbrs = _cached_latent_pca.neighbors  # (k, D_latent) centred
                _ev = np.asarray(_pca_obj.evals, dtype=np.float64)
                _Vt = _pca_obj.evecs.T  # (n_comp, D_latent)
                _ev_norm = np.maximum(_ev, 1e-12) / float(_ev.max())
                _lambda_max_lat = float(_ev[0])
                _sqrt_lambda_max_lat = float(np.sqrt(_lambda_max_lat))
                _anchor_2d = (query_vec - np.asarray(_pca_obj.mean, dtype=np.float64)) @ _Vt[:2].T
                _neigh_2d = _nbrs.astype(np.float64) @ _Vt[:2].T

                _a1 = float(sigma * np.sqrt(_ev_norm[0]))
                _a2 = float(sigma * np.sqrt(_ev_norm[1]))
                _mid_idx = len(_ev_norm) // 2
                _a_mid  = float(sigma * np.sqrt(_ev_norm[_mid_idx]))
                _a_last = float(sigma * np.sqrt(_ev_norm[-1]))
                _pc1_std = float(np.std(_neigh_2d[:, 0]))
                _pc1_range = float(np.max(_neigh_2d[:, 0]) - np.min(_neigh_2d[:, 0]))
                _zoom_mid = 0.10 * _pc1_range
                _zoom_tight = _a1

                _zoom_labels = [
                    (None,        f"[A] Full cloud  σ/std={sigma / (_pc1_std + 1e-12):.4f}", _a1, _a2),
                    (_zoom_mid,   f"[B] Mid-zoom  ±{_zoom_mid:.2f}", _a1, _a2),
                    (_zoom_tight, f"[C] Tight ±σ={_zoom_tight:.4f}  a2={_a2:.4f}", _a1, _a2),
                    (_zoom_tight, f"[D] Mid PC (k={_mid_idx})  a_mid={_a_mid:.4f}\na_mid/a1={_a_mid/_a1:.4f}", _a1, _a_mid),
                    (_zoom_tight, f"[E] Last PC (k={len(_ev_norm)-1})  a_last={_a_last:.6f}\na_last/a1={_a_last/_a1:.6f}", _a1, _a_last),
                ]

                # Generate MC noisy samples in latent manifold space → project to PCA-2D
                _N_mc = cfg.smoothing.n_samples
                _mc_samples_2d = []
                for _ in range(_N_mc):
                    _ns = manifold_sm_latent.sample_from_cached(_cached_latent_pca)
                    _ns_centred = _ns - np.asarray(_pca_obj.mean, dtype=np.float64)
                    _mc_samples_2d.append(_ns_centred @ _Vt[:2].T)
                _mc_2d = np.array(_mc_samples_2d)  # (N_mc, 2)

                # OOD flag per latent neighbour
                _nn_has_attr = np.zeros(len(_neigh_2d), dtype=bool)
                if ood_attr_map is not None and index is not None and hasattr(index, "filenames"):
                    _nn_ids_ood = index.index.get_nns_by_vector(
                        query_vec.astype(np.float32),
                        len(_neigh_2d), include_distances=False,
                    ) if hasattr(index.index, "get_nns_by_vector") else []
                    for _ni, _nid in enumerate(_nn_ids_ood[:len(_neigh_2d)]):
                        _fname = Path(index.filenames[_nid]).name if hasattr(index, "filenames") else ""
                        _nn_has_attr[_ni] = bool(ood_attr_map.get(_fname, 0))

                # Per-MC-sample OOD label via nearest KNN neighbour in 2D PCA space
                if ood_attr_map is not None and len(_neigh_2d) > 0:
                    _mc_nn_dists = np.linalg.norm(
                        _mc_2d[:, None, :] - _neigh_2d[None, :, :], axis=-1
                    )
                    _mc_nn_idx = np.argmin(_mc_nn_dists, axis=1)
                    _mc_is_ood = _nn_has_attr[_mc_nn_idx]
                else:
                    _mc_is_ood = np.zeros(len(_mc_2d), dtype=bool)

                _gfig, _gaxes = _plt_geom.subplots(1, 5, figsize=(22, 5.5), facecolor="white")
                _x_all = np.concatenate([_neigh_2d[:, 0], [_anchor_2d[0]]])
                _y_all = np.concatenate([_neigh_2d[:, 1], [_anchor_2d[1]]])
                _pad = 0.05 * max(float(np.max(_x_all) - np.min(_x_all)),
                                  float(np.max(_y_all) - np.min(_y_all)), 1e-6)
                _n_ood_mc_total    = int(_mc_is_ood.sum())
                _n_notood_mc_total = int((~_mc_is_ood).sum())

                for _gax, (_zr, _gtitle, _ea1, _ea2) in zip(_gaxes, _zoom_labels):
                    if _zr is None:
                        _mask = np.ones(len(_neigh_2d), dtype=bool)
                        _mask_mc = np.ones(len(_mc_2d), dtype=bool)
                    else:
                        _mask = (
                            (_neigh_2d[:, 0] >= _anchor_2d[0] - _zr) & (_neigh_2d[:, 0] <= _anchor_2d[0] + _zr) &
                            (_neigh_2d[:, 1] >= _anchor_2d[1] - _zr) & (_neigh_2d[:, 1] <= _anchor_2d[1] + _zr)
                        )
                        _mask_mc = (
                            (_mc_2d[:, 0] >= _anchor_2d[0] - _zr) & (_mc_2d[:, 0] <= _anchor_2d[0] + _zr) &
                            (_mc_2d[:, 1] >= _anchor_2d[1] - _zr) & (_mc_2d[:, 1] <= _anchor_2d[1] + _zr)
                        )
                    _has_attr = _mask & _nn_has_attr
                    _no_attr  = _mask & ~_nn_has_attr
                    _gax.scatter(_neigh_2d[_has_attr, 0], _neigh_2d[_has_attr, 1],
                                 s=5, alpha=0.22, color="#aaaaaa", linewidths=0, zorder=1,
                                 label="KNN neighbours (OOD attr=1)")
                    if _no_attr.any():
                        _gax.scatter(_neigh_2d[_no_attr, 0], _neigh_2d[_no_attr, 1],
                                     s=10, alpha=0.35, color="#ffaa00", linewidths=0, zorder=2,
                                     label="KNN (OOD attr=0, risky)")
                    _mc_vis_ood     = _mask_mc & _mc_is_ood
                    _mc_vis_not_ood = _mask_mc & ~_mc_is_ood
                    _gax.scatter(_mc_2d[_mc_vis_ood, 0], _mc_2d[_mc_vis_ood, 1],
                                 s=8, alpha=0.50, color="#2ca02c", marker="o", linewidths=0,
                                 zorder=3, label=f"MC in OOD territory ({_n_ood_mc_total}/{_N_mc})")
                    _gax.scatter(_mc_2d[_mc_vis_not_ood, 0], _mc_2d[_mc_vis_not_ood, 1],
                                 s=8, alpha=0.50, color="#17becf", marker="o", linewidths=0,
                                 zorder=3, label=f"MC in non-OOD territory ({_n_notood_mc_total}/{_N_mc})")
                    _gax.add_patch(_plt_geom.Circle(
                        (_anchor_2d[0], _anchor_2d[1]), sigma,
                        fill=False, edgecolor="tab:blue", linewidth=2.5,
                        linestyle=(0, (4, 2)), alpha=0.95, zorder=3,
                        label=f"Iso circle r=σ={sigma}",
                    ))
                    _gax.add_patch(_Ellipse(
                        (_anchor_2d[0], _anchor_2d[1]),
                        width=2.0 * _ea1, height=2.0 * _ea2,
                        fill=False, edgecolor="tab:orange", linewidth=2.5,
                        linestyle="solid", alpha=0.95, zorder=3,
                        label=f"Mani ellipse a1={_ea1:.4f} a2={_ea2:.4f}",
                    ))
                    _gax.scatter(_anchor_2d[0], _anchor_2d[1], s=160, marker="*",
                                 c="black", edgecolors="white", linewidths=1.0, zorder=5, label="Anchor")
                    if _zr is None:
                        _gax.set_xlim(float(np.min(_x_all)) - _pad, float(np.max(_x_all)) + _pad)
                        _gax.set_ylim(float(np.min(_y_all)) - _pad, float(np.max(_y_all)) + _pad)
                    else:
                        _gax.set_xlim(_anchor_2d[0] - _zr, _anchor_2d[0] + _zr)
                        _gax.set_ylim(_anchor_2d[1] - _zr, _anchor_2d[1] + _zr)
                    _gax.set_aspect("equal")
                    _gax.set_title(_gtitle, fontsize=8.5)
                    _gax.set_xlabel(f"PC1 (std={_pc1_std:.2f})")
                    _gax.set_ylabel("PC2")
                    _gax.grid(alpha=0.25)
                    _gax.legend(fontsize=7, loc="upper right")

                _ood_tag_lat = f"  |  OOD: {getattr(cfg.dataset, 'ood_attribute', None)}=1" if ood_attr_map is not None and getattr(cfg.dataset, 'ood_attribute', None) else ""
                _gfig.suptitle(
                    f"Latent Manifold: Circle + Ellipse Geometry (idx={sample_idx})  "
                    f"| λ_max={_lambda_max_lat:.4f}  √λ_max={_sqrt_lambda_max_lat:.4f}  "
                    f"|  α=σ/√λ_max={alpha_display:.4f}  (α/σ={alpha_display/sigma:.4f})\n"
                    f"σ={sigma}  |  K={manifold_sm_latent.knn_k if hasattr(manifold_sm_latent, 'knn_k') else '?'}  |  PC1 std={_pc1_std:.3f}{_ood_tag_lat}",
                    fontsize=11, y=1.02,
                )
                _plt_geom.tight_layout()
                _geom_lat_path = viz_dir / f"sample_{sample_idx:04d}_geometry_latent.png"
                _gfig.savefig(_geom_lat_path, dpi=150, bbox_inches="tight")
                _plt_geom.close(_gfig)
            except Exception as _geom_err_lat:
                _log(f"Latent geometry figure skipped for sample {sample_idx}: {_geom_err_lat}")

    # ------------------------------------------------------------------
    # Title with certification result
    # ------------------------------------------------------------------
    cert_status = "ABSTAIN" if abstained else ("CORRECT" if pred == label else "WRONG")
    pred_label = "smile" if pred == 1 else "no smile"
    true_label = "smile" if label == 1 else "no smile"
    smoothing_type = "Manifold" if cfg.smoothing.use_manifold else "Isotropic"
    _ood_title = ""
    _ood_attr_name = getattr(cfg.dataset, "ood_attribute", None) if hasattr(cfg, "dataset") else None
    if _ood_attr_name:
        _ood_title = f"  |  OOD: {_ood_attr_name}=1"
    fig.suptitle(
        f"Sample {sample_idx} | {cfg.smoothing.mode.capitalize()} {smoothing_type} | "
        f"σ={sigma} | True: {true_label} | Pred: {pred_label} | "
        f"Radius: {radius:.4f} | {cert_status}{_ood_title}",
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
    img_tensor: torch.Tensor,
    smoother,
    n_samples: int,
    n0_samples: int,
    classifier_transform: transforms.Compose,
    device: torch.device,
    alpha_conf: float = 0.001,
    sigma: float = 0.25,
    gpu_cache: Optional[dict] = None,
    sample_index: Optional[int] = None,
    sample_fn: Optional[callable] = None,
    collect_n_samples: bool = False,
) -> Tuple[TokenCertificate, Optional[List[torch.Tensor]]]:
    """Certify one image using batched GPU noise sampling.

    All n0+n noise copies are generated and classified in one GPU batch
    (or mini-batches if memory is tight), replacing the old Python loop.

    If collect_n_samples=True and sample_fn is provided, returns the raw
    n-phase sample tensors (in smooth_transform space) alongside the cert.
    These are the exact samples used for certification, useful for OOD analysis.
    Returns (cert, n_phase_raw_samples) or (cert, None).
    """
    classifier.eval()
    if n0_samples <= 0 or n_samples <= 0:
        raise ValueError("n0_samples and n_samples must be > 0.")

    # Pre-process the clean image once
    x_clean = classifier_transform(
        transforms.ToPILImage()(img_tensor.clamp(-1, 1) * 0.5 + 0.5)
    ).unsqueeze(0).to(device)  # (1, C, H, W)

    total = n0_samples + n_samples
    class_counts_n0 = np.zeros(2, dtype=np.int64)
    class_counts_n  = np.zeros(2, dtype=np.int64)
    # Collect raw n-phase samples (smooth_transform space) if requested
    _raw_n_samples: Optional[List[torch.Tensor]] = [] if collect_n_samples else None

    # Build index tensor for manifold cache lookup
    if gpu_cache is not None and sample_index is not None:
        idx = torch.tensor([sample_index], device=device)  # (1,)

    processed = 0
    CHUNK = 256  # max samples per GPU forward to avoid OOM
    # Pre-compute normalization constants for CPU fallback denormalization
    _smooth_mean = torch.tensor(CELEBA_MEAN).view(3, 1, 1)
    _smooth_std  = torch.tensor(CELEBA_STD).view(3, 1, 1)
    while processed < total:
        chunk = min(CHUNK, total - processed)
        # Repeat clean image chunk times: (chunk, C, H, W)
        x_rep = x_clean.expand(chunk, -1, -1, -1).clone()

        # Add noise entirely on GPU
        if gpu_cache is not None and sample_index is not None:
            idx_rep = idx.expand(chunk)  # (chunk,)
            x_noisy = smoother.sample_batch_gpu(x_rep, gpu_cache=gpu_cache, indices=idx_rep)
        elif sample_fn is not None:
            # CPU fallback: use pre-built sample_fn (kNN+SVD cached, works for ManifoldSmoother).
            # sample_fn() returns a normalized tensor in smooth_transform space; undo normalization
            # → PIL → apply classifier_transform to match the GPU path's input space.
            processed_samples = []
            raw_chunk: List[torch.Tensor] = []
            for _ in range(chunk):
                s = sample_fn()  # (C, H, W) in smooth_transform space
                raw_chunk.append(s)
                s_01 = (s * _smooth_std + _smooth_mean).clamp(0, 1)
                pil = transforms.ToPILImage()(s_01)
                processed_samples.append(classifier_transform(pil))
            x_noisy = torch.stack(processed_samples, dim=0).to(device)  # (chunk, C, H, W)
            # Collect raw n-phase samples only
            if _raw_n_samples is not None:
                for _ci, _raw in enumerate(raw_chunk):
                    if processed + _ci >= n0_samples:
                        _raw_n_samples.append(_raw)
        else:
            x_noisy = smoother.sample_batch_gpu(x_rep)

        # Clamp, classify
        logits = classifier(x_noisy.clamp(0, 1)).squeeze(-1)  # (chunk,)
        preds  = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()

        for j, pred in enumerate(preds):
            global_idx = processed + j
            if global_idx < n0_samples:
                class_counts_n0[pred] += 1
            else:
                class_counts_n[pred]  += 1
        processed += chunk

    cert = certify_token_from_counts_two_stage_paper(
        class_counts_n0=class_counts_n0,
        class_counts_n=class_counts_n,
        alpha_noise=sigma,
        alpha_conf=alpha_conf,
        abstain_label=-1,
    )
    return cert, (_raw_n_samples if collect_n_samples else None)


def run_certification(cfg: CertifyConfig) -> Dict:
    """Run full certification pipeline."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if int(cfg.smoothing.n0_samples) <= 0:
        raise ValueError("Paper-aligned CERTIFY requires smoothing.n0_samples > 0 in config.")
    if int(cfg.smoothing.n_samples) <= 0:
        raise ValueError("Paper-aligned CERTIFY requires smoothing.n_samples > 0 in config.")
    
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

    # Apply OOD attribute filter (or standard subset) — replaces old subset_size block
    test_samples = get_ood_test_samples(cfg, test_samples)

    ood_attr = getattr(cfg.dataset, "ood_attribute", None)
    if ood_attr:
        _log(f"OOD attribute: '{ood_attr}'  test subset: {len(test_samples)} images")
    
    dataset_info = {
        "dataset": cfg.dataset.name,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "seed": cfg.seed,
        "ood_attribute": ood_attr if ood_attr else None,
        "ood_attr_value": 1 if ood_attr else None,  # only attr=1 images are certified
        "ood_balanced": cfg.dataset.ood_balanced if ood_attr else None,
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

    ckpt_path = resolve_classifier_checkpoint(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
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
    _log(f"Classifier: {ckpt_path}")
    
    classifier_transform = transforms.Compose([
        transforms.Resize((cfg.model.input_size, cfg.model.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
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
    # Use model.input_size so index, smoothing, and classifier all operate at the same resolution.
    # CelebA: 224×224, CelebA-HQ: 512×512 (set in config).
    pixel_size = cfg.model.input_size
    
    # Build pixel index for pixel manifold mode, latent manifold comparison viz,
    # or OOD MC-sample tracking (needed for mc_ood_count/mc_ood_frac in both ISO and manifold).
    _need_pixel_index_for_ood = bool(getattr(cfg.dataset, "ood_attribute", None))
    if cfg.smoothing.use_manifold or (cfg.smoothing.mode in ("latent", "both") and cfg.output.save_visualizations) or _need_pixel_index_for_ood:
        pixel_index = load_or_build_pixel_index(
            train_samples, pixel_size, paths.pixel_index_dir, cfg.index.n_trees,
            metric=cfg.index.metric,
        )
    
    # Load latent index for manifold mode, AND for isotropic latent when OOD tracking is needed
    # (latent_index is used as the query index for nn_ood_count in latent mode).
    _need_latent_index = (
        cfg.smoothing.mode in ("latent", "both") and vae is not None and
        (cfg.smoothing.use_manifold or _need_pixel_index_for_ood)
    )
    if _need_latent_index:
        latent_index = load_or_build_latent_index(
            train_samples, vae, paths.latent_index_dir, cfg.index.n_trees, device,
            metric=cfg.index.metric,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Transform for smoothing
    # ─────────────────────────────────────────────────────────────────────────
    smooth_size = pixel_size  # same resolution for index, smoothing, and classifier
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

    # Build OOD attr map once (filename -> 1/0) — used for both KNN OOD fraction and viz
    _ood_attr_map_global: Optional[dict] = None
    _ood_attr_name_global = getattr(cfg.dataset, "ood_attribute", None)
    if _ood_attr_name_global:
        _attr_path_g = Path(cfg.dataset.root_dir) / cfg.dataset.annotation_file
        _lines_g = [l.strip() for l in _attr_path_g.read_text().splitlines() if l.strip()]
        _attr_names_g = _lines_g[1].split()
        if _ood_attr_name_global in _attr_names_g:
            _aidx_g = _attr_names_g.index(_ood_attr_name_global)
            _ood_attr_map_global = {}
            for _row_g in _lines_g[2:]:
                _parts_g = _row_g.split()
                _ood_attr_map_global[_parts_g[0]] = 1 if int(_parts_g[1 + _aidx_g]) == 1 else 0

    # Precompute a fast int8 label array indexed by Annoy ID for mc_ood_count.
    # Avoids per-sample Python dict + Path() overhead inside the tight MC loop.
    _pixel_ood_labels: Optional[np.ndarray] = None  # shape (N_train,) int8
    if (_ood_attr_map_global is not None
            and pixel_index is not None
            and hasattr(pixel_index, "filenames")):
        _pixel_ood_labels = np.array(
            [_ood_attr_map_global.get(Path(fn).name, 0) for fn in pixel_index.filenames],
            dtype=np.int8,
        )

    for idx in tqdm(range(start_idx, len(test_samples)), desc="Certifying (test)", initial=start_idx, total=len(test_samples)):
        img_path, label = test_samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = smooth_transform(img)
        
        # Create sample function with cached PCA (kNN+SVD computed once per image)
        if cfg.smoothing.mode == "pixel":
            sample_fn = make_pixel_sample_fn(img_tensor, pixel_smoother)
        elif cfg.smoothing.mode == "latent" and vae is not None:
            sample_fn = make_latent_sample_fn(img_tensor, vae, latent_smoother, device)
        else:
            # Default to pixel isotropic
            sample_fn = make_pixel_sample_fn(img_tensor, pixel_smoother)
        
        # Determine which smoother to use for this sample
        _active_smoother = latent_smoother if cfg.smoothing.mode == "latent" and vae is not None else pixel_smoother

        # Collect raw n-phase MC samples for OOD hit-rate computation.
        # Works for BOTH isotropic and manifold — pixel_index is now loaded whenever
        # ood_attribute is configured (see index-loading block above).
        _collect = (_ood_attr_map_global is not None
                    and pixel_index is not None
                    and hasattr(pixel_index, "index")
                    and hasattr(pixel_index.index, "get_nns_by_vector")
                    and hasattr(pixel_index, "filenames"))
        cert, _cert_raw_samples = certify_single_sample(
            classifier=classifier,
            img_tensor=img_tensor,
            smoother=_active_smoother,
            n_samples=cfg.smoothing.n_samples,
            n0_samples=int(cfg.smoothing.n0_samples),
            classifier_transform=classifier_transform,
            device=device,
            alpha_conf=cfg.alpha_conf,
            sigma=cfg.smoothing.sigma,
            sample_fn=sample_fn,
            collect_n_samples=_collect,
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
            "ood_attribute": ood_attr if ood_attr else None,
            "ood_attr_value": 1 if ood_attr else None,  # all certified images have attr=1
        }

        # KNN neighbour OOD fraction: how many of the k neighbours have ood_attr == 1
        # Works for both manifold and isotropic runs whenever an index is available.
        result["nn_ood_count"] = None
        result["nn_ood_frac"]  = None
        if _ood_attr_map_global is not None:
            _knn_index = latent_index if cfg.smoothing.mode == "latent" else pixel_index
            if _knn_index is not None and hasattr(_knn_index, "index") and hasattr(_knn_index.index, "get_nns_by_vector") and hasattr(_knn_index, "filenames"):
                # Use latent encoding as query for latent index; pixel flatten for pixel index
                if cfg.smoothing.mode == "latent" and vae is not None:
                    with torch.no_grad():
                        _x_q = img_tensor.unsqueeze(0).to(device)
                        if _x_q.shape[-1] != vae.image_size or _x_q.shape[-2] != vae.image_size:
                            _x_q = F.interpolate(_x_q, size=vae.image_size, mode="bilinear", align_corners=False)
                        _mu_q, _ = vae.encode(_x_q)
                    _qvec = _mu_q.squeeze(0).cpu().numpy().astype(np.float32)
                else:
                    _qvec = img_tensor.numpy().flatten().astype(np.float32)
                _k_nn = cfg.smoothing.knn_k
                _nn_ids = _knn_index.index.get_nns_by_vector(_qvec.tolist(), _k_nn, include_distances=False)
                _nn_ood_count = sum(_ood_attr_map_global.get(Path(_knn_index.filenames[_nid]).name, 0) for _nid in _nn_ids)
                result["nn_ood_count"] = int(_nn_ood_count)
                result["nn_ood_frac"]  = float(_nn_ood_count) / len(_nn_ids) if _nn_ids else None

        # MC OOD fraction from actual certification n-phase samples.
        # Each raw sample is looked up in pixel_index (Annoy) → inherits OOD label of nearest neighbour.
        result["mc_ood_count"] = None
        result["mc_ood_frac"]  = None
        if (_cert_raw_samples is not None and len(_cert_raw_samples) > 0
                and _pixel_ood_labels is not None
                and pixel_index is not None
                and hasattr(pixel_index, "index")):
            _mc_hits = 0
            for _rs in _cert_raw_samples:
                _nid = pixel_index.index.get_nns_by_vector(
                    _rs.numpy().flatten().astype(np.float32).tolist(), 1, include_distances=False
                )
                if _nid:
                    _mc_hits += int(_pixel_ood_labels[_nid[0]])
            result["mc_ood_count"] = _mc_hits
            result["mc_ood_frac"]  = float(_mc_hits) / len(_cert_raw_samples)

        # Volume computation: extract eigenvalues, lambda_max and alpha from manifold smoother
        if isinstance(pixel_smoother, ManifoldSmoother) and cfg.smoothing.mode == "pixel":
            cached = pixel_smoother.compute_pca(img_tensor.numpy().reshape(-1))
            _lmax = float(cached.pca.evals[0])
            result["eigenvalues"] = cached.pca.evals.tolist()
            result["lambda_max"] = _lmax
            result["alpha"] = float(cfg.smoothing.sigma) / float(np.sqrt(max(_lmax, 1e-12)))
        elif isinstance(latent_smoother, ManifoldSmoother) and cfg.smoothing.mode == "latent" and vae is not None:
            with torch.no_grad():
                x_vae = img_tensor.unsqueeze(0).to(device)
                if x_vae.shape[-1] != vae.image_size or x_vae.shape[-2] != vae.image_size:
                    x_vae = F.interpolate(x_vae, size=vae.image_size, mode="bilinear", align_corners=False)
                mu, _ = vae.encode(x_vae)
            cached = latent_smoother.compute_pca(mu.cpu().numpy().reshape(-1))
            _lmax = float(cached.pca.evals[0])
            result["eigenvalues"] = cached.pca.evals.tolist()
            result["lambda_max"] = _lmax
            result["alpha"] = float(cfg.smoothing.sigma) / float(np.sqrt(max(_lmax, 1e-12)))

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
                ood_attr_map=_ood_attr_map_global,
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
    
    # Keep the latest partial state on disk for inspection / manual recovery.
    # It is overwritten on future checkpoints and safely ignored unless
    # cfg.checkpoint.resume is enabled.
    
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
        "n0_samples": int(cfg.smoothing.n0_samples),
        "n_samples": cfg.smoothing.n_samples,
        "total_samples": int(cfg.smoothing.n0_samples) + int(cfg.smoothing.n_samples),
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

    # ── OOD neighbour + MC hit-rate aggregates (reported for both iso and manifold) ──
    _nn_fracs = [r["nn_ood_frac"] for r in results if r.get("nn_ood_frac") is not None]
    _mc_fracs = [r["mc_ood_frac"] for r in results if r.get("mc_ood_frac") is not None]
    _nn_counts = [r["nn_ood_count"] for r in results if r.get("nn_ood_count") is not None]
    _mc_counts = [r["mc_ood_count"] for r in results if r.get("mc_ood_count") is not None]
    metrics["ood_stats"] = {
        "ood_attribute": ood_attr if ood_attr else None,
        # KNN neighbour OOD fraction (how many of k nearest neighbours have ood_attr=1)
        "nn_samples_with_data": len(_nn_fracs),
        "mean_nn_ood_frac":   float(np.mean(_nn_fracs))   if _nn_fracs   else None,
        "median_nn_ood_frac": float(np.median(_nn_fracs)) if _nn_fracs   else None,
        "mean_nn_ood_count":  float(np.mean(_nn_counts))  if _nn_counts  else None,
        # MC sample OOD hit-rate (fraction of n-phase certification samples landing in OOD territory)
        "mc_samples_with_data": len(_mc_fracs),
        "mean_mc_ood_frac":   float(np.mean(_mc_fracs))   if _mc_fracs   else None,
        "median_mc_ood_frac": float(np.median(_mc_fracs)) if _mc_fracs   else None,
        "mean_mc_ood_count":  float(np.mean(_mc_counts))  if _mc_counts  else None,
    }

    # Print results
    _log("=" * 70)
    _log(f"CERTIFICATION RESULTS: {cfg.experiment_name}")
    _log("=" * 70)
    _log(f"Dataset:            {cfg.dataset.name}")
    _log(f"Index split:        TRAIN ({len(train_samples)} samples)")
    _log(f"Certify split:      TEST ({total} samples)")
    _log(f"Smoothing:          {cfg.smoothing.mode} ({'manifold' if cfg.smoothing.use_manifold else 'isotropic'})")
    _log(f"Sigma:              {cfg.smoothing.sigma}")
    _log(f"Sampling:           n0={metrics['n0_samples']}, n={metrics['n_samples']}, total={metrics['total_samples']}")
    _log(f"Certified accuracy: {100*metrics['certified_accuracy']:.2f}%")
    _log(f"Abstain rate:       {100*metrics['abstain_rate']:.2f}%")
    _log(f"Mean radius:        {metrics['mean_radius']:.4f}")
    _log(f"Class No-Smile:     {metrics['class_no_smile_correct']}/{metrics['class_no_smile_total']} = {100*metrics['class_no_smile_accuracy']:.2f}%")
    _log(f"Class Smile:        {metrics['class_smile_correct']}/{metrics['class_smile_total']} = {100*metrics['class_smile_accuracy']:.2f}%")
    _log("=" * 70)

    # ── Volume computation — 4-quantity framework ──
    # The full picture requires cross-referencing BOTH iso and mani runs (done in notebook).
    # Per-run, we store raw ingredients + what we CAN compute locally:
    #
    # Qty 1: V_iso,D      = C_D · r_iso^D              (classical RS — from ISO run only)
    # Qty 2: V_iso,k      = C_k · r_iso^k              (fair semantic baseline — needs r_iso + k from mani)
    # Qty 3: V_mani,pred  = C_k · r_iso^k · √det(Λ)   (geometry-only gain — needs r_iso + eigenvalues)
    # Qty 4: V_mani,actual = C_k · r_mani^k · √det(Λ)  (real manifold certificate — from MANI run)
    #
    # geometry_factor = 0.5 · Σ log(λ_i) = log(√det(Λ))  — r-independent, pure eigenvalue gain
    # This equals log(Qty3) - log(Qty2) = log(Qty4) - log(C_k · r_mani^k)
    #
    # MANIFOLD RUN stores: Qty 4, geometry_factor, k, D, per-sample eigenvalues + radii
    # ISOTROPIC RUN stores: Qty 1, D, per-sample radii
    # NOTEBOOK cross-references to compute Qty 2 and Qty 3.
    from src.certify.randomized import (
        log_volume_isotropic,
        log_volume_manifold,
        eigenvalue_diagnostics,
        normalize_eigenvalues,
        log_volume_geo_iso,
        log_volume_geo_mani,
        axis_lengths,
        anisotropy_ratio,
        cumulative_stretch_energy,
    )

    k_pca = None
    
    # Ambient dimension
    if cfg.smoothing.mode == "latent" and vae is not None:
        ambient_dim = vae.latent_dim
    else:
        ambient_dim = cfg.model.input_size * cfg.model.input_size * 3  # pixel space: H*W*C
    
    # Try to load companion iso radii for cross-reference (Qty 1, 2, 3)
    iso_companion_radii = None
    sigma_tag = f"sigma_{cfg.smoothing.sigma:.2f}".replace(".", "_")
    iso_mode_tag = f"{cfg.smoothing.mode}_isotropic"
    iso_companion_dir = paths.certify_dir / iso_mode_tag / sigma_tag
    iso_csv = iso_companion_dir / "results.csv"
    if iso_csv.exists():
        import csv as csv_mod
        with open(iso_csv) as f:
            reader = csv_mod.DictReader(f)
            iso_companion_radii = np.array([float(row["radius"]) for row in reader if float(row["radius"]) > 0])
        _log(f"Loaded {len(iso_companion_radii)} iso radii from companion: {iso_csv}")
    else:
        _log(f"No companion iso results at {iso_csv} — Qty 1,2,3 will be unavailable")

    geometry_factors = []
    lv_mani_actuals = []
    lv_iso_Ds = []
    # Geometry-first per-sample lists
    per_sample_log_geo_ratio = []
    per_sample_anisotropy = []
    per_sample_axis_lengths = []
    per_sample_cum_energy = []
    per_sample_effective_rank = []
    sigma_val = float(cfg.smoothing.sigma)

    for r in results:
        if r["radius"] <= 0:
            continue
        evals = r.get("eigenvalues")
        if evals is not None:
            # Manifold run → Qty 4 + geometry_factor (legacy) + NEW geo metrics
            evals_arr = np.array(evals, dtype=np.float64)
            k_pca = len(evals_arr)
            lv_mani = log_volume_manifold(r["radius"], evals_arr)       # Qty 4
            geom = 0.5 * np.sum(np.log(np.maximum(evals_arr, 1e-30)))   # geometry factor
            diag = eigenvalue_diagnostics(evals_arr)
            r["log_vol_mani_actual"] = lv_mani
            r["geometry_factor"] = geom
            r["eigen_k"] = k_pca
            r["ambient_D"] = ambient_dim
            r["eigen_effective_rank"] = diag.effective_rank
            r["eigen_condition_number"] = diag.condition_number
            r["eigen_sum"] = diag.eigenvalue_sum
            lv_mani_actuals.append(lv_mani)
            geometry_factors.append(geom)
            # ── NEW: geometry metrics per sample ──
            evals_norm_max = normalize_eigenvalues(evals_arr, mode="max")
            lv_gm = log_volume_geo_mani(sigma_val, evals_norm_max)
            lv_gi = log_volume_geo_iso(sigma_val, k_pca)
            log_geo_ratio = lv_gm - lv_gi   # = 0.5·Σlog(λ̃_i) normalized
            ani = anisotropy_ratio(evals_norm_max)
            ax = axis_lengths(sigma_val, evals_norm_max)
            cum = cumulative_stretch_energy(evals_norm_max)
            p = evals_arr / np.maximum(evals_arr.sum(), 1e-30)
            eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
            per_sample_log_geo_ratio.append(log_geo_ratio)
            per_sample_anisotropy.append(ani)
            per_sample_axis_lengths.append(ax.tolist())  # save ALL axes
            per_sample_cum_energy.append(cum.tolist())
            per_sample_effective_rank.append(eff_rank)
            r["log_v_geo_iso"] = lv_gi
            r["log_v_geo_mani"] = lv_gm
            r["log_geo_ratio"] = log_geo_ratio
            r["anisotropy_ratio"] = ani
        else:
            # Isotropic run → Qty 1: V_iso,D = C_D · r_iso^D
            lv_iso_D = log_volume_isotropic(r["radius"], ambient_dim)
            r["log_vol_iso_D"] = lv_iso_D
            r["log_vol_mani_actual"] = None
            r["geometry_factor"] = None
            r["eigen_k"] = None
            r["ambient_D"] = ambient_dim
            lv_iso_Ds.append(lv_iso_D)

    # Aggregate volume stats
    if lv_mani_actuals or lv_iso_Ds:
        # Compute Qty 1, 2, 3 from companion iso radii if available
        mean_log_vol_iso_D = None
        mean_log_vol_iso_k = None
        mean_log_vol_mani_pred = None
        if iso_companion_radii is not None and len(iso_companion_radii) > 0 and k_pca is not None:
            mean_geom = float(np.mean(geometry_factors)) if geometry_factors else 0.0
            lv_iso_D_from_iso = [log_volume_isotropic(r, ambient_dim) for r in iso_companion_radii]
            lv_iso_k_from_iso = [log_volume_isotropic(r, k_pca) for r in iso_companion_radii]
            lv_mani_pred_from_iso = [log_volume_isotropic(r, k_pca) + mean_geom for r in iso_companion_radii]
            mean_log_vol_iso_D = float(np.mean(lv_iso_D_from_iso))
            mean_log_vol_iso_k = float(np.mean(lv_iso_k_from_iso))
            mean_log_vol_mani_pred = float(np.mean(lv_mani_pred_from_iso))
        elif lv_iso_Ds:
            # This IS the isotropic run — Qty 1 directly
            mean_log_vol_iso_D = float(np.mean(lv_iso_Ds))

        # ── Geometry-first summary ──
        geo_summary = None
        if per_sample_log_geo_ratio:
            k_for_geo = k_pca  # from last manifold sample
            ax_arr_np = np.array(per_sample_axis_lengths)
            geo_summary = {
                "sigma": sigma_val,
                "k_pca": k_for_geo,
                "ambient_D": ambient_dim,
                # V_iso,geo = C_k·σ^k  (scalar — same for all samples)
                "log_v_iso_geo": float(log_volume_geo_iso(sigma_val, k_for_geo)) if k_for_geo else None,
                # V_mani,geo = C_k·σ^k·√det(Λ̃)  per-sample averaged
                "mean_log_v_mani_geo": float(np.mean([r["log_v_geo_mani"] for r in results if r.get("log_v_geo_mani") is not None])),
                # Geometry gain = log(V_mani,geo) - log(V_iso,geo) — pure shape
                "mean_log_geo_ratio": float(np.mean(per_sample_log_geo_ratio)),
                "median_log_geo_ratio": float(np.median(per_sample_log_geo_ratio)),
                "std_log_geo_ratio": float(np.std(per_sample_log_geo_ratio)),
                # Axis lengths a_i = σ·√λ̃_i averaged across samples (top-10)
                # Axis lengths a_i = σ·√λ̃_i averaged across samples (full spectrum)
                "mean_axis_lengths": np.nanmean(ax_arr_np, axis=0).tolist() if ax_arr_np.size else [],
                # Anisotropy = a_1/a_k
                "mean_anisotropy_ratio": float(np.mean(per_sample_anisotropy)),
                "median_anisotropy_ratio": float(np.median(per_sample_anisotropy)),
                "std_anisotropy_ratio": float(np.std(per_sample_anisotropy)),
                # Effective rank
                "mean_effective_rank": float(np.mean(per_sample_effective_rank)),
            }

        metrics["volume"] = {
            "k_pca": k_pca,
            "ambient_D": ambient_dim,
            # ── Legacy 4-quantity framework (radius-based) — kept for reference ──
            # Qty 1: V_iso,D = C_D · r_iso^D
            "mean_log_vol_iso_D": mean_log_vol_iso_D,
            # Qty 2: V_iso,k = C_k · r_iso^k
            "mean_log_vol_iso_k": mean_log_vol_iso_k,
            # Qty 3: V_mani,pred = C_k · r_iso^k · √det(Λ)
            "mean_log_vol_mani_pred": mean_log_vol_mani_pred,
            # Qty 4: V_mani,actual = C_k · r_mani^k · √det(Λ)
            "mean_log_vol_mani_actual": float(np.mean(lv_mani_actuals)) if lv_mani_actuals else None,
            # Geometry factor (r-independent): 0.5·Σlog(λ_i) raw eigenvalues
            "mean_geometry_factor": float(np.mean(geometry_factors)) if geometry_factors else None,
            "median_geometry_factor": float(np.median(geometry_factors)) if geometry_factors else None,
            # Diagnostics
            "mean_effective_rank": float(np.mean([r["eigen_effective_rank"] for r in results if r.get("eigen_effective_rank")])) if any(r.get("eigen_effective_rank") for r in results) else None,
            "mean_condition_number": float(np.mean([r["eigen_condition_number"] for r in results if r.get("eigen_condition_number")])) if any(r.get("eigen_condition_number") for r in results) else None,
            # ── NEW: geometry-first metrics (sigma-based, normalized) ──
            "geometry": geo_summary,
        }
        vol = metrics["volume"]
        if vol["mean_log_vol_mani_actual"] is not None:
            _log(f"Volume (k={k_pca}, D={ambient_dim}): mani_actual={vol['mean_log_vol_mani_actual']:.2f}, "
                 f"geom_factor={vol['mean_geometry_factor']:.2f}, eff_rank={vol['mean_effective_rank']:.1f}")
        elif vol["mean_log_vol_iso_D"] is not None:
            _log(f"Volume (iso, D={ambient_dim}): mean_log_vol_iso_D={vol['mean_log_vol_iso_D']:.2f}")

    # Log OOD stats summary
    _ood_s = metrics.get("ood_stats", {})
    if _ood_s.get("mean_nn_ood_frac") is not None or _ood_s.get("mean_mc_ood_frac") is not None:
        _log(f"OOD stats  [{_ood_s.get('ood_attribute')}=1]:")
        if _ood_s.get("mean_nn_ood_frac") is not None:
            _log(f"  KNN neighbour OOD frac: mean={_ood_s['mean_nn_ood_frac']:.3f}  "
                 f"median={_ood_s['median_nn_ood_frac']:.3f}  "
                 f"(n={_ood_s['nn_samples_with_data']})")
        if _ood_s.get("mean_mc_ood_frac") is not None:
            _log(f"  MC sample   OOD frac:   mean={_ood_s['mean_mc_ood_frac']:.3f}  "
                 f"median={_ood_s['median_mc_ood_frac']:.3f}  "
                 f"(n={_ood_s['mc_samples_with_data']})")

    # Save results
    if cfg.output.save_results:
        (paths.experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        _log(f"Metrics: {paths.experiment_dir / 'metrics.json'}")
        
        if cfg.output.save_per_sample:
            csv_path = paths.experiment_dir / "results.csv"
            fieldnames = ["idx", "image_path", "label", "pred", "radius", "abstained", "p_a_lower", "p_b_upper", "correct", "certified_correct",
                          "ood_attribute", "ood_attr_value",
                          "nn_ood_count", "nn_ood_frac",
                          "mc_ood_count", "mc_ood_frac",
                          "lambda_max", "alpha",
                          "log_vol_mani_actual", "log_vol_iso_D", "geometry_factor",
                          "eigen_k", "ambient_D", "eigen_effective_rank", "eigen_condition_number", "eigen_sum"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            _log(f"Results: {csv_path}")

            # Save eigenvalue spectra + geometry arrays for all samples
            eigen_samples = [(r["idx"], r["eigenvalues"]) for r in results if r.get("eigenvalues") is not None]
            if eigen_samples:
                eigen_path = paths.experiment_dir / "eigenvalues.npz"
                raw_evals_list = [np.array(e[1], dtype=np.float64) for e in eigen_samples]
                # Build geometry arrays (mirrors NER eigenvalues.npz format)
                _ax_max_len = max((len(e) for e in raw_evals_list), default=0)
                _ax_arr = np.full((len(raw_evals_list), _ax_max_len), np.nan)
                _cum_rows = []
                _geo_ratios = []
                _aniso_arr = []
                _eff_rank_arr = []
                _norm_evals_list = []
                k_geo = len(raw_evals_list[0]) if raw_evals_list else 0
                for i, ev in enumerate(raw_evals_list):
                    ev_norm = normalize_eigenvalues(ev, mode="max")
                    _norm_evals_list.append(ev_norm)
                    ax = axis_lengths(sigma_val, ev_norm)
                    _ax_arr[i, :len(ax)] = ax  # save ALL axes
                    cum = cumulative_stretch_energy(ev_norm)
                    _cum_rows.append(cum.tolist())
                    lv_gm = log_volume_geo_mani(sigma_val, ev_norm)
                    lv_gi = log_volume_geo_iso(sigma_val, len(ev_norm))
                    _geo_ratios.append(lv_gm - lv_gi)
                    _aniso_arr.append(anisotropy_ratio(ev_norm))
                    p = ev / np.maximum(ev.sum(), 1e-30)
                    _eff_rank_arr.append(float(np.exp(-np.sum(p * np.log(p + 1e-30)))))
                cum_max_len = max((len(c) for c in _cum_rows), default=0)
                _cum_arr = np.full((len(_cum_rows), cum_max_len), np.nan)
                for i, c in enumerate(_cum_rows):
                    _cum_arr[i, :len(c)] = c
                np.savez_compressed(
                    eigen_path,
                    # ── Original arrays ──
                    indices=np.array([e[0] for e in eigen_samples]),
                    eigenvalues=np.array(raw_evals_list, dtype=np.float64),
                    # ── Geometry arrays (match NER eigenvalues.npz format) ──
                    eigenvalues_norm_max=np.array(_norm_evals_list, dtype=np.float64),
                    # a_i = σ·√λ̃_i  (top-10 per sample)
                    # a_i = σ·√λ̃_i  (full spectrum per sample)
                    axis_lengths_all=_ax_arr,
                    # Anisotropy = a_1/a_k
                    anisotropy_ratios=np.array(_aniso_arr, dtype=np.float64),
                    # Cumulative stretch energy
                    cumulative_stretch_energy=_cum_arr,
                    # Geometry gain = log(V_mani,geo) - log(V_iso,geo) = 0.5·Σlog(λ̃_i)
                    log_geo_ratio_per_sample=np.array(_geo_ratios, dtype=np.float64),
                    # Effective rank (entropy-based)
                    effective_rank_per_sample=np.array(_eff_rank_arr, dtype=np.float64),
                    # Sigma used
                    sigma=np.float64(sigma_val),
                )
                _log(f"Eigenvalues: {eigen_path}")
        
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
