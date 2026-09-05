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
from src.dataloaders.celeba_smile import build_smile_dataloaders, build_dataloader_from_samples
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

    OOD auto-resolution (applies to both explicit and smoothed paths):
        If cfg.dataset.ood_attribute is set, the checkpoint is loaded from
        the OOD-specific classifier directory:
            .../smile_resnet_celeba_ood_<attr>/best.pt
        This overrides the standard checkpoint_path so you never need to
        manually edit the certify config when switching OOD attributes.

    If cfg.model.use_smoothed_classifier is True the path is auto-built as:
        {base_dir}/{mode}/{dataset}/smile_resnet_{dataset}_sigma_{s}/best.pt

    Otherwise cfg.model.checkpoint_path is returned (with OOD suffix if needed).
    """
    ood_attr = getattr(cfg.dataset, "ood_attribute", None)

    if not cfg.model.use_smoothed_classifier:
        ckpt = Path(cfg.model.checkpoint_path)
        use_ood_clf = getattr(cfg.model, "use_ood_classifier", True)
        if ood_attr and use_ood_clf:
            # e.g. output/pretrained_model/smile_resnet_celeba/best.pt
            #   → output/pretrained_model/smile_resnet_celeba_ood_wearing_hat/best.pt
            ood_tag = f"_ood_{ood_attr.lower()}"
            ckpt = ckpt.parent.parent / (ckpt.parent.name + ood_tag) / ckpt.name
            _log(f"OOD classifier checkpoint: {ckpt}")
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"OOD classifier not found: {ckpt}\n"
                    f"Train it first with:\n"
                    f"  ./server_scripts/submit_smile_celeba_ood_classifiers.sh "
                    f"--attrs \"{ood_attr}\""
                )
        else:
            _log(f"Standard classifier checkpoint: {ckpt}"
                 + (f"  (baseline: use_ood_classifier=False)" if ood_attr else ""))
        return str(ckpt)

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
            mode_dir = base_dir / "certify_ood" / ood_attr.lower() / mode_tag
        else:
            mode_dir = base_dir / "certify" / mode_tag

        ablation_study = getattr(cfg.output, "ablation_study", None)
        ablation_variant = getattr(cfg.output, "ablation_variant", None)
        if bool(ablation_study) != bool(ablation_variant):
            raise ValueError(
                "output.ablation_study and output.ablation_variant must either "
                "both be set or both be null"
            )
        if ablation_study:
            experiment_dir = (
                mode_dir / "ablation" / sigma_tag
                / str(ablation_study) / str(ablation_variant)
            )
        else:
            experiment_dir = mode_dir / sigma_tag
        
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
    # Do NOT pass ood_exclude_attribute here — train_samples must be the FULL
    # train partition so the index filenames match the built index (162,079 vectors).
    # certify_ood_samples (train-partition attr=1) is extracted manually below.
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
        split_seed=cfg.dataset.split_seed,
    )
    
    loader_cfg = SmileDataloaderConfig(batch_size=64, shuffle_train=False, pin_memory=True)
    model_cfg = SmileModelConfig(name=cfg.model.name, pretrained=False, dropout=0.0, input_size=cfg.model.input_size)
    
    bundle = build_smile_dataloaders(dataset_cfg, loader_cfg, model_cfg)
    # Full train partition — must match the built index exactly (all 162,079 vectors)
    train_samples = list(bundle.train_loader.dataset.samples)

    # OOD certification: extract train-partition attr=1 samples manually.
    # These are truly unseen (excluded from OOD classifier training, not in val/test).
    # We do NOT filter train_samples itself so the index filenames stay consistent.
    ood_attr_certify = getattr(cfg.dataset, "ood_attribute", None)
    if ood_attr_certify:
        attr_path = Path(cfg.dataset.root_dir) / cfg.dataset.annotation_file
        lines = [l.strip() for l in attr_path.read_text().splitlines() if l.strip()]
        attr_names = lines[1].split()
        if ood_attr_certify in attr_names:
            aidx = attr_names.index(ood_attr_certify)
            attr_map = {row.split()[0]: int(row.split()[1 + aidx])
                        for row in lines[2:] if len(row.split()) > aidx + 1}
            test_samples = [(p, l) for p, l in train_samples
                            if attr_map.get(Path(p).name, -1) == 1]
            _log(f"OOD certification: {len(test_samples)} train-partition attr=1 samples "
                 f"(index consistent — full {len(train_samples)} train vectors kept)")
        else:
            test_samples = list(bundle.test_loader.dataset.samples)
    else:
        test_samples = list(bundle.test_loader.dataset.samples)

    _log(f"Train samples (index): {len(train_samples)}, Certify samples: {len(test_samples)}")
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
        
        dataloader = build_dataloader_from_samples(train_samples, image_size)

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
        
        dataloader = build_dataloader_from_samples(train_samples, vae.image_size)
        
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
            scale_noise=getattr(cfg.smoothing, 'scale_noise', True),
            pca_dim=getattr(cfg.smoothing, "pca_dim", None),
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
            scale_noise=getattr(cfg.smoothing, 'scale_noise', True),
            pca_dim=getattr(cfg.smoothing, "pca_dim", None),
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

# ── Visualization colour palette ──────────────────────────────────────────────
_VC = {
    'knn_ood1':   '#aec7e8',   # light blue  — KNN neighbours with OOD attr
    'knn_ood0':   '#6b3fa0',   # purple      — KNN neighbours without OOD attr
    'knn_plain':  '#aec7e8',   # light blue  — KNN (no OOD info)
    'mc':         '#74c476',   # green       — MC noise samples
    'iso_circle': '#2166ac',   # deep blue   — isotropic circle
    'mani_ellip': '#f5c518',   # yellow/gold — manifold ellipse
    'anchor':     'black',     # black       — anchor star
    'iso_border': '#2166ac',   # deep blue   — iso image border
    'mani_border':'#6b3fa0',   # purple      — manifold image border
    'nn_border':  '#006d2c',   # dark green  — neighbour image border
}

_VIZ_STYLE = {
    'font.family':        'serif',
    'font.size':          10,
    'axes.titlesize':     10,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    7.5,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linestyle':     '--',
    'savefig.dpi':        150,
    'savefig.bbox':       'tight',
}

def _apply_viz_style():
    import matplotlib
    matplotlib.rcParams.update(_VIZ_STYLE)


def _draw_geometry_figure(
    pca_obj,
    nbrs,           # (k, D) centred neighbour matrix
    query_vec,      # (D,) anchor in PCA space
    smoother,       # ManifoldSmoother — used to draw MC samples
    cached_pca,     # cached PCA result passed to sample_from_cached
    sigma: float,
    n_mc: int,
    save_path,
    sample_idx: int,
    space_label: str,       # "Pixel" or "Latent"
    ood_attr_name: str,     # e.g. "Wearing_Hat" or "" for non-OOD
    ood_attr_map,           # filename→0/1 dict or None
    index,                  # NeighborIndex for neighbour OOD lookup
    vae_decode_fn=None,     # unused here, kept for signature compat
) -> None:
    """Shared geometry figure for both pixel-manifold and latent-manifold.

    Shows 5 zoom panels (A–E):
      A  Full neighbour cloud
      B  Mid-zoom ±10% PC1 range
      C  Tight ±a1 (ellipse semi-axis)
      D  Mid PC axis
      E  Last PC axis

    Each panel shows:
      - KNN neighbours (coloured by OOD attr if available, grey otherwise)
      - MC noise samples (single colour — no MC OOD matching)
      - Isotropic circle (dashed blue)
      - Manifold ellipse (solid orange)
      - Anchor star

    Legend is placed outside the plot (right of each panel) to avoid overlap.
    """
    _apply_viz_style()
    try:
        import matplotlib.pyplot as _plt
        from matplotlib.patches import Ellipse as _Ellipse
    except ImportError:
        return

    _ev = np.asarray(pca_obj.evals, dtype=np.float64)
    _Vt = pca_obj.evecs.T                          # (n_comp, D)
    _ev_norm = np.maximum(_ev, 1e-12) / float(_ev.max())
    _lambda_max = float(_ev[0])
    _sqrt_lmax  = float(np.sqrt(_lambda_max))
    _alpha      = sigma / np.sqrt(max(_lambda_max, 1e-12))

    _anchor_2d = (query_vec - np.asarray(pca_obj.mean, dtype=np.float64)) @ _Vt[:2].T
    _neigh_2d  = nbrs.astype(np.float64) @ _Vt[:2].T

    _a1 = float(sigma * np.sqrt(_ev_norm[0]))
    _a2 = float(sigma * np.sqrt(_ev_norm[1]))
    _mid_idx = len(_ev_norm) // 2
    _a_mid   = float(sigma * np.sqrt(_ev_norm[_mid_idx]))
    _a_last  = float(sigma * np.sqrt(_ev_norm[-1]))
    _pc1_std   = float(np.std(_neigh_2d[:, 0]))
    _pc1_range = float(np.max(_neigh_2d[:, 0]) - np.min(_neigh_2d[:, 0]))
    _zoom_mid  = 0.10 * _pc1_range
    _zoom_tight = _a1

    _zoom_panels = [
        (None,         f"(a) Full neighbourhood cloud",                _a1, _a2),
        (_zoom_mid,    f"(b) Mid-zoom  ±{_zoom_mid:.2f}",             _a1, _a2),
        (_zoom_tight,  f"(c) Ellipse axes  a₁={_a1:.3f}  a₂={_a2:.3f}",  _a1, _a2),
        (_zoom_tight,  f"(d) Mid axis  a_mid={_a_mid:.3f}\na_mid/a₁={_a_mid/(_a1+1e-12):.3f}", _a1, _a_mid),
        (_zoom_tight,  f"(e) Last axis  a_last={_a_last:.2e}\na_last/a₁={_a_last/(_a1+1e-12):.2e}", _a1, _a_last),
    ]

    # MC samples — plain colour, no OOD matching
    _mc_2d_list = []
    for _ in range(n_mc):
        _ns = smoother.sample_from_cached(cached_pca)
        _ns_c = _ns - np.asarray(pca_obj.mean, dtype=np.float64)
        _mc_2d_list.append(_ns_c @ _Vt[:2].T)
    _mc_2d = np.array(_mc_2d_list)

    # Neighbour OOD colouring (if ood_attr_map provided)
    _nn_has_attr = np.zeros(len(_neigh_2d), dtype=bool)
    if ood_attr_map is not None and index is not None and hasattr(index, "filenames") and hasattr(index, "index") and hasattr(index.index, "get_nns_by_vector"):
        _nn_ids = index.index.get_nns_by_vector(
            query_vec.astype(np.float32), len(_neigh_2d), include_distances=False)
        for _ni, _nid in enumerate(_nn_ids[:len(_neigh_2d)]):
            _nn_has_attr[_ni] = bool(ood_attr_map.get(Path(index.filenames[_nid]).name, 0))

    _x_all = np.concatenate([_neigh_2d[:, 0], [_anchor_2d[0]]])
    _y_all = np.concatenate([_neigh_2d[:, 1], [_anchor_2d[1]]])
    _pad = 0.05 * max(float(np.ptp(_x_all)), float(np.ptp(_y_all)), 1e-6)

    _fig, _axes = _plt.subplots(1, 5, figsize=(26, 5.5), facecolor="white")

    for _ax, (_zr, _ptitle, _ea1, _ea2) in zip(_axes, _zoom_panels):
        if _zr is None:
            _mask    = np.ones(len(_neigh_2d), dtype=bool)
            _mask_mc = np.ones(len(_mc_2d),    dtype=bool)
        else:
            _mask = (
                (np.abs(_neigh_2d[:, 0] - _anchor_2d[0]) <= _zr) &
                (np.abs(_neigh_2d[:, 1] - _anchor_2d[1]) <= _zr)
            )
            _mask_mc = (
                (np.abs(_mc_2d[:, 0] - _anchor_2d[0]) <= _zr) &
                (np.abs(_mc_2d[:, 1] - _anchor_2d[1]) <= _zr)
            )

        # Neighbours
        _has = _mask & _nn_has_attr
        _no  = _mask & ~_nn_has_attr
        if ood_attr_map is not None:
            if _has.any():
                _ax.scatter(_neigh_2d[_has, 0], _neigh_2d[_has, 1],
                            s=5, alpha=0.30, color=_VC['knn_ood1'], linewidths=0, zorder=1,
                            label=f"KNN ({ood_attr_name}=1)")
            if _no.any():
                _ax.scatter(_neigh_2d[_no, 0], _neigh_2d[_no, 1],
                            s=8, alpha=0.40, color=_VC['knn_ood0'], linewidths=0, zorder=2,
                            label=f"KNN ({ood_attr_name}=0)")
        else:
            _ax.scatter(_neigh_2d[_mask, 0], _neigh_2d[_mask, 1],
                        s=5, alpha=0.30, color=_VC['knn_plain'], linewidths=0, zorder=1,
                        label="KNN neighbours")

        # MC samples
        if _mask_mc.any():
            _ax.scatter(_mc_2d[_mask_mc, 0], _mc_2d[_mask_mc, 1],
                        s=6, alpha=0.45, color=_VC['mc'], marker="o", linewidths=0,
                        zorder=3, label=f"MC samples (n={n_mc})")

        # Isotropic circle
        _ax.add_patch(_plt.Circle(
            (_anchor_2d[0], _anchor_2d[1]), sigma,
            fill=False, edgecolor=_VC['iso_circle'], linewidth=2.0,
            linestyle=(0, (4, 2)), alpha=0.95, zorder=4,
            label=f"Iso circle  r=σ={sigma}",
        ))
        # Manifold ellipse
        _ax.add_patch(_Ellipse(
            (_anchor_2d[0], _anchor_2d[1]),
            width=2.0 * _ea1, height=2.0 * _ea2,
            fill=False, edgecolor=_VC['mani_ellip'], linewidth=2.0,
            linestyle="solid", alpha=0.95, zorder=4,
            label=f"Manifold ellipse  a1={_ea1:.4f}",
        ))
        # Anchor — always black star
        _ax.scatter(_anchor_2d[0], _anchor_2d[1], s=180, marker="*",
                    c=_VC['anchor'], edgecolors="white", linewidths=0.8, zorder=5, label="Anchor")

        if _zr is None:
            _ax.set_xlim(float(np.min(_x_all)) - _pad, float(np.max(_x_all)) + _pad)
            _ax.set_ylim(float(np.min(_y_all)) - _pad, float(np.max(_y_all)) + _pad)
        else:
            _ax.set_xlim(_anchor_2d[0] - _zr, _anchor_2d[0] + _zr)
            _ax.set_ylim(_anchor_2d[1] - _zr, _anchor_2d[1] + _zr)

        _ax.set_aspect("equal")
        _ax.set_title(_ptitle, fontsize=8.5)
        _ax.set_xlabel(f"PC1  (std={_pc1_std:.2f})", fontsize=8)
        _ax.set_ylabel("PC2", fontsize=8)
        _ax.grid(alpha=0.25)
        # No per-axis legend — shared one drawn below

    # Single shared horizontal legend at bottom (same handles for all subplots)
    _handles, _labels = _axes[0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="lower center", ncol=len(_handles),
                fontsize=7.5, framealpha=0.9,
                bbox_to_anchor=(0.5, -0.08), borderaxespad=0)

    _ood_line = f"\nOOD: {ood_attr_name}=1" if ood_attr_name else ""
    _fig.suptitle(
        f"{space_label} Manifold Smoothing Geometry   σ={sigma}"
        f"\nλ_max={_lambda_max:.4f}   √λ_max={_sqrt_lmax:.4f}   α=σ/√λ_max={_alpha:.4f}"
        f"{_ood_line}",
        fontsize=11, y=1.02,
    )
    _plt.tight_layout()
    _fig.savefig(save_path, dpi=150, bbox_inches="tight")
    _plt.close(_fig)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [0,1] tensor (C,H,W) to PIL Image."""
    img = tensor.clamp(0, 1)
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


def _save_pixel_manifold_report_visualization(
    viz_dir: Path,
    sample_idx: int,
    img_tensor: torch.Tensor,
    manifold_imgs: List[torch.Tensor],
    isotropic_imgs: List[torch.Tensor],
    nn_imgs: List[torch.Tensor],
) -> None:
    """Save a compact, paper-ready manifold comparison and its raw panels.

    This is an additional output; the existing ``sample_NNNN.png`` grid is
    deliberately left unchanged.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Export individual panels so they can be rearranged directly in a report.
    panel_dir = viz_dir / "report_samples" / f"sample_{sample_idx:04d}"
    groups = {
        "manifold": manifold_imgs,
        "isotropic": isotropic_imgs,
        "neighbours": nn_imgs,
    }
    panel_dir.mkdir(parents=True, exist_ok=True)
    _tensor_to_pil(img_tensor).save(panel_dir / "original.png")
    for group_name, images in groups.items():
        group_dir = panel_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        singular = "neighbour" if group_name == "neighbours" else f"{group_name}_noise"
        for image_idx, image in enumerate(images, start=1):
            _tensor_to_pil(image).save(group_dir / f"{singular}_{image_idx:02d}.png")

    n_cols = 5
    top_row = [img_tensor] + list(nn_imgs[: n_cols - 1])
    rows = [top_row, list(manifold_imgs[:n_cols]), list(isotropic_imgs[:n_cols])]
    row_labels = [
        r"Original ($x$) / NN ($\mathcal{N}_K(x)$)",
        r"Manifold ($x + \delta_x$)",
        r"Isotropic ($x + \epsilon$)",
    ]

    # STIX closely matches the serif/math typography used by common paper
    # templates while remaining available through Matplotlib itself.
    with plt.rc_context({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 14,
        "figure.facecolor": "white",
    }):
        fig, axes = plt.subplots(3, n_cols, figsize=(10.2, 6.05))
        for row_idx, images in enumerate(rows):
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.axis("off")
                if col_idx < len(images):
                    ax.imshow(_tensor_to_pil(images[col_idx]))
                    ax.set_aspect("equal")

        # Keep the rotated labels in a narrow gutter beside the image grid.
        row_centres = (0.835, 0.505, 0.175)
        for y_pos, label_text in zip(row_centres, row_labels):
            fig.text(
                0.021, y_pos, label_text,
                rotation=90, va="center", ha="center",
                fontsize=11, fontweight="normal",
            )

        fig.subplots_adjust(
            left=0.030, right=0.997, bottom=0.005, top=0.995,
            wspace=0.025, hspace=0.035,
        )
        fig.savefig(
            viz_dir / f"sample_{sample_idx:04d}_report.png",
            dpi=300, bbox_inches="tight", pad_inches=0.025,
        )
        plt.close(fig)


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
    _apply_viz_style()
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
        row_labels = ["Original", "Isotropic noise"]
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_noisy_samples,
                                 figsize=(3 * n_noisy_samples, 3.2 * n_rows))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")

        # Row descriptions are already shown as titles above the first image.
        # for r, lbl in enumerate(row_labels):
        #     _add_row_label(axes, r, lbl)

        axes[0, 0].imshow(_tensor_to_pil(img_tensor))
        axes[0, 0].set_title("Original", fontsize=9)

        for i in range(n_noisy_samples):
            axes[1, i].imshow(_tensor_to_pil(sample_pixel(img_tensor, iso_pixel)))
            if i == 0:
                axes[1, i].set_title(f"Isotropic noise at  σ={sigma}", fontsize=9)

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
                (None,        f"(a) Full noise cloud  σ/std={sigma/(_pc1_std+1e-12):.2f}"),
                (_zoom_mid,   f"(b) Mid-zoom  ±3σ={_zoom_mid:.3f}"),
                (_zoom_tight, f"(c) Tight zoom  ±σ={sigma:.3f}"),
            ]

            # MC OOD colouring — disabled (per-sample Annoy queries slow); re-enable if needed
            # _iso_mc_is_ood = np.zeros(len(_mc_flat), dtype=bool)
            # if (ood_attr_map is not None and index is not None
            #         and hasattr(index, "index") and hasattr(index.index, "get_nns_by_vector")
            #         and hasattr(index, "filenames")):
            #     for _mi, _ms in enumerate(_mc_flat):
            #         _nid = index.index.get_nns_by_vector(_ms.tolist(), 1, include_distances=False)
            #         if _nid:
            #             _fname = Path(index.filenames[_nid[0]]).name
            #             _iso_mc_is_ood[_mi] = bool(ood_attr_map.get(_fname, 0))

            _gfig, _gaxes = _plt_geom.subplots(1, 3, figsize=(15, 5), facecolor="white")
            _mc_pad = 0.05 * max(float(np.ptp(_mc_2d[:, 0])), float(np.ptp(_mc_2d[:, 1])), 1e-6)
            # Total counts — only needed when OOD split is re-enabled above
            # _n_iso_ood_total     = int(_iso_mc_is_ood.sum())
            # _n_iso_not_ood_total = int((~_iso_mc_is_ood).sum())

            for _gax, (_zr, _gtitle) in zip(_gaxes, _zoom_specs):
                if _zr is None:
                    _mask_mc = np.ones(len(_mc_2d), dtype=bool)
                else:
                    _mask_mc = (
                        (np.abs(_mc_2d[:, 0] - _anchor_2d[0]) <= _zr) &
                        (np.abs(_mc_2d[:, 1] - _anchor_2d[1]) <= _zr)
                    )

                # MC OOD split — disabled; uncomment to re-enable
                # _iso_vis_ood     = _mask_mc & _iso_mc_is_ood
                # _iso_vis_not_ood = _mask_mc & ~_iso_mc_is_ood
                # if not _iso_vis_ood.any() and not _iso_vis_not_ood.any():
                #     _gax.scatter(_mc_2d[_mask_mc, 0], _mc_2d[_mask_mc, 1],
                #                  s=6, alpha=0.40, color="#4c78a8", marker="o", linewidths=0,
                #                  zorder=3, label=f"Iso MC samples (n={_mask_mc.sum()})")
                # else:
                #     _gax.scatter(_mc_2d[_iso_vis_ood, 0], _mc_2d[_iso_vis_ood, 1],
                #                  s=6, alpha=0.50, color="#2ca02c", marker="o", linewidths=0,
                #                  zorder=3, label=f"MC in OOD territory ({_n_iso_ood_total}/{_N_mc})")
                #     _gax.scatter(_mc_2d[_iso_vis_not_ood, 0], _mc_2d[_iso_vis_not_ood, 1],
                #                  s=6, alpha=0.50, color="#17becf", marker="o", linewidths=0,
                #                  zorder=3, label=f"MC in non-OOD territory ({_n_iso_not_ood_total}/{_N_mc})")

                # MC samples — single colour, no OOD split
                _gax.scatter(_mc_2d[_mask_mc, 0], _mc_2d[_mask_mc, 1],
                             s=6, alpha=0.45, color=_VC['mc'], marker="o", linewidths=0,
                             zorder=3, label=f"Iso MC samples (n={_N_mc})")
                # iso circle
                _gax.add_patch(_plt_geom.Circle(
                    (_anchor_2d[0], _anchor_2d[1]), sigma,
                    fill=False, edgecolor=_VC['iso_circle'], linewidth=2,
                    linestyle=(0, (4, 2)), alpha=0.9, zorder=4, label=f"σ-circle  r={sigma}",
                ))
                _gax.scatter(_anchor_2d[0], _anchor_2d[1], s=180, marker="*",
                             c=_VC['anchor'], edgecolors="white", linewidths=0.8, zorder=5, label="Anchor")
                if _zr is None:
                    _gax.set_xlim(float(np.min(_mc_2d[:, 0])) - _mc_pad, float(np.max(_mc_2d[:, 0])) + _mc_pad)
                    _gax.set_ylim(float(np.min(_mc_2d[:, 1])) - _mc_pad, float(np.max(_mc_2d[:, 1])) + _mc_pad)
                else:
                    _gax.set_xlim(_anchor_2d[0] - _zr, _anchor_2d[0] + _zr)
                    _gax.set_ylim(_anchor_2d[1] - _zr, _anchor_2d[1] + _zr)
                _gax.set_aspect("equal")
                _gax.set_title(_gtitle, fontsize=9)
                _gax.set_xlabel(f"PC1  (std={_pc1_std:.4f})", fontsize=8)
                _gax.set_ylabel(f"PC2  (std={_pc2_std:.4f})", fontsize=8)
                _gax.grid(alpha=0.25)
                _gax.get_legend_handles_labels()  # collect but don't draw per-axis

            # Single shared horizontal legend at bottom
            _handles, _labels = _gaxes[0].get_legend_handles_labels()
            _gfig.legend(_handles, _labels, loc="lower center", ncol=len(_handles),
                         fontsize=8, framealpha=0.9,
                         bbox_to_anchor=(0.5, -0.08), borderaxespad=0)

            _ood_attr_iso = getattr(cfg.dataset, 'ood_attribute', None) if ood_attr_map is not None else None
            _ood_line_iso = f"\nOOD: {_ood_attr_iso}=1" if _ood_attr_iso else ""
            _gfig.suptitle(
                f"Isotropic Smoothing Geometry   σ={sigma}   MC samples={_N_mc}{_ood_line_iso}",
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
            "Original",
            "Isotropic\nlatent noise σ",
            "Isotropic\npixel noise σ",
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
        scaling_on = True
        if isinstance(manifold_sm, ManifoldSmoother):
            _cached_pca   = manifold_sm.compute_pca(query_vec)
            lambda_max    = float(_cached_pca.pca.evals[0])
            alpha_display = sigma / np.sqrt(max(lambda_max, 1e-12))
            scaling_on    = manifold_sm._scale_noise

        noise_label = (f"Manifold noise\nα=σ/√λ_max={alpha_display:.4f}"
                       if scaling_on else f"Manifold noise\nσ={sigma}")
        # (no scaling)
        row_labels = ["Original", noise_label]
        if scaling_on:
            row_labels.append(f"Manifold noise\nσ={sigma}")
            # unscaled
        row_labels += [f"Isotropic\npixel noise σ={sigma}", "Neighbours"]
        n_rows = len(row_labels)
        fig, axes = plt.subplots(n_rows, n_noisy_samples,
                                 figsize=(3 * n_noisy_samples, 3.5 * n_rows))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")

        # Row descriptions are already shown as titles above the first image.
        # for r, lbl in enumerate(row_labels):
        #     _add_row_label(axes, r, lbl)

        # Row 0: Original in col 0 only
        axes[0, 0].imshow(_tensor_to_pil(img_tensor))
        axes[0, 0].set_title("Original", fontsize=9)

        # Row 1: Manifold noise (alpha scaled) — uses smoother.sample_from_cached()
        report_manifold_imgs: List[torch.Tensor] = []
        for i in range(n_noisy_samples):
            if isinstance(manifold_sm, ManifoldSmoother):
                noisy_flat = manifold_sm.sample_from_cached(_cached_pca)
                noisy_t    = torch.from_numpy(noisy_flat.reshape(img_tensor.shape)).float()
            else:
                noisy_t = sample_pixel(img_tensor, manifold_sm)
            report_manifold_imgs.append(noisy_t)
            axes[1, i].imshow(_tensor_to_pil(noisy_t))
            if i == 0:
                axes[1, i].set_title(f"Manifold noise  α={alpha_display:.4f}" if scaling_on else f"Manifold noise\nσ={sigma}", fontsize=9)

        # Row 2: Manifold noise unscaled — only shown when scale_noise=True
        if scaling_on:
            for i in range(n_noisy_samples):
                if isinstance(manifold_sm, ManifoldSmoother):
                    pca = _cached_pca.pca
                    noise_w    = np.random.normal(0.0, sigma, size=len(pca.evals)).astype(np.float32)
                    noise_orig = (noise_w * np.sqrt(pca.evals)) @ pca.evecs.T
                    noisy_flat = query_vec + noise_orig
                    noisy_t    = torch.from_numpy(noisy_flat.reshape(img_tensor.shape)).float()
                else:
                    noisy_t = sample_pixel(img_tensor, iso_pixel)
                axes[2, i].imshow(_tensor_to_pil(noisy_t))
                if i == 0:
                    axes[2, i].set_title(f"Manifold noise  σ={sigma}", fontsize=9)
                    # unscaled

        # Dynamic row offset — if scaling is off, unscaled row was removed
        iso_row = 3 if scaling_on else 2
        nn_row  = 4 if scaling_on else 3

        # Isotropic pixel noise
        report_isotropic_imgs: List[torch.Tensor] = []
        for i in range(n_noisy_samples):
            isotropic_t = sample_pixel(img_tensor, iso_pixel)
            report_isotropic_imgs.append(isotropic_t)
            axes[iso_row, i].imshow(_tensor_to_pil(isotropic_t))
            if i == 0:
                axes[iso_row, i].set_title(f"Isotropic noise σ={sigma}", fontsize=9)

        # Neighbours
        nn_imgs = _get_nn_images(index, query_vec, n_noisy_samples,
                                 img_tensor.shape, vae, device, is_latent=False)
        for i in range(n_noisy_samples):
            if i < len(nn_imgs):
                axes[nn_row, i].imshow(_tensor_to_pil(nn_imgs[i]))
            if i == 0:
                axes[nn_row, i].set_title("Neighbours", fontsize=9)

        _save_pixel_manifold_report_visualization(
            viz_dir=viz_dir,
            sample_idx=sample_idx,
            img_tensor=img_tensor,
            manifold_imgs=report_manifold_imgs,
            isotropic_imgs=report_isotropic_imgs,
            nn_imgs=nn_imgs,
        )

        # ── Geometry figure ───────────────────────────────────────────────
        if isinstance(manifold_sm, ManifoldSmoother):
            try:
                _draw_geometry_figure(
                    pca_obj=_cached_pca.pca,
                    nbrs=_cached_pca.neighbors,
                    query_vec=query_vec,
                    smoother=manifold_sm,
                    cached_pca=_cached_pca,
                    sigma=sigma,
                    n_mc=cfg.smoothing.n_samples,
                    save_path=viz_dir / f"sample_{sample_idx:04d}_geometry.png",
                    sample_idx=sample_idx,
                    space_label="Pixel",
                    ood_attr_name=getattr(cfg.dataset, "ood_attribute", None) or "",
                    ood_attr_map=ood_attr_map,
                    index=index,
                )
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
            "Original &\nPCA Recon (Latent)",
            f"Latent manifold\nα=σ/√λ_max={alpha_display:.4f}\n(certified)",
            f"Latent manifold\nσ={sigma} unscaled\n(ref)",
            "Pixel manifold\nα=σ/√λ_max (decoded)",
            f"Latent iso\nnoise σ={sigma}",
            f"Pixel iso\nnoise σ={sigma}",
            "Neighbours",
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
                _draw_geometry_figure(
                    pca_obj=_cached_latent_pca.pca,
                    nbrs=_cached_latent_pca.neighbors,
                    query_vec=query_vec,
                    smoother=manifold_sm_latent,
                    cached_pca=_cached_latent_pca,
                    sigma=sigma,
                    n_mc=cfg.smoothing.n_samples,
                    save_path=viz_dir / f"sample_{sample_idx:04d}_geometry_latent.png",
                    sample_idx=sample_idx,
                    space_label="Latent",
                    ood_attr_name=getattr(cfg.dataset, "ood_attribute", None) or "",
                    ood_attr_map=ood_attr_map,
                    index=index,
                )
            except Exception as _geom_err_lat:
                _log(f"Latent geometry figure skipped for sample {sample_idx}: {_geom_err_lat}")

    # ------------------------------------------------------------------
    # Title with certification result
    # ------------------------------------------------------------------
    smoothing_type = "Manifold" if cfg.smoothing.use_manifold else "Isotropic"
    pred_label = "smile" if pred == 1 else "no smile"
    true_label = "smile" if label == 1 else "no smile"
    _ood_attr_name = getattr(cfg.dataset, "ood_attribute", None) if hasattr(cfg, "dataset") else None
    _ood_line_samp = f"\nOOD: {_ood_attr_name}=1" if _ood_attr_name else ""
    fig.suptitle(
        # {cfg.smoothing.mode.capitalize()}
        f"{smoothing_type} Smoothing at σ={sigma}\n"
        f"True: {true_label}   Predicted: {pred_label}"
        f"{_ood_line_samp}",
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
        transforms.ToPILImage()(img_tensor.clamp(0, 1))
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
    while processed < total:
        chunk = min(CHUNK, total - processed)
        # Repeat clean image chunk times: (chunk, C, H, W)
        x_rep = x_clean.expand(chunk, -1, -1, -1).clone()

        # Add noise entirely on GPU
        if gpu_cache is not None and sample_index is not None:
            idx_rep = idx.expand(chunk)  # (chunk,)
            x_noisy = smoother.sample_batch_gpu(x_rep, gpu_cache=gpu_cache, indices=idx_rep)
        elif sample_fn is not None:
            # CPU loop: use pre-built sample_fn (kNN+SVD cached, needed for ManifoldSmoother).
            # sample_fn() returns a [0,1] tensor; convert to PIL → apply classifier_transform.
            processed_samples = []
            raw_chunk: List[torch.Tensor] = []
            for _ in range(chunk):
                s = sample_fn()  # (C, H, W) in [0, 1]
                raw_chunk.append(s)
                pil = transforms.ToPILImage()(s.clamp(0, 1))
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

        # Collect exact GPU-generated n-phase samples (for iso GPU batch path where sample_fn=None).
        # x_noisy is already classified — pulling it to CPU here adds only a memcpy, not a
        # re-forward. This gives the exact same samples that were used for certification.
        if _raw_n_samples is not None and sample_fn is None:
            for _ci in range(chunk):
                if processed + _ci >= n0_samples:
                    _raw_n_samples.append(x_noisy[_ci].detach().cpu())

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

    import time as _time
    _certify_start = _time.time()

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

        _is_iso = not cfg.smoothing.use_manifold
        # MC OOD tracking disabled — to re-enable: set collect_n_samples=_collect,
        # restore _collect flag, _pixel_ood_labels precompute, and MC hit loop below
        # _collect = (_ood_attr_map_global is not None
        #             and pixel_index is not None
        #             and hasattr(pixel_index, "index")
        #             and hasattr(pixel_index.index, "get_nns_by_vector")
        #             and hasattr(pixel_index, "filenames"))
        # cert, _cert_raw_samples = certify_single_sample(..., collect_n_samples=_collect)
        cert, _ = certify_single_sample(
            classifier=classifier,
            img_tensor=img_tensor,
            smoother=_active_smoother,
            n_samples=cfg.smoothing.n_samples,
            n0_samples=int(cfg.smoothing.n0_samples),
            classifier_transform=classifier_transform,
            device=device,
            alpha_conf=cfg.alpha_conf,
            sigma=cfg.smoothing.sigma,
            sample_fn=None if _is_iso else sample_fn,
            collect_n_samples=False,
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
            "ood_attr_value": 1 if ood_attr else None,
            "nn_ood_count": None,
            "nn_ood_frac": None,
            "mc_ood_count": None,   # disabled — kept for notebook compatibility
            "mc_ood_frac": None,    # disabled — kept for notebook compatibility
        }

        # KNN neighbour OOD fraction
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

        # MC OOD fraction — disabled (per-sample Annoy queries slow; re-enable if needed)
        # Each raw MC sample is looked up in pixel_index → inherits OOD label of nearest neighbour
        # if (_cert_raw_samples is not None and len(_cert_raw_samples) > 0
        #         and _pixel_ood_labels is not None
        #         and pixel_index is not None
        #         and hasattr(pixel_index, "index")):
        #     _mc_mat = np.stack([_rs.numpy().flatten().astype(np.float32)
        #                         for _rs in _cert_raw_samples])  # (N, D)
        #     _nn_ids_mc = np.array([
        #         pixel_index.index.get_nns_by_vector(_mc_mat[_i].tolist(), 1, include_distances=False)[0]
        #         for _i in range(len(_mc_mat))
        #     ], dtype=np.int32)
        #     _mc_hits = int(_pixel_ood_labels[_nn_ids_mc].sum())
        #     result["mc_ood_count"] = _mc_hits
        #     result["mc_ood_frac"]  = float(_mc_hits) / len(_cert_raw_samples)

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
    _certify_elapsed = _time.time() - _certify_start
    total = len(test_samples)
    _samples_processed = total - start_idx  # excludes resumed-over samples
    metrics = {
        # ── Identity ──────────────────────────────────────────────────────────
        "experiment": cfg.experiment_name,
        "dataset": cfg.dataset.name,
        "output_dir": str(paths.experiment_dir),

        # ── Classifier ────────────────────────────────────────────────────────
        "classifier": {
            "model": cfg.model.name,
            "checkpoint": resolve_classifier_checkpoint(cfg),
            "input_size": cfg.model.input_size,
            "dropout": cfg.model.dropout,
            "use_ood_classifier": getattr(cfg.model, "use_ood_classifier", True),
        },

        # ── Smoothing config ──────────────────────────────────────────────────
        "smoothing": {
            "mode": cfg.smoothing.mode,
            "use_manifold": cfg.smoothing.use_manifold,
            "sigma": cfg.smoothing.sigma,
            "n0_samples": int(cfg.smoothing.n0_samples),
            "n_samples": int(cfg.smoothing.n_samples),
            "total_mc_samples": int(cfg.smoothing.n0_samples) + int(cfg.smoothing.n_samples),
            "knn_k": cfg.smoothing.knn_k,
            "pca_dim": getattr(cfg.smoothing, "pca_dim", None),
            "eps_eig": cfg.smoothing.eps_eig,
            "alpha_conf": cfg.alpha_conf,
        },

        # ── VAE (latent mode) ─────────────────────────────────────────────────
        "vae": {
            "enabled": cfg.vae.enabled,
            "checkpoint": cfg.vae.checkpoint_path if cfg.vae.enabled else None,
            "image_size": cfg.vae.image_size if cfg.vae.enabled else None,
            "latent_dim": cfg.vae.latent_dim if cfg.vae.enabled else None,
        },

        # ── Index ─────────────────────────────────────────────────────────────
        "index": {
            "backend": cfg.index.backend,
            "metric": cfg.index.metric,
            "n_trees": cfg.index.n_trees,
            "split": "train",
            "n_train_samples": len(train_samples),
        },

        # ── Dataset / split ───────────────────────────────────────────────────
        "dataset_info": {
            "name": cfg.dataset.name,
            "root_dir": cfg.dataset.root_dir,
            "train_samples": len(train_samples),
            "test_samples": total,
            "split_seed": cfg.dataset.split_seed,
            "train_ratio": cfg.dataset.train_ratio,
            "val_ratio": cfg.dataset.val_ratio,
            "ood_attribute": ood_attr if ood_attr else None,
            "ood_attr_value": 1 if ood_attr else None,
            "ood_balanced": cfg.dataset.ood_balanced if ood_attr else None,
        },

        # ── Runtime ───────────────────────────────────────────────────────────
        "runtime": {
            "certify_seconds": round(_certify_elapsed, 2),
            "certify_minutes": round(_certify_elapsed / 60, 2),
            "samples_processed": _samples_processed,
            "seconds_per_sample": round(_certify_elapsed / max(_samples_processed, 1), 4),
            "device": str(device),
            "resumed_from_sample": start_idx if start_idx > 0 else None,
        },

        # ── Results ───────────────────────────────────────────────────────────
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

    # ── OOD neighbour stats (KNN only — MC index queries disabled) ───────────
    _nn_fracs  = [r["nn_ood_frac"]  for r in results if r.get("nn_ood_frac")  is not None]
    _nn_counts = [r["nn_ood_count"] for r in results if r.get("nn_ood_count") is not None]
    metrics["ood_stats"] = {
        "ood_attribute": ood_attr if ood_attr else None,
        "nn_samples_with_data": len(_nn_fracs),
        "mean_nn_ood_frac":   float(np.mean(_nn_fracs))   if _nn_fracs  else None,
        "median_nn_ood_frac": float(np.median(_nn_fracs)) if _nn_fracs  else None,
        "mean_nn_ood_count":  float(np.mean(_nn_counts))  if _nn_counts else None,
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
    _log(f"Sampling:           n0={metrics['smoothing']['n0_samples']}, n={metrics['smoothing']['n_samples']}, total={metrics['smoothing']['total_mc_samples']}")
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
    if _ood_s.get("mean_nn_ood_frac") is not None:
        _log(f"OOD stats  [{_ood_s.get('ood_attribute')}=1]:")
        _log(f"  KNN neighbour OOD frac: mean={_ood_s['mean_nn_ood_frac']:.3f}  "
             f"median={_ood_s['median_nn_ood_frac']:.3f}  "
             f"(n={_ood_s['nn_samples_with_data']})")
        # MC OOD logging disabled — uncomment to re-enable
        # if _ood_s.get("mean_mc_ood_frac") is not None:
        #     _log(f"  MC sample OOD frac: mean={_ood_s['mean_mc_ood_frac']:.3f}  "
        #          f"median={_ood_s['median_mc_ood_frac']:.3f}  "
        #          f"(n={_ood_s['mc_samples_with_data']})")

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
            if eigen_samples and getattr(cfg.output, "save_eigenvalues", True):
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
# Multi-sigma certification (one-time PCA, loop over sigmas)
# ─────────────────────────────────────────────────────────────────────────────


def run_certification_multi_sigma(cfg: CertifyConfig, sigma_values: List[float]) -> List[Dict]:
    """Certify across multiple sigmas with a single shared setup pass.

    For each test sample:
      - Image is loaded once
      - kNN + PCA is computed once  (manifold mode)
      - VAE encode is computed once (latent mode)
    Then for every sigma the noise is applied and the sample is certified.

    Each sigma writes its results to the *identical* folder hierarchy as
    run_certification() — certify/{mode_tag}/sigma_{s}/ — so downstream
    scripts, the quad-array skip logic, and the analysis notebooks all work
    unchanged.

    Partial-state checkpoints and skip-if-metrics-exists are maintained
    per sigma, exactly as in the single-sigma pipeline.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if int(cfg.smoothing.n0_samples) <= 0:
        raise ValueError("n0_samples must be > 0")
    if int(cfg.smoothing.n_samples) <= 0:
        raise ValueError("n_samples must be > 0")
    if not sigma_values:
        raise ValueError("sigma_values must not be empty")

    sigma_values = sorted(set(float(s) for s in sigma_values))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")
    _log(f"Multi-sigma sweep: {sigma_values}")

    # ── shared paths (index dirs are sigma-independent) ───────────────────────
    paths = CertifyPaths.from_config(cfg)
    paths.ensure_dirs()

    # ── train / test samples (same split for all sigmas) ─────────────────────
    train_samples, test_samples = get_train_test_samples(cfg)
    test_samples = get_ood_test_samples(cfg, test_samples)
    ood_attr = getattr(cfg.dataset, "ood_attribute", None)
    _log(f"Train: {len(train_samples)}  Test: {len(test_samples)}")

    dataset_info = {
        "dataset": cfg.dataset.name,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "seed": cfg.seed,
        "ood_attribute": ood_attr if ood_attr else None,
        "ood_attr_value": 1 if ood_attr else None,
        "ood_balanced": cfg.dataset.ood_balanced if ood_attr else None,
    }
    (paths.dataset_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))

    # ── classifier ────────────────────────────────────────────────────────────
    classifier = build_resnet_classifier(
        name=cfg.model.name,
        pretrained=False,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
    ).to(device)
    ckpt_path = resolve_classifier_checkpoint(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        classifier.load_state_dict(ckpt["model_state_dict"])
    elif "model_state" in ckpt:
        classifier.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and "conv1.weight" not in ckpt:
        for key in ["state_dict", "model"]:
            if key in ckpt:
                classifier.load_state_dict(ckpt[key])
                break
        else:
            raise ValueError(f"Cannot find weights in checkpoint. Keys: {list(ckpt.keys())}")
    else:
        classifier.load_state_dict(ckpt)
    classifier.eval()
    _log(f"Classifier: {ckpt_path}")

    classifier_transform = transforms.Compose([
        transforms.Resize((cfg.model.input_size, cfg.model.input_size)),
        transforms.ToTensor(),
    ])

    # ── VAE ───────────────────────────────────────────────────────────────────
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

    # ── indices ───────────────────────────────────────────────────────────────
    pixel_index = None
    latent_index = None
    pixel_size = cfg.model.input_size
    _need_pixel_index_for_ood = bool(ood_attr)
    if cfg.smoothing.use_manifold or (cfg.smoothing.mode in ("latent", "both") and cfg.output.save_visualizations) or _need_pixel_index_for_ood:
        pixel_index = load_or_build_pixel_index(
            train_samples, pixel_size, paths.pixel_index_dir, cfg.index.n_trees,
            metric=cfg.index.metric,
        )
    _need_latent_index = (
        cfg.smoothing.mode in ("latent", "both") and vae is not None and
        (cfg.smoothing.use_manifold or _need_pixel_index_for_ood)
    )
    if _need_latent_index:
        latent_index = load_or_build_latent_index(
            train_samples, vae, paths.latent_index_dir, cfg.index.n_trees, device,
            metric=cfg.index.metric,
        )

    smooth_size = pixel_size
    smooth_transform = transforms.Compose([
        transforms.Resize((smooth_size, smooth_size)),
        transforms.ToTensor(),
    ])

    # ── OOD map (shared across all sigmas) ────────────────────────────────────
    _ood_attr_map_global: Optional[dict] = None
    if ood_attr:
        _attr_path_g = Path(cfg.dataset.root_dir) / cfg.dataset.annotation_file
        _lines_g = [l.strip() for l in _attr_path_g.read_text().splitlines() if l.strip()]
        _attr_names_g = _lines_g[1].split()
        if ood_attr in _attr_names_g:
            _aidx_g = _attr_names_g.index(ood_attr)
            _ood_attr_map_global = {}
            for _row_g in _lines_g[2:]:
                _parts_g = _row_g.split()
                _ood_attr_map_global[_parts_g[0]] = 1 if int(_parts_g[1 + _aidx_g]) == 1 else 0

    # MC OOD tracking disabled — uncomment to re-enable per-sample index queries
    # _pixel_ood_labels: Optional[np.ndarray] = None
    # if (_ood_attr_map_global is not None and pixel_index is not None and hasattr(pixel_index, "filenames")):
    #     _pixel_ood_labels = np.array(
    #         [_ood_attr_map_global.get(Path(fn).name, 0) for fn in pixel_index.filenames],
    #         dtype=np.int8,
    #     )

    # ── per-sigma state — load partial checkpoints, skip completed ────────────
    def _sigma_experiment_dir(sigma: float) -> Path:
        """Same path formula as CertifyPaths.from_config but for a given sigma."""
        sigma_tag = f"sigma_{sigma:.2f}".replace(".", "_")
        mode_tag = f"{cfg.smoothing.mode}_{'manifold' if cfg.smoothing.use_manifold else 'isotropic'}"
        if ood_attr:
            mode_dir = paths.base_dir / "certify_ood" / ood_attr.lower() / mode_tag
        else:
            mode_dir = paths.base_dir / "certify" / mode_tag
        ablation_study = getattr(cfg.output, "ablation_study", None)
        ablation_variant = getattr(cfg.output, "ablation_variant", None)
        if ablation_study:
            return mode_dir / "ablation" / sigma_tag / str(ablation_study) / str(ablation_variant)
        return mode_dir / sigma_tag

    # Build per-sigma mutable state dicts
    sigma_states: Dict[float, Dict] = {}
    active_sigmas: List[float] = []
    for sigma in sigma_values:
        exp_dir = _sigma_experiment_dir(sigma)
        exp_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            _log(f"SKIP sigma={sigma} — metrics.json already exists: {metrics_path}")
            continue

        state: Dict = {
            "exp_dir": exp_dir,
            "results": [],
            "total_correct": 0,
            "total_certified": 0,
            "total_abstained": 0,
            "radii": [],
            "start_idx": 0,
        }
        # Resume from partial checkpoint if enabled
        if cfg.checkpoint.resume:
            partial = _load_partial_state(exp_dir, len(test_samples))
            if partial is not None:
                state["start_idx"] = partial["next_idx"]
                state["results"] = partial["results"]
                state["total_correct"] = partial["total_correct"]
                state["total_certified"] = partial["total_certified"]
                state["total_abstained"] = partial["total_abstained"]
                state["radii"] = partial["radii"]
                _log(f"sigma={sigma}: resuming from sample {state['start_idx']}/{len(test_samples)}")

        sigma_states[sigma] = state
        active_sigmas.append(sigma)
        # Save config once per sigma dir
        _cfg_copy = load_certify_config.__module__  # just to have the import; we write manually
        import copy as _copy
        import dataclasses as _dc
        _cfg_sigma = _copy.deepcopy(cfg)
        _cfg_sigma.smoothing.sigma = sigma
        save_certify_config(_cfg_sigma, exp_dir / "config.yaml")

    if not active_sigmas:
        _log("All sigmas already completed — nothing to do.")
        return []

    # Global start = minimum start_idx across all active sigmas
    global_start = min(s["start_idx"] for s in sigma_states.values())
    checkpoint_every = cfg.checkpoint.checkpoint_every if cfg.checkpoint.enabled else 0

    import time as _time
    _certify_start = _time.time()

    _log(f"Active sigmas: {active_sigmas}  (global_start={global_start})")

    # ── main loop — one pass over test samples ────────────────────────────────
    for idx in tqdm(range(global_start, len(test_samples)), desc="Certifying (multi-sigma)",
                    initial=global_start, total=len(test_samples)):

        img_path, label = test_samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = smooth_transform(img)

        # ── compute PCA / VAE encode ONCE per sample ──────────────────────────
        # For manifold: cache PCA so all sigmas reuse the same kNN+SVD result.
        # For isotropic: no PCA needed, sample_fn is None (GPU batch path).
        _is_manifold = cfg.smoothing.use_manifold

        # Latent vector (reused across sigmas for latent mode)
        _lat_z: Optional[np.ndarray] = None
        if cfg.smoothing.mode == "latent" and vae is not None:
            with torch.no_grad():
                x_vae = img_tensor.unsqueeze(0).to(device)
                if x_vae.shape[-1] != vae.image_size or x_vae.shape[-2] != vae.image_size:
                    x_vae = F.interpolate(x_vae, size=vae.image_size, mode="bilinear", align_corners=False)
                mu, _ = vae.encode(x_vae)
            _lat_z = mu.squeeze(0).cpu().numpy().astype(np.float32)

        # PCA cache — computed once, reused for all sigmas (manifold only)
        _pca_cached = None
        if _is_manifold:
            if cfg.smoothing.mode == "pixel":
                # Build a temporary pixel smoother with knn_k to compute PCA
                _tmp_smoother = ManifoldSmoother(
                    sigma=1.0,  # sigma doesn't matter for PCA computation
                    index=pixel_index,
                    knn_k=cfg.smoothing.knn_k,
                    eps_eig=cfg.smoothing.eps_eig,
                    pca_dim=getattr(cfg.smoothing, "pca_dim", None),
                )
                _pca_cached = _tmp_smoother.compute_pca(img_tensor.numpy().flatten().astype(np.float32))
            elif cfg.smoothing.mode == "latent" and vae is not None and latent_index is not None:
                _tmp_smoother = ManifoldSmoother(
                    sigma=1.0,
                    index=latent_index,
                    knn_k=cfg.smoothing.knn_k,
                    eps_eig=cfg.smoothing.eps_eig,
                    pca_dim=getattr(cfg.smoothing, "pca_dim", None),
                )
                _pca_cached = _tmp_smoother.compute_pca(_lat_z)

        # KNN OOD fraction (sigma-independent — same neighbours for all sigmas)
        _nn_ood_count: Optional[int] = None
        _nn_ood_frac: Optional[float] = None
        if _ood_attr_map_global is not None:
            _knn_index = latent_index if cfg.smoothing.mode == "latent" else pixel_index
            if _knn_index is not None and hasattr(_knn_index, "index") and hasattr(_knn_index.index, "get_nns_by_vector") and hasattr(_knn_index, "filenames"):
                _qvec = _lat_z if (cfg.smoothing.mode == "latent" and _lat_z is not None) else img_tensor.numpy().flatten().astype(np.float32)
                _k_nn = cfg.smoothing.knn_k
                _nn_ids = _knn_index.index.get_nns_by_vector(_qvec.tolist(), _k_nn, include_distances=False)
                _nn_ood_count = int(sum(_ood_attr_map_global.get(Path(_knn_index.filenames[_nid]).name, 0) for _nid in _nn_ids))
                _nn_ood_frac = float(_nn_ood_count) / len(_nn_ids) if _nn_ids else None

        # ── inner loop: certify this sample for each active sigma ─────────────
        for sigma in active_sigmas:
            state = sigma_states[sigma]

            # Skip samples already processed for this sigma
            if idx < state["start_idx"]:
                continue

            # Build sigma-specific smoother (cheap — no kNN, just sets sigma)
            if cfg.smoothing.use_manifold:
                _pix_sm = ManifoldSmoother(sigma=sigma, index=pixel_index,
                                           knn_k=cfg.smoothing.knn_k, eps_eig=cfg.smoothing.eps_eig,
                                           scale_noise=getattr(cfg.smoothing, 'scale_noise', True),
                                           pca_dim=getattr(cfg.smoothing, "pca_dim", None)) if pixel_index else IsotropicSmoother(sigma=sigma)
                _lat_sm = ManifoldSmoother(sigma=sigma, index=latent_index,
                                           knn_k=cfg.smoothing.knn_k, eps_eig=cfg.smoothing.eps_eig,
                                           scale_noise=getattr(cfg.smoothing, 'scale_noise', True),
                                           pca_dim=getattr(cfg.smoothing, "pca_dim", None)) if latent_index else IsotropicSmoother(sigma=sigma)
            else:
                _pix_sm = IsotropicSmoother(sigma=sigma)
                _lat_sm = IsotropicSmoother(sigma=sigma)

            _active_smoother = _lat_sm if cfg.smoothing.mode == "latent" and vae is not None else _pix_sm

            # sample_fn: for manifold reuse cached PCA, for iso use None (GPU path)
            _collect = (_ood_attr_map_global is not None and pixel_index is not None
                        and hasattr(pixel_index, "index") and hasattr(pixel_index.index, "get_nns_by_vector")
                        and hasattr(pixel_index, "filenames"))

            if _is_manifold and _pca_cached is not None:
                # Patch sigma into the smoother used by sample_from_cached
                _active_smoother.sigma = sigma
                if cfg.smoothing.mode == "pixel":
                    _pix_sm.sigma = sigma
                    def _sfn(_sm=_pix_sm, _pc=_pca_cached, _shape=img_tensor.shape):
                        noisy_flat = _sm.sample_from_cached(_pc)
                        return torch.from_numpy(noisy_flat.reshape(_shape)).float()
                    _sample_fn = _sfn
                else:  # latent
                    _lat_sm.sigma = sigma
                    def _sfn(_sm=_lat_sm, _pc=_pca_cached, _z=_lat_z):
                        z_noised = _sm.sample_from_cached(_pc)
                        with torch.no_grad():
                            z_t = torch.from_numpy(z_noised[None, :]).to(device=device, dtype=torch.float32)
                            return vae.decode(z_t).squeeze(0).cpu()
                    _sample_fn = _sfn
            else:
                _sample_fn = None  # isotropic → GPU batch path in certify_single_sample

            cert, _ = certify_single_sample(
                classifier=classifier,
                img_tensor=img_tensor,
                smoother=_active_smoother,
                n_samples=cfg.smoothing.n_samples,
                n0_samples=int(cfg.smoothing.n0_samples),
                classifier_transform=classifier_transform,
                device=device,
                alpha_conf=cfg.alpha_conf,
                sigma=sigma,
                sample_fn=_sample_fn,
                collect_n_samples=False,
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
                "ood_attr_value": 1 if ood_attr else None,
                "nn_ood_count": _nn_ood_count,
                "nn_ood_frac": _nn_ood_frac,
                "mc_ood_count": None,   # disabled — kept for notebook compatibility
                "mc_ood_frac": None,    # disabled — kept for notebook compatibility
            }

            # MC OOD fraction — disabled (per-sample index queries slow; re-enable if needed)
            # if (_cert_raw_samples is not None and len(_cert_raw_samples) > 0
            #         and _pixel_ood_labels is not None and pixel_index is not None
            #         and hasattr(pixel_index, "index")):
            #     _mc_mat = np.stack([_rs.numpy().flatten().astype(np.float32)
            #                         for _rs in _cert_raw_samples])
            #     _nn_ids = np.array([
            #         pixel_index.index.get_nns_by_vector(_mc_mat[_i].tolist(), 1, include_distances=False)[0]
            #         for _i in range(len(_mc_mat))
            #     ], dtype=np.int32)
            #     _mc_hits = int(_pixel_ood_labels[_nn_ids].sum())
            #     result["mc_ood_count"] = _mc_hits
            #     result["mc_ood_frac"] = float(_mc_hits) / len(_cert_raw_samples)

            # Eigenvalue / geometry (reuse cached PCA — no recompute)
            if _pca_cached is not None:
                if cfg.smoothing.mode == "pixel":
                    _lmax = float(_pca_cached.pca.evals[0])
                    result["eigenvalues"] = _pca_cached.pca.evals.tolist()
                elif cfg.smoothing.mode == "latent":
                    _lmax = float(_pca_cached.pca.evals[0])
                    result["eigenvalues"] = _pca_cached.pca.evals.tolist()
                else:
                    _lmax = None
                if _lmax is not None:
                    result["lambda_max"] = _lmax
                    result["alpha"] = float(sigma) / float(np.sqrt(max(_lmax, 1e-12)))

            state["results"].append(result)
            if cert.abstained:
                state["total_abstained"] += 1
            else:
                state["total_certified"] += 1
                state["radii"].append(cert.radius)
                if cert.pred == label:
                    state["total_correct"] += 1

            # Visualization (first N samples, first sigma only to avoid duplication)
            if cfg.output.save_visualizations and idx < cfg.output.num_viz_samples and sigma == active_sigmas[0]:
                _cfg_viz = load_certify_config.__module__  # sentinel
                import copy as _copy2
                _cfg_s = _copy2.deepcopy(cfg)
                _cfg_s.smoothing.sigma = sigma
                viz_dir = state["exp_dir"] / "visualizations"
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
                    cfg=_cfg_s,
                    device=device,
                    pixel_smoother=_pix_sm,
                    latent_smoother=_lat_sm,
                    ood_attr_map=_ood_attr_map_global,
                )

        # ── checkpoint all active sigmas periodically ─────────────────────────
        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
            for sigma in active_sigmas:
                state = sigma_states[sigma]
                _save_partial_state(
                    experiment_dir=state["exp_dir"],
                    next_idx=idx + 1,
                    results=state["results"],
                    total_correct=state["total_correct"],
                    total_certified=state["total_certified"],
                    total_abstained=state["total_abstained"],
                    radii=state["radii"],
                    num_test_samples=len(test_samples),
                )

    # ── finalise each sigma: compute metrics, save outputs ────────────────────
    all_outputs = []
    for sigma in active_sigmas:
        state = sigma_states[sigma]
        exp_dir = state["exp_dir"]
        results = state["results"]
        total_certified = state["total_certified"]
        total_abstained = state["total_abstained"]
        total_correct = state["total_correct"]
        radii = state["radii"]
        total = len(test_samples)

        # Clone cfg with this sigma so _save_summary_visualization and metrics are correct
        import copy as _copy3
        _cfg_s = _copy3.deepcopy(cfg)
        _cfg_s.smoothing.sigma = sigma

        _sigma_elapsed = _time.time() - _certify_start
        _samples_processed = total - sigma_states[sigma].get("start_idx", 0)
        metrics = {
            # ── Identity ──────────────────────────────────────────────────────
            "experiment": cfg.experiment_name,
            "dataset": cfg.dataset.name,
            "output_dir": str(exp_dir),

            # ── Classifier ────────────────────────────────────────────────────
            "classifier": {
                "model": cfg.model.name,
                "checkpoint": resolve_classifier_checkpoint(cfg),
                "input_size": cfg.model.input_size,
                "dropout": cfg.model.dropout,
                "use_ood_classifier": getattr(cfg.model, "use_ood_classifier", True),
            },

            # ── Smoothing config ──────────────────────────────────────────────
            "smoothing": {
                "mode": cfg.smoothing.mode,
                "use_manifold": cfg.smoothing.use_manifold,
                "sigma": sigma,
                "n0_samples": int(cfg.smoothing.n0_samples),
                "n_samples": int(cfg.smoothing.n_samples),
                "total_mc_samples": int(cfg.smoothing.n0_samples) + int(cfg.smoothing.n_samples),
                "knn_k": cfg.smoothing.knn_k,
                "pca_dim": getattr(cfg.smoothing, "pca_dim", None),
                "eps_eig": cfg.smoothing.eps_eig,
                "alpha_conf": cfg.alpha_conf,
            },

            # ── VAE ───────────────────────────────────────────────────────────
            "vae": {
                "enabled": cfg.vae.enabled,
                "checkpoint": cfg.vae.checkpoint_path if cfg.vae.enabled else None,
                "image_size": cfg.vae.image_size if cfg.vae.enabled else None,
                "latent_dim": cfg.vae.latent_dim if cfg.vae.enabled else None,
            },

            # ── Index ─────────────────────────────────────────────────────────
            "index": {
                "backend": cfg.index.backend,
                "metric": cfg.index.metric,
                "n_trees": cfg.index.n_trees,
                "split": "train",
                "n_train_samples": len(train_samples),
            },

            # ── Dataset / split ───────────────────────────────────────────────
            "dataset_info": {
                "name": cfg.dataset.name,
                "root_dir": cfg.dataset.root_dir,
                "train_samples": len(train_samples),
                "test_samples": total,
                "split_seed": cfg.dataset.split_seed,
                "train_ratio": cfg.dataset.train_ratio,
                "val_ratio": cfg.dataset.val_ratio,
                "ood_attribute": ood_attr if ood_attr else None,
                "ood_attr_value": 1 if ood_attr else None,
                "ood_balanced": cfg.dataset.ood_balanced if ood_attr else None,
            },

            # ── Runtime ───────────────────────────────────────────────────────
            "runtime": {
                "certify_seconds": round(_sigma_elapsed, 2),
                "certify_minutes": round(_sigma_elapsed / 60, 2),
                "samples_processed": _samples_processed,
                "seconds_per_sample": round(_sigma_elapsed / max(_samples_processed, 1), 4),
                "device": str(device),
                "n_active_sigmas": len(active_sigmas),
            },

            # ── Results ───────────────────────────────────────────────────────
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
        for cls in [0, 1]:
            cls_name = "smile" if cls == 1 else "no_smile"
            cls_results = [r for r in results if r["label"] == cls]
            cls_correct = sum(1 for r in cls_results if r["certified_correct"])
            cls_radii = [r["radius"] for r in cls_results if not r["abstained"]]
            metrics[f"class_{cls_name}_total"] = len(cls_results)
            metrics[f"class_{cls_name}_correct"] = cls_correct
            metrics[f"class_{cls_name}_accuracy"] = cls_correct / len(cls_results) if cls_results else 0.0
            metrics[f"class_{cls_name}_mean_radius"] = float(np.mean(cls_radii)) if cls_radii else 0.0

        _nn_fracs  = [r["nn_ood_frac"]  for r in results if r.get("nn_ood_frac")  is not None]
        _nn_counts = [r["nn_ood_count"] for r in results if r.get("nn_ood_count") is not None]
        # MC OOD tracking disabled — uncomment to re-enable
        # _mc_fracs  = [r["mc_ood_frac"]  for r in results if r.get("mc_ood_frac")  is not None]
        # _mc_counts = [r["mc_ood_count"] for r in results if r.get("mc_ood_count") is not None]
        metrics["ood_stats"] = {
            "ood_attribute": ood_attr if ood_attr else None,
            "nn_samples_with_data":  len(_nn_fracs),
            "mean_nn_ood_frac":      float(np.mean(_nn_fracs))   if _nn_fracs  else None,
            "median_nn_ood_frac":    float(np.median(_nn_fracs)) if _nn_fracs  else None,
            "mean_nn_ood_count":     float(np.mean(_nn_counts))  if _nn_counts else None,
            # "mc_samples_with_data":  len(_mc_fracs),
            # "mean_mc_ood_frac":      float(np.mean(_mc_fracs))   if _mc_fracs   else None,
            # "median_mc_ood_frac":    float(np.median(_mc_fracs)) if _mc_fracs   else None,
            # "mean_mc_ood_count":     float(np.mean(_mc_counts))  if _mc_counts  else None,
        }

        # Volume + geometry metrics (same logic as run_certification)
        from src.certify.randomized import (
            log_volume_isotropic, log_volume_manifold, eigenvalue_diagnostics,
            normalize_eigenvalues, log_volume_geo_iso, log_volume_geo_mani,
            axis_lengths, anisotropy_ratio, cumulative_stretch_energy,
        )
        if cfg.smoothing.mode == "latent" and vae is not None:
            ambient_dim = vae.latent_dim
        else:
            ambient_dim = cfg.model.input_size * cfg.model.input_size * 3
        sigma_val = float(sigma)
        k_pca = None
        geometry_factors, lv_mani_actuals, lv_iso_Ds = [], [], []
        per_sample_log_geo_ratio, per_sample_anisotropy = [], []
        per_sample_axis_lengths, per_sample_cum_energy, per_sample_effective_rank = [], [], []

        iso_companion_radii = None
        _stag = f"sigma_{sigma:.2f}".replace(".", "_")
        iso_companion_dir = paths.certify_dir / f"{cfg.smoothing.mode}_isotropic" / _stag
        iso_csv = iso_companion_dir / "results.csv"
        if iso_csv.exists():
            import csv as csv_mod2
            with open(iso_csv) as f:
                reader = csv_mod2.DictReader(f)
                iso_companion_radii = np.array([float(row["radius"]) for row in reader if float(row["radius"]) > 0])

        for r in results:
            if r["radius"] <= 0:
                continue
            evals = r.get("eigenvalues")
            if evals is not None:
                evals_arr = np.array(evals, dtype=np.float64)
                k_pca = len(evals_arr)
                lv_mani = log_volume_manifold(r["radius"], evals_arr)
                geom = 0.5 * np.sum(np.log(np.maximum(evals_arr, 1e-30)))
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
                evals_norm_max = normalize_eigenvalues(evals_arr, mode="max")
                lv_gm = log_volume_geo_mani(sigma_val, evals_norm_max)
                lv_gi = log_volume_geo_iso(sigma_val, k_pca)
                log_geo_ratio = lv_gm - lv_gi
                ani = anisotropy_ratio(evals_norm_max)
                ax = axis_lengths(sigma_val, evals_norm_max)
                cum = cumulative_stretch_energy(evals_norm_max)
                p = evals_arr / np.maximum(evals_arr.sum(), 1e-30)
                eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
                per_sample_log_geo_ratio.append(log_geo_ratio)
                per_sample_anisotropy.append(ani)
                per_sample_axis_lengths.append(ax.tolist())
                per_sample_cum_energy.append(cum.tolist())
                per_sample_effective_rank.append(eff_rank)
                r["log_v_geo_iso"] = lv_gi
                r["log_v_geo_mani"] = lv_gm
                r["log_geo_ratio"] = log_geo_ratio
                r["anisotropy_ratio"] = ani
            else:
                lv_iso_D = log_volume_isotropic(r["radius"], ambient_dim)
                r["log_vol_iso_D"] = lv_iso_D
                r["log_vol_mani_actual"] = None
                r["geometry_factor"] = None
                r["eigen_k"] = None
                r["ambient_D"] = ambient_dim
                lv_iso_Ds.append(lv_iso_D)

        if lv_mani_actuals or lv_iso_Ds:
            mean_log_vol_iso_D = mean_log_vol_iso_k = mean_log_vol_mani_pred = None
            if iso_companion_radii is not None and len(iso_companion_radii) > 0 and k_pca is not None:
                mean_geom = float(np.mean(geometry_factors)) if geometry_factors else 0.0
                mean_log_vol_iso_D = float(np.mean([log_volume_isotropic(r, ambient_dim) for r in iso_companion_radii]))
                mean_log_vol_iso_k = float(np.mean([log_volume_isotropic(r, k_pca) for r in iso_companion_radii]))
                mean_log_vol_mani_pred = float(np.mean([log_volume_isotropic(r, k_pca) + mean_geom for r in iso_companion_radii]))
            elif lv_iso_Ds:
                mean_log_vol_iso_D = float(np.mean(lv_iso_Ds))
            geo_summary = None
            if per_sample_log_geo_ratio:
                ax_arr_np = np.array(per_sample_axis_lengths)
                geo_summary = {
                    "sigma": sigma_val, "k_pca": k_pca, "ambient_D": ambient_dim,
                    "log_v_iso_geo": float(log_volume_geo_iso(sigma_val, k_pca)) if k_pca else None,
                    "mean_log_v_mani_geo": float(np.mean([r["log_v_geo_mani"] for r in results if r.get("log_v_geo_mani") is not None])),
                    "mean_log_geo_ratio": float(np.mean(per_sample_log_geo_ratio)),
                    "median_log_geo_ratio": float(np.median(per_sample_log_geo_ratio)),
                    "std_log_geo_ratio": float(np.std(per_sample_log_geo_ratio)),
                    "mean_axis_lengths": np.nanmean(ax_arr_np, axis=0).tolist() if ax_arr_np.size else [],
                    "mean_anisotropy_ratio": float(np.mean(per_sample_anisotropy)),
                    "median_anisotropy_ratio": float(np.median(per_sample_anisotropy)),
                    "std_anisotropy_ratio": float(np.std(per_sample_anisotropy)),
                    "mean_effective_rank": float(np.mean(per_sample_effective_rank)),
                }
            metrics["volume"] = {
                "k_pca": k_pca, "ambient_D": ambient_dim,
                "mean_log_vol_iso_D": mean_log_vol_iso_D,
                "mean_log_vol_iso_k": mean_log_vol_iso_k,
                "mean_log_vol_mani_pred": mean_log_vol_mani_pred,
                "mean_log_vol_mani_actual": float(np.mean(lv_mani_actuals)) if lv_mani_actuals else None,
                "mean_geometry_factor": float(np.mean(geometry_factors)) if geometry_factors else None,
                "median_geometry_factor": float(np.median(geometry_factors)) if geometry_factors else None,
                "mean_effective_rank": float(np.mean([r["eigen_effective_rank"] for r in results if r.get("eigen_effective_rank")])) if any(r.get("eigen_effective_rank") for r in results) else None,
                "mean_condition_number": float(np.mean([r["eigen_condition_number"] for r in results if r.get("eigen_condition_number")])) if any(r.get("eigen_condition_number") for r in results) else None,
                "geometry": geo_summary,
            }

        _log(f"sigma={sigma}  acc={metrics['certified_accuracy']*100:.2f}%  "
             f"abstain={metrics['abstain_rate']*100:.2f}%  mean_r={metrics['mean_radius']:.4f}")

        if cfg.output.save_results:
            (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            _log(f"  Metrics: {exp_dir / 'metrics.json'}")
            if cfg.output.save_per_sample:
                csv_path = exp_dir / "results.csv"
                fieldnames = ["idx", "image_path", "label", "pred", "radius", "abstained",
                              "p_a_lower", "p_b_upper", "correct", "certified_correct",
                              "ood_attribute", "ood_attr_value",
                              "nn_ood_count", "nn_ood_frac", "mc_ood_count", "mc_ood_frac",
                              "lambda_max", "alpha", "log_vol_mani_actual", "log_vol_iso_D",
                              "geometry_factor", "eigen_k", "ambient_D",
                              "eigen_effective_rank", "eigen_condition_number", "eigen_sum"]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for r in results:
                        writer.writerow(r)
                _log(f"  Results CSV: {csv_path}")

                eigen_samples = [(r["idx"], r["eigenvalues"]) for r in results if r.get("eigenvalues") is not None]
                if eigen_samples and getattr(cfg.output, "save_eigenvalues", True):
                    eigen_path = exp_dir / "eigenvalues.npz"
                    raw_evals_list = [np.array(e[1], dtype=np.float64) for e in eigen_samples]
                    _ax_max_len = max((len(e) for e in raw_evals_list), default=0)
                    _ax_arr = np.full((len(raw_evals_list), _ax_max_len), np.nan)
                    _cum_rows, _geo_ratios, _aniso_arr, _eff_rank_arr, _norm_evals_list = [], [], [], [], []
                    for i, ev in enumerate(raw_evals_list):
                        ev_norm = normalize_eigenvalues(ev, mode="max")
                        _norm_evals_list.append(ev_norm)
                        ax = axis_lengths(sigma_val, ev_norm)
                        _ax_arr[i, :len(ax)] = ax
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
                        indices=np.array([e[0] for e in eigen_samples]),
                        eigenvalues=np.array(raw_evals_list, dtype=np.float64),
                        eigenvalues_norm_max=np.array(_norm_evals_list, dtype=np.float64),
                        axis_lengths_all=_ax_arr,
                        anisotropy_ratios=np.array(_aniso_arr, dtype=np.float64),
                        cumulative_stretch_energy=_cum_arr,
                        log_geo_ratio_per_sample=np.array(_geo_ratios, dtype=np.float64),
                        effective_rank_per_sample=np.array(_eff_rank_arr, dtype=np.float64),
                        sigma=np.float64(sigma_val),
                    )
                    _log(f"  Eigenvalues: {eigen_path}")

            if cfg.output.save_visualizations:
                _save_summary_visualization(exp_dir, results, metrics, _cfg_s)

        # Clean up partial state now that metrics.json is written
        _remove_partial_state(exp_dir)
        all_outputs.append({"sigma": sigma, "metrics": metrics, "results": results, "exp_dir": exp_dir})

    return all_outputs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def run_viz_only(cfg: CertifyConfig, sigma_values: List[float]) -> None:
    """Re-generate visualizations only for existing completed sigma runs.

    Skips full certification — requires metrics.json to already exist.
    Re-runs the smoother on the first num_viz_samples test images only.
    """
    import copy as _copy

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _log(f"VIZ-ONLY  device={device}  sigmas={sigma_values}")

    train_samples, test_samples = get_train_test_samples(cfg)
    test_samples = get_ood_test_samples(cfg, test_samples)
    ood_attr = getattr(cfg.dataset, "ood_attribute", None)

    # Classifier
    classifier = build_resnet_classifier(
        name=cfg.model.name, pretrained=False,
        dropout=cfg.model.dropout, num_classes=cfg.model.num_classes,
    ).to(device)
    ckpt_path = resolve_classifier_checkpoint(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:       classifier.load_state_dict(ckpt["model_state_dict"])
    elif "model_state" in ckpt:          classifier.load_state_dict(ckpt["model_state"])
    else:                                classifier.load_state_dict(ckpt)
    classifier.eval()

    # VAE
    vae = None
    if cfg.vae.enabled and cfg.smoothing.mode in ("latent", "both"):
        vae = ConvVAE(in_channels=cfg.vae.in_channels, image_size=cfg.vae.image_size,
                      latent_dim=cfg.vae.latent_dim).to(device)
        load_vae_checkpoint(vae, cfg.vae.checkpoint_path, device)
        vae.eval()

    # Indexes
    pixel_size = cfg.model.input_size
    _need_pixel = cfg.smoothing.use_manifold or (cfg.smoothing.mode in ("latent","both") and cfg.output.save_visualizations)
    pixel_index = None
    latent_index = None

    # OOD attr map
    _ood_attr_map: Optional[dict] = None
    if ood_attr:
        _attr_path = Path(cfg.dataset.root_dir) / cfg.dataset.annotation_file
        _lines = [l.strip() for l in _attr_path.read_text().splitlines() if l.strip()]
        _attr_names = _lines[1].split()
        if ood_attr in _attr_names:
            _aidx = _attr_names.index(ood_attr)
            _ood_attr_map = {}
            for _row in _lines[2:]:
                _parts = _row.split()
                _ood_attr_map[_parts[0]] = 1 if int(_parts[1 + _aidx]) == 1 else 0

    smooth_transform = transforms.Compose([
        transforms.Resize((pixel_size, pixel_size)),
        transforms.ToTensor(),
    ])

    n_viz = cfg.output.num_viz_samples

    for sigma in sorted(set(float(s) for s in sigma_values)):
        _cfg_s = _copy.deepcopy(cfg)
        _cfg_s.smoothing.sigma = sigma
        paths = CertifyPaths.from_config(_cfg_s)

        if not (paths.experiment_dir / "metrics.json").exists():
            _log(f"SKIP sigma={sigma} — metrics.json not found")
            continue

        _log(f"sigma={sigma}  generating {n_viz} viz → {paths.experiment_dir / 'visualizations'}")

        # Build index and smoother for this sigma
        if _need_pixel and pixel_index is None:
            pixel_index = load_or_build_pixel_index(
                train_samples, pixel_size, paths.pixel_index_dir,
                cfg.index.n_trees, metric=cfg.index.metric,
            )
        if cfg.smoothing.mode in ("latent","both") and vae is not None and latent_index is None and cfg.smoothing.use_manifold:
            latent_index = load_or_build_latent_index(
                train_samples, vae, paths.latent_index_dir,
                cfg.index.n_trees, device, metric=cfg.index.metric,
            )

        pixel_smoother = create_pixel_smoother(_cfg_s, pixel_index)
        latent_smoother = create_latent_smoother(_cfg_s, latent_index)

        current_index = latent_index if cfg.smoothing.mode == "latent" else pixel_index

        for idx in range(min(n_viz, len(test_samples))):
            img_path, label = test_samples[idx]
            img_tensor = smooth_transform(Image.open(img_path).convert("RGB"))

            if cfg.smoothing.mode == "pixel":
                sample_fn = make_pixel_sample_fn(img_tensor, pixel_smoother)
            elif cfg.smoothing.mode == "latent" and vae is not None:
                sample_fn = make_latent_sample_fn(img_tensor, vae, latent_smoother, device)
            else:
                sample_fn = make_pixel_sample_fn(img_tensor, pixel_smoother)

            _active_smoother = latent_smoother if cfg.smoothing.mode == "latent" and vae is not None else pixel_smoother
            _is_iso = not cfg.smoothing.use_manifold
            cert, _ = certify_single_sample(
                classifier=classifier,
                img_tensor=img_tensor,
                smoother=_active_smoother,
                n_samples=cfg.smoothing.n_samples,
                n0_samples=int(cfg.smoothing.n0_samples),
                classifier_transform=transforms.Compose([
                    transforms.Resize((cfg.model.input_size, cfg.model.input_size)),
                    transforms.ToTensor(),
                ]),
                device=device,
                alpha_conf=cfg.alpha_conf,
                sigma=sigma,
                sample_fn=None if _is_iso else sample_fn,
                collect_n_samples=False,
            )

            save_sample_visualization(
                viz_dir=paths.experiment_dir / "visualizations",
                sample_idx=idx,
                img_tensor=img_tensor,
                label=label,
                pred=cert.pred,
                radius=cert.radius,
                abstained=cert.abstained,
                index=current_index,
                vae=vae,
                cfg=_cfg_s,
                device=device,
                pixel_smoother=pixel_smoother,
                latent_smoother=latent_smoother,
                ood_attr_map=_ood_attr_map,
            )
            _log(f"  sample {idx:04d} done")

        _log(f"sigma={sigma} viz complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="CelebA Certification Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=None,
        help="Override sigma_values from config. E.g. --sigmas 0.10 0.25 0.50. "
             "If not given, reads sigma_values (or sigma) from config.",
    )
    parser.add_argument("--viz-only", action="store_true",
                        help="Re-generate visualizations only — skips certification, "
                             "requires metrics.json to already exist for each sigma.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_certify_config(args.config)

    # Resolve sigma list: CLI --sigmas > config sigma_values > config sigma
    if args.sigmas:
        sigma_values = [float(s) for s in args.sigmas]
    else:
        sigma_values = cfg.smoothing.sigma_values if hasattr(cfg.smoothing, "sigma_values") and cfg.smoothing.sigma_values else [cfg.smoothing.sigma]

    if args.viz_only:
        run_viz_only(cfg, sigma_values)
    elif len(sigma_values) == 1:
        # Single sigma — use original single-sigma path (no overhead)
        cfg.smoothing.sigma = sigma_values[0]
        run_certification(cfg)
    else:
        run_certification_multi_sigma(cfg, sigma_values)
