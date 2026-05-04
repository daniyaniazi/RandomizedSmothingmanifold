"""Certification workflow for CelebA / CelebA-HQ smile classification.

Directory Structure (following NER pattern):
    output/smile_classification/{celeba|celebahq}/
    ├── dataset/                          # Dataset info
    ├── index/
    │   ├── pixel/
    │   │   └── annoy/euclidean/
    │   │       ├── index.ann
    │   │       └── index_metadata.json
    │   └── latent/
    │       └── annoy/euclidean/
    │           ├── index.ann
    │           └── index_metadata.json
    └── certify/
        ├── pixel_manifold/
        │   ├── sigma_0_25/
        │   │   ├── metrics.json
        │   │   ├── results.csv
        │   │   └── visualizations/
        │   └── sigma_0_50/
        └── latent_manifold/
            └── sigma_0_50/

Key Design:
    - Index is built from TRAIN split
    - Certification runs on TEST split
    - Uses existing SmileDataBundle from src.dataloaders.celeba_smile

Usage:
    python -m src.certify.celeba_workflow --config src/configs/experiments/certify_celeba_pixel_128.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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
from src.models.resnet import build_resnet_classifier
from src.models.VAE import ConvVAE, load_checkpoint as load_vae_checkpoint
from src.smoothing.workflow import fit_local_pca, whiten, unwhiten


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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
# Index Building (on TRAIN set)
# ─────────────────────────────────────────────────────────────────────────────


def build_pixel_index(
    train_samples: List[Tuple[str, int]],
    image_size: int,
    index_dir: Path,
    n_trees: int = 50,
) -> NeighborIndex:
    """Build Annoy index over flattened pixel vectors from TRAIN set."""
    import annoy
    
    index_path = index_dir / "index.ann"
    meta_path = index_dir / "index_metadata.json"
    
    dim = 3 * image_size * image_size
    ann_index = annoy.AnnoyIndex(dim, "euclidean")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
    ])
    
    index_map = {}
    for idx, (path, label) in enumerate(tqdm(train_samples, desc="Building pixel index (train)")):
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)
        flat = img_tensor.numpy().flatten().astype(np.float32)
        ann_index.add_item(idx, flat)
        index_map[idx] = {"path": str(path), "label": int(label)}
    
    ann_index.build(n_trees)
    ann_index.save(str(index_path))
    
    metadata = {
        "num_items": len(train_samples),
        "dim": dim,
        "image_size": image_size,
        "n_trees": n_trees,
        "split": "train",
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    
    # Save index map separately
    (index_dir / "index_map.json").write_text(json.dumps(index_map, indent=2))
    
    _log(f"Pixel index saved: {index_path} ({len(train_samples)} items, dim={dim})")
    return load_index(dim=dim, index_path=str(index_path), backend="annoy")


def build_latent_index(
    train_samples: List[Tuple[str, int]],
    vae: ConvVAE,
    index_dir: Path,
    n_trees: int = 50,
    device: torch.device = torch.device("cuda"),
) -> NeighborIndex:
    """Build Annoy index over VAE latent vectors from TRAIN set."""
    import annoy
    
    index_path = index_dir / "index.ann"
    meta_path = index_dir / "index_metadata.json"
    
    dim = vae.latent_dim
    ann_index = annoy.AnnoyIndex(dim, "euclidean")
    
    transform = transforms.Compose([
        transforms.Resize((vae.image_size, vae.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
    ])
    
    index_map = {}
    latent_vectors = []
    
    vae.eval()
    with torch.no_grad():
        for idx, (path, label) in enumerate(tqdm(train_samples, desc="Building latent index (train)")):
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            mu, _ = vae.encode(img_tensor)
            z = mu.squeeze(0).cpu().numpy().astype(np.float32)
            ann_index.add_item(idx, z)
            latent_vectors.append(z)
            index_map[idx] = {"path": str(path), "label": int(label)}
    
    ann_index.build(n_trees)
    ann_index.save(str(index_path))
    
    # Save latent vectors for neighbor lookup
    np.savez(index_dir / "latent_vectors.npz", vectors=np.stack(latent_vectors))
    
    metadata = {
        "num_items": len(train_samples),
        "dim": dim,
        "image_size": vae.image_size,
        "latent_dim": vae.latent_dim,
        "n_trees": n_trees,
        "split": "train",
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    (index_dir / "index_map.json").write_text(json.dumps(index_map, indent=2))
    
    _log(f"Latent index saved: {index_path} ({len(train_samples)} items, dim={dim})")
    return load_index(dim=dim, index_path=str(index_path), backend="annoy")


def load_or_build_pixel_index(
    train_samples: List[Tuple[str, int]],
    image_size: int,
    index_dir: Path,
    n_trees: int = 50,
    force_rebuild: bool = False,
) -> NeighborIndex:
    index_path = index_dir / "index.ann"
    meta_path = index_dir / "index_metadata.json"
    
    if index_path.exists() and meta_path.exists() and not force_rebuild:
        meta = json.loads(meta_path.read_text())
        _log(f"Loading existing pixel index: {index_path} ({meta['num_items']} items)")
        return load_index(dim=meta["dim"], index_path=str(index_path), backend="annoy")
    
    index_dir.mkdir(parents=True, exist_ok=True)
    return build_pixel_index(train_samples, image_size, index_dir, n_trees)


def load_or_build_latent_index(
    train_samples: List[Tuple[str, int]],
    vae: ConvVAE,
    index_dir: Path,
    n_trees: int = 50,
    device: torch.device = torch.device("cuda"),
    force_rebuild: bool = False,
) -> NeighborIndex:
    index_path = index_dir / "index.ann"
    meta_path = index_dir / "index_metadata.json"
    
    if index_path.exists() and meta_path.exists() and not force_rebuild:
        meta = json.loads(meta_path.read_text())
        _log(f"Loading existing latent index: {index_path} ({meta['num_items']} items)")
        return load_index(dim=meta["dim"], index_path=str(index_path), backend="annoy")
    
    index_dir.mkdir(parents=True, exist_ok=True)
    return build_latent_index(train_samples, vae, index_dir, n_trees, device)


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing Functions
# ─────────────────────────────────────────────────────────────────────────────


def get_neighbors_from_index(index: NeighborIndex, vector: np.ndarray, k: int) -> np.ndarray:
    """Get k nearest neighbor vectors from Annoy index."""
    if hasattr(index.index, "get_nns_by_vector"):
        nn_ids = index.index.get_nns_by_vector(vector.tolist(), k)
        neighbors = np.array([index.index.get_item_vector(i) for i in nn_ids], dtype=np.float32)
        return neighbors
    raise ValueError("Index does not support get_nns_by_vector")


def sample_pixel_isotropic(img_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    return img_tensor + torch.randn_like(img_tensor) * sigma


def sample_pixel_manifold(
    img_tensor: torch.Tensor,
    index: NeighborIndex,
    sigma: float,
    knn_k: int,
    eps_eig: float = 1e-6,
) -> torch.Tensor:
    flat = img_tensor.numpy().flatten().astype(np.float32)
    neighbors = get_neighbors_from_index(index, flat, knn_k)
    pca = fit_local_pca(neighbors, eps_eig=eps_eig)
    white = whiten(flat, pca)
    white_noised = white + np.random.randn(len(white)).astype(np.float32) * sigma
    unwhite_vec = unwhiten(white_noised, pca)
    return torch.from_numpy(unwhite_vec.reshape(img_tensor.shape)).float()


def sample_latent_isotropic(
    img_tensor: torch.Tensor,
    vae: ConvVAE,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        mu, _ = vae.encode(x)
        z_noised = mu + torch.randn_like(mu) * sigma
        x_hat = vae.decode(z_noised)
    return x_hat.squeeze(0).cpu()


def sample_latent_manifold(
    img_tensor: torch.Tensor,
    vae: ConvVAE,
    index: NeighborIndex,
    sigma: float,
    knn_k: int,
    device: torch.device,
    eps_eig: float = 1e-6,
) -> torch.Tensor:
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        mu, _ = vae.encode(x)
        z = mu.squeeze(0).cpu().numpy().astype(np.float32)
    
    neighbors = get_neighbors_from_index(index, z, knn_k)
    pca = fit_local_pca(neighbors, eps_eig=eps_eig)
    white = whiten(z, pca)
    white_noised = white + np.random.randn(len(white)).astype(np.float32) * sigma
    z_noised = unwhiten(white_noised, pca)
    
    with torch.no_grad():
        z_t = torch.from_numpy(z_noised[None, :]).to(device=device, dtype=torch.float32)
        x_hat = vae.decode(z_t)
    return x_hat.squeeze(0).cpu()


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
        num_classes=cfg.model.num_classes,
    ).to(device)
    
    ckpt = torch.load(cfg.model.checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        classifier.load_state_dict(ckpt["model_state_dict"])
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
    
    if cfg.smoothing.mode in ("pixel", "both") and cfg.smoothing.use_manifold:
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
    # Run certification on TEST set
    # ─────────────────────────────────────────────────────────────────────────
    _log(f"Certifying {len(test_samples)} TEST samples")
    _log(f"Mode: {cfg.smoothing.mode}, Manifold: {cfg.smoothing.use_manifold}, Sigma: {cfg.smoothing.sigma}")
    
    results: List[Dict] = []
    total_correct = 0
    total_certified = 0
    total_abstained = 0
    radii: List[float] = []
    
    for idx, (img_path, label) in enumerate(tqdm(test_samples, desc="Certifying (test)")):
        img = Image.open(img_path).convert("RGB")
        img_tensor = smooth_transform(img)
        
        if cfg.smoothing.mode == "pixel":
            if cfg.smoothing.use_manifold and pixel_index is not None:
                sample_fn = lambda t=img_tensor: sample_pixel_manifold(
                    t, pixel_index, cfg.smoothing.sigma, cfg.smoothing.knn_k, cfg.smoothing.eps_eig
                )
            else:
                sample_fn = lambda t=img_tensor: sample_pixel_isotropic(t, cfg.smoothing.sigma)
        elif cfg.smoothing.mode == "latent" and vae is not None:
            if cfg.smoothing.use_manifold and latent_index is not None:
                sample_fn = lambda t=img_tensor: sample_latent_manifold(
                    t, vae, latent_index, cfg.smoothing.sigma, cfg.smoothing.knn_k, device, cfg.smoothing.eps_eig
                )
            else:
                sample_fn = lambda t=img_tensor: sample_latent_isotropic(t, vae, cfg.smoothing.sigma, device)
        else:
            sample_fn = lambda t=img_tensor: sample_pixel_isotropic(t, cfg.smoothing.sigma)
        
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
    
    return {"metrics": metrics, "results": results, "paths": paths}


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
