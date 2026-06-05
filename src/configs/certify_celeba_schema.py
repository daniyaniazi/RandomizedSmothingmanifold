"""Schema for image certification experiments (CelebA / CelebA-HQ)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CertifyDatasetConfig:
    name: str = "CelebA"
    root_dir: str = ""
    image_dir: str = ""
    annotation_file: str = ""
    annotation_format: str = "celeba"  # celeba | csv
    image_column: str = "image"
    label_column: str = "smile"
    file_extension: str = ""
    num_workers: int = 4
    subset_size: Optional[int] = None  # limit samples for testing
    # Split ratios + seed — must be identical across training, indexing, and certification
    train_ratio: float = 0.8
    val_ratio: float = 0.1   # test = 1 - train_ratio - val_ratio
    split_seed: int = 73     # controls dataset shuffle; keep in sync with training split_seed
    # OOD attribute subset — restrict test images to those where attribute=1
    ood_attribute: Optional[str] = None   # e.g. "Mouth_Slightly_Open", null = standard test split
    ood_balanced: bool = True             # sample exactly subset_size/2 smile + subset_size/2 non-smile
    ood_seed: int = 73                    # random seed for OOD sampling


@dataclass
class CertifyModelConfig:
    name: str = "resnet18"
    checkpoint_path: str = ""  # explicit path — used when use_smoothed_classifier=False
    input_size: int = 224  # CelebA: 224, CelebA-HQ: 512 — used for ALL transforms
    num_classes: int = 1  # binary for smile
    dropout: float = 0.5  # must match training config to load weights correctly
    # ── Smoothed-classifier lookup ──────────────────────────────────────────
    # When True, checkpoint_path is IGNORED and the path is auto-resolved:
    #   {smoothed_classifier_base_dir}/{iso|manifold}/smile_resnet_{dataset}_sigma_{s}/best.pt
    use_smoothed_classifier: bool = False
    smoothed_classifier_base_dir: str = "output/pretrained_model"


@dataclass
class CertifyVAEConfig:
    enabled: bool = False
    checkpoint_path: str = ""
    image_size: int = 128
    latent_dim: int = 256
    in_channels: int = 3


@dataclass
class CertifySmoothingConfig:
    mode: str = "pixel"  # pixel | latent | both
    sigma: float = 0.25
    n0_samples: int = 64  # pilot samples for class selection (paper CERTIFY stage-1)
    n_samples: int = 100  # Monte Carlo samples for certification
    knn_k: int = 64
    eps_eig: float = 1e-6
    use_manifold: bool = True  # True = manifold PCA smoothing, False = isotropic


@dataclass
class CertifyIndexConfig:
    backend: str = "annoy"  # annoy | faiss | torch
    metric: str = "euclidean"  # angular (cosine) | euclidean — set in config yaml
    n_trees: int = 50
    pixel_index_path: Optional[str] = None  # pre-built .ann for pixel space
    latent_index_path: Optional[str] = None  # pre-built .ann for latent space
    build_if_missing: bool = True


@dataclass
class CertifyOutputConfig:
    output_dir: str = "output/certification"
    save_results: bool = True
    save_per_sample: bool = True  # save per-sample certification results
    save_visualizations: bool = True
    num_viz_samples: int = 10


@dataclass
class CertifyCheckpointConfig:
    """Checkpoint configuration for long-running certification jobs."""
    enabled: bool = True
    checkpoint_every: int = 100  # save partial state every N samples
    resume: bool = False  # resume from partial state if available


@dataclass
class CertifyConfig:
    experiment_name: str = "celeba_certify"
    seed: int = 73
    device: str = "cuda"
    alpha_conf: float = 0.001  # confidence level for Clopper-Pearson

    dataset: CertifyDatasetConfig = field(default_factory=CertifyDatasetConfig)
    model: CertifyModelConfig = field(default_factory=CertifyModelConfig)
    vae: CertifyVAEConfig = field(default_factory=CertifyVAEConfig)
    smoothing: CertifySmoothingConfig = field(default_factory=CertifySmoothingConfig)
    index: CertifyIndexConfig = field(default_factory=CertifyIndexConfig)
    output: CertifyOutputConfig = field(default_factory=CertifyOutputConfig)
    checkpoint: CertifyCheckpointConfig = field(default_factory=CertifyCheckpointConfig)
