"""Schema for CelebAMask-HQ segmentation certification experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SegDatasetConfig:
    name: str = "CelebAMask-HQ"
    root_dir: str = "/BS/dniazi_thesis/static00/CelebAMask-HQ/CelebAMask-HQ"
    image_dir: str = "CelebA-HQ-img"
    mask_dir: str = "CelebAMask-HQ-mask-anno"
    file_extension: str = ".jpg"
    image_size: int = 512          # native resolution
    n_classes: int = 19            # BiSeNet CelebAMask-HQ classes
    num_workers: int = 4
    # Split strategy
    # True  = official face-parsing.PyTorch split: train=0-27999, test=28000-29999
    # False = ratio split (train_ratio/val_ratio) for custom experiments
    use_official_split: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    split_seed: int = 73
    subset_size: Optional[int] = None   # limit test set size for fast testing


@dataclass
class SegModelConfig:
    n_classes: int = 19
    checkpoint_path: str = ""       # path to pretrained BiSeNet weights
    input_size: int = 512


@dataclass
class SegSmoothingConfig:
    mode: str = "pixel"             # pixel | latent
    sigma: float = 0.25
    tau: float = 0.75               # SEGCERTIFY threshold τ ∈ [0.5, 1)
    n0_samples: int = 10            # pilot samples for class selection
    n_samples: int = 100            # MC samples for Holm test
    use_manifold: bool = False
    knn_k: int = 500
    eps_eig: float = 1e-6
    scale_noise: bool = True   # True=alpha=sigma/sqrt(lambda_max), False=alpha=sigma
    # List of sigmas for multi-sigma sweep (overrides sigma if set)
    sigma_values: Optional[list] = None


@dataclass
class SegIndexConfig:
    backend: str = "annoy"
    metric: str = "euclidean"
    n_trees: int = 50


@dataclass
class SegOutputConfig:
    output_dir: str = "output"
    save_results: bool = True
    save_per_sample: bool = True
    save_visualizations: bool = True
    num_viz_samples: int = 5


@dataclass
class SegCheckpointConfig:
    enabled: bool = True
    checkpoint_every: int = 10
    resume: bool = True


@dataclass
class SegCertifyConfig:
    experiment_name: str = "celebahq_seg_certify"
    seed: int = 73
    device: str = "cuda"
    alpha_conf: float = 0.001       # FWER level α for Holm correction (paper uses 0.001)

    dataset: SegDatasetConfig = field(default_factory=SegDatasetConfig)
    model: SegModelConfig = field(default_factory=SegModelConfig)
    smoothing: SegSmoothingConfig = field(default_factory=SegSmoothingConfig)
    index: SegIndexConfig = field(default_factory=SegIndexConfig)
    output: SegOutputConfig = field(default_factory=SegOutputConfig)
    checkpoint: SegCheckpointConfig = field(default_factory=SegCheckpointConfig)


# CelebAMask-HQ class names (19 classes, index 0 = background)
SEG_CLASS_NAMES = [
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye",
    "eye_g", "l_ear", "r_ear", "ear_r", "nose", "mouth",
    "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat",
]
