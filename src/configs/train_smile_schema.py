"""Typed configuration schema for CelebA/CelebA-HQ smile classification training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SmileDatasetConfig:
    name: str = "CelebA"
    root_dir: str = "/BS/databases08/CelebA"
    image_dir: str = "img_align_celeba"
    annotation_file: str = "list_attr_celeba.txt"
    partition_file: Optional[str] = "list_eval_partition.txt"
    annotation_format: str = "celeba"  # celeba | csv
    image_column: str = "image"
    label_column: str = "smile"
    file_extension: str = ""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 73
    num_workers: int = 4


@dataclass
class SmileDataloaderConfig:
    batch_size: int = 64
    shuffle_train: bool = True
    pin_memory: bool = True


@dataclass
class SmileModelConfig:
    name: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.0
    input_size: int = 224


@dataclass
class SmileTrainConfig:
    seed: int = 73
    device: str = "cuda"
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    save_every_n_epochs: int = 1


@dataclass
class SmileLoggingConfig:
    log_every_n_steps: int = 50


@dataclass
class SmileWandbConfig:
    enabled: bool = False
    project: str = "randomized-smoothing-smile"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class SmileCheckpointConfig:
    keep_last_k: int = 2
    base_dir: str = "output/pretrained_model"


@dataclass
class SmileSmoothingAugConfig:
    """Smoothing augmentation applied to each training image.

    Mirrors the Cohen et al. training scheme: for every (x, y) pair the model
    sees x + noise drawn once per forward pass, so the classifier learns the
    noisy distribution it will be certified against.

    mode = "isotropic"  -> x' = x + N(0, sigma^2 I)   (no index needed)
    mode = "manifold"   -> x' = x + manifold noise     (requires pixel index)
    """
    enabled: bool = False
    mode: str = "isotropic"   # isotropic | manifold
    sigma: float = 0.25
    knn_k: int = 500
    eps_eig: float = 1e-6
    # Pre-built Annoy index path (required for manifold mode).
    # If None and mode=manifold the index is built on the fly from the train set.
    index_path: Optional[str] = None


@dataclass
class SmileTrainingConfig:
    experiment_name: str = "smile_resnet_train"
    output_dir: str = "output"
    dataset: SmileDatasetConfig = field(default_factory=SmileDatasetConfig)
    dataloader: SmileDataloaderConfig = field(default_factory=SmileDataloaderConfig)
    model: SmileModelConfig = field(default_factory=SmileModelConfig)
    train: SmileTrainConfig = field(default_factory=SmileTrainConfig)
    logging: SmileLoggingConfig = field(default_factory=SmileLoggingConfig)
    wandb: SmileWandbConfig = field(default_factory=SmileWandbConfig)
    checkpoint: SmileCheckpointConfig = field(default_factory=SmileCheckpointConfig)
    smoothing_aug: SmileSmoothingAugConfig = field(default_factory=SmileSmoothingAugConfig)