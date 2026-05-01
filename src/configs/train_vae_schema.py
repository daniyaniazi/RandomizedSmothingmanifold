"""Typed configuration schema for VAE training on image datasets."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VaeDatasetConfig:
    name: str = "CelebA"
    root_dir: str = "/BS/databases08/CelebA"
    image_dir: str = "img_align_celeba"
    file_extension: Optional[str] = None
    num_workers: int = 4


@dataclass
class VaeDataloaderConfig:
    batch_size: int = 64
    shuffle_train: bool = True
    pin_memory: bool = True


@dataclass
class VaeModelConfig:
    in_channels: int = 3
    image_size: int = 64
    latent_dim: int = 128


@dataclass
class VaeTrainConfig:
    seed: int = 73
    device: str = "cuda"
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta: float = 1e-4


@dataclass
class VaeLoggingConfig:
    log_every_n_steps: int = 50


@dataclass
class VaeWandbConfig:
    enabled: bool = False
    project: str = "randomized-smoothing-vae"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class VaeCheckpointConfig:
    save_every_n_epochs: int = 1
    keep_last_k: int = 2
    base_dir: str = "output/pretrained_model"


@dataclass
class VaeTrainingConfig:
    experiment_name: str = "vae_celeba"
    output_dir: str = "output/vae_celeba"
    dataset: VaeDatasetConfig = field(default_factory=VaeDatasetConfig)
    dataloader: VaeDataloaderConfig = field(default_factory=VaeDataloaderConfig)
    model: VaeModelConfig = field(default_factory=VaeModelConfig)
    train: VaeTrainConfig = field(default_factory=VaeTrainConfig)
    logging: VaeLoggingConfig = field(default_factory=VaeLoggingConfig)
    wandb: VaeWandbConfig = field(default_factory=VaeWandbConfig)
    checkpoint: VaeCheckpointConfig = field(default_factory=VaeCheckpointConfig)
