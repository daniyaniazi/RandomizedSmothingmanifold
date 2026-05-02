"""Typed configuration schema for smoothing experiments."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    name: str = "conll2003"
    split_train: str = "train"
    split_val: str = "validation"
    split_test: str = "test"
    text_field: str = "tokens"
    label_field: str = "ner_tags"
    max_length: int = 192
    num_workers: int = 2


@dataclass
class DataloaderConfig:
    batch_size: int = 16
    shuffle_train: bool = True


@dataclass
class ModelConfig:
    encoder_name: str = "distilbert-base-uncased"
    dropout: float = 0.1


@dataclass
class SmoothingConfig:
    enabled: bool = True
    mode: str = "isotropic"  # isotropic | manifold (local kNN PCA)
    target: str = "hidden_states"  # hidden_states | input_embeddings | latent | image_tensor
    space: str = "input_embeddings"  # deprecated alias retained for compatibility
    sigma: float = 0.10
    layer_index: Optional[int] = None
    knn_k: int = 64
    eps_eig: float = 1e-6
    index_backend: str = "torch"  # torch | annoy | faiss
    index_metric: str = "euclidean"
    index_path: Optional[str] = None
    index_n_trees: int = 20


@dataclass
class CertificationConfig:
    enabled: bool = True
    alpha: float = 0.001
    n0: int = 64
    n: int = 512
    abstain_label: int = -1


@dataclass
class TrainConfig:
    seed: int = 73
    lr: float = 3e-5
    weight_decay: float = 0.01
    epochs: int = 2
    device: str = "cuda"
    grad_clip_norm: float = 1.0


@dataclass
class EvalConfig:
    max_batches: Optional[int] = None


@dataclass
class MaskingConfig:
    enabled: bool = False
    mode: str = "none"  # none | context | entity | hybrid
    entity_label_ids: list[int] = field(default_factory=lambda: [1])
    mask_ratio: float = 0.15
    max_masks_per_sentence: Optional[int] = None
    cap_by_batch_avg_tokens: bool = True
    seed: int = 73


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "randomized-smoothing-ner"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class TaskConfig:
    name: str = "ner"
    module: str = "src.models.transformer.ner.train"
    eval_plugin: Optional[str] = None


@dataclass
class ExperimentConfig:
    experiment_name: str = "ner_conll2003_smoothing"
    output_dir: str = "outputs/ner_conll2003"
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    certification: CertificationConfig = field(default_factory=CertificationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
