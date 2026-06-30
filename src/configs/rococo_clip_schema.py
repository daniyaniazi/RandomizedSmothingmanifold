"""Schema for RoCOCO CLIP image-text retrieval with randomized smoothing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class RoCoCoSmoothingConfig:
    mode: str = "baseline"   # baseline | isotropic | manifold
    sigma: float = 0.10
    knn_k: int = 500         # always 500 neighbours for PCA
    # n_samples: how many noisy embeddings to average for a stable smoothed query
    # Lower than certification (10-20 is enough for retrieval averaging)
    n_samples: int = 10
    eps_eig: float = 1e-6
    scale_noise: bool = True   # True=alpha=sigma/sqrt(lambda_max), False=alpha=sigma
    # Multi-sigma sweep
    sigma_values: Optional[List[float]] = None


@dataclass
class RoCoCoIndexConfig:
    backend: str = "annoy"
    metric: str = "angular"  # angular = cosine on unit sphere — correct for CLIP embeddings
    n_trees: int = 50


@dataclass
class RoCoCoConfig:
    experiment_name: str = "rococo_clip"
    clip_model: str = "ViT-B/32"

    # Data paths
    image_dir: str = "/BS/dniazi_thesis/static00/rococo/images"
    annotation_dir: str = "/BS/dniazi_thesis/static00/rococo/annotation"
    annotation_files: List[str] = field(default_factory=lambda: [
        "coco_karpathy_test.json",
        "danger.json",
        "same_concept.json",
        "diff_concept.json",
        "rand_voca.json",
    ])

    # Cache and output
    embedding_cache_dir: str = "/BS/dniazi_thesis/static00/rococo/clip_embeddings"
    index_dir: str = "output/rococo/index/clip/annoy/angular"
    output_dir: str = "output/rococo"

    # Runtime
    device: str = "cuda"
    batch_size: int = 256
    seed: int = 73
    num_workers: int = 4

    smoothing: RoCoCoSmoothingConfig = field(default_factory=RoCoCoSmoothingConfig)
    index: RoCoCoIndexConfig = field(default_factory=RoCoCoIndexConfig)


# ── IO ────────────────────────────────────────────────────────────────────────

def _merge(obj, d: dict) -> None:
    for k, v in d.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def load_rococo_config(path: str | Path) -> RoCoCoConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = RoCoCoConfig()
    for key in ("experiment_name", "clip_model", "image_dir", "annotation_dir",
                "annotation_files", "embedding_cache_dir", "index_dir",
                "output_dir", "device", "batch_size", "seed", "num_workers"):
        if key in raw:
            setattr(cfg, key, raw[key])
    if "smoothing" in raw: _merge(cfg.smoothing, raw["smoothing"])
    if "index"     in raw: _merge(cfg.index,     raw["index"])
    return cfg


def save_rococo_config(cfg: RoCoCoConfig, path: str | Path) -> None:
    import dataclasses
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
