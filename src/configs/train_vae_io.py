"""Configuration loading and serialization helpers for VAE training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import yaml

from .train_vae_schema import (
    VaeCheckpointConfig,
    VaeDataloaderConfig,
    VaeDatasetConfig,
    VaeLoggingConfig,
    VaeModelConfig,
    VaeTrainConfig,
    VaeTrainingConfig,
    VaeWandbConfig,
)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_vae_training_config(path: str | Path) -> VaeTrainingConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}

    defaults = asdict(VaeTrainingConfig())
    merged = _merge_dict(defaults, raw)

    return VaeTrainingConfig(
        experiment_name=merged["experiment_name"],
        output_dir=merged["output_dir"],
        dataset=VaeDatasetConfig(**merged["dataset"]),
        dataloader=VaeDataloaderConfig(**merged["dataloader"]),
        model=VaeModelConfig(**merged["model"]),
        train=VaeTrainConfig(**merged["train"]),
        logging=VaeLoggingConfig(**merged["logging"]),
        wandb=VaeWandbConfig(**merged["wandb"]),
        checkpoint=VaeCheckpointConfig(**merged["checkpoint"]),
    )


def save_vae_resolved_config(config: VaeTrainingConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
