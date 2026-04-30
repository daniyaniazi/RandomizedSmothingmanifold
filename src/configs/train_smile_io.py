"""Configuration loading and serialization helpers for smile classification training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import yaml

from .train_smile_schema import (
    SmileCheckpointConfig,
    SmileDataloaderConfig,
    SmileDatasetConfig,
    SmileLoggingConfig,
    SmileModelConfig,
    SmileTrainConfig,
    SmileTrainingConfig,
    SmileWandbConfig,
)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_smile_training_config(path: str | Path) -> SmileTrainingConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}

    defaults = asdict(SmileTrainingConfig())
    merged = _merge_dict(defaults, raw)

    return SmileTrainingConfig(
        experiment_name=merged["experiment_name"],
        output_dir=merged["output_dir"],
        dataset=SmileDatasetConfig(**merged["dataset"]),
        dataloader=SmileDataloaderConfig(**merged["dataloader"]),
        model=SmileModelConfig(**merged["model"]),
        train=SmileTrainConfig(**merged["train"]),
        logging=SmileLoggingConfig(**merged["logging"]),
        wandb=SmileWandbConfig(**merged["wandb"]),
        checkpoint=SmileCheckpointConfig(**merged["checkpoint"]),
    )


def save_smile_resolved_config(config: SmileTrainingConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))