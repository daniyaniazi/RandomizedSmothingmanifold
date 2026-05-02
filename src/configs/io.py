"""Configuration loading and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import (
    CertificationConfig,
    DataloaderConfig,
    DatasetConfig,
    EvalConfig,
    ExperimentConfig,
    MaskingConfig,
    ModelConfig,
    SmoothingConfig,
    TaskConfig,
    TrainConfig,
    WandbConfig,
)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}

    defaults = asdict(ExperimentConfig())
    merged = _merge_dict(defaults, raw)

    task_cfg = dict(merged["task"])
    if task_cfg.get("eval_plugin") and not task_cfg.get("module"):
        task_cfg["module"] = task_cfg["eval_plugin"]

    smoothing_cfg = dict(merged["smoothing"])
    if smoothing_cfg.get("target") == "hidden_states" and smoothing_cfg.get("space"):
        space = smoothing_cfg["space"]
        if space == "final_hidden":
            smoothing_cfg["target"] = "hidden_states"
        elif space == "hidden_layer":
            smoothing_cfg["target"] = "hidden_states"
        elif space == "input_embeddings":
            smoothing_cfg["target"] = "input_embeddings"

    return ExperimentConfig(
        experiment_name=merged["experiment_name"],
        output_dir=merged["output_dir"],
        task=TaskConfig(**task_cfg),
        dataset=DatasetConfig(**merged["dataset"]),
        dataloader=DataloaderConfig(**merged["dataloader"]),
        model=ModelConfig(**merged["model"]),
        smoothing=SmoothingConfig(**smoothing_cfg),
        certification=CertificationConfig(**merged["certification"]),
        train=TrainConfig(**merged["train"]),
        eval=EvalConfig(**merged["eval"]),
        masking=MaskingConfig(**merged["masking"]),
        wandb=WandbConfig(**merged["wandb"]),
    )


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
