"""Load/save helpers for segmentation certification config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.configs.certify_seg_schema import (
    SegCertifyConfig, SegDatasetConfig, SegModelConfig,
    SegSmoothingConfig, SegIndexConfig, SegOutputConfig, SegCheckpointConfig,
)


def _merge(dataclass_obj, d: Dict[str, Any]):
    """Recursively apply dict values onto a dataclass instance."""
    for k, v in d.items():
        if hasattr(dataclass_obj, k):
            setattr(dataclass_obj, k, v)


def load_seg_certify_config(path: str) -> SegCertifyConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = SegCertifyConfig()
    for key in ("experiment_name", "seed", "device", "alpha_conf"):
        if key in raw:
            setattr(cfg, key, raw[key])
    if "dataset"   in raw: _merge(cfg.dataset,   raw["dataset"])
    if "model"     in raw: _merge(cfg.model,      raw["model"])
    if "smoothing" in raw: _merge(cfg.smoothing,  raw["smoothing"])
    if "index"     in raw: _merge(cfg.index,      raw["index"])
    if "output"    in raw: _merge(cfg.output,     raw["output"])
    if "checkpoint" in raw: _merge(cfg.checkpoint, raw["checkpoint"])
    return cfg


def save_seg_certify_config(cfg: SegCertifyConfig, path: Path) -> None:
    import dataclasses
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
