"""IO utilities for loading/saving certification configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .certify_celeba_schema import (
    CertifyConfig,
    CertifyDatasetConfig,
    CertifyIndexConfig,
    CertifyModelConfig,
    CertifyOutputConfig,
    CertifySmoothingConfig,
    CertifyVAEConfig,
)


def _dict_to_dataclass(dc_cls, d: dict | None):
    if d is None:
        return dc_cls()
    return dc_cls(**{k: v for k, v in d.items() if k in dc_cls.__dataclass_fields__})


def load_certify_config(path: str | Path) -> CertifyConfig:
    """Load a certification YAML config into a CertifyConfig dataclass."""
    raw = yaml.safe_load(Path(path).read_text())
    return CertifyConfig(
        experiment_name=raw.get("experiment_name", "celeba_certify"),
        seed=raw.get("seed", 73),
        device=raw.get("device", "cuda"),
        alpha_conf=raw.get("alpha_conf", 0.001),
        dataset=_dict_to_dataclass(CertifyDatasetConfig, raw.get("dataset")),
        model=_dict_to_dataclass(CertifyModelConfig, raw.get("model")),
        vae=_dict_to_dataclass(CertifyVAEConfig, raw.get("vae")),
        smoothing=_dict_to_dataclass(CertifySmoothingConfig, raw.get("smoothing")),
        index=_dict_to_dataclass(CertifyIndexConfig, raw.get("index")),
        output=_dict_to_dataclass(CertifyOutputConfig, raw.get("output")),
    )


def save_certify_config(cfg: CertifyConfig, path: str | Path) -> None:
    """Save a CertifyConfig to YAML."""
    from dataclasses import asdict

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.dump(asdict(cfg), default_flow_style=False, sort_keys=False))
