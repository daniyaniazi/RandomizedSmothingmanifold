"""Generic evaluation runner with task-plugin dispatch."""

from __future__ import annotations

import argparse
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.configs import load_experiment_config, save_resolved_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_cfg(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _load_plugin(module_path: str):
    plugin = importlib.import_module(module_path)
    required = ["build_model_and_data", "evaluate_task_accuracy", "certify_task"]
    for fn in required:
        if not hasattr(plugin, fn):
            raise AttributeError(f"Task plugin {module_path} is missing required function: {fn}")
    return plugin


def run(cfg_path: str, checkpoint_path: str | None = None, metrics_out_dir: str | None = None):
    cfg = load_experiment_config(cfg_path)
    set_seed(cfg.train.seed)
    device = device_from_cfg(cfg.train.device)

    out_dir = Path(metrics_out_dir) if metrics_out_dir is not None else Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, out_dir / "resolved_config.yaml")

    plugin_path = cfg.task.module or cfg.task.eval_plugin
    plugin = _load_plugin(plugin_path)
    model, data = plugin.build_model_and_data(cfg, device)

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else Path(cfg.output_dir) / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    task_metrics = plugin.evaluate_task_accuracy(model, data, cfg, device)
    cert_metrics = plugin.certify_task(model, data, cfg, device)
    final_metrics = {**task_metrics, **cert_metrics}

    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
    print(json.dumps(final_metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Generic evaluation and certification runner")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/experiments/ner_conll2003_distilbert.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint path. Defaults to <output_dir>/model.pt from config.",
    )
    parser.add_argument(
        "--metrics-out-dir",
        type=str,
        default=None,
        help="Optional directory for evaluation outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, checkpoint_path=args.checkpoint, metrics_out_dir=args.metrics_out_dir)
