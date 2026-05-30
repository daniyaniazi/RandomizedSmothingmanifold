"""Sigma sweep: train one smoothed classifier per (mode, sigma) pair.

Produces checkpoints at:
  output/pretrained_model/isotropic/{dataset}/smile_resnet_{dataset}_sigma_{s}/best.pt
  output/pretrained_model/manifold/{dataset}/smile_resnet_{dataset}_sigma_{s}/best.pt

Usage:
    # Isotropic sweep only:
    python -m src.experiments.training.resnet_smile.train_smooth_sweep \\
        --config src/configs/training/smile_resnet_celeba.yaml \\
        --modes isotropic \\
        --sigmas 0.12 0.25 0.50 1.00

    # Both modes (manifold requires a pre-built pixel index):
    python -m src.experiments.training.resnet_smile.train_smooth_sweep \\
        --config src/configs/training/smile_resnet_celeba.yaml \\
        --modes isotropic manifold \\
        --sigmas 0.12 0.25 0.50 1.00 \\
        --index_path output/smile_classification/celeba/index/pixel/annoy/euclidean/index.ann

SLURM note:
    Each (mode, sigma) pair is an independent training run so you can also
    submit them as separate SLURM array jobs by passing a single sigma per job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def sigma_tag(sigma: float) -> str:
    """Format sigma as a directory-safe string, e.g. 0.25 -> 'sigma_0_25'."""
    return f"sigma_{sigma:.2f}".replace(".", "_")


def dataset_tag(dataset_name: str) -> str:
    return dataset_name.lower().replace("-", "").replace("_", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sigma sweep for smoothed classifier training")
    parser.add_argument("--config", required=True, help="Base training YAML config")
    parser.add_argument(
        "--modes", nargs="+", default=["isotropic"],
        choices=["isotropic", "manifold"],
        help="Smoothing modes to sweep (default: isotropic)",
    )
    parser.add_argument(
        "--sigmas", nargs="+", type=float,
        default=[0.12, 0.25, 0.50, 1.00],
        help="List of sigma values to train (default: 0.12 0.25 0.50 1.00)",
    )
    parser.add_argument(
        "--index_path", type=str, default=None,
        help="Pre-built pixel Annoy index (.ann) — required for manifold mode",
    )
    parser.add_argument(
        "--base_ckpt_dir", type=str,
        default="output/pretrained_model",
        help="Root directory for all sigma checkpoints (default: output/pretrained_model)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    base_cfg = Path(args.config)
    if not base_cfg.exists():
        print(f"ERROR: config not found: {base_cfg}")
        sys.exit(1)

    # Read dataset name from config to build the output path
    import yaml
    raw = yaml.safe_load(base_cfg.read_text()) or {}
    ds_name = raw.get("dataset", {}).get("name", "celeba")
    model_name = raw.get("model", {}).get("name", "resnet18")
    dtag = dataset_tag(ds_name)

    jobs = []
    for mode in args.modes:
        for sigma in args.sigmas:
            stag = sigma_tag(sigma)
            run_name = f"smooth_{mode}_{dtag}_{stag}"
            ckpt_dir = f"{args.base_ckpt_dir}/{mode}/{dtag}/smile_{model_name}_{dtag}_{stag}"
            output_dir = f"output/smoothed_training/{mode}/{dtag}/{stag}"

            cmd = [
                sys.executable, "-m",
                "src.experiments.training.resnet_smile.main",
                "--config", str(base_cfg),
                "--sigma", str(sigma),
                "--aug_mode", mode,
                "--ckpt_dir", ckpt_dir,
                "--output_dir", output_dir,
            ]
            if mode == "manifold":
                if args.index_path is None:
                    print(f"  SKIP  mode=manifold σ={sigma}  (--index_path not provided)")
                    continue
                cmd += ["--index_path", args.index_path]

            jobs.append((run_name, cmd))

    print(f"\nSweep: {len(jobs)} job(s)  modes={args.modes}  sigmas={args.sigmas}\n")

    for run_name, cmd in jobs:
        print(f"── {run_name} ──────────────────────────────────────")
        print("  " + " ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"  WARNING: job exited with code {result.returncode}")
        print()

    if args.dry_run:
        print("[dry_run] No jobs were executed.")
    else:
        print(f"Sweep complete. Checkpoints saved under: {args.base_ckpt_dir}/{{mode}}/{dtag}/smile_{model_name}_{dtag}_sigma_*/best.pt")


if __name__ == "__main__":
    main()
