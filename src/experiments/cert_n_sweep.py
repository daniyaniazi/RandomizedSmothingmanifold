"""Generic certification experiment sweeping Monte Carlo sample count n."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

from src.eval.run_experiment import run as run_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep certification sample size n")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/experiments/ner_conll2003_distilbert.yaml",
        help="Base experiment config path.",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Certification sample counts n to evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_cfg_path = Path(args.config)
    base_cfg = json.loads(json.dumps(yaml.safe_load(base_cfg_path.read_text())))
    base_output_dir = Path(base_cfg.get("output_dir", "outputs/default"))
    checkpoint_path = base_output_dir / "model.pt"

    summary = []
    for n in args.n_values:
        cfg = deepcopy(base_cfg)
        cfg.setdefault("certification", {})
        cfg["certification"]["n"] = int(n)

        out_dir = base_output_dir / f"cert_n_{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_cfg_path = out_dir / "tmp_eval_config.yaml"
        with tmp_cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        run_eval(
            str(tmp_cfg_path),
            checkpoint_path=str(checkpoint_path),
            metrics_out_dir=str(out_dir),
        )

        metrics = json.loads((out_dir / "metrics.json").read_text())
        summary.append({"n": n, **metrics})

    summary_path = base_output_dir / "cert_n_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
