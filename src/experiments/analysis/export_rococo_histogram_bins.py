"""Export exact binned histogram data from RoCOCO instability sample CSV.

This is for LaTeX/PGFPlots when the full sample CSV is too large. It mirrors
the histogram behavior in rococo_retrieval_instability.py: each mode is binned
separately with numpy/matplotlib-style `bins=40` unless overridden.

Example:
    python -m src.experiments.analysis.export_rococo_histogram_bins \
        --samples-csv output/rococo/retrieval_instability/danger_global_sigma_0_10/danger_global_sigma_0_10_samples.csv \
        --metric abs_margin_delta
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "score_delta_std",
    "score_delta_l2",
    "abs_margin_delta",
    "best_adv_rank_delta_abs",
]


def _export_metric(df: pd.DataFrame, metric: str, bins: int, output_csv: Path) -> None:
    rows = []
    for mode in ["isotropic", "manifold"]:
        values = df.loc[df["mode"] == mode, metric].dropna().to_numpy(dtype=float)
        counts, edges = np.histogram(values, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        for idx, (count, left, right, center, width) in enumerate(
            zip(counts, edges[:-1], edges[1:], centers, widths)
        ):
            rows.append({
                "metric": metric,
                "mode": mode,
                "bin_index": idx,
                "bin_left": left,
                "bin_right": right,
                "bin_center": center,
                "bin_width": width,
                "count": int(count),
                "density_count_fraction": float(count / max(len(values), 1)),
                "n_values": int(len(values)),
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export exact RoCOCO histogram bins")
    p.add_argument("--samples-csv", type=Path, required=True)
    p.add_argument("--metric", choices=DEFAULT_METRICS + ["all"], default="all")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.samples_csv)
    metrics = DEFAULT_METRICS if args.metric == "all" else [args.metric]

    if len(metrics) == 1:
        output_csv = args.output_csv or args.samples_csv.with_name(
            f"{args.samples_csv.stem}_{metrics[0]}_hist_bins_exact.csv"
        )
        _export_metric(df, metrics[0], args.bins, output_csv)
        print(output_csv)
        return

    for metric in metrics:
        output_csv = args.samples_csv.with_name(
            f"{args.samples_csv.stem}_{metric}_hist_bins_exact.csv"
        )
        _export_metric(df, metric, args.bins, output_csv)
        print(output_csv)


if __name__ == "__main__":
    main()
