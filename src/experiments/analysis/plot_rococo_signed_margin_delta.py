"""Plot signed RoCOCO margin changes from an existing instability sample CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_csv", type=Path)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.samples_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    values = {"isotropic": [], "manifold": []}
    with args.samples_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            mode = row.get("mode")
            if mode in values:
                values[mode].append(float(row["margin_delta"]))

    iso = np.asarray(values["isotropic"], dtype=np.float64)
    mani = np.asarray(values["manifold"], dtype=np.float64)
    if iso.size == 0 or mani.size == 0:
        raise ValueError("CSV must contain isotropic and manifold margin_delta rows")

    limit = float(max(np.max(np.abs(iso)), np.max(np.abs(mani)), 1e-12))
    edges = np.linspace(-limit, limit, args.bins + 1)
    iso_count, _ = np.histogram(iso, bins=edges)
    mani_count, _ = np.histogram(mani, bins=edges)

    stem = args.samples_csv.stem.removesuffix("_samples")
    bins_path = output_dir / f"{stem}_signed_margin_delta_histogram_bins.csv"
    with bins_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_left", "bin_right", "bin_center", "isotropic_count", "manifold_count"])
        for left, right, iso_n, mani_n in zip(edges[:-1], edges[1:], iso_count, mani_count):
            writer.writerow([left, right, (left + right) / 2.0, int(iso_n), int(mani_n)])

    summary_path = output_dir / f"{stem}_signed_margin_delta_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "mode", "n", "mean", "median", "p05", "p95",
            "toward_danger_fraction", "toward_gt_fraction",
        ])
        writer.writeheader()
        for mode, arr in (("isotropic", iso), ("manifold", mani)):
            writer.writerow({
                "mode": mode,
                "n": arr.size,
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p05": float(np.quantile(arr, 0.05)),
                "p95": float(np.quantile(arr, 0.95)),
                "toward_danger_fraction": float(np.mean(arr < 0)),
                "toward_gt_fraction": float(np.mean(arr > 0)),
            })

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.hist(iso, bins=edges, alpha=0.58, label="isotropic", color="#4c78a8")
    ax.hist(mani, bins=edges, alpha=0.58, label="manifold", color="#7b3294")
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Signed GT-Danger margin change")
    ax.set_xlabel("margin change (negative: toward Danger; positive: toward GT)")
    ax.set_ylabel("image-samples")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plot_path = output_dir / f"{stem}_signed_margin_delta_histogram.png"
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {plot_path}")
    print(f"Saved histogram bins: {bins_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
