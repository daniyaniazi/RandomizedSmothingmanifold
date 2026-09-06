"""Aggregate CelebA manifold ablation results into CSVs and plots.

This is a post-processing script. It does not run certification. It scans
completed ablation folders produced by submit_celeba_manifold_ablation.sh:

  output/smile_classification/celeba/certify/pixel_manifold/ablation/
    sigma_0_10/local_manifold_size/knn_128_pca_auto/metrics.json
    sigma_0_10/pca_dimension/knn_500_pca_128/metrics.json

It saves compact CSVs and appendix-friendly plots for:
  - certified accuracy vs sigma and ablation value
  - mean/median radius vs sigma and ablation value
  - certified accuracy at radius thresholds
  - optional geometry/volume quantities when present
  - the theoretical k tradeoff: sampling error + curvature bias
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


SIGMA_RE = re.compile(r"sigma_(\d+)_(\d+)")
KNN_RE = re.compile(r"knn_(\d+)")
PCA_RE = re.compile(r"pca_(\d+)")


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _sigma_from_name(name: str) -> Optional[float]:
    m = SIGMA_RE.search(name)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _flatten_metrics(metrics: dict, metrics_path: Path, study: str, variant: str) -> dict:
    smoothing = metrics.get("smoothing", {})
    volume = metrics.get("volume", {}) or {}
    geometry = volume.get("geometry", {}) or {}

    sigma = float(smoothing.get("sigma", _sigma_from_name(metrics_path.parts[-4]) or 0.0))
    knn_k = smoothing.get("knn_k")
    pca_dim = smoothing.get("pca_dim")
    if knn_k is None:
        m = KNN_RE.search(variant)
        knn_k = int(m.group(1)) if m else None
    if pca_dim is None:
        m = PCA_RE.search(variant)
        pca_dim = int(m.group(1)) if m else None

    if study == "local_manifold_size":
        ablation_value = int(knn_k) if knn_k is not None else None
        ablation_axis = "knn_k"
    elif study == "pca_dimension":
        ablation_value = int(pca_dim) if pca_dim is not None else None
        ablation_axis = "pca_dim"
    else:
        ablation_value = None
        ablation_axis = "unknown"

    return {
        "study": study,
        "variant": variant,
        "ablation_axis": ablation_axis,
        "ablation_value": ablation_value,
        "sigma": sigma,
        "knn_k": knn_k,
        "pca_dim": pca_dim,
        "metrics_path": str(metrics_path),
        "results_path": str(metrics_path.with_name("results.csv")),
        "experiment": metrics.get("experiment"),
        "total_test_samples": metrics.get("total_test_samples"),
        "certified_samples": metrics.get("certified_samples"),
        "abstained_samples": metrics.get("abstained_samples"),
        "certified_correct": metrics.get("certified_correct"),
        "certified_accuracy": metrics.get("certified_accuracy"),
        "certified_accuracy_pct": _pct(metrics.get("certified_accuracy")),
        "abstain_rate": metrics.get("abstain_rate"),
        "abstain_rate_pct": _pct(metrics.get("abstain_rate")),
        "mean_radius": metrics.get("mean_radius"),
        "median_radius": metrics.get("median_radius"),
        "max_radius": metrics.get("max_radius"),
        "std_radius": metrics.get("std_radius"),
        "class_no_smile_accuracy": metrics.get("class_no_smile_accuracy"),
        "class_no_smile_accuracy_pct": _pct(metrics.get("class_no_smile_accuracy")),
        "class_smile_accuracy": metrics.get("class_smile_accuracy"),
        "class_smile_accuracy_pct": _pct(metrics.get("class_smile_accuracy")),
        "class_no_smile_mean_radius": metrics.get("class_no_smile_mean_radius"),
        "class_smile_mean_radius": metrics.get("class_smile_mean_radius"),
        "mean_log_vol_mani_actual": volume.get("mean_log_vol_mani_actual"),
        "mean_geometry_factor": volume.get("mean_geometry_factor"),
        "median_geometry_factor": volume.get("median_geometry_factor"),
        "mean_effective_rank": volume.get("mean_effective_rank"),
        "mean_condition_number": volume.get("mean_condition_number"),
        "mean_log_geo_ratio": geometry.get("mean_log_geo_ratio"),
        "median_log_geo_ratio": geometry.get("median_log_geo_ratio"),
        "mean_anisotropy_ratio": geometry.get("mean_anisotropy_ratio"),
    }


def _pct(value) -> Optional[float]:
    return None if value is None else 100.0 * float(value)


def _collect(base_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for metrics_path in sorted(base_dir.glob("sigma_*/**/*/metrics.json")):
        try:
            rel = metrics_path.relative_to(base_dir)
            if len(rel.parts) < 4:
                continue
            study = rel.parts[1]
            variant = rel.parts[2]
            metrics = _read_json(metrics_path)
            rows.append(_flatten_metrics(metrics, metrics_path, study, variant))
        except Exception as exc:
            _log(f"Skipping {metrics_path}: {exc}")
    return rows


def _read_result_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _radius_threshold_rows(summary_rows: List[dict], thresholds: List[float]) -> List[dict]:
    out: List[dict] = []
    for row in summary_rows:
        result_rows = _read_result_rows(Path(row["results_path"]))
        if not result_rows:
            continue
        total = len(result_rows)
        for threshold in thresholds:
            certified_at_r = 0
            correct_at_r = 0
            for r in result_rows:
                radius = float(r.get("radius") or 0.0)
                is_correct = str(r.get("correct", "")).lower() == "true"
                is_cert_correct = str(r.get("certified_correct", "")).lower() == "true"
                if is_cert_correct and radius >= threshold:
                    certified_at_r += 1
                if is_correct and radius >= threshold:
                    correct_at_r += 1
            out.append({
                "study": row["study"],
                "variant": row["variant"],
                "ablation_axis": row["ablation_axis"],
                "ablation_value": row["ablation_value"],
                "sigma": row["sigma"],
                "radius_threshold": threshold,
                "certified_accuracy_at_radius": certified_at_r / total if total else 0.0,
                "certified_accuracy_at_radius_pct": 100.0 * certified_at_r / total if total else 0.0,
                "correct_and_radius_at_threshold": correct_at_r / total if total else 0.0,
                "correct_and_radius_at_threshold_pct": 100.0 * correct_at_r / total if total else 0.0,
                "n_total": total,
                "n_certified_correct_at_radius": certified_at_r,
            })
    return out


def _write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_pairs(rows: List[dict], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for r in rows:
        x = r.get(x_key)
        y = r.get(y_key)
        if x is None or y is None:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if not pairs:
        return np.array([]), np.array([])
    arr = np.array(pairs, dtype=float)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def _plot_metric_grid(rows: List[dict], out_dir: Path, study: str, x_key: str) -> None:
    study_rows = [r for r in rows if r["study"] == study]
    if not study_rows:
        return
    metrics = [
        ("certified_accuracy_pct", "Certified accuracy (%)"),
        ("mean_radius", "Mean radius over certified samples"),
        ("median_radius", "Median radius over certified samples"),
        ("abstain_rate_pct", "Abstain rate (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), facecolor="white")
    axes = axes.ravel()
    sigmas = sorted({float(r["sigma"]) for r in study_rows if r.get("sigma") is not None})
    colors = plt.cm.viridis(np.linspace(0.10, 0.90, max(len(sigmas), 1)))

    for ax, (metric, ylabel) in zip(axes, metrics):
        any_line = False
        for color, sigma in zip(colors, sigmas):
            sigma_rows = [r for r in study_rows if float(r["sigma"]) == sigma]
            x, y = _finite_pairs(sigma_rows, x_key, metric)
            if len(x) == 0:
                continue
            ax.plot(x, y, "o-", lw=1.7, ms=4, color=color, label=f"sigma={sigma:.2f}")
            any_line = True
        ax.set_xlabel(x_key)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)
        if any_line:
            ax.legend(fontsize=7)

    title = "Local k ablation" if study == "local_manifold_size" else "PCA dimension ablation"
    fig.suptitle(f"{title}: certification outcomes", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"{study}_certification_grid.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved plot: {path}")


def _mean_by_x(rows: List[dict], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    grouped: Dict[int, List[float]] = {}
    for r in rows:
        x = r.get(x_key)
        y = r.get(y_key)
        if x is None or y is None:
            continue
        try:
            xi = int(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(yf):
            grouped.setdefault(xi, []).append(yf)
    if not grouped:
        return np.array([]), np.array([])
    xs = np.array(sorted(grouped), dtype=float)
    ys = np.array([np.mean(grouped[int(x)]) for x in xs], dtype=float)
    return xs, ys


def _plot_geometry_grid(rows: List[dict], out_dir: Path, study: str, x_key: str) -> None:
    study_rows = [r for r in rows if r["study"] == study]
    if not study_rows:
        return
    metrics = [
        ("mean_effective_rank", "Mean effective rank"),
        ("mean_condition_number", "Mean condition number"),
        ("mean_geometry_factor", "Mean geometry factor"),
        ("mean_log_geo_ratio", "Mean log volume ratio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), facecolor="white")
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        x, y = _mean_by_x(study_rows, x_key, metric)
        if len(x) > 0:
            ax.plot(x, y, "o-", lw=1.8, ms=4, color="#2f4b7c")
        ax.set_xlabel(x_key)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)

    title = "Local k ablation" if study == "local_manifold_size" else "PCA dimension ablation"
    fig.suptitle(f"{title}: local covariance geometry", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"{study}_geometry_grid.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved plot: {path}")


def _plot_heatmap(rows: List[dict], out_dir: Path, study: str, value_key: str, metric: str) -> None:
    study_rows = [r for r in rows if r["study"] == study and r.get(value_key) is not None]
    if not study_rows:
        return
    xs = sorted({int(r[value_key]) for r in study_rows})
    sigmas = sorted({float(r["sigma"]) for r in study_rows})
    grid = np.full((len(sigmas), len(xs)), np.nan)
    for r in study_rows:
        if r.get(metric) is None:
            continue
        y = sigmas.index(float(r["sigma"]))
        x = xs.index(int(r[value_key]))
        grid[y, x] = float(r[metric])

    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor="white")
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="magma")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([f"{s:.2f}" for s in sigmas])
    ax.set_xlabel(value_key)
    ax.set_ylabel("sigma")
    label = metric.replace("_", " ")
    ax.set_title(f"{study}: {label}")
    for yi in range(len(sigmas)):
        for xi in range(len(xs)):
            val = grid[yi, xi]
            if np.isfinite(val):
                ax.text(xi, yi, f"{val:.1f}" if "pct" in metric else f"{val:.3f}",
                        ha="center", va="center", fontsize=7, color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    path = out_dir / f"{study}_{metric}_heatmap.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved plot: {path}")


def _plot_radius_thresholds(rows: List[dict], out_dir: Path, study: str, x_key: str) -> None:
    study_rows = [r for r in rows if r["study"] == study]
    if not study_rows:
        return
    thresholds = sorted({float(r["radius_threshold"]) for r in study_rows})
    sigmas = sorted({float(r["sigma"]) for r in study_rows})
    if not thresholds or not sigmas:
        return

    for sigma in sigmas:
        sigma_rows = [r for r in study_rows if float(r["sigma"]) == sigma]
        fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor="white")
        colors = plt.cm.plasma(np.linspace(0.10, 0.90, len(thresholds)))
        any_line = False
        for color, thr in zip(colors, thresholds):
            thr_rows = [r for r in sigma_rows if float(r["radius_threshold"]) == thr]
            x, y = _finite_pairs(thr_rows, x_key, "certified_accuracy_at_radius_pct")
            if len(x) == 0:
                continue
            ax.plot(x, y, "o-", lw=1.6, ms=4, color=color, label=f"r >= {thr:g}")
            any_line = True
        if not any_line:
            plt.close(fig)
            continue
        ax.set_xlabel(x_key)
        ax.set_ylabel("Certified accuracy at radius (%)")
        ax.set_title(f"{study}: certified accuracy at radius, sigma={sigma:.2f}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{study}_radius_thresholds_sigma_{sigma:.2f}.png".replace(".", "_")
        path = path.with_suffix(".png")
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        _log(f"Saved plot: {path}")


def _plot_theory_tradeoff(out_dir: Path, m: int, n: int, delta: float, tau: float) -> None:
    ks = np.unique(np.round(np.geomspace(10, max(20, min(n // 2, 5000)), 240)).astype(int))
    sampling = np.sqrt((m + np.log(1.0 / delta)) / ks)
    curvature = (1.0 / tau) * (ks / n) ** (1.0 / m)
    total = sampling + curvature
    best_idx = int(np.argmin(total))

    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
    ax.plot(ks, sampling, lw=2, label="sampling error")
    ax.plot(ks, curvature, lw=2, label="curvature bias")
    ax.plot(ks, total, lw=2.5, color="#222222", label="sum")
    ax.scatter([ks[best_idx]], [total[best_idx]], color="#d62728", zorder=5,
               label=f"minimum near k={ks[best_idx]}")
    ax.set_xscale("log")
    ax.set_xlabel("neighborhood size k")
    ax.set_ylabel("bound term")
    ax.set_title("Theory-motivated local k tradeoff")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "theory_knn_tradeoff.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    data = [
        {
            "k": int(k),
            "sampling_error": float(s),
            "curvature_bias": float(c),
            "total": float(t),
            "m": m,
            "n": n,
            "delta": delta,
            "tau": tau,
        }
        for k, s, c, t in zip(ks, sampling, curvature, total)
    ]
    _write_csv(data, out_dir / "theory_knn_tradeoff.csv")
    _log(f"Saved theory plot: {path}")


def _best_by_sigma(rows: List[dict]) -> List[dict]:
    out = []
    for study in sorted({r["study"] for r in rows}):
        for sigma in sorted({float(r["sigma"]) for r in rows if r["study"] == study}):
            group = [
                r for r in rows
                if r["study"] == study
                and float(r["sigma"]) == sigma
                and r.get("certified_accuracy") is not None
            ]
            if not group:
                continue
            best = max(group, key=lambda r: (float(r["certified_accuracy"]), float(r.get("mean_radius") or 0.0)))
            out.append(dict(best))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate CelebA manifold ablation outputs")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=_ROOT / "output" / "smile_classification" / "celeba" / "certify" / "pixel_manifold" / "ablation",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "output" / "smile_classification" / "celeba" / "analysis" / "manifold_ablation",
    )
    p.add_argument("--radius-thresholds", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--theory-m", type=int, default=20)
    p.add_argument("--theory-n", type=int, default=162079)
    p.add_argument("--theory-delta", type=float, default=0.05)
    p.add_argument("--theory-tau", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect(args.base_dir)
    _log(f"Found {len(rows)} completed ablation metric files under {args.base_dir}")
    if not rows:
        raise FileNotFoundError(
            f"No metrics.json files found under {args.base_dir}. "
            "Run server_scripts/submit_celeba_manifold_ablation.sh first."
        )

    summary_path = args.output_dir / "ablation_summary_all.csv"
    _write_csv(rows, summary_path)
    _write_csv([r for r in rows if r["study"] == "local_manifold_size"],
               args.output_dir / "ablation_local_size_summary.csv")
    _write_csv([r for r in rows if r["study"] == "pca_dimension"],
               args.output_dir / "ablation_pca_dim_summary.csv")
    _write_csv(_best_by_sigma(rows), args.output_dir / "ablation_best_by_sigma.csv")

    radius_rows = _radius_threshold_rows(rows, args.radius_thresholds)
    _write_csv(radius_rows, args.output_dir / "ablation_radius_thresholds.csv")

    _plot_metric_grid(rows, args.output_dir, "local_manifold_size", "knn_k")
    _plot_metric_grid(rows, args.output_dir, "pca_dimension", "pca_dim")
    _plot_geometry_grid(rows, args.output_dir, "local_manifold_size", "knn_k")
    _plot_geometry_grid(rows, args.output_dir, "pca_dimension", "pca_dim")
    _plot_heatmap(rows, args.output_dir, "local_manifold_size", "knn_k", "certified_accuracy_pct")
    _plot_heatmap(rows, args.output_dir, "pca_dimension", "pca_dim", "certified_accuracy_pct")
    _plot_heatmap(rows, args.output_dir, "local_manifold_size", "knn_k", "mean_radius")
    _plot_heatmap(rows, args.output_dir, "pca_dimension", "pca_dim", "mean_radius")
    _plot_radius_thresholds(radius_rows, args.output_dir, "local_manifold_size", "knn_k")
    _plot_radius_thresholds(radius_rows, args.output_dir, "pca_dimension", "pca_dim")
    _plot_theory_tradeoff(
        args.output_dir,
        m=args.theory_m,
        n=args.theory_n,
        delta=args.theory_delta,
        tau=args.theory_tau,
    )

    manifest = {
        "base_dir": str(args.base_dir),
        "output_dir": str(args.output_dir),
        "n_metric_files": len(rows),
        "studies": sorted({r["study"] for r in rows}),
        "sigmas": sorted({float(r["sigma"]) for r in rows}),
        "files": [
            "ablation_summary_all.csv",
            "ablation_local_size_summary.csv",
            "ablation_pca_dim_summary.csv",
            "ablation_best_by_sigma.csv",
            "ablation_radius_thresholds.csv",
            "theory_knn_tradeoff.csv",
            "local_manifold_size_certification_grid.png",
            "pca_dimension_certification_grid.png",
            "local_manifold_size_geometry_grid.png",
            "pca_dimension_geometry_grid.png",
            "local_manifold_size_certified_accuracy_pct_heatmap.png",
            "pca_dimension_certified_accuracy_pct_heatmap.png",
            "theory_knn_tradeoff.png",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"Saved CSV summary: {summary_path}")
    _log(f"Saved manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
