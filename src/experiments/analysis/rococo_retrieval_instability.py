"""RoCOCO retrieval-instability diagnostic for Iso vs Mani smoothing.

The Q diagnostic checks covariance along one direction. This script checks the
actual retrieval surface: how much caption scores, margins, and ranks change
after sampled perturbations.

For each image:
  1. Pick the best GT caption.
  2. Pick the best adversarial caption, either paired or global.
  3. Sample isotropic and manifold perturbations.
  4. Measure score-vector change, GT/adversarial score change, margin change,
     and adversarial rank changes.

Outputs CSV/JSON summaries and comparison plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.rococo_clip_schema import load_rococo_config
from src.experiments.eval.rococo_clip_eval import _normalize
from src.smoothing.isotropic import IsotropicSmoother
from src.smoothing.manifold import ManifoldSmoother


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _caption_indices_for_image(
    img_idx: int,
    img2txt: Dict[int, List[int]],
    n_texts: int,
) -> Tuple[List[int], List[int]]:
    gt = list(img2txt.get(img_idx, []))
    if not gt:
        return [], []
    starts = sorted(min(v) for v in img2txt.values() if v)
    block_start = min(gt)
    next_starts = [s for s in starts if s > block_start]
    block_end = next_starts[0] if next_starts else n_texts
    adv = list(range(max(gt) + 1, block_end))
    return gt, adv


def _rank_of_best(scores: np.ndarray, indices: List[int]) -> int:
    ranked = np.argsort(-scores)
    positions = np.where(np.isin(ranked, indices))[0]
    return int(positions[0]) + 1 if len(positions) else -1


def _summarize(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: List[dict], out_dir: Path, tag: str) -> None:
    metrics = [
        ("score_delta_std", "Std of caption score changes"),
        ("score_delta_l2", "L2 of caption score changes"),
        ("abs_margin_delta", "Absolute GT-adv margin change"),
        ("best_adv_rank_delta_abs", "Absolute best-adv rank change"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        iso = np.asarray([r[metric] for r in rows if r["mode"] == "isotropic"])
        mani = np.asarray([r[metric] for r in rows if r["mode"] == "manifold"])
        bins = 40
        ax.hist(iso, bins=bins, alpha=0.60, label="isotropic", color="#4c78a8")
        ax.hist(mani, bins=bins, alpha=0.60, label="manifold", color="#e45756")
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("image-samples")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"RoCOCO retrieval instability: {tag}", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"{tag}_instability_histograms.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved plot: {out_path}")


def _plot_signed_margin_delta(rows: List[dict], out_dir: Path, tag: str) -> None:
    iso = np.asarray([
        r["margin_delta"] for r in rows if r["mode"] == "isotropic"
    ], dtype=np.float64)
    mani = np.asarray([
        r["margin_delta"] for r in rows if r["mode"] == "manifold"
    ], dtype=np.float64)
    if iso.size == 0 or mani.size == 0:
        return

    # Shared, symmetric bins make left/right movement and both methods comparable.
    limit = float(max(np.max(np.abs(iso)), np.max(np.abs(mani)), 1e-12))
    bins = np.linspace(-limit, limit, 81)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.hist(iso, bins=bins, alpha=0.58, label="isotropic", color="#4c78a8")
    ax.hist(mani, bins=bins, alpha=0.58, label="manifold", color="#7b3294")
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Signed GT-Danger margin change")
    ax.set_xlabel(
        "margin_delta (negative: toward Danger; positive: toward GT)"
    )
    ax.set_ylabel("image-samples")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"{tag}_signed_margin_delta_histogram.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved signed-margin plot: {out_path}")


def _plot_score_delta_components(rows: List[dict], out_dir: Path, tag: str) -> None:
    metrics = [
        ("gt_score_delta", "Ground-truth similarity change"),
        ("adv_score_delta", "Danger similarity change"),
    ]
    all_values = np.asarray([
        r[metric] for r in rows for metric, _ in metrics
    ], dtype=np.float64)
    if all_values.size == 0:
        return

    limit = float(max(np.max(np.abs(all_values)), 1e-12))
    bins = np.linspace(-limit, limit, 81)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="white")
    for ax, (metric, title) in zip(axes, metrics):
        iso = np.asarray([r[metric] for r in rows if r["mode"] == "isotropic"])
        mani = np.asarray([r[metric] for r in rows if r["mode"] == "manifold"])
        ax.hist(iso, bins=bins, alpha=0.58, label="isotropic", color="#4c78a8")
        ax.hist(mani, bins=bins, alpha=0.58, label="manifold", color="#7b3294")
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel(f"{metric} (negative: similarity decreases)")
        ax.set_ylabel("image-samples")
        ax.legend()
        ax.grid(alpha=0.25)

    fig.suptitle("Decomposition of signed GT-Danger margin change", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"{tag}_gt_danger_score_delta_histograms.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved score-delta plot: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoCOCO retrieval instability: Iso vs Mani")
    p.add_argument("--config", required=True)
    p.add_argument("--ann-stem", default="danger")
    p.add_argument("--adv-scope", choices=["paired", "global"], default="global")
    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--knn-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_rococo_config(args.config)
    if args.knn_k is None:
        args.knn_k = cfg.smoothing.knn_k
    cfg.smoothing.knn_k = args.knn_k

    np.random.seed(args.seed)
    tag = f"{args.ann_stem}_{args.adv_scope}_sigma_{args.sigma:.2f}".replace(".", "_")
    out_dir = args.output_dir or (_ROOT / cfg.output_dir / "retrieval_instability" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg.embedding_cache_dir)
    image_cache = torch.load(cache_dir / "image_embeddings.pt", map_location="cpu")
    cap_cache = torch.load(cache_dir / f"{args.ann_stem}_captions.pt", map_location="cpu")

    image_embs = _normalize_rows(image_cache["embeddings"].numpy().astype(np.float32))
    text_embs = _normalize_rows(cap_cache["embeddings"].numpy().astype(np.float32))
    img2txt = {int(k): list(v) for k, v in cap_cache["img2txt"].items()}
    all_adv_indices = [int(i) for i in cap_cache.get("wrongtext", [])]
    if not all_adv_indices:
        for i in range(len(image_embs)):
            _, adv = _caption_indices_for_image(i, img2txt, len(text_embs))
            all_adv_indices.extend(adv)

    n_eval = min(args.n_images, len(image_embs))
    _log(
        f"Instability analysis ann={args.ann_stem} scope={args.adv_scope} "
        f"sigma={args.sigma} images={n_eval} samples={args.n_samples}"
    )

    from src.experiments.indexing.rococo_clip_images import build_rococo_clip_index

    clip_index = build_rococo_clip_index(cfg, rebuild=False)
    iso = IsotropicSmoother(sigma=args.sigma)
    mani = ManifoldSmoother(
        sigma=args.sigma,
        index=clip_index,
        knn_k=args.knn_k,
        eps_eig=cfg.smoothing.eps_eig,
        scale_noise=cfg.smoothing.scale_noise,
    )

    rows: List[dict] = []
    pair_rows: List[dict] = []

    for i in tqdm(range(n_eval), desc="Retrieval instability"):
        anchor = image_embs[i]
        gt_idx, paired_adv_idx = _caption_indices_for_image(i, img2txt, len(text_embs))
        adv_idx = paired_adv_idx if args.adv_scope == "paired" else all_adv_indices
        if not gt_idx or not adv_idx:
            continue

        base_scores = anchor @ text_embs.T
        best_gt_idx = gt_idx[int(np.argmax(base_scores[gt_idx]))]
        best_adv_idx = adv_idx[int(np.argmax(base_scores[adv_idx]))]
        base_margin = float(base_scores[best_gt_idx] - base_scores[best_adv_idx])
        base_adv_rank = _rank_of_best(base_scores, adv_idx)

        cached_pca = mani.compute_pca(anchor)

        for mode in ("isotropic", "manifold"):
            sample_rows = []
            for sample_idx in range(args.n_samples):
                if mode == "isotropic":
                    noisy = _normalize(iso.sample(anchor))
                else:
                    noisy = _normalize(mani.sample_from_cached(cached_pca))

                scores = noisy @ text_embs.T
                delta = scores - base_scores
                margin = float(scores[best_gt_idx] - scores[best_adv_idx])
                adv_rank = _rank_of_best(scores, adv_idx)
                sample_rows.append({
                    "image_index": i,
                    "sample_index": sample_idx,
                    "mode": mode,
                    "ann_stem": args.ann_stem,
                    "adv_scope": args.adv_scope,
                    "sigma": args.sigma,
                    "base_margin": base_margin,
                    "margin": margin,
                    "margin_delta": margin - base_margin,
                    "abs_margin_delta": abs(margin - base_margin),
                    "base_best_adv_rank": base_adv_rank,
                    "best_adv_rank": adv_rank,
                    "best_adv_rank_delta": adv_rank - base_adv_rank,
                    "best_adv_rank_delta_abs": abs(adv_rank - base_adv_rank),
                    "score_delta_mean": float(delta.mean()),
                    "score_delta_std": float(delta.std()),
                    "score_delta_l2": float(np.linalg.norm(delta)),
                    "gt_score_delta": float(scores[best_gt_idx] - base_scores[best_gt_idx]),
                    "adv_score_delta": float(scores[best_adv_idx] - base_scores[best_adv_idx]),
                    "adv_beats_gt_pair": bool(scores[best_adv_idx] > scores[best_gt_idx]),
                })
            rows.extend(sample_rows)

        iso_rows = rows[-2 * args.n_samples : -args.n_samples]
        mani_rows = rows[-args.n_samples :]
        pair_rows.append({
            "image_index": i,
            "base_margin": base_margin,
            "iso_score_delta_std_mean": float(np.mean([r["score_delta_std"] for r in iso_rows])),
            "mani_score_delta_std_mean": float(np.mean([r["score_delta_std"] for r in mani_rows])),
            "iso_abs_margin_delta_mean": float(np.mean([r["abs_margin_delta"] for r in iso_rows])),
            "mani_abs_margin_delta_mean": float(np.mean([r["abs_margin_delta"] for r in mani_rows])),
            "iso_margin_delta_mean": float(np.mean([r["margin_delta"] for r in iso_rows])),
            "mani_margin_delta_mean": float(np.mean([r["margin_delta"] for r in mani_rows])),
            "iso_margin_toward_danger_fraction": float(np.mean([r["margin_delta"] < 0 for r in iso_rows])),
            "mani_margin_toward_danger_fraction": float(np.mean([r["margin_delta"] < 0 for r in mani_rows])),
            "iso_adv_flip_rate": float(np.mean([r["adv_beats_gt_pair"] for r in iso_rows])),
            "mani_adv_flip_rate": float(np.mean([r["adv_beats_gt_pair"] for r in mani_rows])),
        })

    csv_path = out_dir / f"{tag}_samples.csv"
    image_csv_path = out_dir / f"{tag}_per_image.csv"
    _write_csv(rows, csv_path)
    _write_csv(pair_rows, image_csv_path)

    summary = {
        "tag": tag,
        "ann_stem": args.ann_stem,
        "adv_scope": args.adv_scope,
        "sigma": args.sigma,
        "n_images": n_eval,
        "n_samples": args.n_samples,
        "scale_noise": cfg.smoothing.scale_noise,
        "metrics": {},
        "paired_comparison": {
            "iso_score_delta_std_gt_mani_fraction": float(np.mean([
                r["iso_score_delta_std_mean"] > r["mani_score_delta_std_mean"]
                for r in pair_rows
            ])),
            "iso_abs_margin_delta_gt_mani_fraction": float(np.mean([
                r["iso_abs_margin_delta_mean"] > r["mani_abs_margin_delta_mean"]
                for r in pair_rows
            ])),
            "iso_adv_flip_rate_mean": float(np.mean([r["iso_adv_flip_rate"] for r in pair_rows])),
            "mani_adv_flip_rate_mean": float(np.mean([r["mani_adv_flip_rate"] for r in pair_rows])),
            "iso_margin_toward_danger_fraction_mean": float(np.mean([
                r["iso_margin_toward_danger_fraction"] for r in pair_rows
            ])),
            "mani_margin_toward_danger_fraction_mean": float(np.mean([
                r["mani_margin_toward_danger_fraction"] for r in pair_rows
            ])),
        },
    }
    for mode in ("isotropic", "manifold"):
        mode_rows = [r for r in rows if r["mode"] == mode]
        summary["metrics"][mode] = {
            "score_delta_std": _summarize([r["score_delta_std"] for r in mode_rows]),
            "score_delta_l2": _summarize([r["score_delta_l2"] for r in mode_rows]),
            "gt_score_delta": _summarize([r["gt_score_delta"] for r in mode_rows]),
            "adv_score_delta": _summarize([r["adv_score_delta"] for r in mode_rows]),
            "margin_delta": _summarize([r["margin_delta"] for r in mode_rows]),
            "abs_margin_delta": _summarize([r["abs_margin_delta"] for r in mode_rows]),
            "margin_toward_danger_fraction": float(np.mean([
                r["margin_delta"] < 0 for r in mode_rows
            ])),
            "margin_toward_gt_fraction": float(np.mean([
                r["margin_delta"] > 0 for r in mode_rows
            ])),
            "best_adv_rank_delta_abs": _summarize([r["best_adv_rank_delta_abs"] for r in mode_rows]),
            "adv_flip_rate": float(np.mean([r["adv_beats_gt_pair"] for r in mode_rows])),
        }

    summary_path = out_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _plot(rows, out_dir, tag)
    _plot_signed_margin_delta(rows, out_dir, tag)
    _plot_score_delta_components(rows, out_dir, tag)

    _log(f"Saved sample CSV: {csv_path}")
    _log(f"Saved per-image CSV: {image_csv_path}")
    _log(f"Saved summary: {summary_path}")
    _log(
        "Headline: "
        f"Iso score-std > Mani fraction="
        f"{summary['paired_comparison']['iso_score_delta_std_gt_mani_fraction']:.3f}; "
        f"Iso margin-change > Mani fraction="
        f"{summary['paired_comparison']['iso_abs_margin_delta_gt_mani_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()
