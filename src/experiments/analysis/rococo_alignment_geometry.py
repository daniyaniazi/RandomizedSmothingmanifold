"""RoCOCO GT-vs-adversarial alignment geometry diagnostic.

This script asks whether the local image-neighbourhood covariance used by
manifold smoothing gives unusually low variance to the direction that changes
the GT-vs-adversarial caption competition.

For each image and annotation stem, it computes:
    d_sep = normalize(t_gt_best - t_adv_best)
    Q_sep = d_sep^T Sigma_loc d_sep

and compares Q_sep to Q values for random unit directions. If Q_sep is smaller
than random, Mani smoothing has less variance along the caption-separation
direction than an arbitrary direction, which can help explain lower RSMS than
isotropic smoothing.

Usage:
    python -m src.experiments.analysis.rococo_alignment_geometry \
        --config src/configs/experiments/rococo_clip_manifold.yaml \
        --ann-stem danger \
        --n-images 1000 \
        --knn-k 500 \
        --n-random 128 \
        --n-viz 12
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from src.smoothing.pca import fit_local_pca


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _q_value(direction: np.ndarray, evals: np.ndarray, evecs: np.ndarray) -> float:
    """Compute d^T Sigma d where Sigma = U diag(lambda) U^T."""
    proj = direction @ evecs
    return float(np.sum(evals * proj * proj))


def _topk_alignment(direction: np.ndarray, evecs: np.ndarray, k: int) -> float:
    k = min(k, evecs.shape[1])
    proj = direction @ evecs[:, :k]
    return float(np.sum(proj * proj))


def _caption_indices_for_image(
    img_idx: int,
    img2txt: Dict[int, List[int]],
    n_texts: int,
) -> Tuple[List[int], List[int]]:
    """Return GT and adversarial caption indices for one image.

    RoCOCO adversarial caption files are saved as contiguous image blocks:
    first GT captions, then adversarial captions. The cache directly stores
    img2txt for the GT part, so the adversarial indices are the rest of the
    block until the next image's GT block begins.
    """
    gt = list(img2txt.get(img_idx, []))
    if not gt:
        return [], []

    starts = sorted(min(v) for v in img2txt.values() if v)
    block_start = min(gt)
    next_starts = [s for s in starts if s > block_start]
    block_end = next_starts[0] if next_starts else n_texts
    adv = list(range(max(gt) + 1, block_end))
    return gt, adv


def _random_unit_directions(
    rng: np.random.Generator,
    n_random: int,
    dim: int,
) -> np.ndarray:
    r = rng.normal(size=(n_random, dim)).astype(np.float32)
    return _normalize_rows(r)


def _plot_summary(rows: List[dict], out_dir: Path, ann_stem: str) -> None:
    q_sep = np.array([r["q_sep"] for r in rows], dtype=np.float64)
    q_rand = np.array([r["q_rand_mean"] for r in rows], dtype=np.float64)
    ratio = np.array([r["q_ratio_sep_over_rand"] for r in rows], dtype=np.float64)
    percentile = np.array([r["q_sep_random_percentile"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor="white")

    ax = axes[0]
    bins = 40
    ax.hist(q_rand, bins=bins, alpha=0.65, label="random directions", color="#4c78a8")
    ax.hist(q_sep, bins=bins, alpha=0.65, label="GT-adv separator", color="#e45756")
    ax.set_xlabel("Q = d^T Sigma_loc d")
    ax.set_ylabel("Images")
    ax.set_title("Local covariance variance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(q_rand, q_sep, s=10, alpha=0.45, color="#2f4b7c", linewidths=0)
    lim = max(float(q_rand.max()), float(q_sep.max()))
    ax.plot([0, lim], [0, lim], "--", lw=1, color="#777777")
    ax.set_xlabel("Mean random-direction Q")
    ax.set_ylabel("GT-adv separator Q")
    ax.set_title("Per-image comparison")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.hist(ratio, bins=40, alpha=0.75, color="#54a24b", label="Q_sep / Q_rand")
    ax.axvline(1.0, color="#333333", lw=1.5, ls="--")
    ax2 = ax.twinx()
    ax2.hist(percentile, bins=20, alpha=0.30, color="#f58518", label="random percentile")
    ax.set_xlabel("Ratio or percentile")
    ax.set_ylabel("Images")
    ax2.set_ylabel("Images")
    ax.set_title("Suppression strength")
    ax.grid(alpha=0.25)

    fig.suptitle(f"RoCOCO {ann_stem}: GT-adversarial alignment geometry", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"{ann_stem}_qsep_vs_random.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved summary plot: {out_path}")


def _plot_examples(
    examples: List[dict],
    out_dir: Path,
    ann_stem: str,
) -> None:
    viz_dir = out_dir / "examples"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for ex in examples:
        anchor = ex["anchor"]
        neighbours = ex["neighbours"]
        pca = ex["pca"]
        d_sep = ex["d_sep"]
        d_gt = ex["d_gt"]
        d_adv = ex["d_adv"]
        row = ex["row"]

        evecs = pca.evecs
        centered = neighbours - pca.mean
        anchor_c = anchor - pca.mean
        neigh_2d = centered @ evecs[:, :2]
        anchor_2d = anchor_c @ evecs[:, :2]

        def arrow_components(direction: np.ndarray) -> np.ndarray:
            comp = direction @ evecs[:, :2]
            norm = np.linalg.norm(comp)
            if norm > 1e-12:
                comp = comp / norm
            scale = max(float(np.std(neigh_2d[:, 0])), 1e-3) * 2.0
            return comp * scale

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

        ax = axes[0]
        ax.scatter(neigh_2d[:, 0], neigh_2d[:, 1], s=7, alpha=0.28,
                   color="#9aa0a6", linewidths=0, label="image kNN")
        ax.scatter(anchor_2d[0], anchor_2d[1], s=140, marker="*",
                   color="#f2c94c", edgecolors="white", linewidths=0.8,
                   label="image")

        arrows = [
            ("GT direction", d_gt, "#2ca02c"),
            ("Danger direction", d_adv, "#d62728"),
            ("GT-Danger separator", d_sep, "#1f77b4"),
        ]
        for label, direction, color in arrows:
            comp = arrow_components(direction)
            ax.arrow(anchor_2d[0], anchor_2d[1], comp[0], comp[1],
                     color=color, width=0.0008, head_width=0.012,
                     length_includes_head=True, alpha=0.9, label=label)

        ax.set_aspect("equal")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Local image PCA plane")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

        ax = axes[1]
        n_show = min(50, len(pca.evals))
        ax.plot(range(1, n_show + 1), pca.evals[:n_show], "o-",
                ms=3, lw=1.2, color="#6f4e7c")
        ax.set_yscale("log")
        ax.set_xlabel("PC index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(
            "Q values\n"
            f"Q_sep={row['q_sep']:.4g}, "
            f"Q_rand={row['q_rand_mean']:.4g}, "
            f"ratio={row['q_ratio_sep_over_rand']:.3f}"
        )
        ax.grid(alpha=0.25)

        fig.suptitle(
            f"{ann_stem} image={row['image_index']} "
            f"margin={row['gt_minus_adv_margin']:.4f} "
            f"percentile={row['q_sep_random_percentile']:.3f}",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = viz_dir / f"{ann_stem}_image{row['image_index']:05d}_geometry.png"
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)

    _log(f"Saved {len(examples)} example geometry plots: {viz_dir}")


def _write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = [k for k in rows[0].keys() if not k.startswith("_")]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _summary(rows: List[dict], args: argparse.Namespace, cfg) -> dict:
    def vals(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=np.float64)

    q_sep = vals("q_sep")
    q_rand = vals("q_rand_mean")
    ratio = vals("q_ratio_sep_over_rand")
    percentile = vals("q_sep_random_percentile")
    q_gt = vals("q_gt_image_to_gt")
    q_adv = vals("q_adv_image_to_adv")

    return {
        "ann_stem": args.ann_stem,
        "config": args.config,
        "clip_model": cfg.clip_model,
        "n_images": len(rows),
        "knn_k": args.knn_k,
        "n_random": args.n_random,
        "seed": args.seed,
        "scale_noise": cfg.smoothing.scale_noise,
        "interpretation": {
            "q_sep": "Variance assigned by local image PCA covariance to the GT-vs-adversarial separating direction.",
            "q_ratio_sep_over_rand": "Values below 1 mean the separator receives less local manifold variance than random directions.",
            "q_sep_random_percentile": "Fraction of sampled random directions with Q <= Q_sep; small values mean Q_sep is unusually low.",
        },
        "means": {
            "q_sep": float(q_sep.mean()),
            "q_rand_mean": float(q_rand.mean()),
            "q_ratio_sep_over_rand": float(ratio.mean()),
            "q_sep_random_percentile": float(percentile.mean()),
            "q_gt_image_to_gt": float(q_gt.mean()),
            "q_adv_image_to_adv": float(q_adv.mean()),
        },
        "medians": {
            "q_sep": float(np.median(q_sep)),
            "q_rand_mean": float(np.median(q_rand)),
            "q_ratio_sep_over_rand": float(np.median(ratio)),
            "q_sep_random_percentile": float(np.median(percentile)),
        },
        "fractions": {
            "q_sep_below_rand_mean": float(np.mean(q_sep < q_rand)),
            "ratio_below_1": float(np.mean(ratio < 1.0)),
            "percentile_below_0_25": float(np.mean(percentile < 0.25)),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoCOCO alignment geometry diagnostic")
    p.add_argument("--config", required=True)
    p.add_argument("--ann-stem", default="danger",
                   help="Adversarial annotation stem: danger, same_concept, diff_concept, rand_voca")
    p.add_argument("--n-images", type=int, default=None,
                   help="Optional first-N subset for quick analysis.")
    p.add_argument("--knn-k", type=int, default=None)
    p.add_argument("--n-random", type=int, default=128)
    p.add_argument("--n-viz", type=int, default=12)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--topk", type=int, default=50,
                   help="Top-k PCA overlap A_k reported as an auxiliary number.")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_rococo_config(args.config)
    if args.knn_k is None:
        args.knn_k = cfg.smoothing.knn_k
    cfg.smoothing.knn_k = args.knn_k

    rng = np.random.default_rng(args.seed)
    out_dir = args.output_dir or (_ROOT / cfg.output_dir / "alignment_geometry" / args.ann_stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg.embedding_cache_dir)
    image_path = cache_dir / "image_embeddings.pt"
    cap_path = cache_dir / f"{args.ann_stem}_captions.pt"
    if not image_path.exists():
        raise FileNotFoundError(f"Image embeddings not found: {image_path}")
    if not cap_path.exists():
        raise FileNotFoundError(f"Caption embeddings not found: {cap_path}")

    _log(f"Loading image embeddings: {image_path}")
    image_cache = torch.load(image_path, map_location="cpu")
    image_embs = image_cache["embeddings"].numpy().astype(np.float32)
    image_embs = _normalize_rows(image_embs)
    image_ids = image_cache["image_ids"]

    _log(f"Loading caption embeddings: {cap_path}")
    cap_cache = torch.load(cap_path, map_location="cpu")
    text_embs = cap_cache["embeddings"].numpy().astype(np.float32)
    text_embs = _normalize_rows(text_embs)
    img2txt = {int(k): list(v) for k, v in cap_cache["img2txt"].items()}

    n_total = len(image_embs)
    n_eval = min(args.n_images, n_total) if args.n_images else n_total
    _log(f"Images: {n_eval:,}/{n_total:,}; captions: {len(text_embs):,}")

    from src.experiments.indexing.rococo_clip_images import build_rococo_clip_index

    index_path = _ROOT / cfg.index_dir / "index.ann"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Annoy index not found: {index_path}\n"
            f"Run: python -m src.experiments.indexing.rococo_clip_images --config {args.config}"
        )
    clip_index = build_rococo_clip_index(cfg, rebuild=False)

    viz_indices = set()
    if args.n_viz > 0 and n_eval > 0:
        viz_indices = set(rng.choice(n_eval, min(args.n_viz, n_eval), replace=False).tolist())

    rows: List[dict] = []
    examples: List[dict] = []

    for i in tqdm(range(n_eval), desc=f"Alignment geometry [{args.ann_stem}]"):
        anchor = image_embs[i]
        gt_idx, adv_idx = _caption_indices_for_image(i, img2txt, len(text_embs))
        if not gt_idx or not adv_idx:
            continue

        gt_scores = text_embs[gt_idx] @ anchor
        adv_scores = text_embs[adv_idx] @ anchor
        best_gt_idx = gt_idx[int(np.argmax(gt_scores))]
        best_adv_idx = adv_idx[int(np.argmax(adv_scores))]
        t_gt = text_embs[best_gt_idx]
        t_adv = text_embs[best_adv_idx]

        d_sep = _normalize(t_gt - t_adv)
        d_gt = _normalize(t_gt - anchor)
        d_adv = _normalize(t_adv - anchor)

        nn_ids = clip_index.index.get_nns_by_vector(
            anchor.tolist(), args.knn_k + 1, include_distances=False
        )
        nn_ids = [n for n in nn_ids if n != i][: args.knn_k]
        neighbours = image_embs[nn_ids]
        pca = fit_local_pca(neighbours, eps_eig=cfg.smoothing.eps_eig)

        q_sep = _q_value(d_sep, pca.evals, pca.evecs)
        q_gt = _q_value(d_gt, pca.evals, pca.evecs)
        q_adv = _q_value(d_adv, pca.evals, pca.evecs)
        a_sep = _topk_alignment(d_sep, pca.evecs, args.topk)
        a_gt = _topk_alignment(d_gt, pca.evecs, args.topk)
        a_adv = _topk_alignment(d_adv, pca.evecs, args.topk)

        random_dirs = _random_unit_directions(rng, args.n_random, image_embs.shape[1])
        q_random = np.array([
            _q_value(d, pca.evals, pca.evecs) for d in random_dirs
        ], dtype=np.float64)
        q_rand_mean = float(q_random.mean())
        q_rand_p05, q_rand_p50, q_rand_p95 = np.quantile(q_random, [0.05, 0.50, 0.95])

        row = {
            "image_index": i,
            "image_id": image_ids[i],
            "best_gt_caption_index": best_gt_idx,
            "best_adv_caption_index": best_adv_idx,
            "best_gt_score": float(gt_scores.max()),
            "best_adv_score": float(adv_scores.max()),
            "gt_minus_adv_margin": float(gt_scores.max() - adv_scores.max()),
            "q_sep": q_sep,
            "q_rand_mean": q_rand_mean,
            "q_rand_p05": float(q_rand_p05),
            "q_rand_p50": float(q_rand_p50),
            "q_rand_p95": float(q_rand_p95),
            "q_ratio_sep_over_rand": float(q_sep / max(q_rand_mean, 1e-12)),
            "q_sep_random_percentile": float(np.mean(q_random <= q_sep)),
            "q_gt_image_to_gt": q_gt,
            "q_adv_image_to_adv": q_adv,
            f"a{args.topk}_sep": a_sep,
            f"a{args.topk}_gt": a_gt,
            f"a{args.topk}_adv": a_adv,
            "lambda_max": float(pca.evals[0]),
            "lambda_trace": float(np.sum(pca.evals)),
        }
        rows.append(row)

        if i in viz_indices:
            examples.append({
                "anchor": anchor,
                "neighbours": neighbours,
                "pca": pca,
                "d_sep": d_sep,
                "d_gt": d_gt,
                "d_adv": d_adv,
                "row": row,
            })

    if not rows:
        raise RuntimeError("No usable image/caption pairs were found.")

    csv_path = out_dir / f"{args.ann_stem}_alignment_geometry.csv"
    json_path = out_dir / f"{args.ann_stem}_alignment_summary.json"
    _write_csv(rows, csv_path)
    summary = _summary(rows, args, cfg)
    json_path.write_text(json.dumps(summary, indent=2))
    _plot_summary(rows, out_dir, args.ann_stem)
    _plot_examples(examples, out_dir, args.ann_stem)

    _log(f"Saved CSV: {csv_path}")
    _log(f"Saved summary: {json_path}")
    _log(
        "Result headline: "
        f"mean ratio Q_sep/Q_rand={summary['means']['q_ratio_sep_over_rand']:.3f}, "
        f"fraction below random mean={summary['fractions']['q_sep_below_rand_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
