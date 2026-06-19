"""Visualization for RoCOCO CLIP retrieval experiments.

Saves one PNG per annotation file per method (10 sample images each):

Baseline row:
  [original image] | GT caption | top-5 retrieved captions

Isotropic row:
  [original image] | [noisy image sample] | top-5 retrieved captions (from smoothed emb)

Manifold row:
  [original image] | [5 kNN images] | top-5 retrieved captions (from smoothed emb)

Usage:
    python -m src.experiments.eval.rococo_viz \\
        --config src/configs/experiments/rococo_clip_manifold.yaml \\
        --ann-file danger.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.rococo_clip_schema import RoCoCoConfig, load_rococo_config
from src.dataloaders.rococo import RoCoCoDataset
from src.experiments.eval.rococo_clip_eval import smooth_iso, smooth_manifold, _normalize
from src.models.CLIP.model import CLIPWrapper
from src.smoothing.isotropic import IsotropicSmoother
from src.smoothing.manifold import ManifoldSmoother


def _load_pil(path: str, size: int = 224) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size))


def _wrap(text: str, max_len: int = 55) -> str:
    """Wrap long text for display."""
    words = text.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > max_len:
            lines.append(" ".join(cur[:-1]))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def save_retrieval_viz(
    samples: List[Dict],
    save_path: Path,
    mode: str,
    ann_stem: str,
    n_show: int = 10,
) -> None:
    """Save visualization grid.

    Layout per sample (1 row):
      Col 0:    Query image (original)
      Col 1-5:  Top-5 retrieved images (the images whose caption scored highest)
      Col 6:    Text panel — GT caption + top-5 captions with scores

    For manifold mode, kNN neighbour images shown instead of top-5 matched images
    (since kNN images explain why the smoothed embedding moved).

    Each sample dict:
      image_path:         str — query image path
      gt_caption:         str — first GT caption
      top5_captions:      List[str]
      top5_scores:        List[float]
      top5_image_paths:   List[str] — images corresponding to top-5 captions
      knn_image_paths:    Optional[List[str]] — manifold: 5 kNN neighbours
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
    except ImportError:
        print("matplotlib not available — skipping visualization")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    samples  = samples[:n_show]
    n_rows   = len(samples)
    N_COLS   = 7   # query + 5 images + text
    fig_w    = N_COLS * 2.2
    fig_h    = n_rows * 2.6

    fig, axes = plt.subplots(
        n_rows, N_COLS, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 2.8],
                     "wspace": 0.04, "hspace": 0.35}
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ax in axes.flat:
        ax.axis("off")

    for r, s in enumerate(samples):
        # Col 0: query image
        axes[r, 0].imshow(_load_pil(s["image_path"]))
        axes[r, 0].set_title("Query", fontsize=7, color="#333333")

        # Cols 1-5: top-5 retrieved images (or kNN images for manifold)
        show_paths = s.get("knn_image_paths") or s.get("top5_image_paths", [])
        col_label  = "NN" if s.get("knn_image_paths") else "Top"
        for k, img_path in enumerate(show_paths[:5], start=1):
            try:
                axes[r, k].imshow(_load_pil(img_path))
                axes[r, k].set_title(f"{col_label}-{k}", fontsize=6, color="#555")
            except Exception:
                pass

        # Col 6: text panel
        tax = axes[r, 6]
        tax.axis("off")
        lines = [f"GT: {_wrap(s['gt_caption'], 42)}\n"]
        for rank, (cap, sc) in enumerate(zip(s["top5_captions"], s["top5_scores"]), 1):
            hit = "✓" if rank == 1 and cap == s["gt_caption"] else " "
            lines.append(f"[{rank}]{hit} {_wrap(cap, 40)}  ({sc:.3f})")
        tax.text(0.02, 0.97, "\n".join(lines),
                 transform=tax.transAxes, va="top", fontsize=6.2,
                 fontfamily="monospace")

    fig.suptitle(f"{mode.title()} — {ann_stem.replace('_',' ').title()}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_viz(
    cfg: RoCoCoConfig,
    ann_file: str,
    n_show: int = 10,
) -> None:
    mode     = cfg.smoothing.mode
    ann_stem = Path(ann_file).stem
    cache_dir = Path(cfg.embedding_cache_dir)
    viz_dir   = _ROOT / cfg.output_dir / mode / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    img_cache  = torch.load(cache_dir / "image_embeddings.pt", map_location="cpu")
    image_embs = img_cache["embeddings"].numpy().astype(np.float32)
    image_ids  = img_cache["image_ids"]

    cap_path   = cache_dir / f"{ann_stem}_captions.pt"
    cap_cache = torch.load(cap_path, map_location="cpu")
    text_embs = cap_cache["embeddings"].numpy().astype(np.float32)
    captions  = cap_cache["captions"]

    # Use img2txt directly from new format; reconstruct from legacy if needed
    if "img2txt" in cap_cache:
        img2txt = cap_cache["img2txt"]
    else:
        raw_indices = cap_cache["image_indices"]
        img2txt = {}
        for ci, ii in enumerate(raw_indices):
            img2txt.setdefault(ii, []).append(ci)

    # Reverse map: caption_idx → image_idx (for top-5 image display)
    cap_to_img: Dict[int, int] = {}
    for img_idx, cap_idxs in img2txt.items():
        for ci in cap_idxs:
            cap_to_img[ci] = img_idx

    # Dataset for image paths and GT captions
    dataset = RoCoCoDataset(cfg.image_dir, cfg.annotation_dir, cfg.annotation_files)

    # kNN index for manifold
    pixel_index = None
    if mode == "manifold":
        from src.experiments.indexing.rococo_clip_images import build_rococo_clip_index
        pixel_index = build_rococo_clip_index(cfg, rebuild=False)

    smoother = None
    if mode == "isotropic":
        smoother = IsotropicSmoother(sigma=cfg.smoothing.sigma)
    elif mode == "manifold":
        smoother = ManifoldSmoother(sigma=cfg.smoothing.sigma, index=pixel_index,
                                    knn_k=cfg.smoothing.knn_k, eps_eig=cfg.smoothing.eps_eig)

    # Sample n_show images evenly
    indices = np.linspace(0, len(dataset)-1, n_show, dtype=int)
    samples_out = []

    for i in indices:
        s      = dataset.samples[i]
        emb    = image_embs[i]

        # Smooth embedding — no "noisy image" since smoothing is in CLIP embedding space
        if mode == "baseline":
            q_emb     = emb
            knn_paths = None
        elif mode == "isotropic":
            q_emb     = smooth_iso(emb, smoother, cfg.smoothing.n_samples)
            knn_paths = None
        elif mode == "manifold":
            q_emb     = smooth_manifold(emb, smoother, cfg.smoothing.n_samples)
            # kNN neighbours in CLIP embedding space
            nn_ids    = pixel_index.index.get_nns_by_vector(
                emb.tolist(), 6, include_distances=False)[1:6]
            knn_paths = [str(Path(cfg.image_dir) / image_ids[nid])
                         for nid in nn_ids if nid < len(image_ids)]
        else:
            q_emb     = emb
            knn_paths = None

        # Retrieve top-5 captions from smoothed embedding
        scores      = q_emb @ text_embs.T
        top5_ci     = np.argsort(-scores)[:5]
        top5_caps   = [captions[ci]       for ci in top5_ci]
        top5_scores = [float(scores[ci])  for ci in top5_ci]

        # Map top-5 caption indices → their image paths
        top5_img_paths = []
        for ci in top5_ci:
            ii = cap_to_img.get(int(ci))
            if ii is not None and ii < len(dataset.samples):
                top5_img_paths.append(dataset.samples[ii].image_path)

        gt_cap = s.gt_captions[0] if s.gt_captions else ""
        samples_out.append({
            "image_path":       s.image_path,
            "gt_caption":       gt_cap,
            "top5_captions":    top5_caps,
            "top5_scores":      top5_scores,
            "top5_image_paths": top5_img_paths,
            "knn_image_paths":  knn_paths,     # manifold only: kNN neighbour images
        })

    save_retrieval_viz(
        samples_out,
        save_path=viz_dir / f"{ann_stem}_viz.png",
        mode=mode,
        ann_stem=ann_stem,
        n_show=n_show,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   required=True)
    p.add_argument("--ann-file", default=None)
    p.add_argument("--n-show",   type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_rococo_config(args.config)
    ann_files = [args.ann_file] if args.ann_file else cfg.annotation_files
    for af in ann_files:
        run_viz(cfg, af, n_show=args.n_show)
