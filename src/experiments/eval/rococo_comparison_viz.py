"""Side-by-side comparison visualization: CLIP vs ISO vs Manifold for the same samples.

For each annotation × sigma combination, generates one figure per sample batch (5 samples),
each sample showing 3 rows:
  Row 1 (CLIP)     — baseline retrieval
  Row 2 (ISO)      — isotropic smoothed retrieval at given sigma
  Row 3 (Manifold) — manifold smoothed retrieval at given sigma

Reads pre-computed metric files from output/rococo/{mode}/sigma_{s}/ — does NOT
re-run smoothing or disturb existing per-mode visualizations.

Output: output/rococo/comparison/{ann_stem}/sigma_{s}/compare_part{N}.png

Usage:
    python -m src.experiments.eval.rococo_comparison_viz \\
        --config src/configs/experiments/rococo_clip_manifold.yaml \\
        --sigma 0.05 0.10 \\
        --ann danger same_concept \\
        --n-samples 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.rococo_clip_schema import RoCoCoConfig, load_rococo_config
from src.dataloaders.rococo import RoCoCoDataset
from src.experiments.eval.rococo_clip_eval import smooth_iso, smooth_manifold, _normalize
from src.experiments.eval.rococo_viz import _load_pil, _caption_color, _wrap
from src.smoothing.isotropic import IsotropicSmoother
from src.smoothing.manifold import ManifoldSmoother


# ── helpers ───────────────────────────────────────────────────────────────────

def _top5(q_emb: np.ndarray, text_embs: np.ndarray, captions: List[str],
          cap_to_img: Dict[int, int], dataset: RoCoCoDataset):
    scores    = q_emb @ text_embs.T
    top5_ci   = np.argsort(-scores)[:5]
    top5_caps = [captions[ci] for ci in top5_ci]
    top5_sc   = [float(scores[ci]) for ci in top5_ci]
    top5_imgs = []
    for ci in top5_ci:
        ii = cap_to_img.get(int(ci))
        if ii is not None and int(ii) < len(dataset.samples):
            top5_imgs.append(dataset.samples[int(ii)].image_path)
        else:
            top5_imgs.append(None)
    return top5_caps, top5_sc, top5_imgs


def _draw_row(axes_row, label: str, label_color: str,
              query_path: str, top5_imgs: List[Optional[str]],
              top5_caps: List[str], top5_sc: List[float],
              gt_set: set, adv_set: set) -> None:
    """Fill one row of axes: [label | query | top5 images | caption panel]."""
    import matplotlib.pyplot as plt

    # Col 0: row label
    axes_row[0].axis("off")
    axes_row[0].text(0.5, 0.5, label, transform=axes_row[0].transAxes,
                     ha="center", va="center", fontsize=7, fontweight="bold",
                     color=label_color, rotation=90)

    # Col 1: query image
    axes_row[1].imshow(_load_pil(query_path))
    axes_row[1].axis("off")

    # Cols 2-6: top-5 retrieved images
    for k in range(5):
        ax   = axes_row[2 + k]
        ax.axis("off")
        cap  = top5_caps[k] if k < len(top5_caps) else ""
        col  = _caption_color(cap, gt_set, adv_set)
        path = top5_imgs[k] if k < len(top5_imgs) else None
        if path is not None:
            try:
                ax.imshow(_load_pil(path))
                ax.set_title(f"Top-{k+1}", fontsize=5, color=col, pad=2)
            except Exception:
                ax.set_title(f"Top-{k+1}", fontsize=5, color=col, pad=2)
        else:
            ax.text(0.5, 0.5, f"Top-{k+1}\n(adv)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=5, color=col)

    # Col 7: caption panel
    tax = axes_row[7]
    tax.axis("off")
    y = 0.97
    for rank, (cap, sc) in enumerate(zip(top5_caps, top5_sc), 1):
        col     = _caption_color(cap, gt_set, adv_set)
        wrapped = _wrap(cap, max_len=42)
        n_lines = wrapped.count('\n') + 1
        tax.text(0.02, y, f"[{rank}] {wrapped}  ({sc:.3f})",
                 transform=tax.transAxes, va="top", fontsize=5,
                 color=col, fontfamily="monospace")
        y -= 0.04 + 0.055 * n_lines


# ── per-sample data builder ───────────────────────────────────────────────────

def _build_sample(
    idx: int,
    dataset: RoCoCoDataset,
    raw_embs: np.ndarray,
    text_embs: np.ndarray,
    captions: List[str],
    cap_to_img: Dict[int, int],
    global_adv_captions: set,
    iso_smoother: Optional[IsotropicSmoother],
    mani_smoother: Optional[ManifoldSmoother],
    n_samples: int,
) -> Dict:
    s   = dataset.samples[idx]
    emb = raw_embs[idx]

    # Baseline
    base_caps, base_sc, base_imgs = _top5(
        _normalize(emb), text_embs, captions, cap_to_img, dataset)

    # ISO
    if iso_smoother is not None:
        iso_emb = smooth_iso(emb, iso_smoother, n_samples)
    else:
        iso_emb = _normalize(emb)
    iso_caps, iso_sc, iso_imgs = _top5(iso_emb, text_embs, captions, cap_to_img, dataset)

    # Manifold
    if mani_smoother is not None:
        mani_emb = smooth_manifold(emb, mani_smoother, n_samples)
    else:
        mani_emb = _normalize(emb)
    mani_caps, mani_sc, mani_imgs = _top5(mani_emb, text_embs, captions, cap_to_img, dataset)

    return {
        "image_path":  s.image_path,
        "gt_caption":  s.gt_captions[0] if s.gt_captions else "",
        "gt_set":      set(s.gt_captions),
        "adv_set":     global_adv_captions,
        "baseline":    (base_caps, base_sc, base_imgs),
        "iso":         (iso_caps,  iso_sc,  iso_imgs),
        "manifold":    (mani_caps, mani_sc, mani_imgs),
    }


# ── figure saver ─────────────────────────────────────────────────────────────

ROW_LABELS = {
    "baseline": ("CLIP",     "#636363"),
    "iso":      ("ISO",      "#2166ac"),
    "manifold": ("Manifold", "#b8860b"),
}

N_COLS = 8   # label | query | top5 | captions


def _save_comparison_batch(
    batch: List[Dict],
    save_path: Path,
    sigma: float,
    ann_stem: str,
    batch_idx: int,
    n_batches: int,
) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7})

    n_samples = len(batch)
    n_rows    = n_samples * 3   # 3 methods per sample
    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(N_COLS * 2.0, n_rows * 2.2),
        gridspec_kw={
            "width_ratios": [0.18, 1, 1, 1, 1, 1, 1, 2.8],
            "wspace": 0.04,
            "hspace": 0.35,
        },
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        ax.axis("off")

    for s_idx, sample in enumerate(batch):
        for m_idx, mode in enumerate(["baseline", "iso", "manifold"]):
            row       = s_idx * 3 + m_idx
            label, lc = ROW_LABELS[mode]
            caps, scs, imgs = sample[mode]

            # Divider line between samples
            if m_idx == 0 and s_idx > 0:
                for c in range(N_COLS):
                    axes[row, c].axhline(y=1.0, color="#cccccc", lw=0.8,
                                         transform=axes[row, c].transAxes, clip_on=False)

            # GT caption header on first method row
            if m_idx == 0:
                axes[row, 1].set_title(
                    f"GT: {sample['gt_caption'][:60]}{'…' if len(sample['gt_caption'])>60 else ''}",
                    fontsize=5.5, color="#1a7a1a", loc="left", pad=3)

            _draw_row(axes[row], label, lc,
                      sample["image_path"], imgs, caps, scs,
                      sample["gt_set"], sample["adv_set"])

    title = (f"CLIP vs ISO vs Manifold  |  σ={sigma}  |  "
             f"{ann_stem.replace('_',' ').title()}")
    if n_batches > 1:
        title += f"  [{batch_idx+1}/{n_batches}]"

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0],[0], color="#1a7a1a", lw=3, label="GT caption"),
        Line2D([0],[0], color="#c0392b", lw=3, label="Adversarial caption"),
        Line2D([0],[0], color="#1a5fa8", lw=3, label="Other retrieved"),
    ], loc="lower center", ncol=3, fontsize=7,
       bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    fig.suptitle(title, fontsize=10, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_comparison(
    cfg: RoCoCoConfig,
    sigmas: List[float],
    ann_stems: List[str],
    n_show: int = 5,
) -> None:
    import torch

    cache_dir = Path(cfg.embedding_cache_dir)
    out_base  = _ROOT / cfg.output_dir / "rococo" / "comparison"

    # Load raw image embeddings once
    emb_cache  = torch.load(cache_dir / "image_embeddings.pt", map_location="cpu")
    raw_embs   = emb_cache["embeddings"].numpy().astype(np.float32)
    print(f"Image embeddings: {raw_embs.shape}")

    # kNN index for manifold
    from src.experiments.indexing.rococo_clip_images import build_rococo_clip_index
    index_path = _ROOT / cfg.index_dir / "index.ann"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Annoy index not found: {index_path}\n"
            f"Run: python -m src.experiments.indexing.rococo_clip_images "
            f"--config src/configs/experiments/rococo_clip_manifold.yaml"
        )
    pixel_index = build_rococo_clip_index(cfg, rebuild=False)

    dataset = RoCoCoDataset(cfg.image_dir, cfg.annotation_dir, cfg.annotation_files)

    # Sample indices (evenly spaced, bounded by available embeddings)
    n_avail = len(raw_embs)
    indices = np.linspace(0, n_avail - 1, min(n_show, n_avail), dtype=int)

    for sigma in sigmas:
        print(f"\n── sigma={sigma} ────────────────────────────────────")
        iso_smoother  = IsotropicSmoother(sigma=sigma)
        mani_smoother = ManifoldSmoother(sigma=sigma, index=pixel_index,
                                         knn_k=cfg.smoothing.knn_k,
                                         eps_eig=cfg.smoothing.eps_eig)

        for ann_stem in ann_stems:
            cap_path = cache_dir / f"{ann_stem}_captions.pt"
            if not cap_path.exists():
                print(f"  SKIP {ann_stem} — caption cache not found")
                continue

            cap_cache = torch.load(cap_path, map_location="cpu")
            text_embs = cap_cache["embeddings"].numpy().astype(np.float32)
            captions  = cap_cache["captions"]

            img2txt = cap_cache.get("img2txt", {})
            cap_to_img: Dict[int, int] = {
                int(ci): int(img_idx)
                for img_idx, cap_idxs in img2txt.items()
                for ci in cap_idxs
            }
            global_adv_captions = {
                captions[ci] for ci in cap_cache.get("wrongtext", [])
                if ci < len(captions)
            }

            print(f"  Building samples for {ann_stem} …")
            sigma_str = f"sigma_{sigma:.2f}".replace(".", "_")
            for s_num, i in enumerate(indices, 1):
                sample = _build_sample(
                    i, dataset, raw_embs, text_embs, captions,
                    cap_to_img, global_adv_captions,
                    iso_smoother, mani_smoother, cfg.smoothing.n_samples)
                save_path = out_base / ann_stem / sigma_str / f"sample_{s_num:02d}.png"
                _save_comparison_batch([sample], save_path, sigma, ann_stem, 0, 1)


def parse_args():
    p = argparse.ArgumentParser(description="CLIP vs ISO vs Manifold comparison figures")
    p.add_argument("--config",    required=True)
    p.add_argument("--sigma",     type=float, nargs="+", default=None,
                   help="Sigma values to visualize (default: all from config)")
    p.add_argument("--ann",       nargs="+", default=None,
                   help="Annotation stems, e.g. danger same_concept (default: all)")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Number of query images per figure (default: 5)")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_rococo_config(args.config)
    sigmas = args.sigma or [float(s) for s in (cfg.smoothing.sigma_values or [cfg.smoothing.sigma])]
    anns   = args.ann   or [Path(f).stem for f in cfg.annotation_files]
    run_comparison(cfg, sigmas=sigmas, ann_stems=anns, n_show=args.n_samples)
