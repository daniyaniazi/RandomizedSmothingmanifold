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
from src.experiments.eval.rococo_viz import _load_pil, _caption_color


def _wrap(text: str, max_len: int = 42) -> str:
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


def _draw_query_row(axes_row, query_path: str, gt_caption: str) -> None:
    """Query row: [empty label | query image | empty cols | empty caption col]
    GT caption shown as title above the query image cell.
    """
    for ax in axes_row:
        ax.axis("off")

    # Col 1: query image with GT caption as title above it
    axes_row[1].imshow(_load_pil(query_path, size=256))
    axes_row[1].axis("off")
    gt_wrapped = _wrap(gt_caption, max_len=60)
    axes_row[1].set_title(f"GT: {gt_wrapped}",
                          fontsize=6, color="#1a7a1a",
                          loc="left", pad=4)


def _draw_method_row(axes_row, label: str, label_color: str,
                     top5_imgs: List[Optional[str]],
                     top5_caps: List[str], top5_sc: List[float],
                     gt_set: set, adv_set: set) -> None:
    """Method row: [label | top5 images | caption panel]."""

    # Col 0: method label
    axes_row[0].axis("off")
    axes_row[0].text(0.5, 0.5, label, transform=axes_row[0].transAxes,
                     ha="center", va="center", fontsize=7, fontweight="bold",
                     color=label_color, rotation=90)

    # Cols 1-5: top-5 retrieved images (aligned with query in row above)
    axes_row[6].axis("off")   # col 6 unused
    for k in range(5):
        ax   = axes_row[1 + k]
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

    # Col 7: caption panel — vertically centred
    tax = axes_row[7]
    tax.axis("off")

    entries = []
    for rank, (cap, sc) in enumerate(zip(top5_caps, top5_sc), 1):
        w  = _wrap(cap, max_len=45)
        nl = w.count('\n') + 1
        col = _caption_color(cap, gt_set, adv_set)
        entries.append((f"[{rank}] {w}  ({sc:.3f})", col, 5.5, 0.04 + 0.055 * nl))

    total_h = sum(e[3] for e in entries)
    y = 0.5 + total_h / 2

    for txt, col, fs, step in entries:
        tax.text(0.02, y, txt,
                 transform=tax.transAxes, va="top", fontsize=fs,
                 color=col, fontfamily="monospace", clip_on=True)
        y -= step


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
) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7})

    n_samples = len(batch)
    # 4 rows per sample: 1 query row + 3 method rows
    ROWS_PER = 4
    n_rows   = n_samples * ROWS_PER
    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(N_COLS * 2.2, n_samples * (1.8 + 3 * 3.0)),
        gridspec_kw={
            "width_ratios": [0.18, 1, 1, 1, 1, 1, 1, 3.5],
            "wspace": 0.04,
            "hspace": 0.20,
        },
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        ax.axis("off")

    for s_idx, sample in enumerate(batch):
        base_row = s_idx * ROWS_PER

        # Divider between samples
        if s_idx > 0:
            for c in range(N_COLS):
                axes[base_row, c].axhline(y=1.0, color="#aaaaaa", lw=1.0,
                                          transform=axes[base_row, c].transAxes,
                                          clip_on=False)

        # Row 0: query image + GT caption
        _draw_query_row(axes[base_row], sample["image_path"], sample["gt_caption"])

        # Rows 1-3: one per method
        for m_idx, mode in enumerate(["baseline", "iso", "manifold"]):
            row        = base_row + 1 + m_idx
            label, lc  = ROW_LABELS[mode]
            caps, scs, imgs = sample[mode]
            _draw_method_row(axes[row], label, lc, imgs, caps, scs,
                             sample["gt_set"], sample["adv_set"])

    ann_label = ann_stem.replace('_', ' ').title()
    title = f"CLIP vs ISO vs Manifold\nAnnotation: {ann_label}   σ = {sigma}"

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0],[0], color="#1a7a1a", lw=3, label="GT caption"),
        Line2D([0],[0], color="#c0392b", lw=3, label="Adversarial caption"),
        Line2D([0],[0], color="#1a5fa8", lw=3, label="Other retrieved"),
    ], loc="lower center", ncol=3, fontsize=7,
       bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.01)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── per-sample individual image saver (for paper figures) ────────────────────

_MODE_DIR = {
    "baseline": "baseline",
    "iso":      "iso",
    "manifold": "manifold",
}


def _save_individual_images(sample: Dict, sample_dir: Path) -> None:
    """Save original.png + per-method top-N images and captions.txt.

    Layout inside *sample_dir*::

        original.png
        baseline/top1.png … top5.png  captions.txt
        iso/     top1.png … top5.png  captions.txt
        manifold/top1.png … top5.png  captions.txt
    """
    sample_dir.mkdir(parents=True, exist_ok=True)

    # ── query / original image ────────────────────────────────────────────
    try:
        _load_pil(sample["image_path"]).save(sample_dir / "original.png")
    except Exception as exc:
        print(f"    [warn] original.png not saved: {exc}")

    gt_lines = [f"gt_caption: {cap}" for cap in sorted(sample["gt_set"])]
    (sample_dir / "captions.txt").write_text(
        "\n".join(gt_lines) + "\n", encoding="utf-8"
    )

    gt_set  = sample["gt_set"]
    adv_set = sample["adv_set"]

    for mode_key, dir_name in _MODE_DIR.items():
        caps, scs, imgs = sample[mode_key]
        mode_dir = sample_dir / dir_name
        mode_dir.mkdir(parents=True, exist_ok=True)

        caption_lines = []
        for rank, (cap, sc, img_path) in enumerate(zip(caps, scs, imgs), 1):
            # Save retrieved image (or a labeled placeholder for adv-only captions)
            if img_path is not None:
                try:
                    _load_pil(img_path).save(mode_dir / f"top{rank}.png")
                except Exception as exc:
                    print(f"    [warn] {dir_name}/top{rank}.png not saved: {exc}")
            else:
                # Adversarial caption — no associated image; save a white placeholder
                from PIL import Image, ImageDraw
                ph = Image.new("RGB", (256, 256), color=(255, 255, 255))
                draw = ImageDraw.Draw(ph)
                draw.text((10, 110), f"Top-{rank}\n(adv)",
                          fill=(180, 50, 50))
                ph.save(mode_dir / f"top{rank}.png")

            # Caption annotation
            tag = " [GT]" if cap in gt_set else (" [ADV]" if cap in adv_set else "")
            caption_lines.append(f"top{rank} (score={sc:.4f}){tag}: {cap}")

        (mode_dir / "captions.txt").write_text(
            "\n".join(caption_lines) + "\n", encoding="utf-8"
        )

    print(f"    Individual images → {sample_dir}")


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
                save_path  = out_base / ann_stem / sigma_str / f"sample_{s_num:02d}.png"
                sample_dir = out_base / ann_stem / sigma_str / f"sample_{s_num:02d}"
                _save_comparison_batch([sample], save_path, sigma, ann_stem)
                _save_individual_images(sample, sample_dir)


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
