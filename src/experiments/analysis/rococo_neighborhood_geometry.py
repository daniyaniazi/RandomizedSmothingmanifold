"""RoCOCO CLIP embedding neighbourhood geometry analysis.

Same pattern as celeba_neighborhood_geometry.py but for 512-dim CLIP unit-sphere embeddings.

For each anchor shows:
  - kNN neighbours projected onto PC1/PC2
  - Isotropic circle (r = sigma)
  - Manifold ellipse (axes = alpha * sqrt(lambda))
  - n_mc smoothed CLIP embeddings (viz only — actual eval uses n_samples=1)
  - Eigenvalue spectrum
  - Alpha/noise scale: alpha = sigma when scale_noise=false, else sigma / sqrt(lambda_max)

Uses the CLIP embedding Annoy index (not pixel-space index).
Also checks whether saved CLIP embeddings are already on the unit sphere.

Usage:
    python -m src.experiments.analysis.rococo_neighborhood_geometry \\
        --config src/configs/experiments/rococo_clip_manifold.yaml \\
        --sigmas 0.05 0.1 0.2 \\
        --n-anchors 5 \\
        --knn-k 500 \\
        --n-mc 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.rococo_clip_schema import load_rococo_config
from src.smoothing.pca import fit_local_pca
from src.smoothing.manifold import ManifoldSmoother
from src.smoothing.isotropic import IsotropicSmoother
from src.experiments.eval.rococo_clip_eval import _normalize

# ── colour palette (matches celeba script) ───────────────────────────────────
VC = {
    'knn':        '#aaaaaa',
    'mc':         '#e07b39',
    'iso_circle': '#2166ac',
    'mani_ellip': '#d6604d',
    'anchor':     '#f5c518',
}


# ── helpers ───────────────────────────────────────────────────────────────────


def _draw_geometry(
    anchor_vec: np.ndarray,      # (512,) normalised
    neighbours: np.ndarray,      # (k, 512) normalised
    smoother: ManifoldSmoother,
    sigma: float,
    n_mc: int,
    save_path: Path,
    anchor_idx: int,
) -> dict:
    """3-panel geometry figure + eigenvalue panel — same layout as celeba."""
    from src.smoothing.pca import fit_local_pca

    # Fit PCA on neighbours
    pca = fit_local_pca(neighbours)
    ev        = np.asarray(pca.evals, dtype=np.float64)
    Vt        = pca.evecs.T                          # (n_comp, 512)
    ev_norm   = ev / float(ev.max())
    lambda_max = float(ev[0])
    scale_noise = bool(getattr(smoother, "_scale_noise", True))
    alpha = sigma / np.sqrt(max(lambda_max, 1e-12)) if scale_noise else sigma

    # Project onto PC1/PC2
    centered   = neighbours - pca.mean
    anchor_c   = anchor_vec  - pca.mean
    neigh_2d   = centered @ Vt[:2].T      # (k, 2)
    anchor_2d  = anchor_c @ Vt[:2].T      # (2,)

    # Ellipse semi-axes
    a1 = float(alpha * np.sqrt(ev[0]))
    a2 = float(alpha * np.sqrt(ev[1]))

    # MC noise samples — only if requested (n_mc=0 skips, cleaner figures)
    mc_2d = np.empty((0, 2), dtype=np.float32)
    if n_mc > 0:
        from src.smoothing.manifold import CachedPCA
        cached = CachedPCA(anchor=anchor_vec.astype(np.float32), pca=pca,
                           neighbors=neighbours.astype(np.float32))
        mc_samples = np.array([smoother.sample_from_cached(cached) for _ in range(n_mc)])
        mc_c  = mc_samples - pca.mean
        mc_2d = mc_c @ Vt[:2].T

    pc1_std   = float(np.std(neigh_2d[:, 0]))
    pc1_range = float(np.ptp(neigh_2d[:, 0]))
    zoom_mid   = 0.10 * pc1_range
    zoom_tight = a1

    zoom_panels = [
        (None,       f"(a) Full cloud\nPC1 std={pc1_std:.4f}"),
        (zoom_mid,   f"(b) Mid-zoom  ±{zoom_mid:.4f}"),
        (zoom_tight, f"(c) Tight  a₁={a1:.4f}  a₂={a2:.4f}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), facecolor="white")

    for ax, (zr, title) in zip(axes[:3], zoom_panels):
        if zr is None:
            mask    = np.ones(len(neigh_2d), dtype=bool)
            mask_mc = np.ones(len(mc_2d), dtype=bool)
        else:
            mask    = (np.abs(neigh_2d[:, 0] - anchor_2d[0]) <= zr) & \
                      (np.abs(neigh_2d[:, 1] - anchor_2d[1]) <= zr)
            mask_mc = (np.abs(mc_2d[:, 0]   - anchor_2d[0]) <= zr) & \
                      (np.abs(mc_2d[:, 1]    - anchor_2d[1]) <= zr)

        ax.scatter(neigh_2d[mask, 0], neigh_2d[mask, 1],
                   s=6, alpha=0.30, color=VC['knn'], linewidths=0, zorder=1,
                   label=f"kNN ({mask.sum()})")
        if mask_mc.any():
            ax.scatter(mc_2d[mask_mc, 0], mc_2d[mask_mc, 1],
                       s=8, alpha=0.50, color=VC['mc'], linewidths=0, zorder=3,
                       label=f"MC samples (n={n_mc})")

        # Isotropic circle  r = sigma
        ax.add_patch(plt.Circle(
            (anchor_2d[0], anchor_2d[1]), sigma,
            fill=False, edgecolor=VC['iso_circle'], linewidth=2.0,
            linestyle=(0, (4, 2)), alpha=0.95, zorder=4,
            label=f"ISO circle  r=σ={sigma}",
        ))
        # Manifold ellipse
        ax.add_patch(Ellipse(
            (anchor_2d[0], anchor_2d[1]),
            width=2*a1, height=2*a2,
            fill=False, edgecolor=VC['mani_ellip'], linewidth=2.0,
            linestyle="solid", alpha=0.95, zorder=4,
            label=f"Mani ellipse  a₁={a1:.4f}",
        ))
        ax.scatter(anchor_2d[0], anchor_2d[1], s=180, marker="*",
                   c=VC['anchor'], edgecolors="white", linewidths=0.8, zorder=5,
                   label="Anchor")

        if zr is not None:
            ax.set_xlim(anchor_2d[0]-zr, anchor_2d[0]+zr)
            ax.set_ylim(anchor_2d[1]-zr, anchor_2d[1]+zr)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(f"PC1  (std={pc1_std:.4f})", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.grid(alpha=0.25)

    # Panel 4: eigenvalue spectrum
    ax = axes[3]
    n_show = min(50, len(ev))
    ax.plot(range(1, n_show+1), ev[:n_show], 'o-', ms=3, lw=1.5, color='tab:purple',
            label='raw eigenvalue')
    ax.plot(range(1, n_show+1), ev_norm[:n_show], 's--', ms=3, lw=1.5, color='tab:green',
            label='normalized (÷λ_max)')
    ax.axhline(lambda_max, color='#c0392b', lw=1, ls=':', label=f'λ_max={lambda_max:.5f}')
    ax.set_xlabel('PC index')
    ax.set_ylabel('Eigenvalue')
    alpha_label = "σ/√λ_max" if scale_noise else "σ"
    ax.set_title(f'Eigenvalue spectrum\nλ_max={lambda_max:.5f}  α={alpha_label}={alpha:.3f}',
                 fontsize=8.5)
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        f"CLIP Embedding Manifold Geometry  —  anchor={anchor_idx}  σ={sigma}\n"
        f"λ_max={lambda_max:.5f}   √λ_max={np.sqrt(lambda_max):.5f}   "
        f"scale_noise={scale_noise}   α={alpha_label}={alpha:.4f}   (α/σ = {alpha/sigma:.1f}×)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return {
        'anchor_idx': anchor_idx,
        'lambda_max': lambda_max,
        'alpha':      alpha,
        'scale_noise': scale_noise,
        'a1': a1, 'a2': a2,
        'pc1_std': pc1_std,
    }


def _draw_amplification_curve(lambda_maxs: list, sigmas: list, save_path: Path, scale_noise: bool) -> None:
    """Panel 3 from notebook — alpha amplification for all anchors × all sigmas."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sig_arr = np.linspace(0.005, 1.0, 200)

    mean_lmax = float(np.mean(lambda_maxs))
    min_lmax  = float(np.min(lambda_maxs))
    max_lmax  = float(np.max(lambda_maxs))

    if scale_noise:
        ax.fill_between(sig_arr,
                        sig_arr / np.sqrt(max_lmax),
                        sig_arr / np.sqrt(min_lmax),
                        alpha=0.15, color='#c0392b', label='alpha range (min/max λ_max)')
        ax.plot(sig_arr, sig_arr / np.sqrt(mean_lmax), lw=2, color='#c0392b',
                label=f'alpha = σ/√λ_max  (mean λ_max={mean_lmax:.5f})')
    else:
        ax.plot(sig_arr, sig_arr, lw=2, color='#c0392b',
                label='alpha = σ  (unscaled manifold setting)')
    ax.plot(sig_arr, sig_arr, lw=1.5, ls='--', color='#636363',
            label='alpha = σ  (isotropic / no amplification)')
    ax.axhline(1.0, color='orange', lw=1, ls=':', label='alpha = 1  (unit noise)')

    for s in sigmas:
        a = s / np.sqrt(mean_lmax) if scale_noise else s
        ax.annotate(f'σ={s}\nα={a:.1f}',
                    xy=(s, a), xytext=(s+0.02, a+0.5),
                    fontsize=7, color='#c0392b',
                    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))

    ax.set_xlabel('σ  (config value)')
    ax.set_ylabel('Effective noise std  (alpha)')
    title = 'Noise amplification in CLIP embedding space'
    subtitle = 'alpha = σ / √λ_max' if scale_noise else 'unscaled setting: alpha = σ'
    ax.set_title(f'{title}\n{subtitle}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--sigmas",    type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    p.add_argument("--n-anchors", type=int,   default=5)
    p.add_argument("--knn-k",     type=int,   default=500)
    p.add_argument("--n-mc",      type=int,   default=50,
                   help="Number of smoothed samples to draw for visualization only "
                        "(each = one noisy CLIP embedding, shown as orange dots). "
                        "In actual eval only 1 sample is used per image.")
    p.add_argument("--output-dir",type=Path,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_rococo_config(args.config)
    cfg.smoothing.knn_k = args.knn_k

    out_dir = args.output_dir or (_ROOT / cfg.output_dir / "rococo" / "geometry")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    cache = torch.load(Path(cfg.embedding_cache_dir) / "image_embeddings.pt",
                       map_location="cpu")
    embs_raw  = cache["embeddings"].numpy().astype(np.float32)   # (N, 512)
    embs_norm = embs_raw / np.linalg.norm(embs_raw, axis=1, keepdims=True)

    # Check unit sphere
    norms = np.linalg.norm(embs_raw, axis=1)
    print(f"Embedding L2 norms — mean={norms.mean():.4f}  "
          f"min={norms.min():.4f}  max={norms.max():.4f}")
    print(f"  → {'ON unit sphere (norm≈1)' if abs(norms.mean()-1)<0.01 else 'NOT on unit sphere — raw embeddings'}")

    # Load kNN index
    from src.experiments.indexing.rococo_clip_images import build_rococo_clip_index
    index_path = _ROOT / cfg.index_dir / "index.ann"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}\n"
            f"Run: python -m src.experiments.indexing.rococo_clip_images --config {args.config}"
        )
    clip_index = build_rococo_clip_index(cfg, rebuild=False)

    # Pick anchor indices evenly
    rng     = np.random.default_rng(42)
    anchors = rng.choice(len(embs_norm), args.n_anchors, replace=False).tolist()
    print(f"Anchors: {anchors}")

    all_stats = []
    for sigma in args.sigmas:
        smoother = ManifoldSmoother(
            sigma=sigma,
            index=clip_index,
            knn_k=args.knn_k,
            eps_eig=cfg.smoothing.eps_eig,
            scale_noise=getattr(cfg.smoothing, 'scale_noise', True),
        )
        for ai in anchors:
            anchor = embs_norm[ai]
            # Use Annoy index — same as eval pipeline
            nn_ids     = clip_index.index.get_nns_by_vector(
                anchor.tolist(), args.knn_k + 1, include_distances=False)
            nn_ids     = [n for n in nn_ids if n != ai][:args.knn_k]
            neighbours = embs_norm[nn_ids]

            # Fit PCA and print key numbers
            pca_info = fit_local_pca(neighbours)
            ev       = pca_info.evals
            lmax     = float(ev[0])
            scale_noise = bool(getattr(cfg.smoothing, 'scale_noise', True))
            alpha = sigma / np.sqrt(max(lmax, 1e-12)) if scale_noise else sigma
            alpha_label = "σ/√λ_max" if scale_noise else "σ"
            print(f"\nAnchor={ai}  σ={sigma}")
            print(f"  Top-10 eigenvalues (raw): " +
                  "  ".join(f"{v:.5f}" for v in ev[:10]))
            print(f"  λ_max={lmax:.5f}  √λ_max={np.sqrt(lmax):.5f}")
            print(f"  scale_noise={scale_noise}")
            print(f"  alpha = {alpha_label} = {alpha:.4f}  ({alpha/sigma:.1f}× sigma)")
            print(f"  → noise added in whitened space has std={alpha:.4f}  "
                  f"(vs σ={sigma} you set)")

            save_path = out_dir / f"geometry_anchor{ai}_sigma{sigma:.2f}.png".replace('.','_').replace('_png','.png')
            stats = _draw_geometry(anchor, neighbours, smoother,
                                   sigma=sigma, n_mc=args.n_mc,
                                   save_path=save_path, anchor_idx=ai)
            all_stats.append({**stats, 'sigma': sigma})

    # Amplification curve across all sigmas
    lambda_maxs = [s['lambda_max'] for s in all_stats]
    _draw_amplification_curve(lambda_maxs, args.sigmas,
                               out_dir / "alpha_amplification.png",
                               bool(getattr(cfg.smoothing, 'scale_noise', True)))

    # Summary
    import json
    summary_path = out_dir / "geometry_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2))
    print(f"\nSummary: {summary_path}")
    print(f"\nlambda_max across all anchors:")
    lmaxs = [s['lambda_max'] for s in all_stats if s['sigma'] == args.sigmas[0]]
    print(f"  mean={np.mean(lmaxs):.5f}  min={np.min(lmaxs):.5f}  max={np.max(lmaxs):.5f}")
    scale_noise = bool(getattr(cfg.smoothing, 'scale_noise', True))
    for sigma in args.sigmas:
        alphas = [s['alpha'] for s in all_stats if s['sigma'] == sigma]
        suffix = "amplification" if scale_noise else "sigma"
        print(f"  σ={sigma}  →  alpha mean={np.mean(alphas):.3f}  "
              f"({np.mean(alphas)/sigma:.1f}× {suffix})")


if __name__ == "__main__":
    main()
