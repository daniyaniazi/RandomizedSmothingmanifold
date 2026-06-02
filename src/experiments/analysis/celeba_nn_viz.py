"""Pixel + Latent 5-NN visualization for OOD-feasible CelebA attributes.

Saves one PNG per attribute showing:
  Row 0: pixel-space nearest neighbours (blue border)
  Row 1: latent-space nearest neighbours (orange border)

Usage:
    python -m src.experiments.analysis.celeba_nn_viz \
        --output-dir output/analysis/nn_viz \
        --n-attrs 8 \
        --k 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed on the cluster
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.certify_celeba_io import load_certify_config
from src.indexing.annoy_indexing.backend import load_annoy_index
from src.models.VAE import ConvVAE, load_checkpoint as load_vae_checkpoint


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_celeba_attrs(attr_path: Path):
    import pandas as pd
    lines = [l.strip() for l in attr_path.read_text().splitlines() if l.strip()]
    attr_names = lines[1].split()
    rows = []
    for line in lines[2:]:
        parts = line.split()
        rows.append([parts[0]] + [int(v) for v in parts[1:]])
    df = __import__("pandas").DataFrame(rows, columns=["filename"] + attr_names)
    for col in attr_names:
        df[col] = (df[col] == 1).astype(int)
    return df


def _assign_splits(df, train_ratio, val_ratio, seed):
    shuffled = df["filename"].tolist()
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val   = int(len(shuffled) * val_ratio)
    split_map = (
        {f: "train" for f in shuffled[:n_train]}
        | {f: "val"   for f in shuffled[n_train:n_train + n_val]}
        | {f: "test"  for f in shuffled[n_train + n_val:]}
    )
    df = df.copy()
    df["split"] = df["filename"].map(split_map)
    return df, shuffled[:n_train]


def _show_img(ax, img_dir, fname, title, border_color="white"):
    try:
        ax.imshow(Image.open(img_dir / fname).convert("RGB"))
    except FileNotFoundError:
        ax.text(0.5, 0.5, "not found", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
    ax.set_title(title, fontsize=7, pad=2)
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color); sp.set_linewidth(3); sp.set_visible(True)


def _query_nn(ann_index, fnames, vec, k, anchor_fname):
    raw_ids = ann_index.index.get_nns_by_vector(vec.tolist(), k + 1, include_distances=False)
    return [fnames[i] for i in raw_ids if fnames[i] != anchor_fname][:k]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Save pixel+latent NN visualizations")
    parser.add_argument("--pixel-cfg",  default="src/configs/experiments/certify_celeba_pixel.yaml")
    parser.add_argument("--latent-cfg", default="src/configs/experiments/certify_celeba_latent_128.yaml")
    parser.add_argument("--metric",     default="euclidean", choices=["euclidean", "angular"])
    parser.add_argument("--output-dir", default="output/analysis/nn_viz")
    parser.add_argument("--n-attrs",    type=int, default=40,  help="Top N OOD-feasible attrs")
    parser.add_argument("--k",          type=int, default=5,  help="Number of neighbours")
    parser.add_argument("--anchor-seed",type=int, default=42)
    parser.add_argument("--train-ratio",type=float, default=0.8)
    parser.add_argument("--val-ratio",  type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=73)
    args = parser.parse_args()

    out_dir = _ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pcfg = load_certify_config(str(_ROOT / args.pixel_cfg))
    lcfg = load_certify_config(str(_ROOT / args.latent_cfg))

    celeba_root = Path(pcfg.dataset.root_dir)
    image_dir   = celeba_root / pcfg.dataset.image_dir
    attr_file   = celeba_root / pcfg.dataset.annotation_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load attrs & splits ───────────────────────────────────────────────────
    print("Loading attributes...")
    df, tr_fnames = _assign_splits(
        _load_celeba_attrs(attr_file),
        args.train_ratio, args.val_ratio, args.split_seed,
    )
    df_test  = df[df["split"] == "test"].copy()
    attr_cols = [c for c in df.columns if c not in ("filename", "split")]
    print(f"  total={len(df):,}  train={len(tr_fnames):,}  test={len(df_test):,}")

    # ── OOD feasible attrs ────────────────────────────────────────────────────
    feasible = []
    for attr in attr_cols:
        sub = df_test[df_test[attr] == 1]
        if sub["Smiling"].sum() >= 50 and (len(sub) - sub["Smiling"].sum()) >= 50:
            feasible.append((attr, len(sub[sub[attr] == 1])))
    feasible.sort(key=lambda x: -x[1])
    top_attrs = [a for a, _ in feasible[:args.n_attrs]]
    print(f"OOD-feasible: {len(feasible)} attrs, using top {len(top_attrs)}: {top_attrs}")

    # ── load pixel index ──────────────────────────────────────────────────────
    img_sz  = pcfg.model.input_size
    pix_dim = 3 * img_sz * img_sz
    pix_idx_path = (
        _ROOT / pcfg.output.output_dir / "smile_classification"
        / pcfg.dataset.name.lower().replace("-","").replace("_","")
        / "index" / "pixel" / "annoy" / args.metric / "index.ann"
    )
    print(f"Pixel index: {pix_idx_path}")
    if not pix_idx_path.exists():
        print(f"  SKIP — pixel index not found. Run: ./server_scripts/submit_build_indexes.sh celeba-pixel")
        pix_ann = None
    else:
        pix_ann = load_annoy_index(dim=pix_dim, index_path=str(pix_idx_path), metric=args.metric)
        print(f"  {pix_ann.index.get_n_items():,} vectors, dim={pix_dim}")

    # ── load latent index ─────────────────────────────────────────────────────
    lat_dim = lcfg.vae.latent_dim
    lat_idx_path = (
        _ROOT / lcfg.output.output_dir / "smile_classification"
        / lcfg.dataset.name.lower().replace("-","").replace("_","")
        / "index" / "latent" / "annoy" / args.metric / "index.ann"
    )
    print(f"Latent index: {lat_idx_path}")
    if not lat_idx_path.exists():
        print(f"  SKIP — latent index not found. Run: ./server_scripts/submit_build_indexes.sh celeba-latent")
        lat_ann = None
    else:
        lat_ann = load_annoy_index(dim=lat_dim, index_path=str(lat_idx_path), metric=args.metric)
        print(f"  {lat_ann.index.get_n_items():,} vectors, dim={lat_dim}")

    # ── load VAE ──────────────────────────────────────────────────────────────
    vae = ConvVAE(
        image_size=lcfg.vae.image_size,
        latent_dim=lcfg.vae.latent_dim,
        in_channels=lcfg.vae.in_channels,
    ).to(device)
    load_vae_checkpoint(vae, _ROOT / lcfg.vae.checkpoint_path, device=device)
    vae.eval()
    print(f"VAE loaded: image_size={lcfg.vae.image_size}, latent_dim={lat_dim}")

    tf_pixel  = T.Compose([T.Resize((img_sz, img_sz)), T.ToTensor()])
    tf_latent = T.Compose([
        T.Resize((lcfg.vae.image_size, lcfg.vae.image_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def load_pix_vec(fname):
        return tf_pixel(Image.open(image_dir / fname).convert("RGB")).numpy().astype("float32").flatten()

    def load_lat_vec(fname):
        t = tf_latent(Image.open(image_dir / fname).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(t)
        return mu.squeeze(0).cpu().numpy().astype("float32")

    fname_to_smile = dict(zip(df["filename"], df["Smiling"]))

    def attr_val(fname, attr):
        rows = df.loc[df["filename"] == fname, attr].values
        return int(rows[0]) if len(rows) else "?"

    # ── generate figures ──────────────────────────────────────────────────────
    for attr in top_attrs:
        print(f"Processing: {attr} ...", flush=True)
        anchor_row   = df_test[df_test[attr] == 1].sample(1, random_state=args.anchor_seed).iloc[0]
        anchor_fname = anchor_row["filename"]
        anchor_smile = int(anchor_row["Smiling"])

        pix_vec = load_pix_vec(anchor_fname) if pix_ann else None
        lat_vec = load_lat_vec(anchor_fname) if lat_ann else None

        pix_nns = _query_nn(pix_ann, tr_fnames, pix_vec, args.k, anchor_fname) if pix_ann else []
        lat_nns = _query_nn(lat_ann, tr_fnames, lat_vec, args.k, anchor_fname) if lat_ann else []

        n_rows = sum([pix_ann is not None, lat_ann is not None]) + 1  # anchor row shared
        n_cols = 1 + args.k
        fig, axes = plt.subplots(max(2, 1 + (lat_ann is not None) + (pix_ann is not None)),
                                 n_cols, figsize=(3 * n_cols, 7))
        axes = np.atleast_2d(axes)
        row = 0
        axes[row, 0].set_ylabel("Anchor", fontsize=9, fontweight="bold", labelpad=6)
        for c in range(n_cols):
            _show_img(axes[row, c], image_dir, anchor_fname,
                      f"ANCHOR\n{attr}=1\n{'smile' if anchor_smile else 'no-smile'}",
                      border_color="gold") if c == 0 else axes[row, c].axis("off")

        if pix_ann is not None:
            row += 1
            axes[row, 0].set_ylabel("Pixel NNs", fontsize=9, fontweight="bold", labelpad=6)
            for col, nn in enumerate(pix_nns, start=1):
                _show_img(axes[row, col], image_dir, nn,
                          f"NN-{col}\n{attr}={attr_val(nn, attr)}\n"
                          f"{'smile' if fname_to_smile.get(nn,0)==1 else 'no-smile'}",
                          border_color="#4c78a8")

        if lat_ann is not None:
            row += 1
            axes[row, 0].set_ylabel("Latent NNs", fontsize=9, fontweight="bold", labelpad=6)
            for col, nn in enumerate(lat_nns, start=1):
                _show_img(axes[row, col], image_dir, nn,
                          f"NN-{col}\n{attr}={attr_val(nn, attr)}\n"
                          f"{'smile' if fname_to_smile.get(nn,0)==1 else 'no-smile'}",
                          border_color="#e07b54")

        pix_info = f"Pixel: {pix_ann.index.get_n_items():,} @ {img_sz}px" if pix_ann else "Pixel: N/A"
        lat_info = f"Latent: {lat_ann.index.get_n_items():,}, dim={lat_dim}" if lat_ann else "Latent: N/A"
        fig.suptitle(
            f"Attribute: {attr}  |  Anchor (gold)  •  Pixel NNs (blue)  •  Latent NNs (orange)\n"
            f"{pix_info}  |  {lat_info}",
            fontsize=10, y=1.02,
        )
        plt.tight_layout()
        save_path = out_dir / f"nn_viz_{attr.lower()}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print(f"\nDone. {len(top_attrs)} figures saved to {out_dir}")


if __name__ == "__main__":
    main()
