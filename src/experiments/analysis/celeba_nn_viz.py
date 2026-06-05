"""Pixel + Latent NN visualization across all available metrics (euclidean + angular).

Layout per attribute (one PNG):
  Row 0 : Anchor  | NN-1 | NN-2 | ... | NN-k
  Row 1 : Pixel   euclidean NNs
  Row 2 : Pixel   angular   NNs          (if index exists)
  Row 3 : Latent  euclidean NNs          (if index exists)
  Row 4 : Latent  angular   NNs          (if index exists)

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.certify_celeba_io import load_certify_config
from src.indexing.annoy_indexing.backend import load_annoy_index
from src.models.VAE import ConvVAE, load_checkpoint as load_vae_checkpoint


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_celeba_attrs(attr_path: Path):
    lines = [l.strip() for l in attr_path.read_text().splitlines() if l.strip()]
    attr_names = lines[1].split()
    rows = []
    for line in lines[2:]:
        parts = line.split()
        rows.append([parts[0]] + [int(v) for v in parts[1:]])
    import pandas as pd
    df = pd.DataFrame(rows, columns=["filename"] + attr_names)
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
                transform=ax.transAxes, fontsize=7, color="red")
    ax.set_title(title, fontsize=7, pad=2)
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color)
        sp.set_linewidth(3)
        sp.set_visible(True)


def _hide_ax(ax):
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_visible(False)


def _query_nn(ann_index, fnames, vec, k, anchor_fname):
    raw_ids = ann_index.index.get_nns_by_vector(vec.tolist(), k + 10, include_distances=False)
    return [fnames[i] for i in raw_ids if fnames[i] != anchor_fname][:k]


def _try_load_index(base_dir, dataset_tag, space, metric, dim):
    path = base_dir / "smile_classification" / dataset_tag / "index" / space / "annoy" / metric / "index.ann"
    if not path.exists():
        print(f"  SKIP — {space}/{metric} index not found: {path}")
        return None, path
    idx = load_annoy_index(dim=dim, index_path=str(path), metric=metric)
    print(f"  Loaded {space}/{metric}: {idx.index.get_n_items():,} vectors @ {path}")
    return idx, path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NN visualization across all spaces and metrics")
    parser.add_argument("--pixel-cfg",   default="src/configs/experiments/certify_celeba_pixel.yaml")
    parser.add_argument("--latent-cfg",  default="src/configs/experiments/certify_celeba_latent_128.yaml")
    parser.add_argument("--output-dir",  default="output/analysis/nn_viz")
    parser.add_argument("--n-attrs",     type=int,   default=40)
    parser.add_argument("--k",           type=int,   default=5)
    parser.add_argument("--anchor-seed", type=int,   default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    parser.add_argument("--split-seed",  type=int,   default=73)
    args = parser.parse_args()

    out_dir = _ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pcfg = load_certify_config(str(_ROOT / args.pixel_cfg))
    lcfg = load_certify_config(str(_ROOT / args.latent_cfg))

    celeba_root = Path(pcfg.dataset.root_dir)
    image_dir   = celeba_root / pcfg.dataset.image_dir
    attr_file   = celeba_root / pcfg.dataset.annotation_file
    dataset_tag = pcfg.dataset.name.lower().replace("-", "").replace("_", "")
    base_dir    = _ROOT / pcfg.output.output_dir

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_sz  = pcfg.model.input_size
    pix_dim = 3 * img_sz * img_sz
    lat_dim = lcfg.vae.latent_dim

    print(f"Device: {device}")

    # ── load attrs & splits ───────────────────────────────────────────────────
    print("Loading attributes...")
    df, tr_fnames = _assign_splits(
        _load_celeba_attrs(attr_file),
        args.train_ratio, args.val_ratio, args.split_seed,
    )
    df_test   = df[df["split"] == "test"].copy()
    attr_cols = [c for c in df.columns if c not in ("filename", "split")]
    print(f"  total={len(df):,}  train={len(tr_fnames):,}  test={len(df_test):,}")

    # ── OOD-feasible attrs ────────────────────────────────────────────────────
    feasible = []
    for attr in attr_cols:
        sub = df_test[df_test[attr] == 1]
        if sub["Smiling"].sum() >= 50 and (len(sub) - sub["Smiling"].sum()) >= 50:
            feasible.append((attr, len(sub)))
    feasible.sort(key=lambda x: -x[1])
    top_attrs = [a for a, _ in feasible[:args.n_attrs]]
    print(f"OOD-feasible: {len(feasible)} attrs, using top {len(top_attrs)}")

    # ── load all available indexes ────────────────────────────────────────────
    METRICS = ["euclidean", "angular"]
    indexes = {}  # (space, metric) -> annoy index or None
    for metric in METRICS:
        indexes[("pixel",  metric)], _ = _try_load_index(base_dir, dataset_tag, "pixel",  metric, pix_dim)
        indexes[("latent", metric)], _ = _try_load_index(base_dir, dataset_tag, "latent", metric, lat_dim)

    # rows in display order — only include rows where the index exists
    ROW_DEFS = [
        ("pixel",  "euclidean", "#4c78a8", "Pixel\neuclidean"),
        ("pixel",  "angular",   "#1f77b4", "Pixel\nangular"),
        ("latent", "euclidean", "#e07b54", "Latent\neuclidean"),
        ("latent", "angular",   "#d62728", "Latent\nangular"),
    ]
    active_rows = [(sp, mt, col, lbl) for sp, mt, col, lbl in ROW_DEFS if indexes[(sp, mt)] is not None]

    if not active_rows:
        print("No indexes found — nothing to visualize.")
        return

    # ── load VAE (needed for latent rows) ─────────────────────────────────────
    vae = None
    if any(sp == "latent" for sp, *_ in active_rows):
        vae = ConvVAE(
            image_size=lcfg.vae.image_size,
            latent_dim=lat_dim,
            in_channels=lcfg.vae.in_channels,
        ).to(device)
        load_vae_checkpoint(vae, _ROOT / lcfg.vae.checkpoint_path, device=device)
        vae.eval()
        print(f"VAE loaded: image_size={lcfg.vae.image_size}, latent_dim={lat_dim}")

    # ── transforms ────────────────────────────────────────────────────────────
    tf_pixel  = T.Compose([T.Resize((img_sz, img_sz)), T.ToTensor()])
    tf_latent = T.Compose([T.Resize((lcfg.vae.image_size, lcfg.vae.image_size)), T.ToTensor()])

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
    n_cols  = 1 + args.k          # anchor + k neighbours
    n_rows  = 1 + len(active_rows) # anchor row + one per active index
    col_w   = 2.5
    row_h   = 2.8

    for attr in top_attrs:
        print(f"Processing: {attr} ...", flush=True)

        anchor_row   = df_test[df_test[attr] == 1].sample(1, random_state=args.anchor_seed).iloc[0]
        anchor_fname = anchor_row["filename"]
        anchor_smile = int(anchor_row["Smiling"])

        pix_vec = load_pix_vec(anchor_fname)
        lat_vec = load_lat_vec(anchor_fname) if vae is not None else None

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(col_w * n_cols, row_h * n_rows),
            gridspec_kw={"wspace": 0.05, "hspace": 0.45},
        )
        axes = np.atleast_2d(axes)

        # hide all cells by default — paint only the ones we use
        for ax in axes.flat:
            _hide_ax(ax)

        # ── Row 0: anchor (col 0) + empty cols 1..k (kept hidden) ────────────
        _show_img(axes[0, 0], image_dir, anchor_fname,
                  f"ANCHOR\n{attr}=1\n{'smile' if anchor_smile else 'no-smile'}",
                  border_color="gold")
        # row label
        axes[0, 0].set_ylabel("Anchor", fontsize=8, fontweight="bold", labelpad=4)

        # ── NN rows ───────────────────────────────────────────────────────────
        for r, (space, metric, color, label) in enumerate(active_rows, start=1):
            idx   = indexes[(space, metric)]
            vec   = pix_vec if space == "pixel" else lat_vec
            nns   = _query_nn(idx, tr_fnames, vec, args.k, anchor_fname)

            axes[r, 0].set_ylabel(label, fontsize=8, fontweight="bold", labelpad=4)

            # col 0: row label cell — show first NN image there too
            for col_i, nn_fname in enumerate(nns):
                ax = axes[r, col_i + 1] if col_i + 1 < n_cols else None
                if ax is None:
                    continue
                smile_tag = "smile" if fname_to_smile.get(nn_fname, 0) == 1 else "no-smile"
                _show_img(ax, image_dir, nn_fname,
                          f"NN-{col_i+1}\n{attr}={attr_val(nn_fname, attr)}\n{smile_tag}",
                          border_color=color)

            # fill col 0 of NN rows with the row label (no image, just axis label)
            _hide_ax(axes[r, 0])
            axes[r, 0].set_ylabel(label, fontsize=8, fontweight="bold", labelpad=4)

        # ── title ─────────────────────────────────────────────────────────────
        pix_info   = f"Pixel: {indexes[('pixel','euclidean')].index.get_n_items():,} @ {img_sz}px" \
                     if indexes.get(("pixel","euclidean")) else "Pixel: N/A"
        lat_info   = f"Latent: dim={lat_dim}" if vae else "Latent: N/A"
        fig.suptitle(
            f"Attribute: {attr}  |  Anchor (gold)\n"
            f"{pix_info}  |  {lat_info}",
            fontsize=10, y=1.01,
        )

        save_path = out_dir / f"nn_viz_{attr.lower()}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print(f"\nDone. {len(top_attrs)} figures saved to {out_dir}")


if __name__ == "__main__":
    main()
