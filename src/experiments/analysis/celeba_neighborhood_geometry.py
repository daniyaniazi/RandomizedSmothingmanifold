from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.certify.randomized import axis_lengths  # noqa: E402
from src.indexing.base import load_index  # noqa: E402


SIGMA_PALETTE = ["#2166ac", "#d6604d", "#4dac26", "#762a83", "#1b9e77"]
SIGMA_ALPHA = [0.95, 0.90, 0.85, 0.80, 0.75]
SIGMA_LW = 1.6
SCATTER_COLOR = "#d0d0d0"
ANCHOR_COLOR = "#f5c518"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CelebA neighborhood circle/ellipse plots outside the notebook.")
    parser.add_argument("--space", choices=["pixel", "latent"], default="pixel")
    parser.add_argument("--dataset", default="celeba")
    parser.add_argument("--index-root", type=Path, default=None, help="Optional base folder overriding output/smile_classification/<dataset>.")
    parser.add_argument("--manual-dim", type=int, default=None, help="Manual vector dimension if metadata is missing.")
    parser.add_argument("--knn-k", type=int, default=250)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--anchors", type=int, nargs="*", default=None)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--pca-backend", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default=None, help="Optional filename prefix for generated artifacts.")
    return parser.parse_args()


def infer_dimension(index_dir: Path, manual_dim: int | None) -> int:
    if manual_dim:
        return int(manual_dim)

    def find_positive_int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, (int, float)):
            int_value = int(value)
            return int_value if int_value > 0 else 0
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                int_value = int(stripped)
                return int_value if int_value > 0 else 0
            return 0
        if isinstance(value, dict):
            preferred_keys = ["embedding_dim", "dim", "vector_dim", "n_features", "feature_dim"]
            for key in preferred_keys:
                if key in value:
                    found = find_positive_int(value[key])
                    if found > 0:
                        return found
            for nested_value in value.values():
                found = find_positive_int(nested_value)
                if found > 0:
                    return found
        return 0

    candidate_dirs = [index_dir, index_dir.parent, index_dir.parent.parent]
    candidate_meta_names = ["metadata.json", "image_metadata.json", "index_metadata.json"]

    for directory in candidate_dirs:
        for meta_name in candidate_meta_names:
            meta_path = directory / meta_name
            if not meta_path.exists():
                continue
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            dim = find_positive_int(metadata)
            if dim > 0:
                return dim

    raise ValueError("Could not determine index dimension. Pass --manual-dim.")



def resolve_pca_runtime(pca_backend: str, gpu_device: str) -> tuple[str, torch.device | None]:
    if pca_backend == "cpu":
        return "cpu", None

    if pca_backend in {"auto", "gpu"} and torch.cuda.is_available():
        device = torch.device(gpu_device)
        return "gpu", device

    if pca_backend == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("`--pca-backend gpu` requested but CUDA is not available.")

    return "cpu", None


def pca_project_cpu(centered: np.ndarray, anchor_centered: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(centered)
    evals = np.maximum(np.asarray(pca.explained_variance_, dtype=np.float32), 1e-12)
    components = np.asarray(pca.components_.T, dtype=np.float32)
    anchor_proj = anchor_centered @ components
    neighbor_proj = centered @ components
    return evals, anchor_proj, neighbor_proj


def pca_project_gpu(centered: np.ndarray, anchor_centered: np.ndarray, n_components: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered_t = torch.from_numpy(centered).to(device=device, dtype=torch.float32)
    anchor_t = torch.from_numpy(anchor_centered).to(device=device, dtype=torch.float32)
    sample_count = centered_t.shape[0]
    q = int(min(n_components, sample_count, centered_t.shape[1]))

    _, singular_values, right_vecs = torch.pca_lowrank(centered_t, q=q, center=False)
    evals_t = (singular_values.square() / max(sample_count - 1, 1)).clamp(min=1e-12)
    anchor_proj_t = anchor_t @ right_vecs
    neighbor_proj_t = centered_t @ right_vecs

    evals = evals_t.detach().cpu().numpy().astype(np.float32)
    anchor_proj = anchor_proj_t.detach().cpu().numpy().astype(np.float32)
    neighbor_proj = neighbor_proj_t.detach().cpu().numpy().astype(np.float32)

    del centered_t, anchor_t, singular_values, right_vecs, evals_t, anchor_proj_t, neighbor_proj_t
    torch.cuda.empty_cache()
    return evals, anchor_proj, neighbor_proj


def compute_sample_data(
    annoy_idx,
    anchor_indices: list[int],
    knn_k: int,
    pca_components: int,
    pca_backend: str,
    gpu_device: str,
) -> tuple[dict[int, dict[str, np.ndarray | int]], str]:
    sample_data: dict[int, dict[str, np.ndarray | int]] = {}
    resolved_backend, device = resolve_pca_runtime(pca_backend, gpu_device)

    for anchor_idx in anchor_indices:
        anchor_vector = np.asarray(annoy_idx.index.get_item_vector(anchor_idx), dtype=np.float32)
        neighbor_ids = annoy_idx.index.get_nns_by_item(anchor_idx, knn_k + 1, include_distances=False)
        neighbor_ids = [item for item in neighbor_ids if item != anchor_idx][:knn_k]
        neighbor_vectors = np.asarray(
            [annoy_idx.index.get_item_vector(int(item)) for item in neighbor_ids],
            dtype=np.float32,
        )

        local_mean = neighbor_vectors.mean(axis=0, dtype=np.float32)
        centered = neighbor_vectors - local_mean
        anchor_centered = anchor_vector - local_mean

        n_components = int(min(pca_components, centered.shape[0], centered.shape[1]))
        if resolved_backend == "gpu" and device is not None:
            evals, anchor_proj, neighbor_proj = pca_project_gpu(centered, anchor_centered, n_components, device)
        else:
            evals, anchor_proj, neighbor_proj = pca_project_cpu(centered, anchor_centered, n_components)

        evals_norm = evals / evals.max()

        sample_data[anchor_idx] = {
            "anchor_2d": np.asarray(anchor_proj[:2], dtype=np.float32),
            "nb_2d": np.asarray(neighbor_proj[:, :2], dtype=np.float32),
            "evals_norm_2d": np.asarray(evals_norm[:2], dtype=np.float32),
            "evals_topk": np.asarray(evals_norm, dtype=np.float32),
            "n_neighbors": int(len(neighbor_ids)),
        }

        del neighbor_vectors, centered, neighbor_proj, anchor_vector, local_mean, anchor_centered
        gc.collect()

    return sample_data, resolved_backend



def make_geometry_figure_single(sample_id: int, data: dict[str, np.ndarray | int], sigmas: list[float], space: str, output_path: Path) -> None:
    """Save a 3-panel geometry figure for one sample.

    Panel A – full cloud   : matplotlib auto-fits to all neighbors.
                             Circle will be a dot if sigma << pc1_std — physically correct.
    Panel B – mid-zoom     : ±10 % of the PC1 cloud range.
                             Shows circle in context of nearest neighbors.
    Panel C – tight zoom   : ±3 × max_sigma.
                             Circle and ellipse always fill the frame.
    """
    anchor_2d = np.asarray(data["anchor_2d"], dtype=np.float64)
    neighbors_2d = np.asarray(data["nb_2d"], dtype=np.float64)
    evals_norm_2d = np.asarray(data["evals_norm_2d"], dtype=np.float64)

    pc1_std = float(np.std(neighbors_2d[:, 0]))
    pc2_std = float(np.std(neighbors_2d[:, 1]))
    pc1_range = float(np.max(neighbors_2d[:, 0]) - np.min(neighbors_2d[:, 0]))

    max_sigma = float(max(sigmas)) if sigmas else 1.0
    zoom_mid   = 0.10 * pc1_range
    zoom_tight = 3.0 * max_sigma

    zoom_configs = [
        (None,       f"[A] Full cloud\nPC1 std={pc1_std:.2f},  max σ={max_sigma}\nσ/std={max_sigma/(pc1_std+1e-12):.5f}"),
        (zoom_mid,   f"[B] Mid-zoom  ±{zoom_mid:.2f}  (10% of cloud)\nPC1 std={pc1_std:.2f},  PC2 std={pc2_std:.2f}"),
        (zoom_tight, f"[C] Tight zoom  ±{zoom_tight:.4f}  (=3×max σ)\nShapes always fill the frame"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), facecolor="white")
    fig.suptitle(
        f"CelebA {space.title()} Space — Sample {sample_id}  (kNN={data['n_neighbors']})\n"
        f"PC1 std={pc1_std:.2f}  |  PC2 std={pc2_std:.2f}  |  sigmas={sigmas}",
        fontsize=11, fontweight="bold", y=1.03,
    )

    for ax, (zr, panel_title) in zip(axes, zoom_configs):
        ax.scatter(neighbors_2d[:, 0], neighbors_2d[:, 1], c=SCATTER_COLOR, s=6, alpha=0.30, linewidths=0, zorder=1)

        for sigma_index, sigma in enumerate(sigmas):
            color = SIGMA_PALETTE[sigma_index % len(SIGMA_PALETTE)]
            alpha = SIGMA_ALPHA[sigma_index % len(SIGMA_ALPHA)]
            axis_lengths_2d = axis_lengths(sigma, evals_norm_2d)
            if len(axis_lengths_2d) < 2:
                continue
            ax.add_patch(plt.Circle(
                (float(anchor_2d[0]), float(anchor_2d[1])), sigma,
                fill=False, edgecolor=color, linewidth=SIGMA_LW,
                linestyle=(0, (4, 2)), alpha=alpha, zorder=3,
            ))
            ax.add_patch(Ellipse(
                (float(anchor_2d[0]), float(anchor_2d[1])),
                width=float(2 * axis_lengths_2d[0]), height=float(2 * axis_lengths_2d[1]),
                fill=False, edgecolor=color, linewidth=SIGMA_LW,
                linestyle="solid", alpha=alpha, zorder=3,
            ))

        ax.scatter(float(anchor_2d[0]), float(anchor_2d[1]),
                   c=ANCHOR_COLOR, s=150, marker="*", edgecolors="#444444", linewidths=0.7, zorder=5)

        if zr is not None:
            ax.set_xlim(float(anchor_2d[0]) - zr, float(anchor_2d[0]) + zr)
            ax.set_ylim(float(anchor_2d[1]) - zr, float(anchor_2d[1]) + zr)
        # else: auto-fit to full cloud

        ax.set_aspect("equal")
        ax.set_title(panel_title, fontsize=8.5)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(alpha=0.3)

    # shared legend
    legend_handles = []
    for sigma_index, sigma in enumerate(sigmas):
        color = SIGMA_PALETTE[sigma_index % len(SIGMA_PALETTE)]
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=1.8, linestyle=(0, (4, 2)), label=f"σ={sigma} iso circle"))
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=1.8, linestyle="solid", label=f"σ={sigma} mani ellipse"))
    legend_handles.append(plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=ANCHOR_COLOR,
                                      markeredgecolor="#444", markersize=10, label="Anchor"))

    fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(sigmas) * 2 + 1, 6),
               fontsize=7.5, frameon=True, framealpha=0.95, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def make_eigenvalue_figure_single(sample_id: int, evals_topk: np.ndarray, space: str, output_path: Path) -> dict[str, float | int]:
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.8), facecolor="white")
    x = np.arange(1, len(evals_topk) + 1)
    ax.plot(x, evals_topk, "o-", linewidth=2, markersize=4, color="tab:purple")
    ax.set_title(f"Sample {sample_id}: top-{len(evals_topk)} λ̃")
    ax.set_xlabel("PCA axis")
    ax.set_ylabel("normalized eigenvalue")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)

    plt.suptitle(f"CelebA {space.title()} Space — Sample {sample_id} Eigen Spectrum", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    row: dict[str, float | int] = {"sample": int(sample_id)}
    for component_index, value in enumerate(evals_topk, start=1):
        row[f"lambda_norm_{component_index}"] = float(value)
    return row



def main() -> None:
    args = parse_args()

    base = args.index_root or (REPO_ROOT / "output" / "smile_classification" / args.dataset)
    pixel_index_dir = base / "index" / "pixel" / "annoy" / "euclidean"
    latent_index_dir = base / "index" / "latent" / "annoy" / "euclidean"
    index_dir = latent_index_dir if args.space == "latent" else pixel_index_dir
    index_path = index_dir / "index.ann"
    if not index_path.exists():
        raise FileNotFoundError(f"Annoy index not found: {index_path}")

    dim = infer_dimension(index_dir, args.manual_dim)
    annoy_idx = load_index(dim, index_path, backend="annoy")
    n_items = annoy_idx.index.get_n_items()

    if args.anchors:
        anchor_indices = [item for item in args.anchors if item < n_items]
    else:
        anchor_indices = list(range(min(args.n_samples, n_items)))
    if not anchor_indices:
        raise ValueError("No valid anchor indices available for plotting.")

    sample_data, resolved_backend = compute_sample_data(
        annoy_idx,
        anchor_indices,
        args.knn_k,
        args.pca_components,
        args.pca_backend,
        args.gpu_device,
    )

    output_dir = args.output_dir or (base / "analysis" / f"{args.space}_neighborhood_geometry")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"celeba_{args.space}_knn{args.knn_k}_n{len(anchor_indices)}"

    eigen_csv_path = output_dir / f"{prefix}_eigenvalues.csv"
    summary_json_path = output_dir / f"{prefix}_summary.json"

    saved_geometry_paths: list[str] = []
    saved_eigen_paths: list[str] = []
    rows: list[dict[str, float | int]] = []
    for sample_id, data in sample_data.items():
        geometry_path = output_dir / f"{prefix}_sample_{sample_id}_geometry.png"
        eigen_plot_path = output_dir / f"{prefix}_sample_{sample_id}_eigenvalues.png"
        make_geometry_figure_single(sample_id, data, args.sigmas, args.space, geometry_path)
        row = make_eigenvalue_figure_single(sample_id, np.asarray(data["evals_topk"], dtype=np.float32), args.space, eigen_plot_path)
        rows.append(row)
        saved_geometry_paths.append(str(geometry_path))
        saved_eigen_paths.append(str(eigen_plot_path))

    eigen_df = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    eigen_df.to_csv(eigen_csv_path, index=False)

    summary = {
        "space": args.space,
        "index_dir": str(index_dir),
        "anchors": anchor_indices,
        "knn_k": int(args.knn_k),
        "sigmas": [float(sigma) for sigma in args.sigmas],
        "pca_components_analysis": int(args.pca_components),
        "pca_backend_requested": args.pca_backend,
        "pca_backend_resolved": resolved_backend,
        "gpu_device": args.gpu_device,
        "manual_dim": int(dim),
        "outputs": {
            "geometry_plots": saved_geometry_paths,
            "eigen_plots": saved_eigen_paths,
            "eigen_csv": str(eigen_csv_path),
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Saved outputs:")
    for path in saved_geometry_paths:
        print(" -", path)
    for path in saved_eigen_paths:
        print(" -", path)
    print(" -", eigen_csv_path)
    print(" -", summary_json_path)
    print("PCA backend:", resolved_backend)


if __name__ == "__main__":
    main()
