"""
Latent-manifold smoothing utilities.

Covers the full pipeline that maps Jonas' pixel-space method to latent space:
  1. encode_dataset_mu     – encode every sample to z = mu(x)
  2. build_or_load_ann     – Annoy index in latent space
  3. local_latent_pca      – local covariance via kNN in z-space
  4. sampling helpers      – manifold-aligned and isotropic noise
  5. whitening + cert      – whitening transform and Cohen-style certified radius
  6. plot_comparison       – side-by-side original vs. semantic vs. isotropic noise
"""

import json
from pathlib import Path
from typing import List, Tuple

import annoy
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import denormalize, tensor_to_hwc, DEFAULT_MEAN, DEFAULT_STD
from .model import ConvVAE


# ---------------------------------------------------------------------------
# 1. Encode full dataset to latent means
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_dataset_mu(
    model: ConvVAE,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Encode every sample in *dataset* to its latent mean mu(x).

    Returns z_bank of shape (N, latent_dim) as float32 numpy array.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    z_all = []

    for x, _ in tqdm(loader, desc="Encode dataset → latent means"):
        mu, _ = model.encode(x.to(device))
        z_all.append(mu.cpu().numpy())

    return np.concatenate(z_all, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Annoy index in latent space
# ---------------------------------------------------------------------------

def build_or_load_ann(
    z_bank: np.ndarray,
    ann_path: str | Path,
    map_path: str | Path,
    n_trees: int = 50,
) -> Tuple[annoy.AnnoyIndex, dict]:
    """
    Build a new Annoy index or load an existing one.

    Returns (ann_index, id_map).
    """
    latent_dim = z_bank.shape[1]
    index = annoy.AnnoyIndex(latent_dim, "euclidean")

    ann_path = Path(ann_path)
    map_path = Path(map_path)

    if ann_path.exists() and map_path.exists():
        index.load(str(ann_path))
        with open(map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        print(f"Loaded latent ANN index ← {ann_path}")
        return index, id_map

    id_map = {}
    for i in tqdm(range(len(z_bank)), desc="Add vectors to ANN"):
        index.add_item(i, z_bank[i])
        id_map[str(i)] = i

    index.build(n_trees)
    index.save(str(ann_path))
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    print(f"Built latent ANN index → {ann_path}")
    return index, id_map


# ---------------------------------------------------------------------------
# 3. Local PCA in latent space (Jonas method, but in z not x)
# ---------------------------------------------------------------------------

def local_latent_pca(
    z_bank: np.ndarray,
    nn_indices: List[int],
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the local covariance around a set of latent neighbours.

    Returns (mu_local, eigvals, eigvecs) sorted in descending eigenvalue order.
    Eigenvalues are clipped to eps to avoid numerical issues.
    """
    z_local = z_bank[nn_indices]
    mu = z_local.mean(axis=0)
    centered = z_local - mu

    cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    desc = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[desc], eps)
    eigvecs = eigvecs[:, desc]

    return mu, eigvals, eigvecs


# ---------------------------------------------------------------------------
# 4. Sampling helpers
# ---------------------------------------------------------------------------

def manifold_noise_sample(
    z0: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Sample z' = z0 + U * sqrt(Lambda) * eta, where eta ~ N(0, alpha^2 * I).

    This produces noise aligned with the local semantic manifold directions.
    """
    eta = np.random.normal(scale=alpha, size=z0.shape[0]).astype(np.float32)
    delta = eigvecs @ (np.sqrt(eigvals) * eta)
    return z0 + delta


def isotropic_noise_sample(z0: np.ndarray, alpha: float) -> np.ndarray:
    """Baseline: standard isotropic Gaussian noise in latent space."""
    return z0 + np.random.normal(scale=alpha, size=z0.shape[0]).astype(np.float32)


def combined_covariance(
    eigvals_knn: np.ndarray,
    logvar_x: np.ndarray,
    blend_lambda: float = 0.2,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Option C blended covariance (see thesis notes):
        Sigma_z = Sigma_knn + lambda * diag(sigma_vae^2)

    Returns blended eigenvalues (same basis as kNN eigvecs).
    """
    vae_var = np.exp(logvar_x)
    return np.maximum(eigvals_knn + blend_lambda * vae_var, eps)


# ---------------------------------------------------------------------------
# 5. Whitening + certification (Jonas' Sec. 2.4, adapted to latent space)
# ---------------------------------------------------------------------------

def whitening_transform(eigvals: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """
    T = Lambda^{-1/2} U^T   (local whitening in latent space).

    After applying T, noise is isotropic → standard RS guarantees apply.
    """
    return (np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)


def certified_radius_whitened(alpha: float, pA: float, pB: float) -> float:
    """
    Cohen-style certified radius in whitened latent coordinates:
        r = alpha/2 * (Phi^{-1}(pA) - Phi^{-1}(pB))

    The certificate in original latent space is the ellipsoid:
        { z : ||T(z - z0)||_2 <= r }
    """
    return 0.5 * alpha * (norm.ppf(pA) - norm.ppf(pB))


# ---------------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _decode_to_hwc(
    model: ConvVAE,
    z_vec: np.ndarray,
    device: torch.device,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD,
) -> np.ndarray:
    z_t = torch.tensor(z_vec[None, :], dtype=torch.float32, device=device)
    x_hat = model.decode(z_t).squeeze(0)
    return tensor_to_hwc(denormalize(x_hat.cpu(), mean, std))


def plot_latent_comparison(
    model: ConvVAE,
    dataset: Dataset,
    z_bank: np.ndarray,
    ann: annoy.AnnoyIndex,
    sample_idx: int,
    device: torch.device,
    k_neighbors: int = 500,
    n_samples: int = 5,
    alpha: float = 0.8,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD,
):
    """
    5-row comparison figure:
      Row 0 – Original image (repeated across columns)
      Row 1 – Latent PCA reconstruction
      Row 2 – Semantic latent noise (manifold-aligned)
      Row 3 – Isotropic latent noise (baseline)
      Row 4 – k-nearest neighbours in latent space
    """
    # Original image
    x0, _ = dataset[sample_idx]
    x0_hwc = tensor_to_hwc(denormalize(x0, mean, std))

    # Local PCA
    z0 = z_bank[sample_idx]
    nn_indices = ann.get_nns_by_item(sample_idx, k_neighbors)
    mu_loc, eigvals, eigvecs = local_latent_pca(z_bank, nn_indices)

    # PCA reconstruction of z0
    z0_centered = z0 - mu_loc
    z0_recon = mu_loc + eigvecs @ (eigvecs.T @ z0_centered)
    x_recon = _decode_to_hwc(model, z0_recon, device, mean, std)

    fig, axes = plt.subplots(5, n_samples, figsize=(3 * n_samples, 15))
    row_labels = [
        "Original",
        "Latent PCA recon",
        "Semantic latent noise",
        "Isotropic latent noise",
        "Latent kNN",
    ]
    for ax in axes.flat:
        ax.axis("off")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_title(label, fontsize=9)

    for col in range(n_samples):
        axes[0, col].imshow(x0_hwc)
        axes[1, col].imshow(x_recon)
        axes[2, col].imshow(
            _decode_to_hwc(model, manifold_noise_sample(z0, eigvals, eigvecs, alpha), device, mean, std)
        )
        axes[3, col].imshow(
            _decode_to_hwc(model, isotropic_noise_sample(z0, alpha), device, mean, std)
        )
        nn_img, _ = dataset[nn_indices[col]]
        axes[4, col].imshow(tensor_to_hwc(denormalize(nn_img, mean, std)))

    plt.tight_layout()
    plt.show()
