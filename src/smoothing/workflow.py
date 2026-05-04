"""Reusable smoothing workflow.

This module keeps the smoothing path simple:
    1. build or load a kNN index
    2. fetch a local neighborhood
    3. fit local PCA
    4. whiten / unwhiten
    5. sample smoothed vectors

The same functions are intended to work for hidden states, latent vectors,
or image tensors flattened to vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.indexing.base import NeighborIndex, build_index, load_index, neighbor_vectors, query_index, to_numpy_array


@dataclass
class LocalPCA:
    mean: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray


def _flatten_with_mask(x: torch.Tensor, mask: torch.Tensor | None = None):
    if x.dim() == 2:
        flat = x
        restore = ("2d", x.shape)
        valid = torch.ones(flat.shape[0], device=x.device, dtype=torch.bool)
        return flat, valid, restore

    if x.dim() < 2:
        raise ValueError("Expected tensor with at least 2 dimensions.")

    leading = x.shape[:-1]
    dim = x.shape[-1]
    flat = x.reshape(-1, dim)
    if mask is None:
        valid = torch.ones(flat.shape[0], device=x.device, dtype=torch.bool)
    else:
        valid = mask.reshape(-1).bool()
    restore = ("nd", x.shape)
    return flat, valid, restore


def restore_like(flat_values: torch.Tensor, restore) -> torch.Tensor:
    kind, shape = restore
    if kind == "2d":
        return flat_values
    return flat_values.reshape(shape)


def fit_local_pca(neighbors: torch.Tensor | np.ndarray, eps_eig: float = 1e-6) -> LocalPCA:
    neigh = to_numpy_array(neighbors)
    if neigh.ndim != 2:
        raise ValueError("Neighbors must have shape [K, D].")

    mean = neigh.mean(axis=0, keepdims=True)
    centered = neigh - mean
    denom = max(centered.shape[0] - 1, 1)
    cov = (centered.T @ centered) / denom
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], a_min=eps_eig, a_max=None)
    evecs = evecs[:, order]
    return LocalPCA(mean=mean.reshape(-1), evals=evals.astype(np.float32), evecs=evecs.astype(np.float32))


def whiten(vector: torch.Tensor | np.ndarray | list[float], pca: LocalPCA) -> np.ndarray:
    """Whiten a vector using local PCA.
    
    - Project the ORIGINAL vector (not centered) into whitened space
    - X_whitened = X @ Vt.T / sqrt(ev)
    
    Note: The mean is NOT subtracted before whitening. The mean is only used
    during unwhitening to shift the result back to the neighborhood center.
    """
    x = to_numpy_array(vector).reshape(-1)
    # evecs has shape (D, K) where K = n_components, evals has shape (K,)
    # Project and scale: X @ V / sqrt(ev)
    return (x @ pca.evecs) / np.sqrt(pca.evals)


def unwhiten(white_vector: torch.Tensor | np.ndarray | list[float], pca: LocalPCA) -> np.ndarray:
    """Unwhiten a vector back to original space.

    - Scale by sqrt(ev), project back, then ADD the neighborhood mean
    - X_unwhitened = (X_white * sqrt(ev)) @ V.T + mean
    
    This shifts the result towards the neighborhood center.
    """
    z = to_numpy_array(white_vector).reshape(-1)
    # Scale, project back, and add mean
    return (z * np.sqrt(pca.evals)) @ pca.evecs.T + pca.mean


def sample_manifold_point(
    anchor: torch.Tensor | np.ndarray | list[float],
    neighbors: torch.Tensor | np.ndarray,
    sigma: float,
    eps_eig: float = 1e-6,
) -> np.ndarray:
    pca = fit_local_pca(neighbors, eps_eig=eps_eig)
    white_anchor = whiten(anchor, pca)
    white_noise = np.random.randn(white_anchor.shape[0]).astype(np.float32) * sigma
    return unwhiten(white_anchor + white_noise, pca)


def reconstruct_from_local_pca(
    anchor: torch.Tensor | np.ndarray | list[float],
    neighbors: torch.Tensor | np.ndarray,
    eps_eig: float = 1e-6,
) -> np.ndarray:
    pca = fit_local_pca(neighbors, eps_eig=eps_eig)
    return unwhiten(whiten(anchor, pca), pca)


def isotropic_noise_like(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.randn_like(x) * sigma


def smooth_tensor(
    x: torch.Tensor,
    sigma: float,
    mode: str = "manifold",
    knn_k: int = 64,
    eps_eig: float = 1e-6,
    attention_mask: torch.Tensor | None = None,
    index: NeighborIndex | None = None,
) -> torch.Tensor:
    if mode != "manifold":
        return x + isotropic_noise_like(x, sigma=sigma)

    flat, valid_mask, restore = _flatten_with_mask(x, attention_mask)
    valid_vectors = flat[valid_mask]
    if valid_vectors.numel() == 0:
        return x.clone()

    if index is None:
        source = valid_vectors.detach().cpu().numpy()
        index = build_index(source, backend="torch")

    smoothed = torch.zeros_like(valid_vectors)
    for i in range(valid_vectors.shape[0]):
        neighbors = neighbor_vectors(index, k=knn_k, vector=valid_vectors[i])
        smoothed_point = sample_manifold_point(
            anchor=valid_vectors[i],
            neighbors=neighbors,
            sigma=sigma,
            eps_eig=eps_eig,
        )
        smoothed[i] = torch.as_tensor(smoothed_point, device=valid_vectors.device, dtype=valid_vectors.dtype)

    flat_out = flat.clone()
    flat_out[valid_mask] = smoothed
    return restore_like(flat_out, restore)