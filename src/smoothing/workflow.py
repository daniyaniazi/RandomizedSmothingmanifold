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
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class NeighborIndex:
    backend: str
    dim: int
    metric: str
    vectors: np.ndarray | None = None
    index: Any = None


@dataclass
class LocalPCA:
    mean: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray


def to_numpy_array(x: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32, copy=False)
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr


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


def build_index(
    vectors: torch.Tensor | np.ndarray,
    backend: str = "torch",
    metric: str = "euclidean",
    index_path: str | None = None,
    n_trees: int = 20,
) -> NeighborIndex:
    data = to_numpy_array(vectors)
    if data.ndim != 2:
        raise ValueError("Index vectors must have shape [N, D].")

    if backend == "torch":
        return NeighborIndex(backend=backend, dim=data.shape[1], metric=metric, vectors=data)

    if backend == "annoy":
        from annoy import AnnoyIndex

        ann = AnnoyIndex(data.shape[1], metric)
        for idx, row in enumerate(data):
            ann.add_item(idx, row.tolist())
        ann.build(n_trees)
        if index_path is not None:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            ann.save(str(index_path))
        return NeighborIndex(backend=backend, dim=data.shape[1], metric=metric, vectors=data, index=ann)

    if backend == "faiss":
        import faiss

        if metric != "euclidean":
            raise ValueError("FAISS helper currently supports euclidean metric only.")
        faiss_index = faiss.IndexFlatL2(data.shape[1])
        faiss_index.add(data)
        return NeighborIndex(backend=backend, dim=data.shape[1], metric=metric, vectors=data, index=faiss_index)

    raise ValueError(f"Unsupported index backend: {backend}")


def load_index(dim: int, index_path: str, backend: str = "annoy", metric: str = "euclidean") -> NeighborIndex:
    if backend != "annoy":
        raise ValueError("load_index currently supports annoy-backed indices.")
    from annoy import AnnoyIndex

    ann = AnnoyIndex(dim, metric)
    ann.load(str(index_path))
    return NeighborIndex(backend=backend, dim=dim, metric=metric, index=ann)


def query_index(
    index: NeighborIndex,
    k: int,
    vector: torch.Tensor | np.ndarray | list[float] | None = None,
    item_index: int | None = None,
) -> np.ndarray:
    if vector is None and item_index is None:
        raise ValueError("Provide either vector or item_index for neighbor lookup.")

    if index.backend == "torch":
        if index.vectors is None:
            raise ValueError("Torch index requires stored vectors.")
        if item_index is not None:
            anchor = index.vectors[item_index]
        else:
            anchor = to_numpy_array(vector)
            anchor = anchor.reshape(-1)
        dists = np.linalg.norm(index.vectors - anchor[None, :], axis=1)
        k_eff = max(1, min(k, index.vectors.shape[0]))
        return np.argsort(dists)[:k_eff]

    if index.backend == "annoy":
        if item_index is not None:
            return np.asarray(index.index.get_nns_by_item(item_index, k), dtype=np.int64)
        return np.asarray(index.index.get_nns_by_vector(to_numpy_array(vector).reshape(-1).tolist(), k), dtype=np.int64)

    if index.backend == "faiss":
        anchor = index.vectors[item_index] if item_index is not None else to_numpy_array(vector).reshape(1, -1)
        anchor = np.asarray(anchor, dtype=np.float32).reshape(1, -1)
        _, ids = index.index.search(anchor, k)
        return ids[0].astype(np.int64)

    raise ValueError(f"Unsupported index backend: {index.backend}")


def neighbor_vectors(
    index: NeighborIndex,
    k: int,
    vector: torch.Tensor | np.ndarray | list[float] | None = None,
    item_index: int | None = None,
) -> np.ndarray:
    ids = query_index(index=index, k=k, vector=vector, item_index=item_index)
    if index.vectors is None:
        if index.backend != "annoy":
            raise ValueError("Neighbor vectors require stored vectors for this backend.")
        rows = [index.index.get_item_vector(int(idx)) for idx in ids]
        return np.asarray(rows, dtype=np.float32)
    return index.vectors[ids]


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
    x = to_numpy_array(vector).reshape(-1)
    centered = x - pca.mean
    return (centered @ pca.evecs) / np.sqrt(pca.evals)


def unwhiten(white_vector: torch.Tensor | np.ndarray | list[float], pca: LocalPCA) -> np.ndarray:
    z = to_numpy_array(white_vector).reshape(-1)
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