from __future__ import annotations

import numpy as np

from src.indexing.types import NeighborIndex
from src.indexing.utils import to_numpy_array


def build_torch_index(vectors) -> NeighborIndex:
    data = to_numpy_array(vectors)
    if data.ndim != 2:
        raise ValueError("Index vectors must have shape [N, D].")
    return NeighborIndex(backend="torch", dim=data.shape[1], metric="euclidean", vectors=data)


def query_torch_index(
    index: NeighborIndex,
    k: int,
    vector=None,
    item_index: int | None = None,
) -> np.ndarray:
    if index.vectors is None:
        raise ValueError("Torch index requires stored vectors.")
    if vector is None and item_index is None:
        raise ValueError("Provide either vector or item_index for neighbor lookup.")

    if item_index is not None:
        anchor = index.vectors[item_index]
    else:
        anchor = to_numpy_array(vector).reshape(-1)
    dists = np.linalg.norm(index.vectors - anchor[None, :], axis=1)
    k_eff = max(1, min(k, index.vectors.shape[0]))
    return np.argsort(dists)[:k_eff]
