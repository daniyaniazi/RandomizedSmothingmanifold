"""Generic indexing facade.

Backend implementations live in dedicated modules:
- src.indexing.torch_indexing
- src.indexing.annoy_indexing
- src.indexing.faiss_indexing

This file preserves one simple public API for the rest of the codebase.
"""

from __future__ import annotations

import numpy as np
import torch

from .annoy_indexing import build_annoy_index, load_annoy_index, query_annoy_index
from .faiss_indexing import build_faiss_index, load_faiss_index, query_faiss_index
from .torch_indexing.backend import build_torch_index, query_torch_index
from .types import NeighborIndex
from .utils import to_numpy_array


def build_index(
    vectors: torch.Tensor | np.ndarray,
    backend: str = "torch",
    metric: str = "euclidean",
    index_path: str | None = None,
    n_trees: int = 20,
) -> NeighborIndex:
    if backend == "torch":
        return build_torch_index(vectors)

    if backend == "annoy":
        return build_annoy_index(vectors, metric=metric, index_path=index_path, n_trees=n_trees)

    if backend == "faiss":
        return build_faiss_index(vectors, metric=metric)

    raise ValueError(f"Unsupported index backend: {backend}")


def load_index(dim: int, index_path: str, backend: str = "annoy", metric: str = "euclidean") -> NeighborIndex:
    if backend == "annoy":
        return load_annoy_index(dim=dim, index_path=index_path, metric=metric)

    if backend == "faiss":
        return load_faiss_index(dim=dim, index_path=index_path, metric=metric)

    raise ValueError("load_index currently supports annoy- or faiss-backed indices.")


def query_index(
    index: NeighborIndex,
    k: int,
    vector: torch.Tensor | np.ndarray | list[float] | None = None,
    item_index: int | None = None,
) -> np.ndarray:
    if vector is None and item_index is None:
        raise ValueError("Provide either vector or item_index for neighbor lookup.")

    if index.backend == "torch":
        return query_torch_index(index=index, k=k, vector=vector, item_index=item_index)

    if index.backend == "annoy":
        return query_annoy_index(index=index, k=k, vector=vector, item_index=item_index)

    if index.backend == "faiss":
        return query_faiss_index(index=index, k=k, vector=vector, item_index=item_index)

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
