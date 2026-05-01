from __future__ import annotations

from pathlib import Path

import numpy as np

from src.indexing.types import NeighborIndex
from src.indexing.utils import to_numpy_array


def build_annoy_index(
    vectors,
    metric: str = "euclidean",
    index_path: str | None = None,
    n_trees: int = 20,
) -> NeighborIndex:
    from annoy import AnnoyIndex

    data = to_numpy_array(vectors)
    if data.ndim != 2:
        raise ValueError("Index vectors must have shape [N, D].")

    ann = AnnoyIndex(data.shape[1], metric)
    for idx, row in enumerate(data):
        ann.add_item(idx, row.tolist())
    ann.build(n_trees)
    if index_path is not None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        ann.save(str(index_path))
    return NeighborIndex(backend="annoy", dim=data.shape[1], metric=metric, vectors=data, index=ann)


def load_annoy_index(dim: int, index_path: str, metric: str = "euclidean") -> NeighborIndex:
    from annoy import AnnoyIndex

    ann = AnnoyIndex(dim, metric)
    ann.load(str(index_path))
    return NeighborIndex(backend="annoy", dim=dim, metric=metric, index=ann)


def save_annoy_index(index: NeighborIndex, index_path: str) -> None:
    if index.backend != "annoy":
        raise ValueError("save_annoy_index expects an annoy-backed NeighborIndex.")

    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.index.save(str(path))


def query_annoy_index(
    index: NeighborIndex,
    k: int,
    vector=None,
    item_index: int | None = None,
) -> np.ndarray:
    if vector is None and item_index is None:
        raise ValueError("Provide either vector or item_index for neighbor lookup.")
    if item_index is not None:
        return np.asarray(index.index.get_nns_by_item(item_index, k), dtype=np.int64)
    return np.asarray(index.index.get_nns_by_vector(to_numpy_array(vector).reshape(-1).tolist(), k), dtype=np.int64)
