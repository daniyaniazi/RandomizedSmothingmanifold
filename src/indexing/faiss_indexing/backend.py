from __future__ import annotations

from pathlib import Path

import numpy as np

from src.indexing.types import NeighborIndex
from src.indexing.utils import to_numpy_array


def build_faiss_index(vectors, metric: str = "euclidean") -> NeighborIndex:
    import faiss

    data = to_numpy_array(vectors)
    if data.ndim != 2:
        raise ValueError("Index vectors must have shape [N, D].")
    if metric != "euclidean":
        raise ValueError("FAISS helper currently supports euclidean metric only.")

    faiss_index = faiss.IndexFlatL2(data.shape[1])
    faiss_index.add(data)
    return NeighborIndex(backend="faiss", dim=data.shape[1], metric=metric, vectors=data, index=faiss_index)


def query_faiss_index(
    index: NeighborIndex,
    k: int,
    vector=None,
    item_index: int | None = None,
) -> np.ndarray:
    if vector is None and item_index is None:
        raise ValueError("Provide either vector or item_index for neighbor lookup.")

    anchor = index.vectors[item_index] if item_index is not None else to_numpy_array(vector).reshape(1, -1)
    anchor = np.asarray(anchor, dtype=np.float32).reshape(1, -1)
    _, ids = index.index.search(anchor, k)
    return ids[0].astype(np.int64)


def save_faiss_index(index: NeighborIndex, index_path: str) -> None:
    import faiss

    if index.backend != "faiss":
        raise ValueError("save_faiss_index expects a faiss-backed NeighborIndex.")

    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index.index, str(path))


def load_faiss_index(
    dim: int,
    index_path: str,
    metric: str = "euclidean",
    vectors=None,
) -> NeighborIndex:
    import faiss

    if metric != "euclidean":
        raise ValueError("FAISS helper currently supports euclidean metric only.")

    faiss_index = faiss.read_index(str(index_path))
    if faiss_index.d != dim:
        raise ValueError(f"FAISS index dim mismatch: expected {dim}, got {faiss_index.d}")

    data = None if vectors is None else to_numpy_array(vectors)
    return NeighborIndex(backend="faiss", dim=dim, metric=metric, vectors=data, index=faiss_index)
