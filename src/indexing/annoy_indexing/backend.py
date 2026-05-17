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


def build_annoy_index_streaming(
    dataloader,
    dim: int,
    metric: str = "euclidean",
    index_path: str | None = None,
    n_trees: int = 20,
    max_samples: int | None = None,
    image_key: str = "image",
    log_fn=None,
) -> NeighborIndex:
    """Build Annoy index by streaming vectors from a dataloader without
    materializing all vectors in memory. Ideal for high-dimensional pixel
    vectors that would OOM in batch mode.

    Args:
        dataloader: DataLoader yielding batches (dict or tuple).
        dim: Dimensionality of each vector (e.g. 3*H*W).
        metric: Distance metric for Annoy.
        index_path: Path to save the .ann file. None = don't save.
        n_trees: Number of Annoy trees.
        max_samples: Stop after this many vectors (None = all).
        image_key: Key to access images in batch dict.
        log_fn: Optional logging function, e.g. print.

    Returns:
        NeighborIndex (without vectors array to save memory).
    """
    from annoy import AnnoyIndex

    _log = log_fn or (lambda msg: None)

    metric_map = {"euclidean": "euclidean", "cosine": "angular", "manhattan": "manhattan"}
    ann = AnnoyIndex(dim, metric_map.get(metric, "euclidean"))

    item_idx = 0
    for batch in dataloader:
        if isinstance(batch, dict):
            images = batch[image_key]
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        flat = images.reshape(images.shape[0], -1)
        if hasattr(flat, "numpy"):
            flat = flat.numpy()

        for vec in flat:
            ann.add_item(item_idx, vec)
            item_idx += 1
            if max_samples and item_idx >= max_samples:
                break
        if max_samples and item_idx >= max_samples:
            break
        if item_idx % 10000 == 0:
            _log(f"  Added {item_idx} vectors...")

    _log(f"Added {item_idx} vectors. Building {n_trees} trees...")
    ann.build(n_trees)

    if index_path is not None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        ann.save(str(index_path))
        file_size_mb = Path(index_path).stat().st_size / (1024 * 1024)
        _log(f"Index saved: {index_path} ({file_size_mb:.1f} MB)")

    return NeighborIndex(backend="annoy", dim=dim, metric=metric, index=ann)


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
