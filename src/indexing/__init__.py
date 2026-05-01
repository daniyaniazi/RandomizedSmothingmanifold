"""Index-building helpers for smoothing and retrieval workflows."""

from .annoy_indexing import build_annoy_index, load_annoy_index, query_annoy_index, save_annoy_index
from .base import NeighborIndex, build_index, load_index, neighbor_vectors, query_index, to_numpy_array
from .faiss_indexing import build_faiss_index, load_faiss_index, query_faiss_index, save_faiss_index
from .ner_token_index import (
    TokenIndexArtifacts,
    build_or_load_token_index,
    build_token_neighbor_index,
    extract_token_vectors,
    hidden_state_for_layer,
    load_token_index_artifacts,
    save_token_index_artifacts,
)
from .torch_indexing.backend import build_torch_index, query_torch_index

__all__ = [
    "NeighborIndex",
    "build_index",
    "load_index",
    "query_index",
    "neighbor_vectors",
    "to_numpy_array",
    "build_torch_index",
    "query_torch_index",
    "build_annoy_index",
    "save_annoy_index",
    "load_annoy_index",
    "query_annoy_index",
    "build_faiss_index",
    "save_faiss_index",
    "load_faiss_index",
    "query_faiss_index",
    "TokenIndexArtifacts",
    "build_or_load_token_index",
    "build_token_neighbor_index",
    "extract_token_vectors",
    "hidden_state_for_layer",
    "load_token_index_artifacts",
    "save_token_index_artifacts",
]

