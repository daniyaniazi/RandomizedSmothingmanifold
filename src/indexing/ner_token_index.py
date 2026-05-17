"""Token-level hidden-state index utilities for NER smoothing.

This module keeps indexing modular and separate from smoothing:
1. run the trained NER encoder over the train set
2. keep one vector per labeled token position (labels != -100)
3. store vectors + metadata
4. build a reusable neighbor index on top of those vectors
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .base import NeighborIndex, build_index, load_index
from .faiss_indexing.backend import save_faiss_index


@dataclass
class TokenIndexArtifacts:
    vectors: np.ndarray
    token_texts: list[str]
    label_ids: list[int]
    index: NeighborIndex


def hidden_state_for_layer(hidden_states: tuple[torch.Tensor, ...], layer_index: int | None) -> torch.Tensor:
    """Return hidden states for a 0-based transformer layer index."""
    if layer_index is None:
        return hidden_states[-1]

    tuple_index = layer_index + 1
    if tuple_index < 1 or tuple_index >= len(hidden_states):
        raise ValueError(f"Invalid layer_index={layer_index}; available layers: {len(hidden_states) - 1}")
    return hidden_states[tuple_index]


@torch.no_grad()
def extract_token_vectors(
    model,
    loader: DataLoader,
    device: torch.device,
    tokenizer_name: str,
    layer_index: int | None = None,
    max_batches: int | None = None,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Extract one vector per labeled token position from a dataloader."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.eval()

    vectors: list[np.ndarray] = []
    token_texts: list[str] = []
    label_ids: list[int] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )
        hidden = hidden_state_for_layer(out.hidden_states, layer_index)
        valid_mask = (batch["labels"] != -100) & batch["attention_mask"].bool()

        valid_vectors = hidden[valid_mask].detach().cpu().numpy().astype(np.float32)
        valid_ids = batch["input_ids"][valid_mask].detach().cpu().tolist()
        valid_labels = batch["labels"][valid_mask].detach().cpu().tolist()

        vectors.extend(valid_vectors)
        token_texts.extend(tokenizer.convert_ids_to_tokens(valid_ids))
        label_ids.extend(int(label_id) for label_id in valid_labels)

    if not vectors:
        raise RuntimeError("No token vectors extracted from dataloader.")

    return np.stack(vectors).astype(np.float32), token_texts, label_ids


def deduplicate_token_vectors(
    vectors: np.ndarray,
    token_texts: list[str],
    label_ids: list[int],
) -> tuple[np.ndarray, list[str], list[int]]:
    """Average embeddings for duplicate (token_text, label_id) pairs.

    BERT produces a different contextual embedding for every occurrence of a
    token in the corpus.  When the same token (e.g. "swiss") appears many times
    with the same label, KNN neighbors end up being redundant copies of that
    token.  This function collapses them into a single mean embedding per
    unique (token, label) key, giving the index better diversity.

    Args:
        vectors: (N, D) embedding matrix.
        token_texts: length-N list of sub-word strings.
        label_ids: length-N list of integer NER label ids.

    Returns:
        Deduplicated (vectors, token_texts, label_ids).
    """
    from collections import defaultdict

    # Group indices by (token, label)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (tok, lab) in enumerate(zip(token_texts, label_ids)):
        groups[(tok, lab)].append(i)

    dedup_vectors: list[np.ndarray] = []
    dedup_texts: list[str] = []
    dedup_labels: list[int] = []

    for (tok, lab), idxs in groups.items():
        mean_vec = vectors[idxs].mean(axis=0)
        dedup_vectors.append(mean_vec)
        dedup_texts.append(tok)
        dedup_labels.append(lab)

    n_before = len(token_texts)
    n_after = len(dedup_texts)
    if n_before != n_after:
        print(f"[dedup] {n_before} → {n_after} unique (token, label) pairs "
              f"({n_before - n_after} duplicates averaged)")

    return np.stack(dedup_vectors).astype(np.float32), dedup_texts, dedup_labels


def save_token_index_artifacts(
    out_dir: str | Path,
    vectors: np.ndarray,
    token_texts: list[str],
    label_ids: list[int],
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = out_dir / "token_vectors.npz"
    meta_path = out_dir / "token_metadata.json"

    np.savez(vectors_path, vectors=vectors)
    meta_path.write_text(json.dumps({"token_texts": token_texts, "label_ids": label_ids}, indent=2))
    return vectors_path, meta_path


def load_token_index_artifacts(out_dir: str | Path) -> tuple[np.ndarray, list[str], list[int]]:
    out_dir = Path(out_dir)
    vectors_path = out_dir / "token_vectors.npz"
    meta_path = out_dir / "token_metadata.json"

    payload = np.load(vectors_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text())
    return payload["vectors"].astype(np.float32), list(meta["token_texts"]), [int(x) for x in meta["label_ids"]]


def build_token_neighbor_index(
    vectors: np.ndarray,
    backend: str = "torch",
    metric: str = "euclidean",
    index_path: str | None = None,
    n_trees: int = 20,
) -> NeighborIndex:
    return build_index(vectors=vectors, backend=backend, metric=metric, index_path=index_path, n_trees=n_trees)


def build_or_load_token_index(
    out_dir: str | Path,
    model,
    loader: DataLoader,
    device: torch.device,
    tokenizer_name: str,
    layer_index: int | None = None,
    backend: str = "torch",
    metric: str = "euclidean",
    index_path: str | None = None,
    n_trees: int = 20,
    max_batches: int | None = None,
    rebuild: bool = False,
    deduplicate: bool = True,
) -> TokenIndexArtifacts:
    out_dir = Path(out_dir)
    vectors_path = out_dir / "token_vectors.npz"
    meta_path = out_dir / "token_metadata.json"

    if not rebuild and vectors_path.exists() and meta_path.exists():
        vectors, token_texts, label_ids = load_token_index_artifacts(out_dir)
    else:
        vectors, token_texts, label_ids = extract_token_vectors(
            model=model,
            loader=loader,
            device=device,
            tokenizer_name=tokenizer_name,
            layer_index=layer_index,
            max_batches=max_batches,
        )
        if deduplicate:
            vectors, token_texts, label_ids = deduplicate_token_vectors(
                vectors, token_texts, label_ids,
            )
        save_token_index_artifacts(out_dir, vectors, token_texts, label_ids)

    backend_norm = str(backend).strip().lower()
    metric_norm = str(metric).strip().lower()
    index_file = Path(index_path) if index_path is not None else None

    if (
        not rebuild
        and index_file is not None
        and index_file.exists()
        and backend_norm in {"annoy", "faiss"}
    ):
        index = load_index(
            dim=int(vectors.shape[1]),
            index_path=str(index_file),
            backend=backend_norm,
            metric=metric_norm,
        )
        index.vectors = vectors
    else:
        index = build_token_neighbor_index(
            vectors=vectors,
            backend=backend_norm,
            metric=metric_norm,
            index_path=index_path,
            n_trees=n_trees,
        )

        if backend_norm == "faiss" and index_file is not None:
            save_faiss_index(index=index, index_path=str(index_file))

    return TokenIndexArtifacts(vectors=vectors, token_texts=token_texts, label_ids=label_ids, index=index)
