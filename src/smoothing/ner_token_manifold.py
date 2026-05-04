"""Token-wise manifold smoothing helpers for NER.

Workflow:
1. get contextual hidden states for a sentence
2. for each valid token, retrieve train-token neighbours (ONCE per token)
3. fit local PCA (ONCE per token), then sample multiple noisy points
4. inject the delta back into the chosen transformer layer
5. classify the full sentence, then vote per token across noisy samples

OPTIMIZATION: PCA is computed once per token and reused for all samples.
This gives ~100x speedup for the PCA/neighbor lookup phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.certify.randomized import TokenCertificate, certify_token_from_counts
from src.indexing import hidden_state_for_layer
from .workflow import NeighborIndex, fit_local_pca, neighbor_vectors, reconstruct_from_local_pca, sample_manifold_point, LocalPCA, whiten, unwhiten


@dataclass
class TokenDebugRecord:
    token: str
    true_label: int
    neighbor_tokens: list[str]
    neighbor_labels: list[int]
    reconstruction_l2: float
    noisy_l2: float
    explained_variance: list[float]
    # Nearest train-token to the actual noisy sample (where the smoothed point landed)
    noisy_nearest_token: str = ""
    noisy_nearest_label: int = -1


@dataclass
class SmoothedBatchOutput:
    pred_ids: torch.Tensor
    vote_counts: np.ndarray
    certificates: list[list[TokenCertificate | None]]
    debug: list[list[TokenDebugRecord | None]] | None = None


@dataclass
class CachedTokenPCA:
    """Cached PCA info for a single token - reused across all samples."""
    anchor: np.ndarray
    pca: LocalPCA
    neighbors: np.ndarray
    batch_idx: int
    token_idx: int


def precompute_token_pcas(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    neighbor_index: NeighborIndex,
    knn_k: int,
    eps_eig: float,
) -> list[CachedTokenPCA]:
    """Precompute PCA for all valid tokens ONCE (not per sample)."""
    cached_pcas: list[CachedTokenPCA] = []
    batch_size, seq_len, _ = hidden.shape
    
    for batch_idx in range(batch_size):
        for token_idx in range(seq_len):
            if not bool(valid_mask[batch_idx, token_idx]):
                continue
            
            anchor = hidden[batch_idx, token_idx].detach().cpu().numpy().astype(np.float32)
            # Request k+1 neighbours and drop index 0 (self-match at distance 0)
            all_neighbors = neighbor_vectors(neighbor_index, k=knn_k + 1, vector=anchor)
            neighbors = all_neighbors[1:]
            pca = fit_local_pca(neighbors, eps_eig=eps_eig)
            
            cached_pcas.append(CachedTokenPCA(
                anchor=anchor,
                pca=pca,
                neighbors=neighbors,
                batch_idx=batch_idx,
                token_idx=token_idx,
            ))
    
    return cached_pcas


def sample_from_cached_pca(cached: CachedTokenPCA, sigma: float) -> np.ndarray:
    """Sample a single noisy point from cached PCA (fast - no KNN/PCA recomputation)."""
    # Whiten the anchor point
    whitened = whiten(cached.anchor, cached.pca)
    # Add isotropic noise in whitened space
    noise = np.random.randn(len(whitened)).astype(np.float32) * sigma
    whitened_noisy = whitened + noise
    # Unwhiten back to original space
    return unwhiten(whitened_noisy, cached.pca)


def smooth_token_tensor_with_cache(
    hidden: torch.Tensor,
    cached_pcas: list[CachedTokenPCA],
    sigma: float,
) -> torch.Tensor:
    """Apply one round of manifold smoothing using precomputed PCAs."""
    smoothed = hidden.clone()
    
    for cached in cached_pcas:
        noisy = sample_from_cached_pca(cached, sigma)
        smoothed[cached.batch_idx, cached.token_idx] = torch.as_tensor(
            noisy,
            device=hidden.device,
            dtype=hidden.dtype,
        )
    
    return smoothed


def smooth_token_tensor(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    neighbor_index: NeighborIndex,
    token_texts: list[str],
    label_ids: list[int],
    sigma: float,
    knn_k: int,
    eps_eig: float,
    tokenizer,
    collect_debug: bool = False,
) -> tuple[torch.Tensor, list[list[TokenDebugRecord | None]] | None]:
    """Smooth each valid token independently using a global train-token index.
    
    NOTE: This is the legacy function, kept for backward compatibility.
    For better performance, use precompute_token_pcas + smooth_token_tensor_with_cache.
    """
    smoothed = hidden.clone()
    debug_rows: list[list[TokenDebugRecord | None]] | None = [] if collect_debug else None

    batch_size, seq_len, _ = hidden.shape
    for batch_idx in range(batch_size):
        debug_row = [None] * seq_len if collect_debug else None
        for token_idx in range(seq_len):
            if not bool(valid_mask[batch_idx, token_idx]):
                continue

            anchor = hidden[batch_idx, token_idx].detach().cpu().numpy().astype(np.float32)
            # Request k+1 neighbours and drop index 0 (self-match at distance 0)
            all_neighbors = neighbor_vectors(neighbor_index, k=knn_k + 1, vector=anchor)
            neighbors = all_neighbors[1:]
            pca = fit_local_pca(neighbors, eps_eig=eps_eig)
            recon = reconstruct_from_local_pca(anchor, neighbors, eps_eig=eps_eig)
            noisy = sample_manifold_point(anchor, neighbors, sigma=sigma, eps_eig=eps_eig)

            smoothed[batch_idx, token_idx] = torch.as_tensor(
                noisy,
                device=hidden.device,
                dtype=hidden.dtype,
            )

            if collect_debug and debug_row is not None:
                neighbor_ids = np.asarray([], dtype=np.int64)
                try:
                    from .workflow import query_index
                    # k+1 to drop self-match at index 0
                    all_ids = query_index(neighbor_index, k=knn_k + 1, vector=anchor)
                    neighbor_ids = all_ids[1:]
                except Exception:
                    neighbor_ids = np.arange(min(knn_k, len(token_texts)), dtype=np.int64)

                # Query index with the noisy vector to find where the smoothed point landed
                noisy_nbr_id = -1
                noisy_nearest_tok = ""
                noisy_nearest_lbl = -1
                try:
                    from .workflow import query_index
                    noisy_ids = query_index(neighbor_index, k=1, vector=noisy)
                    if len(noisy_ids) > 0:
                        noisy_nbr_id = int(noisy_ids[0])
                        noisy_nearest_tok = token_texts[noisy_nbr_id]
                        noisy_nearest_lbl = int(label_ids[noisy_nbr_id])
                except Exception:
                    pass

                tok = tokenizer.convert_ids_to_tokens([int(input_ids[batch_idx, token_idx].item())])[0]
                debug_row[token_idx] = TokenDebugRecord(
                    token=tok,
                    true_label=int(labels[batch_idx, token_idx].item()),
                    neighbor_tokens=[token_texts[int(i)] for i in neighbor_ids[: min(len(neighbor_ids), 10)]],
                    neighbor_labels=[int(label_ids[int(i)]) for i in neighbor_ids[: min(len(neighbor_ids), 10)]],
                    reconstruction_l2=float(np.linalg.norm(recon - anchor)),
                    noisy_l2=float(np.linalg.norm(noisy - anchor)),
                    explained_variance=pca.evals[: min(10, len(pca.evals))].tolist(),
                    noisy_nearest_token=noisy_nearest_tok,
                    noisy_nearest_label=noisy_nearest_lbl,
                )
        if collect_debug and debug_rows is not None and debug_row is not None:
            debug_rows.append(debug_row)

    return smoothed, debug_rows


def _collect_debug_info(
    cached_pcas: list[CachedTokenPCA],
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    neighbor_index: NeighborIndex,
    token_texts: list[str],
    label_ids: list[int],
    sigma: float,
    tokenizer,
) -> list[list[TokenDebugRecord | None]]:
    """Collect debug info using cached PCAs."""
    batch_size, seq_len, _ = hidden.shape
    debug_rows: list[list[TokenDebugRecord | None]] = []
    
    for batch_idx in range(batch_size):
        debug_row: list[TokenDebugRecord | None] = [None] * seq_len
        debug_rows.append(debug_row)
    
    for cached in cached_pcas:
        batch_idx = cached.batch_idx
        token_idx = cached.token_idx
        anchor = cached.anchor
        pca = cached.pca
        
        # Sample one noisy point for debug
        noisy = sample_from_cached_pca(cached, sigma)
        recon = unwhiten(whiten(anchor, pca), pca)  # Reconstruct through PCA
        
        neighbor_ids = np.asarray([], dtype=np.int64)
        try:
            from .workflow import query_index
            all_ids = query_index(neighbor_index, k=len(cached.neighbors) + 1, vector=anchor)
            neighbor_ids = all_ids[1:]
        except Exception:
            neighbor_ids = np.arange(min(len(cached.neighbors), len(token_texts)), dtype=np.int64)
        
        # Query index with the noisy vector
        noisy_nearest_tok = ""
        noisy_nearest_lbl = -1
        try:
            from .workflow import query_index
            noisy_ids = query_index(neighbor_index, k=1, vector=noisy)
            if len(noisy_ids) > 0:
                noisy_nbr_id = int(noisy_ids[0])
                noisy_nearest_tok = token_texts[noisy_nbr_id]
                noisy_nearest_lbl = int(label_ids[noisy_nbr_id])
        except Exception:
            pass
        
        tok = tokenizer.convert_ids_to_tokens([int(input_ids[batch_idx, token_idx].item())])[0]
        debug_rows[batch_idx][token_idx] = TokenDebugRecord(
            token=tok,
            true_label=int(labels[batch_idx, token_idx].item()),
            neighbor_tokens=[token_texts[int(i)] for i in neighbor_ids[: min(len(neighbor_ids), 10)]],
            neighbor_labels=[int(label_ids[int(i)]) for i in neighbor_ids[: min(len(neighbor_ids), 10)]],
            reconstruction_l2=float(np.linalg.norm(recon - anchor)),
            noisy_l2=float(np.linalg.norm(noisy - anchor)),
            explained_variance=pca.evals[: min(10, len(pca.evals))].tolist(),
            noisy_nearest_token=noisy_nearest_tok,
            noisy_nearest_label=noisy_nearest_lbl,
        )
    
    return debug_rows


@torch.no_grad()
def sample_smoothed_token_predictions(
    model,
    batch: dict[str, torch.Tensor],
    neighbor_index: NeighborIndex,
    token_texts: list[str],
    label_ids: list[int],
    tokenizer,
    num_samples: int,
    sigma: float,
    knn_k: int,
    eps_eig: float,
    layer_index: int | None,
    alpha_conf: float,
    abstain_label: int,
    collect_debug: bool = False,
) -> SmoothedBatchOutput:
    """Sample smoothed predictions with OPTIMIZED PCA caching.
    
    PCA is computed ONCE per token, then reused for all num_samples iterations.
    This gives ~100x speedup for the PCA/neighbor lookup phase.
    """
    model.eval()

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    batch_size, seq_len = input_ids.shape
    num_labels = model.classifier.out_features

    valid_mask = (labels != -100) & attention_mask.bool()
    vote_counts = np.zeros((batch_size, seq_len, num_labels), dtype=np.int64)
    debug_output: list[list[TokenDebugRecord | None]] | None = None

    # Get clean hidden states
    clean_out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    clean_hidden = hidden_state_for_layer(clean_out.hidden_states, layer_index)

    target_layer = model.num_layers() - 1 if layer_index is None else layer_index

    # OPTIMIZATION: Precompute PCA for all tokens ONCE
    cached_pcas = precompute_token_pcas(
        hidden=clean_hidden,
        valid_mask=valid_mask,
        neighbor_index=neighbor_index,
        knn_k=knn_k,
        eps_eig=eps_eig,
    )
    
    # Collect debug info once (using first sample)
    if collect_debug:
        debug_output = _collect_debug_info(
            cached_pcas=cached_pcas,
            hidden=clean_hidden,
            input_ids=input_ids,
            labels=labels,
            valid_mask=valid_mask,
            neighbor_index=neighbor_index,
            token_texts=token_texts,
            label_ids=label_ids,
            sigma=sigma,
            tokenizer=tokenizer,
        )

    # Sample loop - now only samples noise, no PCA recomputation
    for sample_idx in range(num_samples):
        # Use cached PCAs to generate smoothed hidden states (FAST)
        smoothed_hidden = smooth_token_tensor_with_cache(
            hidden=clean_hidden,
            cached_pcas=cached_pcas,
            sigma=sigma,
        )

        delta = smoothed_hidden - clean_hidden

        def _layer_noise_fn(h: torch.Tensor, idx: int) -> torch.Tensor:
            if idx != target_layer:
                return torch.zeros_like(h)
            return delta

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            layer_noise_fn=_layer_noise_fn,
        )
        pred_ids = torch.argmax(out.logits, dim=-1).detach().cpu().numpy()

        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                if not valid_mask[batch_idx, token_idx]:
                    continue
                vote_counts[batch_idx, token_idx, pred_ids[batch_idx, token_idx]] += 1

    majority_pred = vote_counts.argmax(axis=-1)
    pred_tensor = torch.as_tensor(majority_pred, device=input_ids.device)

    certificates: list[list[TokenCertificate | None]] = []
    for batch_idx in range(batch_size):
        row: list[TokenCertificate | None] = []
        for token_idx in range(seq_len):
            if not bool(valid_mask[batch_idx, token_idx]):
                row.append(None)
                continue
            cert = certify_token_from_counts(
                class_counts=vote_counts[batch_idx, token_idx],
                alpha_noise=sigma,
                alpha_conf=alpha_conf,
                abstain_label=abstain_label,
            )
            row.append(cert)
        certificates.append(row)

    return SmoothedBatchOutput(
        pred_ids=pred_tensor,
        vote_counts=vote_counts,
        certificates=certificates,
        debug=debug_output,
    )
