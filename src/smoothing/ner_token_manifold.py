"""Token-wise manifold smoothing for NER certification.

This module provides NER-specific orchestration on top of the generic
ManifoldSmoother. It handles:
1. Token position tracking (batch_idx, token_idx)
2. Model forward pass with noise injection
3. Voting and certification per token

The core smoothing algorithm uses ManifoldSmoother from src/smoothing/manifold.

Workflow:
1. Get contextual hidden states for a sentence
2. For each valid token, cache PCA using ManifoldSmoother (ONCE per token)
3. Sample multiple noisy points using cached PCA
4. Inject the delta back into the chosen transformer layer
5. Classify the full sentence, then vote per token across noisy samples

OPTIMIZATION: PCA is computed once per token and reused for all samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.certify.randomized import TokenCertificate, certify_token_from_counts
from src.indexing import hidden_state_for_layer
from src.indexing.base import NeighborIndex, query_index
from .manifold import ManifoldSmoother, CachedPCA


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
    # Per-token eigenvalues for volume computation: eigenvalues[batch_idx][token_idx] = np.ndarray or None
    eigenvalues: list[list[np.ndarray | None]] | None = None


@dataclass
class CachedTokenPCA:
    """Cached PCA with token position info for NER.
    
    Extends the generic CachedPCA with batch/token indices for
    reconstructing the smoothed hidden state tensor.
    """
    cached_pca: CachedPCA  # Generic cached PCA from ManifoldSmoother
    batch_idx: int
    token_idx: int


def precompute_token_pcas(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    smoother: ManifoldSmoother,
) -> list[CachedTokenPCA]:
    """Precompute PCA for all valid tokens ONCE (not per sample).
    
    Uses ManifoldSmoother.compute_pca() for the actual PCA computation.
    
    Args:
        hidden: Hidden states tensor (batch, seq_len, dim)
        valid_mask: Boolean mask for valid tokens
        smoother: ManifoldSmoother instance with index and parameters
        
    Returns:
        List of CachedTokenPCA for each valid token
    """
    cached_pcas: list[CachedTokenPCA] = []
    batch_size, seq_len, _ = hidden.shape
    
    for batch_idx in range(batch_size):
        for token_idx in range(seq_len):
            if not bool(valid_mask[batch_idx, token_idx]):
                continue
            
            anchor = hidden[batch_idx, token_idx].detach().cpu().numpy().astype(np.float32)
            # Use ManifoldSmoother to compute PCA (handles kNN + PCA)
            cached_pca = smoother.compute_pca(anchor)
            
            cached_pcas.append(CachedTokenPCA(
                cached_pca=cached_pca,
                batch_idx=batch_idx,
                token_idx=token_idx,
            ))
    
    return cached_pcas


def sample_from_cached_token_pca(cached: CachedTokenPCA, smoother: ManifoldSmoother) -> np.ndarray:
    """Sample a single noisy point using ManifoldSmoother.
    
    Args:
        cached: CachedTokenPCA containing the generic CachedPCA
        smoother: ManifoldSmoother instance
        
    Returns:
        Noisy vector in original space
    """
    return smoother.sample_from_cached(cached.cached_pca)


def smooth_token_tensor_with_cache(
    hidden: torch.Tensor,
    cached_pcas: list[CachedTokenPCA],
    smoother: ManifoldSmoother,
) -> torch.Tensor:
    """Apply one round of manifold smoothing using precomputed PCAs.
    
    Args:
        hidden: Hidden states tensor (batch, seq_len, dim)
        cached_pcas: List of CachedTokenPCA from precompute_token_pcas
        smoother: ManifoldSmoother instance
        
    Returns:
        Smoothed hidden states tensor
    """
    smoothed = hidden.clone()
    
    for cached in cached_pcas:
        noisy = sample_from_cached_token_pca(cached, smoother)
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
    smoother: ManifoldSmoother,
    token_texts: list[str],
    label_ids: list[int],
    tokenizer,
    collect_debug: bool = False,
) -> tuple[torch.Tensor, list[list[TokenDebugRecord | None]] | None]:
    """Smooth each valid token independently using ManifoldSmoother.
    
    NOTE: For better performance, use precompute_token_pcas + smooth_token_tensor_with_cache.
    
    Args:
        hidden: Hidden states tensor (batch, seq_len, dim)
        input_ids: Token IDs for debug info
        labels: Labels tensor for debug info
        valid_mask: Boolean mask for valid tokens
        smoother: ManifoldSmoother instance
        token_texts: Token texts from index (for debug)
        label_ids: Label IDs from index (for debug)
        tokenizer: Tokenizer for debug info
        collect_debug: Whether to collect debug info
        
    Returns:
        Tuple of (smoothed hidden states, debug info)
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
            
            # Use ManifoldSmoother for sampling
            result = smoother.sample_with_details(anchor)
            noisy = result.noisy

            smoothed[batch_idx, token_idx] = torch.as_tensor(
                noisy,
                device=hidden.device,
                dtype=hidden.dtype,
            )

            if collect_debug and debug_row is not None:
                neighbor_ids = np.asarray([], dtype=np.int64)
                try:
                    # k+1 to drop self-match at index 0
                    all_ids = query_index(smoother.index, k=smoother.knn_k + 1, vector=anchor)
                    neighbor_ids = all_ids[1:]
                except Exception:
                    neighbor_ids = np.arange(min(smoother.knn_k, len(token_texts)), dtype=np.int64)

                # Query index with the noisy vector to find where the smoothed point landed
                noisy_nearest_tok = ""
                noisy_nearest_lbl = -1
                try:
                    noisy_ids = query_index(smoother.index, k=1, vector=noisy)
                    if len(noisy_ids) > 0:
                        noisy_nbr_id = int(noisy_ids[0])
                        noisy_nearest_tok = token_texts[noisy_nbr_id]
                        noisy_nearest_lbl = int(label_ids[noisy_nbr_id])
                except Exception:
                    pass

                tok = tokenizer.convert_ids_to_tokens([int(input_ids[batch_idx, token_idx].item())])[0]
                
                # Get reconstruction error (anchor through PCA round-trip)
                # Use smoother to compute this
                cached = smoother.compute_pca(anchor)
                from .pca import whiten, unwhiten
                recon = unwhiten(whiten(anchor, cached.pca), cached.pca)
                
                debug_row[token_idx] = TokenDebugRecord(
                    token=tok,
                    true_label=int(labels[batch_idx, token_idx].item()),
                    neighbor_tokens=[token_texts[int(i)] for i in neighbor_ids[: min(len(neighbor_ids), 10)]],
                    neighbor_labels=[int(label_ids[int(i)]) for i in neighbor_ids[: min(len(neighbor_ids), 10)]],
                    reconstruction_l2=float(np.linalg.norm(recon - anchor)),
                    noisy_l2=float(np.linalg.norm(noisy - anchor)),
                    explained_variance=cached.pca.evals[: min(10, len(cached.pca.evals))].tolist(),
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
    smoother: ManifoldSmoother,
    token_texts: list[str],
    label_ids: list[int],
    tokenizer,
) -> list[list[TokenDebugRecord | None]]:
    """Collect debug info using cached PCAs.
    
    Args:
        cached_pcas: List of CachedTokenPCA from precompute_token_pcas
        hidden: Hidden states tensor
        input_ids: Token IDs
        labels: Labels tensor
        valid_mask: Boolean mask
        smoother: ManifoldSmoother instance
        token_texts: Token texts from index
        label_ids: Label IDs from index
        tokenizer: Tokenizer for token text lookup
        
    Returns:
        Debug info for each batch/token position
    """
    from .pca import whiten, unwhiten
    
    batch_size, seq_len, _ = hidden.shape
    debug_rows: list[list[TokenDebugRecord | None]] = []
    
    for batch_idx in range(batch_size):
        debug_row: list[TokenDebugRecord | None] = [None] * seq_len
        debug_rows.append(debug_row)
    
    for cached in cached_pcas:
        batch_idx = cached.batch_idx
        token_idx = cached.token_idx
        anchor = cached.cached_pca.anchor
        pca = cached.cached_pca.pca
        
        # Sample one noisy point for debug
        noisy = sample_from_cached_token_pca(cached, smoother)
        recon = unwhiten(whiten(anchor, pca), pca)  # Reconstruct through PCA
        
        neighbor_ids = np.asarray([], dtype=np.int64)
        try:
            all_ids = query_index(smoother.index, k=smoother.knn_k + 1, vector=anchor)
            neighbor_ids = all_ids[1:]
        except Exception:
            neighbor_ids = np.arange(min(smoother.knn_k, len(token_texts)), dtype=np.int64)
        
        # Query index with the noisy vector
        noisy_nearest_tok = ""
        noisy_nearest_lbl = -1
        try:
            noisy_ids = query_index(smoother.index, k=1, vector=noisy)
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
    
    Uses ManifoldSmoother internally - PCA is computed ONCE per token,
    then reused for all num_samples iterations.
    This gives ~100x speedup for the PCA/neighbor lookup phase.
    
    Args:
        model: NER model with hidden state output
        batch: Batch dict with input_ids, attention_mask, labels
        neighbor_index: NeighborIndex for kNN lookups
        token_texts: Token texts from training index
        label_ids: Label IDs from training index
        tokenizer: Tokenizer for debug info
        num_samples: Number of smoothing samples
        sigma: Noise standard deviation
        knn_k: Number of neighbors for local PCA
        eps_eig: Eigenvalue floor for PCA
        layer_index: Which transformer layer to inject noise
        alpha_conf: Confidence level for certification
        abstain_label: Label ID for abstention
        collect_debug: Whether to collect debug info
        
    Returns:
        SmoothedBatchOutput with predictions, vote counts, and certificates
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

    # Create ManifoldSmoother instance (used for all tokens)
    smoother = ManifoldSmoother(
        sigma=sigma,
        index=neighbor_index,
        knn_k=knn_k,
        eps_eig=eps_eig,
    )

    # OPTIMIZATION: Precompute PCA for all tokens ONCE using smoother
    cached_pcas = precompute_token_pcas(
        hidden=clean_hidden,
        valid_mask=valid_mask,
        smoother=smoother,
    )
    
    # Collect debug info once (using first sample)
    if collect_debug:
        debug_output = _collect_debug_info(
            cached_pcas=cached_pcas,
            hidden=clean_hidden,
            input_ids=input_ids,
            labels=labels,
            valid_mask=valid_mask,
            smoother=smoother,
            token_texts=token_texts,
            label_ids=label_ids,
            tokenizer=tokenizer,
        )

    # Sample loop - now only samples noise, no PCA recomputation
    for sample_idx in range(num_samples):
        # Use cached PCAs to generate smoothed hidden states (FAST)
        smoothed_hidden = smooth_token_tensor_with_cache(
            hidden=clean_hidden,
            cached_pcas=cached_pcas,
            smoother=smoother,
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

    # Extract eigenvalues from cached PCAs for volume computation
    token_eigenvalues: list[list[np.ndarray | None]] = [
        [None] * seq_len for _ in range(batch_size)
    ]
    for cpca in cached_pcas:
        token_eigenvalues[cpca.batch_idx][cpca.token_idx] = cpca.cached_pca.pca.evals

    return SmoothedBatchOutput(
        pred_ids=pred_tensor,
        vote_counts=vote_counts,
        certificates=certificates,
        debug=debug_output,
        eigenvalues=token_eigenvalues,
    )
