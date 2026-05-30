"""Noise sampling utilities for training-time smoothing augmentation.

Used during NER training to add smoothing augmentation to input embeddings.
For certification, use IsotropicSmoother or ManifoldSmoother instead.
"""

from __future__ import annotations

import numpy as np
import torch

from src.indexing.base import build_index, neighbor_vectors
from .base import gaussian_noise
from .pca import fit_local_pca, whiten, unwhiten


def isotropic_noise_like(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Generate isotropic Gaussian noise with same shape as x."""
    return torch.randn_like(x) * sigma


def smooth_input_embeddings(
    embedding_layer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
    sigma: float,
    knn_k: int = 64,
    eps_eig: float = 1e-6,
) -> torch.Tensor:
    """Build smoothed input embeddings for training augmentation.
    
    Args:
        embedding_layer: The model's embedding layer
        input_ids: Token IDs (B, T)
        attention_mask: Attention mask (B, T)
        mode: "manifold" or "isotropic"
        sigma: Noise standard deviation
        knn_k: Number of neighbors for manifold smoothing
        eps_eig: Minimum eigenvalue for PCA
        
    Returns:
        Smoothed embeddings (B, T, D)
    """
    embeds = embedding_layer(input_ids)
    
    if mode != "manifold":
        return embeds + isotropic_noise_like(embeds, sigma=sigma)
    
    # Manifold smoothing: apply local PCA-based smoothing to each token
    smoothed = embeds.clone()
    device = embeds.device
    batch_size, seq_len, dim = embeds.shape
    
    # Flatten valid embeddings for index building
    mask = attention_mask.bool() if attention_mask is not None else torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    flat_embeds = embeds.view(-1, dim)
    flat_mask = mask.view(-1)
    
    valid_embeds = flat_embeds[flat_mask].detach().cpu().numpy()
    
    if len(valid_embeds) < knn_k + 1:
        # Not enough tokens for manifold smoothing, fall back to isotropic
        return embeds + isotropic_noise_like(embeds, sigma=sigma)
    
    # Build index on current batch
    index = build_index(valid_embeds, backend="torch")
    
    # Smooth each valid token
    for b in range(batch_size):
        for t in range(seq_len):
            if not mask[b, t]:
                continue
            
            anchor = embeds[b, t].detach().cpu().numpy().astype(np.float32)
            neighbors = neighbor_vectors(index, k=knn_k + 1, vector=anchor)[1:]  # Skip self
            
            if len(neighbors) < 2:
                smoothed[b, t] += torch.randn_like(smoothed[b, t]) * sigma
                continue
            
            pca = fit_local_pca(neighbors, eps_eig=eps_eig)
            whitened = whiten(anchor, pca)  # mean-subtracted inside whiten()
            lambda_max = float(pca.evals[0])
            alpha = sigma / np.sqrt(max(lambda_max, 1e-12))  # sigma / sqrt(lambda_max)
            noise = gaussian_noise(shape=len(whitened), std=alpha)
            noisy = unwhiten(whitened + noise, pca)
            
            smoothed[b, t] = torch.as_tensor(noisy, device=device, dtype=embeds.dtype)
    
    return smoothed

