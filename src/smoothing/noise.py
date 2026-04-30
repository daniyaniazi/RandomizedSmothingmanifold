"""Noise sampling utilities for isotropic and manifold-aware smoothing."""

from __future__ import annotations

import torch

from .workflow import isotropic_noise_like, smooth_tensor

def manifold_noise_from_batch(
    x: torch.Tensor,
    sigma: float,
    knn_k: int = 64,
    eps_eig: float = 1e-6,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return manifold noise by subtracting the original tensor from its smoothed version."""
    smoothed = smooth_tensor(
        x,
        sigma=sigma,
        mode="manifold",
        knn_k=knn_k,
        eps_eig=eps_eig,
        attention_mask=attention_mask,
    )
    return smoothed - x


def smooth_input_embeddings(
    embedding_layer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
    sigma: float,
    knn_k: int = 64,
    eps_eig: float = 1e-6,
) -> torch.Tensor:
    """Build smoothed input embeddings for downstream prediction/certification."""
    embeds = embedding_layer(input_ids)
    if mode == "manifold":
        return smooth_tensor(
            embeds,
            sigma=sigma,
            mode="manifold",
            knn_k=knn_k,
            eps_eig=eps_eig,
            attention_mask=attention_mask,
        )
    return embeds + isotropic_noise_like(embeds, sigma=sigma)
