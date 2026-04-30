"""Covariance and whitening helpers for manifold certificates."""

from __future__ import annotations

import torch


def batch_covariance(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    denom = max(x.shape[0] - 1, 1)
    cov = (x.transpose(0, 1) @ x) / denom
    cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return cov


def whitening_transform(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(cov)
    inv_sqrt = torch.diag(torch.rsqrt(torch.clamp(evals, min=eps)))
    return inv_sqrt @ evecs.transpose(0, 1)


def whitened_norm(delta: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(transform @ delta, ord=2)
