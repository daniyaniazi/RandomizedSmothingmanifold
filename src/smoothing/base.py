"""Abstract base class for smoothers.

A Smoother defines how to sample noisy points around an anchor vector.
Implementations include:
- IsotropicSmoother: Gaussian N(0, σ²I) noise
- ManifoldSmoother: Local PCA-based manifold smoothing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.indexing.base import NeighborIndex


@dataclass
class SmoothingResult:
    """Result of a smoothing operation.
    
    Attributes:
        noisy: The smoothed/noisy vector
        anchor: The original anchor vector
        noise: The noise vector that was added (in original or whitened space)
        radius: L2 distance from anchor to noisy point
    """
    noisy: np.ndarray
    anchor: np.ndarray
    noise: np.ndarray
    radius: float


class Smoother(ABC):
    """Abstract base class for smoothing algorithms.
    
    A Smoother samples noisy points around an anchor vector.
    The noise distribution depends on the smoothing mode:
    - Isotropic: Gaussian in the original space
    - Manifold: Gaussian in whitened PCA space, respecting local geometry
    
    Args:
        sigma: Standard deviation of the noise
    """
    
    def __init__(self, sigma: float):
        self._sigma = sigma
    
    @property
    def sigma(self) -> float:
        """Noise standard deviation."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        self._sigma = float(value)
    
    @abstractmethod
    def sample(self, anchor: np.ndarray) -> np.ndarray:
        """Sample a single noisy point around the anchor.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            Noisy vector (D,)
        """
        pass
    
    def sample_n(self, anchor: np.ndarray, n: int) -> np.ndarray:
        """Sample n noisy points around the anchor.
        
        Args:
            anchor: Anchor vector (D,)
            n: Number of samples
            
        Returns:
            Noisy vectors (n, D)
        """
        return np.stack([self.sample(anchor) for _ in range(n)])
    
    def sample_with_details(self, anchor: np.ndarray) -> SmoothingResult:
        """Sample with full details about the noise.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            SmoothingResult with noisy vector and metadata
        """
        noisy = self.sample(anchor)
        diff = noisy - anchor
        return SmoothingResult(
            noisy=noisy,
            anchor=anchor,
            noise=diff,
            radius=float(np.linalg.norm(diff)),
        )


def gaussian_noise(shape: int | tuple[int, ...], std: float, dtype=np.float32) -> np.ndarray:
    """Sample zero-mean Gaussian noise with notebook-style API.

    Equivalent to np.random.normal(0.0, std, size=shape), returned in the
    requested dtype.
    """
    return np.random.normal(loc=0.0, scale=float(std), size=shape).astype(dtype, copy=False)


def create_smoother(
    mode: str,
    sigma: float,
    index: Optional["NeighborIndex"] = None,
    knn_k: int = 32,
    eps_eig: float = 1e-6,
) -> Smoother:
    """Factory function to create a smoother.
    
    Args:
        mode: "isotropic" or "manifold"
        sigma: Noise standard deviation
        index: kNN index for manifold smoothing (required if mode="manifold")
        knn_k: Number of neighbors for local PCA
        eps_eig: Minimum eigenvalue for PCA
        
    Returns:
        Smoother instance
    """
    mode = mode.lower().strip()
    
    if mode == "isotropic":
        from .isotropic import IsotropicSmoother
        return IsotropicSmoother(sigma=sigma)
    
    elif mode == "manifold":
        if index is None:
            raise ValueError("Manifold smoothing requires a kNN index")
        from .manifold import ManifoldSmoother
        return ManifoldSmoother(sigma=sigma, index=index, knn_k=knn_k, eps_eig=eps_eig)
    
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}. Choose 'isotropic' or 'manifold'")
