"""Isotropic (Gaussian) smoothing.

Adds Gaussian noise N(0, σ²I) directly to the vector.
This is the standard randomized smoothing approach.
"""

from __future__ import annotations

import numpy as np

from .base import Smoother, SmoothingResult


class IsotropicSmoother(Smoother):
    """Isotropic Gaussian smoothing.
    
    Adds Gaussian noise N(0, σ²I) to the anchor vector.
    
    Args:
        sigma: Standard deviation of the Gaussian noise
        
    Example:
        smoother = IsotropicSmoother(sigma=0.25)
        noisy = smoother.sample(anchor)
    """
    
    def __init__(self, sigma: float):
        super().__init__(sigma)
    
    def sample(self, anchor: np.ndarray) -> np.ndarray:
        """Sample by adding isotropic Gaussian noise.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            Noisy vector (D,) = anchor + N(0, σ²I)
        """
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        noise = np.random.randn(len(anchor)).astype(np.float32) * self._sigma
        return anchor + noise
    
    def sample_n(self, anchor: np.ndarray, n: int) -> np.ndarray:
        """Sample n noisy points efficiently.
        
        Args:
            anchor: Anchor vector (D,)
            n: Number of samples
            
        Returns:
            Noisy vectors (n, D)
        """
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        noise = np.random.randn(n, len(anchor)).astype(np.float32) * self._sigma
        return anchor + noise
    
    def sample_with_details(self, anchor: np.ndarray) -> SmoothingResult:
        """Sample with full noise details.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            SmoothingResult with noise in original space
        """
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        noise = np.random.randn(len(anchor)).astype(np.float32) * self._sigma
        noisy = anchor + noise
        
        return SmoothingResult(
            noisy=noisy,
            anchor=anchor,
            noise=noise,
            radius=float(np.linalg.norm(noise)),
        )
