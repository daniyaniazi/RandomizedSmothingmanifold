"""Manifold-aware smoothing using local PCA.

Adds Gaussian noise in the whitened (PCA) space, then transforms back.
This respects the local geometry of the data manifold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.indexing.base import NeighborIndex, neighbor_vectors

from .base import Smoother, SmoothingResult
from .pca import LocalPCA, fit_local_pca, whiten, unwhiten


@dataclass
class CachedPCA:
    """Cached PCA for a single anchor - reused across samples.
    
    Attributes:
        anchor: Original anchor vector
        pca: Fitted local PCA
        neighbors: Neighbor vectors used to fit PCA
    """
    anchor: np.ndarray
    pca: LocalPCA
    neighbors: np.ndarray


class ManifoldSmoother(Smoother):
    """Manifold-aware smoothing using local PCA.
    
    For each anchor:
    1. Find k nearest neighbors from the index
    2. Fit local PCA on the neighborhood
    3. Whiten the anchor: w = (x @ V) / sqrt(λ)
    4. Add isotropic noise in whitened space: w' = w + N(0, σ²I)
    5. Unwhiten back: x' = (w' * sqrt(λ)) @ V.T + mean
    
    This ensures noise is scaled according to local variance directions,
    respecting the data manifold geometry.
    
    Args:
        sigma: Standard deviation in whitened space
        index: kNN index for neighbor lookup
        knn_k: Number of neighbors for local PCA
        eps_eig: Minimum eigenvalue clamp
        
    Example:
        index = load_index(...)
        smoother = ManifoldSmoother(sigma=0.25, index=index, knn_k=32)
        noisy = smoother.sample(anchor)
    """
    
    def __init__(
        self,
        sigma: float,
        index: NeighborIndex,
        knn_k: int = 32,
        eps_eig: float = 1e-6,
    ):
        super().__init__(sigma)
        self._index = index
        self._knn_k = knn_k
        self._eps_eig = eps_eig
    
    @property
    def index(self) -> NeighborIndex:
        """kNN index for neighbor lookup."""
        return self._index
    
    @property
    def knn_k(self) -> int:
        """Number of neighbors for local PCA."""
        return self._knn_k
    
    def _get_neighbors(self, anchor: np.ndarray) -> np.ndarray:
        """Get k nearest neighbors for the anchor.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            Neighbor vectors (k, D)
        """
        # Request k+1 and drop index 0 (self-match)
        neighbors = neighbor_vectors(self._index, k=self._knn_k + 1, vector=anchor)
        return neighbors[1:]  # Skip self-match
    
    def _fit_pca(self, neighbors: np.ndarray) -> LocalPCA:
        """Fit local PCA on neighbors.
        
        Args:
            neighbors: Neighbor vectors (k, D)
            
        Returns:
            LocalPCA object
        """
        return fit_local_pca(neighbors, eps_eig=self._eps_eig)
    
    def compute_pca(self, anchor: np.ndarray) -> CachedPCA:
        """Compute and cache PCA for an anchor.
        
        Use this when you need to sample multiple times from the same anchor
        without recomputing the PCA.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            CachedPCA object for reuse
        """
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        neighbors = self._get_neighbors(anchor)
        pca = self._fit_pca(neighbors)
        return CachedPCA(anchor=anchor, pca=pca, neighbors=neighbors)
    
    def sample(self, anchor: np.ndarray) -> np.ndarray:
        """Sample by adding noise in whitened space.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            Noisy vector (D,)
        """
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        neighbors = self._get_neighbors(anchor)
        pca = self._fit_pca(neighbors)
        return self._sample_from_pca(anchor, pca)
    
    def sample_from_cached(self, cached: CachedPCA) -> np.ndarray:
        """Sample using a cached PCA (fast - no kNN/PCA recomputation).
        
        Args:
            cached: CachedPCA from compute_pca()
            
        Returns:
            Noisy vector (D,)
        """
        return self._sample_from_pca(cached.anchor, cached.pca)
    
    def _sample_from_pca(self, anchor: np.ndarray, pca: LocalPCA) -> np.ndarray:
        """Sample using precomputed PCA.
        
        Args:
            anchor: Anchor vector (D,)
            pca: LocalPCA object
            
        Returns:
            Noisy vector (D,)
        """
        # Whiten
        w = whiten(anchor, pca)
        # Add isotropic noise in whitened space
        noise = np.random.randn(len(w)).astype(np.float32) * self._sigma
        w_noisy = w + noise
        # Unwhiten back
        return unwhiten(w_noisy, pca)
    
    def sample_n(self, anchor: np.ndarray, n: int) -> np.ndarray:
        """Sample n noisy points with single PCA computation.
        
        Args:
            anchor: Anchor vector (D,)
            n: Number of samples
            
        Returns:
            Noisy vectors (n, D)
        """
        cached = self.compute_pca(anchor)
        return np.stack([self.sample_from_cached(cached) for _ in range(n)])
    
    def sample_with_details(self, anchor: np.ndarray) -> SmoothingResult:
        """Sample with full details.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            SmoothingResult with noise in original space
        """
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        cached = self.compute_pca(anchor)
        noisy = self.sample_from_cached(cached)
        diff = noisy - anchor
        
        return SmoothingResult(
            noisy=noisy,
            anchor=anchor,
            noise=diff,
            radius=float(np.linalg.norm(diff)),
        )
