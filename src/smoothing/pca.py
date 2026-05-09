"""Local PCA utilities for manifold smoothing.

Provides whiten/unwhiten transformations using local PCA
computed from k-nearest neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalPCA:
    """Local PCA fitted on a neighborhood.
    
    Attributes:
        mean: Neighborhood mean vector (D,)
        evals: Eigenvalues in descending order (K,)
        evecs: Eigenvectors as columns (D, K)
    """
    mean: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray
    
    @property
    def n_components(self) -> int:
        """Number of principal components."""
        return len(self.evals)
    
    @property
    def dim(self) -> int:
        """Original dimensionality."""
        return self.evecs.shape[0]


def fit_local_pca(
    neighbors: np.ndarray,
    eps_eig: float = 1e-6,
) -> LocalPCA:
    """Fit local PCA on a set of neighbor vectors.
    
    Args:
        neighbors: Neighbor vectors (K, D)
        eps_eig: Minimum eigenvalue clamp to avoid division by zero
        
    Returns:
        LocalPCA object with mean, eigenvalues, and eigenvectors
    """
    if neighbors.ndim != 2:
        raise ValueError(f"Neighbors must be 2D, got shape {neighbors.shape}")
    
    neighbors = np.asarray(neighbors, dtype=np.float32)
    
    # Compute mean and center data
    mean = neighbors.mean(axis=0)
    centered = neighbors - mean
    
    # Compute covariance matrix
    n_samples = max(centered.shape[0] - 1, 1)
    cov = (centered.T @ centered) / n_samples
    
    # Eigendecomposition
    evals, evecs = np.linalg.eigh(cov)
    
    # Sort by descending eigenvalue
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], a_min=eps_eig, a_max=None).astype(np.float32)
    evecs = evecs[:, order].astype(np.float32)
    
    return LocalPCA(mean=mean, evals=evals, evecs=evecs)


def whiten(vector: np.ndarray, pca: LocalPCA) -> np.ndarray:
    """Whiten a vector using local PCA.
    
    Projects vector into whitened space: w = (x @ V) / sqrt(λ)
    
    Note: The mean is NOT subtracted before whitening. The mean is only
    used during unwhitening to shift the result back to the neighborhood center.
    
    Args:
        vector: Input vector (D,)
        pca: Local PCA object
        
    Returns:
        Whitened vector (K,) where K = n_components
    """
    x = np.asarray(vector, dtype=np.float32).reshape(-1)
    return (x @ pca.evecs) / np.sqrt(pca.evals)


def unwhiten(white_vector: np.ndarray, pca: LocalPCA) -> np.ndarray:
    """Unwhiten a vector back to original space.
    
    Projects from whitened space: x = (w * sqrt(λ)) @ V.T + mean
    
    Args:
        white_vector: Whitened vector (K,)
        pca: Local PCA object
        
    Returns:
        Vector in original space (D,)
    """
    w = np.asarray(white_vector, dtype=np.float32).reshape(-1)
    return (w * np.sqrt(pca.evals)) @ pca.evecs.T + pca.mean


def reconstruct(vector: np.ndarray, pca: LocalPCA) -> np.ndarray:
    """Reconstruct vector through PCA (whiten then unwhiten).
    
    Useful for checking reconstruction error.
    
    Args:
        vector: Input vector (D,)
        pca: Local PCA object
        
    Returns:
        Reconstructed vector (D,)
    """
    return unwhiten(whiten(vector, pca), pca)


def reconstruction_error(vector: np.ndarray, pca: LocalPCA) -> float:
    """Compute L2 reconstruction error through PCA.
    
    Args:
        vector: Input vector (D,)
        pca: Local PCA object
        
    Returns:
        L2 norm of reconstruction error
    """
    recon = reconstruct(vector, pca)
    return float(np.linalg.norm(vector - recon))
