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
    n_components: int | None = None,
) -> LocalPCA:
    """Fit local PCA on a set of neighbor vectors (reference-style).
    
    Uses sklearn PCA like the notebook: centers the data and, by default, fits
    PCA with n_components = K (number of neighbors). An explicit component
    count supports controlled PCA-dimension ablations.
    
    Args:
        neighbors: Neighbor vectors (K, D)
        eps_eig: Minimum eigenvalue clamp to avoid division by zero
        n_components: Optional explicit PCA dimension. Because K samples are
            centered before PCA, an explicit dimension must be at most K-1.
        
    Returns:
        LocalPCA object with mean, eigenvalues, and eigenvectors
    """
    from sklearn.decomposition import PCA
    
    if neighbors.ndim != 2:
        raise ValueError(f"Neighbors must be 2D, got shape {neighbors.shape}")
    
    neighbors = np.asarray(neighbors, dtype=np.float32)
    K, D = neighbors.shape
    
    # Center data
    mean = neighbors.mean(axis=0)
    centered = neighbors - mean
    
    # Fit PCA (same as reference notebook workflow)
    if n_components is None:
        # Preserve the original implementation for baseline reproducibility.
        fitted_components = min(K, D)
    else:
        fitted_components = int(n_components)
        max_meaningful_rank = min(K - 1, D)
        if not 1 <= fitted_components <= max_meaningful_rank:
            raise ValueError(
                f"pca_dim must be in [1, {max_meaningful_rank}] for "
                f"K={K}, D={D}; got {fitted_components}"
            )
    pca = PCA(n_components=fitted_components)
    pca.fit(centered)
    
    ev = np.maximum(pca.explained_variance_.astype(np.float32), eps_eig)
    # pca.components_ has shape (n_components, D) — rows are eigenvectors
    # We store eigenvectors as columns (D, n_components) to match our whiten/unwhiten
    Vt = pca.components_.astype(np.float32)  # (n_components, D)
    evecs = Vt.T  # (D, n_components)
    
    return LocalPCA(mean=mean, evals=ev, evecs=evecs)


def whiten(vector: np.ndarray, pca: LocalPCA) -> np.ndarray:
    """Whiten a vector using local PCA.

    Projects vector into whitened space: w = ((x - mean) @ V) / sqrt(λ)

    The neighborhood mean is subtracted before projecting so that the
    whitened anchor lives at the centre of the local coordinate frame.
    This matches the notebook convention::

        X_whitened = (X_orig - mean) @ (1/sqrt(ev)) * Vt.T

    Args:
        vector: Input vector (D,)
        pca: Local PCA object

    Returns:
        Whitened vector (K,) where K = n_components
    """
    x = np.asarray(vector, dtype=np.float32).reshape(-1)
    x_centered = x - pca.mean
    return (x_centered @ pca.evecs) / np.sqrt(pca.evals)


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
