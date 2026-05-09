"""Smoothing module for randomized smoothing.

Provides isotropic (Gaussian) and manifold-aware smoothing.

Main API:
    - Smoother: Abstract base class
    - IsotropicSmoother: N(0, σ²I) Gaussian noise
    - ManifoldSmoother: Local PCA-based manifold smoothing
    - create_smoother(): Factory function

PCA utilities:
    - LocalPCA: Fitted PCA dataclass
    - fit_local_pca: Fit PCA on neighbors
    - whiten/unwhiten: Transform to/from whitened space
"""

from .base import Smoother, SmoothingResult, create_smoother
from .isotropic import IsotropicSmoother
from .manifold import ManifoldSmoother, CachedPCA
from .pca import LocalPCA, fit_local_pca, whiten, unwhiten, reconstruct, reconstruction_error

__all__ = [
    # Core API
    "Smoother",
    "SmoothingResult",
    "create_smoother",
    "IsotropicSmoother",
    "ManifoldSmoother",
    "CachedPCA",
    # PCA utilities
    "LocalPCA",
    "fit_local_pca",
    "whiten",
    "unwhiten",
    "reconstruct",
    "reconstruction_error",
]

