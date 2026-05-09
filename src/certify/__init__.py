"""Certification module for randomized smoothing.

Provides the core certification logic and abstractions:
    - Certifier: Abstract base class for certification workflows
    - SimpleCertifier: Generic certifier for any Space + Smoother
    - TokenCertificate: Certificate with prediction and radius
    - Voting utilities for counting and aggregating predictions

Example:
    from src.certify import SimpleCertifier
    from src.smoothing import ManifoldSmoother
    from src.spaces import PixelSpace
    
    space = PixelSpace(image_size=64)
    smoother = ManifoldSmoother(sigma=0.25, index=index)
    certifier = SimpleCertifier(
        smoother=smoother,
        space=space,
        classifier=lambda img: model(img).argmax().item(),
        n_samples=100,
    )
    result = certifier.certify(image_tensor)
    print(f"Predicted: {result.pred}, Radius: {result.radius:.4f}")
"""

from .base import (
    Certifier,
    SimpleCertifier,
    CertificationResult,
    BatchCertificationResult,
)
from .voting import (
    VotingResult,
    count_votes,
    majority_vote,
    sample_and_vote,
    top2_counts,
    BatchVoter,
)
from .randomized import (
    TokenCertificate,
    certify_token_from_counts,
    certified_radius,
    clopper_pearson_lower,
    clopper_pearson_upper,
)

__all__ = [
    # Core API
    "Certifier",
    "SimpleCertifier",
    "CertificationResult",
    "BatchCertificationResult",
    "VotingResult",
    "count_votes",
    "majority_vote",
    "sample_and_vote",
    "top2_counts",
    "BatchVoter",
    # Certification math
    "TokenCertificate",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "certified_radius",
    "certify_token_from_counts",
]
