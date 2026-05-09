"""Abstract base certifier and certification result types.

A Certifier encapsulates the full certification workflow:
1. Sample noisy predictions using a smoother
2. Count votes for each class
3. Compute certified radius using Clopper-Pearson bounds
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar

import numpy as np

from src.smoothing.base import Smoother
from src.spaces.base import Space

from .randomized import TokenCertificate, certify_token_from_counts


T = TypeVar("T")  # Input type (image, token embedding, etc.)


@dataclass
class CertificationResult:
    """Result of certifying a single input.
    
    Attributes:
        pred: Predicted class (majority vote), or abstain_label if abstained
        certificate: The TokenCertificate with radius and bounds
        vote_counts: Per-class vote counts from sampling
        clean_pred: Prediction on the clean (unsmoothed) input
        abstained: Whether certification abstained (no confident majority)
    """
    pred: int
    certificate: TokenCertificate
    vote_counts: np.ndarray
    clean_pred: Optional[int] = None
    abstained: bool = False
    
    @property
    def radius(self) -> float:
        """Certified radius."""
        return self.certificate.radius
    
    @property
    def correct(self) -> bool:
        """Whether prediction matches clean prediction."""
        return self.pred == self.clean_pred if self.clean_pred is not None else False


@dataclass
class BatchCertificationResult:
    """Result of certifying a batch of inputs.
    
    Attributes:
        results: List of individual CertificationResults
        predictions: Array of predictions (batch_size,)
        abstained: Array of abstention flags (batch_size,)
        radii: Array of certified radii (batch_size,)
    """
    results: list[CertificationResult]
    predictions: np.ndarray = field(init=False)
    abstained: np.ndarray = field(init=False)
    radii: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.predictions = np.array([r.pred for r in self.results])
        self.abstained = np.array([r.abstained for r in self.results])
        self.radii = np.array([r.radius for r in self.results])
    
    @property
    def num_certified(self) -> int:
        """Number of non-abstained samples."""
        return int((~self.abstained).sum())
    
    @property
    def mean_radius(self) -> float:
        """Mean radius over certified (non-abstained) samples."""
        certified_radii = self.radii[~self.abstained]
        return float(certified_radii.mean()) if len(certified_radii) > 0 else 0.0


class Certifier(ABC, Generic[T]):
    """Abstract base class for certification.
    
    A Certifier runs randomized smoothing certification on inputs:
    1. Encode input to vector space
    2. Sample n noisy vectors using smoother
    3. Decode each noisy vector and classify
    4. Count votes and compute certified radius
    
    Args:
        smoother: Smoother for generating noisy samples
        space: Space for encoding/decoding inputs
        classifier: Callable that takes decoded input and returns class prediction
        n_samples: Number of samples for certification
        alpha: Confidence level for Clopper-Pearson bounds
        abstain_label: Label to return when abstaining
    """
    
    def __init__(
        self,
        smoother: Smoother,
        space: Space[T],
        classifier: Callable[[T], int],
        n_samples: int = 100,
        alpha: float = 0.001,
        abstain_label: int = -1,
        num_classes: Optional[int] = None,
    ):
        self._smoother = smoother
        self._space = space
        self._classifier = classifier
        self._n_samples = n_samples
        self._alpha = alpha
        self._abstain_label = abstain_label
        self._num_classes = num_classes
    
    @property
    def smoother(self) -> Smoother:
        return self._smoother
    
    @property
    def space(self) -> Space[T]:
        return self._space
    
    @property
    def n_samples(self) -> int:
        return self._n_samples
    
    @property
    def sigma(self) -> float:
        return self._smoother.sigma
    
    def certify(self, x: T) -> CertificationResult:
        """Certify a single input.
        
        Args:
            x: Input in original domain
            
        Returns:
            CertificationResult with prediction and certified radius
        """
        # Clean prediction
        clean_pred = self._classifier(x)
        
        # Encode to vector
        anchor = self._space.encode(x)
        
        # Sample and vote
        if self._num_classes is None:
            # Infer from clean prediction + 1
            num_classes = max(clean_pred + 1, 2)
        else:
            num_classes = self._num_classes
        
        vote_counts = np.zeros(num_classes, dtype=np.int64)
        
        for _ in range(self._n_samples):
            noisy = self._smoother.sample(anchor)
            x_noisy = self._space.decode(noisy)
            pred = self._classifier(x_noisy)
            if 0 <= pred < num_classes:
                vote_counts[pred] += 1
        
        # Compute certificate
        certificate = certify_token_from_counts(
            class_counts=vote_counts,
            alpha_noise=self._smoother.sigma,
            alpha_conf=self._alpha,
            abstain_label=self._abstain_label,
        )
        
        return CertificationResult(
            pred=certificate.pred,
            certificate=certificate,
            vote_counts=vote_counts,
            clean_pred=clean_pred,
            abstained=certificate.abstained,
        )
    
    def certify_batch(self, xs: list[T]) -> BatchCertificationResult:
        """Certify a batch of inputs.
        
        Args:
            xs: List of inputs
            
        Returns:
            BatchCertificationResult with all results
        """
        results = [self.certify(x) for x in xs]
        return BatchCertificationResult(results=results)


class SimpleCertifier(Certifier[T]):
    """Simple implementation of Certifier for generic inputs.
    
    Works with any Space and classifier callable.
    
    Example:
        space = PixelSpace(image_size=64)
        smoother = ManifoldSmoother(sigma=0.25, index=index)
        certifier = SimpleCertifier(
            smoother=smoother,
            space=space,
            classifier=lambda img: model(img).argmax().item(),
            n_samples=100,
        )
        result = certifier.certify(image_tensor)
    """
    pass  # Inherits everything from Certifier base class
