"""Image classification certification workflow.

Provides a high-level interface for certifying image classifiers
using randomized smoothing in pixel or latent space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from src.certify import SimpleCertifier, CertificationResult, BatchCertificationResult
from src.indexing.base import NeighborIndex, load_index
from src.smoothing import create_smoother, Smoother
from src.spaces import Space

if TYPE_CHECKING:
    from src.spaces.latent import VAEProtocol

from .spaces import ImagePixelSpace, ImageLatentSpace


@dataclass
class ImageCertificationConfig:
    """Configuration for image certification.
    
    Attributes:
        sigma: Noise standard deviation
        n_samples: Number of samples for certification
        alpha: Confidence level for Clopper-Pearson
        smoothing_mode: "isotropic" or "manifold"
        space_mode: "pixel" or "latent"
        knn_k: Number of neighbors for manifold smoothing
        index_path: Path to kNN index (for manifold mode)
    """
    sigma: float = 0.25
    n_samples: int = 100
    alpha: float = 0.001
    smoothing_mode: str = "manifold"
    space_mode: str = "pixel"
    knn_k: int = 32
    index_path: Optional[str] = None
    device: str = "cuda"


class ImageCertifier:
    """Certifier for image classification tasks.
    
    Combines:
    - Space (pixel or latent)
    - Smoother (isotropic or manifold)
    - Classifier
    
    Example:
        certifier = ImageCertifier.from_config(config, classifier, vae=vae)
        result = certifier.certify(image_tensor)
    """
    
    def __init__(
        self,
        space: Space[torch.Tensor],
        smoother: Smoother,
        classifier: Callable[[torch.Tensor], int],
        n_samples: int = 100,
        alpha: float = 0.001,
        num_classes: int = 2,
    ):
        self._space = space
        self._smoother = smoother
        self._classifier = classifier
        self._n_samples = n_samples
        self._alpha = alpha
        self._num_classes = num_classes
        
        # Create internal certifier
        self._certifier = SimpleCertifier(
            smoother=smoother,
            space=space,
            classifier=classifier,
            n_samples=n_samples,
            alpha=alpha,
            num_classes=num_classes,
        )
    
    @classmethod
    def from_config(
        cls,
        config: ImageCertificationConfig,
        classifier: Callable[[torch.Tensor], int],
        image_size: int = 64,
        vae: Optional["VAEProtocol"] = None,
        index: Optional[NeighborIndex] = None,
    ) -> "ImageCertifier":
        """Create certifier from config.
        
        Args:
            config: Certification configuration
            classifier: Callable that classifies images
            image_size: Size of input images
            vae: VAE model (required for latent space)
            index: kNN index (required for manifold smoothing)
        """
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Create space
        if config.space_mode == "latent":
            if vae is None:
                raise ValueError("VAE required for latent space")
            space = ImageLatentSpace(vae=vae, device=device)
        else:
            space = ImagePixelSpace(image_size=image_size, device=device)
        
        # Load index if needed
        if config.smoothing_mode == "manifold" and index is None:
            if config.index_path is None:
                raise ValueError("Index path required for manifold smoothing")
            index = load_index(dim=space.dim, index_path=config.index_path, backend="annoy")
        
        # Create smoother
        smoother = create_smoother(
            mode=config.smoothing_mode,
            sigma=config.sigma,
            index=index,
            knn_k=config.knn_k,
        )
        
        return cls(
            space=space,
            smoother=smoother,
            classifier=classifier,
            n_samples=config.n_samples,
            alpha=config.alpha,
        )
    
    @property
    def space(self) -> Space[torch.Tensor]:
        return self._space
    
    @property
    def smoother(self) -> Smoother:
        return self._smoother
    
    @property
    def sigma(self) -> float:
        return self._smoother.sigma
    
    def certify(self, image: torch.Tensor) -> CertificationResult:
        """Certify a single image.
        
        Args:
            image: Image tensor (C, H, W)
            
        Returns:
            CertificationResult with prediction and radius
        """
        return self._certifier.certify(image)
    
    def certify_dataset(
        self,
        samples: list[Tuple[torch.Tensor, int]],
        progress: bool = True,
    ) -> list[CertificationResult]:
        """Certify a dataset of (image, label) pairs.
        
        Args:
            samples: List of (image_tensor, label) tuples
            progress: Show progress bar
            
        Returns:
            List of CertificationResults
        """
        results = []
        iterator = tqdm(samples, desc="Certifying") if progress else samples
        
        for image, label in iterator:
            result = self.certify(image)
            results.append(result)
        
        return results


def run_image_certification(
    config: ImageCertificationConfig,
    classifier: Callable[[torch.Tensor], int],
    samples: list[Tuple[torch.Tensor, int]],
    image_size: int = 64,
    vae: Optional["VAEProtocol"] = None,
    index: Optional[NeighborIndex] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run full image certification pipeline.
    
    Args:
        config: Certification configuration
        classifier: Image classifier
        samples: List of (image, label) pairs
        image_size: Input image size
        vae: VAE for latent space (optional)
        index: kNN index for manifold smoothing (optional)
        output_dir: Directory to save results (optional)
        
    Returns:
        Dictionary with certification metrics
    """
    certifier = ImageCertifier.from_config(
        config=config,
        classifier=classifier,
        image_size=image_size,
        vae=vae,
        index=index,
    )
    
    results = certifier.certify_dataset(samples)
    
    # Compute metrics
    labels = [label for _, label in samples]
    preds = [r.pred for r in results]
    radii = [r.radius for r in results]
    abstained = [r.abstained for r in results]
    
    correct = sum(1 for p, l in zip(preds, labels) if p == l and not abstained[preds.index(p)])
    certified = sum(1 for a in abstained if not a)
    
    metrics = {
        "total_samples": len(samples),
        "certified_samples": certified,
        "abstained_samples": len(samples) - certified,
        "accuracy": correct / certified if certified > 0 else 0.0,
        "abstention_rate": (len(samples) - certified) / len(samples),
        "mean_radius": np.mean([r for r, a in zip(radii, abstained) if not a]) if certified > 0 else 0.0,
        "sigma": config.sigma,
        "n_samples": config.n_samples,
        "smoothing_mode": config.smoothing_mode,
        "space_mode": config.space_mode,
    }
    
    # Save results if output_dir provided
    if output_dir:
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics
