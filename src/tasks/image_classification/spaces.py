"""Space adapters for image classification.

Wraps the generic Space classes with image-specific utilities.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torchvision import transforms

from src.spaces import PixelSpace, LatentSpace
from src.spaces.latent import VAEProtocol


# Common normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CELEBA_MEAN = [0.5, 0.5, 0.5]
CELEBA_STD = [0.5, 0.5, 0.5]


class ImagePixelSpace(PixelSpace):
    """Pixel space for image classification.
    
    Extends PixelSpace with:
    - Preset normalizations (CelebA, ImageNet)
    - Classifier transform support (resize to 224x224)
    
    Args:
        image_size: Size of input images
        normalization: "celeba", "imagenet", or None for custom
        mean: Custom mean (if normalization is None)
        std: Custom std (if normalization is None)
        device: Torch device
    """
    
    def __init__(
        self,
        image_size: int | Tuple[int, int],
        normalization: str = "celeba",
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
        device: torch.device | str = "cpu",
    ):
        # Set normalization
        if normalization == "celeba":
            mean = CELEBA_MEAN
            std = CELEBA_STD
        elif normalization == "imagenet":
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
        
        super().__init__(
            image_size=image_size,
            channels=3,
            mean=mean,
            std=std,
            device=device,
        )
        
        self._normalization = normalization
    
    @property
    def normalization(self) -> str:
        return self._normalization
    
    def get_classifier_transform(self, classifier_input_size: int = 224) -> transforms.Compose:
        """Get transform to prepare images for classifier.
        
        Args:
            classifier_input_size: Size expected by classifier (usually 224)
            
        Returns:
            Torchvision transform
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((classifier_input_size, classifier_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class ImageLatentSpace(LatentSpace):
    """Latent space for image classification via VAE.
    
    Extends LatentSpace with:
    - Image loading/preprocessing
    - Classifier transform support
    
    Args:
        vae: VAE model with encode/decode
        normalization: "celeba", "imagenet", or None
        device: Torch device
    """
    
    def __init__(
        self,
        vae: VAEProtocol,
        normalization: str = "celeba",
        device: torch.device | str = "cpu",
    ):
        super().__init__(vae=vae, device=device)
        
        self._normalization = normalization
        
        if normalization == "celeba":
            self._mean = CELEBA_MEAN
            self._std = CELEBA_STD
        elif normalization == "imagenet":
            self._mean = IMAGENET_MEAN
            self._std = IMAGENET_STD
        else:
            self._mean = [0.5, 0.5, 0.5]
            self._std = [0.5, 0.5, 0.5]
    
    @property
    def normalization(self) -> str:
        return self._normalization
    
    def unnormalize(self, img: torch.Tensor) -> np.ndarray:
        """Convert normalized image to [0, 1] for display."""
        arr = img.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
        
        mean = np.array(self._mean)
        std = np.array(self._std)
        arr = arr * std + mean
        
        return np.clip(arr, 0, 1)
    
    def get_classifier_transform(self, classifier_input_size: int = 224) -> transforms.Compose:
        """Get transform to prepare decoded images for classifier."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((classifier_input_size, classifier_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
