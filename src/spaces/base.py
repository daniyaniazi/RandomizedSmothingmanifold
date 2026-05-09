"""Abstract base class for representation spaces.

A Space encapsulates:
1. How to convert input data to a flat vector (encode)
2. How to convert a vector back to the original domain (decode)
3. The dimensionality of the vector space

This abstraction allows smoothing algorithms to work uniformly across
different representations (pixels, latents, hidden states, concepts).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
import torch

# Type variable for the original data type (image tensor, token embedding, etc.)
T = TypeVar("T")


class Space(ABC, Generic[T]):
    """Abstract representation space for smoothing.
    
    A Space maps between the original data domain T and a flat numpy vector
    where smoothing operations can be applied.
    
    Example:
        space = PixelSpace(image_size=64)
        anchor = space.encode(img_tensor)  # (C,H,W) -> (C*H*W,)
        noisy = smoother.sample(anchor, sigma=0.5)
        img_noisy = space.decode(noisy)    # (C*H*W,) -> (C,H,W)
    """
    
    @abstractmethod
    def encode(self, x: T) -> np.ndarray:
        """Encode input to a flat vector in this space.
        
        Args:
            x: Input in the original domain (image, token embedding, etc.)
            
        Returns:
            Flat numpy array of shape (dim,)
        """
        pass
    
    @abstractmethod
    def decode(self, z: np.ndarray) -> T:
        """Decode a vector back to the original domain.
        
        Args:
            z: Flat numpy array of shape (dim,)
            
        Returns:
            Data in the original domain
        """
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the vector space."""
        pass
    
    def encode_batch(self, xs: list[T]) -> np.ndarray:
        """Encode a batch of inputs.
        
        Args:
            xs: List of inputs in the original domain
            
        Returns:
            Numpy array of shape (batch_size, dim)
        """
        return np.stack([self.encode(x) for x in xs])
    
    def decode_batch(self, zs: np.ndarray) -> list[T]:
        """Decode a batch of vectors.
        
        Args:
            zs: Numpy array of shape (batch_size, dim)
            
        Returns:
            List of data in the original domain
        """
        return [self.decode(z) for z in zs]


def to_numpy(x: torch.Tensor | np.ndarray | list) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    if isinstance(x, list):
        return np.array(x, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def to_tensor(x: np.ndarray, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(x).to(device=device, dtype=dtype)
