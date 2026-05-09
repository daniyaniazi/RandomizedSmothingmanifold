"""Pixel space representation for images.

Encodes images as flattened pixel vectors (C*H*W).
Useful for direct pixel-space smoothing on images.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .base import Space, to_numpy, to_tensor


class PixelSpace(Space[torch.Tensor]):
    """Pixel space: flattened image tensors.
    
    Encodes image tensors (C, H, W) as flat vectors (C*H*W,).
    Optionally applies normalization during encode/decode.
    
    Args:
        image_size: Height/width of square images (or tuple for non-square)
        channels: Number of channels (default: 3 for RGB)
        mean: Per-channel mean for normalization (optional)
        std: Per-channel std for normalization (optional)
        
    Example:
        space = PixelSpace(image_size=64, mean=[0.5]*3, std=[0.5]*3)
        vec = space.encode(img_tensor)  # (3,64,64) -> (12288,)
        img = space.decode(vec)          # (12288,) -> (3,64,64)
    """
    
    def __init__(
        self,
        image_size: int | Tuple[int, int],
        channels: int = 3,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        device: torch.device | str = "cpu",
    ):
        if isinstance(image_size, int):
            self._height = image_size
            self._width = image_size
        else:
            self._height, self._width = image_size
        
        self._channels = channels
        self._mean = np.array(mean, dtype=np.float32) if mean else None
        self._std = np.array(std, dtype=np.float32) if std else None
        self._device = device
        self._shape = (channels, self._height, self._width)
    
    @property
    def dim(self) -> int:
        return self._channels * self._height * self._width
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Original tensor shape (C, H, W)."""
        return self._shape
    
    def encode(self, x: torch.Tensor) -> np.ndarray:
        """Encode image tensor to flat vector.
        
        Args:
            x: Image tensor of shape (C, H, W), assumed already normalized
            
        Returns:
            Flat numpy array of shape (C*H*W,)
        """
        return to_numpy(x).reshape(-1)
    
    def decode(self, z: np.ndarray) -> torch.Tensor:
        """Decode flat vector to image tensor.
        
        Args:
            z: Flat numpy array of shape (C*H*W,)
            
        Returns:
            Image tensor of shape (C, H, W)
        """
        arr = z.reshape(self._shape)
        return to_tensor(arr, device=self._device)
    
    def unnormalize(self, img: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert normalized image to [0, 1] range for display.
        
        Args:
            img: Normalized image (C, H, W)
            
        Returns:
            Unnormalized image (H, W, C) in [0, 1] range
        """
        arr = to_numpy(img)
        if arr.ndim == 3 and arr.shape[0] == self._channels:
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
        
        if self._mean is not None and self._std is not None:
            arr = arr * self._std + self._mean
        
        return np.clip(arr, 0, 1)
    
    def normalize(self, img: np.ndarray) -> torch.Tensor:
        """Convert [0, 1] image to normalized tensor.
        
        Args:
            img: Image (H, W, C) in [0, 1] range
            
        Returns:
            Normalized tensor (C, H, W)
        """
        arr = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        if self._mean is not None and self._std is not None:
            arr = (arr - self._mean[:, None, None]) / self._std[:, None, None]
        
        return to_tensor(arr.astype(np.float32), device=self._device)
