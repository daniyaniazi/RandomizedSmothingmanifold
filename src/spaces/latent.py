"""Latent space representation using VAE/autoencoder.

Encodes images through a VAE encoder to latent vectors z.
Decodes by passing z through the VAE decoder.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .base import Space, to_numpy, to_tensor


@runtime_checkable
class VAEProtocol(Protocol):
    """Protocol for VAE-like models with encode/decode methods."""
    
    latent_dim: int
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Encode image to latent. Returns (mu, logvar) or just (z,)."""
        ...
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        ...


class LatentSpace(Space[torch.Tensor]):
    """Latent space via VAE encoder/decoder.
    
    Encodes images through a VAE to get latent vectors z.
    Decodes by passing z through the VAE decoder.
    
    Args:
        vae: VAE model with encode() and decode() methods
        device: Torch device for computations
        
    Example:
        vae = load_vae_model()
        space = LatentSpace(vae, device="cuda")
        z = space.encode(img_tensor)   # (3,64,64) -> (latent_dim,)
        img = space.decode(z)          # (latent_dim,) -> (3,64,64)
    """
    
    def __init__(
        self,
        vae: VAEProtocol,
        device: torch.device | str = "cpu",
    ):
        self._vae = vae
        self._device = torch.device(device) if isinstance(device, str) else device
        self._latent_dim = vae.latent_dim
    
    @property
    def dim(self) -> int:
        return self._latent_dim
    
    @property
    def vae(self) -> VAEProtocol:
        """Access the underlying VAE model."""
        return self._vae
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> np.ndarray:
        """Encode image tensor to latent vector.
        
        Args:
            x: Image tensor of shape (C, H, W)
            
        Returns:
            Latent vector of shape (latent_dim,)
        """
        self._vae.eval()
        
        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self._device)
        
        # VAE encode returns (mu, logvar) or just (z,)
        result = self._vae.encode(x)
        if isinstance(result, tuple):
            z = result[0]  # Use mu (mean)
        else:
            z = result
        
        return to_numpy(z.squeeze(0))
    
    @torch.no_grad()
    def decode(self, z: np.ndarray) -> torch.Tensor:
        """Decode latent vector to image tensor.
        
        Args:
            z: Latent vector of shape (latent_dim,)
            
        Returns:
            Image tensor of shape (C, H, W)
        """
        self._vae.eval()
        
        z_tensor = to_tensor(z, device=self._device).unsqueeze(0)
        x_hat = self._vae.decode(z_tensor)
        
        return x_hat.squeeze(0).cpu()
    
    def encode_batch(self, imgs: torch.Tensor) -> np.ndarray:
        """Encode a batch of images.
        
        Args:
            imgs: Batch of images (B, C, H, W)
            
        Returns:
            Latent vectors (B, latent_dim)
        """
        self._vae.eval()
        
        with torch.no_grad():
            imgs = imgs.to(self._device)
            result = self._vae.encode(imgs)
            if isinstance(result, tuple):
                z = result[0]
            else:
                z = result
        
        return to_numpy(z)
    
    def decode_batch(self, zs: np.ndarray) -> torch.Tensor:
        """Decode a batch of latent vectors.
        
        Args:
            zs: Latent vectors (B, latent_dim)
            
        Returns:
            Batch of images (B, C, H, W)
        """
        self._vae.eval()
        
        with torch.no_grad():
            z_tensor = to_tensor(zs, device=self._device)
            x_hat = self._vae.decode(z_tensor)
        
        return x_hat.cpu()
