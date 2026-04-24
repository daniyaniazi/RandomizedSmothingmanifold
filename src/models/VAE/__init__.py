"""
VAE package — public API.
"""

from .model import ConvVAE, vae_loss
from .trainer import train_vae, save_checkpoint, load_checkpoint
from .data import build_dataset, build_transform, denormalize, tensor_to_hwc
from .latent import (
    encode_dataset_mu,
    build_or_load_ann,
    local_latent_pca,
    manifold_noise_sample,
    isotropic_noise_sample,
    combined_covariance,
    whitening_transform,
    certified_radius_whitened,
    plot_latent_comparison,
)

__all__ = [
    "ConvVAE",
    "vae_loss",
    "train_vae",
    "save_checkpoint",
    "load_checkpoint",
    "build_dataset",
    "build_transform",
    "denormalize",
    "tensor_to_hwc",
    "encode_dataset_mu",
    "build_or_load_ann",
    "local_latent_pca",
    "manifold_noise_sample",
    "isotropic_noise_sample",
    "combined_covariance",
    "whitening_transform",
    "certified_radius_whitened",
    "plot_latent_comparison",
]
