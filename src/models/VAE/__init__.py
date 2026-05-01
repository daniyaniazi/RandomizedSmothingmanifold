"""
VAE package — public API.
"""

from .model import ConvVAE, vae_loss
from .trainer import train_vae, save_checkpoint, load_checkpoint

# Optional utilities are exported only when present in the workspace.
try:
    from .data import DEFAULT_MEAN, DEFAULT_STD, build_dataset, build_transform, denormalize, tensor_to_hwc
except ImportError:
    DEFAULT_MEAN = None
    DEFAULT_STD = None

try:
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
except ImportError:
    pass

__all__ = [
    "ConvVAE",
    "vae_loss",
    "train_vae",
    "save_checkpoint",
    "load_checkpoint",
]

for _name in [
    "build_dataset",
    "build_transform",
    "DEFAULT_MEAN",
    "DEFAULT_STD",
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
]:
    if _name in globals():
        __all__.append(_name)
