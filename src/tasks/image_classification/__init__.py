"""Image classification certification task.

Supports certification in different spaces:
- Pixel space: Direct smoothing on image pixels
- Latent space: Smoothing in VAE latent space

Works with any image classification dataset (CelebA, ImageNet, etc.).
"""

from .spaces import ImagePixelSpace, ImageLatentSpace
from .certify import ImageCertifier, run_image_certification
from .visualize import visualize_certification_samples
from .jonas_visualization import (
    create_jonas_grid,
    generate_image_sample_data,
    visualize_certification_batch,
    plot_radius_distribution,
    plot_accuracy_by_class,
)

__all__ = [
    "ImagePixelSpace",
    "ImageLatentSpace",
    "ImageCertifier",
    "run_image_certification",
    "visualize_certification_samples",
    # Visualization helpers
    "create_jonas_grid",
    "generate_image_sample_data",
    "visualize_certification_batch",
    "plot_radius_distribution",
    "plot_accuracy_by_class",
]
