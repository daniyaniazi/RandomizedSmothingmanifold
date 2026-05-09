"""Visualization utilities for image classification certification."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.smoothing import Smoother
from src.spaces import Space


def visualize_certification_samples(
    image: torch.Tensor,
    space: Space[torch.Tensor],
    smoother: Smoother,
    label: int,
    pred: int,
    radius: float,
    n_samples: int = 5,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> None:
    """Visualize certification samples for a single image.
    
    Creates a grid showing:
    - Original image
    - n_samples of noisy/smoothed images
    
    Args:
        image: Original image tensor (C, H, W)
        space: Space for encoding/decoding
        smoother: Smoother for generating samples
        label: True label
        pred: Predicted label
        radius: Certified radius
        n_samples: Number of noisy samples to show
        output_path: Path to save figure (optional)
        title: Custom title (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    # Encode anchor
    anchor = space.encode(image)
    
    # Generate noisy samples
    noisy_images = []
    for _ in range(n_samples):
        noisy_vec = smoother.sample(anchor)
        noisy_img = space.decode(noisy_vec)
        noisy_images.append(noisy_img)
    
    # Create figure
    fig, axes = plt.subplots(1, n_samples + 1, figsize=(3 * (n_samples + 1), 3))
    
    # Helper to display image
    def show_img(ax, img_tensor, img_title):
        if hasattr(space, 'unnormalize'):
            img_np = space.unnormalize(img_tensor)
        else:
            img_np = img_tensor.cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np, 0, 1)
        
        ax.imshow(img_np)
        ax.set_title(img_title, fontsize=10)
        ax.axis("off")
    
    # Original
    show_img(axes[0], image, f"Original\nLabel: {label}")
    
    # Noisy samples
    for i, noisy_img in enumerate(noisy_images):
        show_img(axes[i + 1], noisy_img, f"Sample {i + 1}")
    
    # Title
    correct_str = "✓" if pred == label else "✗"
    default_title = f"Pred: {pred} {correct_str} | Radius: {radius:.4f}"
    fig.suptitle(title or default_title, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_radius_histogram(
    radii: list[float],
    sigma: float,
    output_path: Optional[Path] = None,
    title: str = "Certified Radius Distribution",
) -> None:
    """Plot histogram of certified radii.
    
    Args:
        radii: List of certified radii
        sigma: Noise sigma used
        output_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(radii, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(radii), color="red", linestyle="--", label=f"Mean: {np.mean(radii):.4f}")
    ax.axvline(np.median(radii), color="green", linestyle="--", label=f"Median: {np.median(radii):.4f}")
    
    ax.set_xlabel("Certified Radius")
    ax.set_ylabel("Count")
    ax.set_title(f"{title} (σ = {sigma})")
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
