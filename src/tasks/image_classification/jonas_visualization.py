"""Visualization utilities for image certification.

Creates a standardized grid showing:
    Row 0: Original | PCA Reconstruction
    Row 1: Samples with noise in WHITENED space (manifold-aware)
    Row 2: Samples with noise in ORIGINAL space (isotropic Gaussian)  
    Row 3: k Nearest Neighbors from training set

This allows direct comparison of manifold vs isotropic smoothing effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.smoothing.pca import LocalPCA, fit_local_pca, whiten, unwhiten
from src.indexing.base import NeighborIndex, neighbor_vectors

from ..output_schema import ImageSampleData, save_sample_data, ExperimentPaths


def create_jonas_grid(
    sample_data: ImageSampleData,
    image_shape: Tuple[int, int, int],
    unnormalize_fn: callable,
    output_path: Optional[Path] = None,
    n_whitened_samples: int = 5,
    n_isotropic_samples: int = 5,
    n_neighbors: int = 5,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 16),
) -> None:
    """Create visualization grid for a single image.
    
    Args:
        sample_data: ImageSampleData with all vectors
        image_shape: (C, H, W) shape for reshaping vectors
        unnormalize_fn: Function to convert normalized tensor to displayable image
        output_path: Path to save figure (shows if None)
        n_whitened_samples: Number of whitened noise samples to show
        n_isotropic_samples: Number of isotropic noise samples to show
        n_neighbors: Number of neighbors to show
        title: Optional title override
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for visualization")
        return
    
    n_cols = max(n_whitened_samples, n_isotropic_samples, n_neighbors, 2)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, n_cols, figure=fig, hspace=0.3, wspace=0.1)
    
    def vec_to_img(vec):
        """Convert flat vector to displayable image."""
        img = vec.reshape(image_shape)
        return unnormalize_fn(img)
    
    # Row 0: Original and PCA Reconstruction
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(vec_to_img(sample_data.original_vector))
    ax.set_title("Original", fontsize=12, fontweight="bold")
    ax.axis("off")
    
    if sample_data.reconstructed_vector is not None:
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(vec_to_img(sample_data.reconstructed_vector))
        ax.set_title("PCA Reconstruction", fontsize=12)
        ax.axis("off")
    
    # Row 1: Samples with noise in WHITENED space
    if sample_data.whitened_samples is not None:
        for i in range(min(n_whitened_samples, len(sample_data.whitened_samples))):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(vec_to_img(sample_data.whitened_samples[i]))
            if i == 0:
                ax.set_title("Samples with noise in WHITENED space\n(manifold-aware)", fontsize=10)
            ax.axis("off")
    
    # Row 2: Samples with noise in ORIGINAL space (isotropic)
    if sample_data.isotropic_samples is not None:
        for i in range(min(n_isotropic_samples, len(sample_data.isotropic_samples))):
            ax = fig.add_subplot(gs[2, i])
            ax.imshow(vec_to_img(sample_data.isotropic_samples[i]))
            if i == 0:
                ax.set_title("Samples with noise in ORIGINAL space\n(isotropic Gaussian)", fontsize=10)
            ax.axis("off")
    
    # Row 3: Nearest Neighbors
    if sample_data.neighbor_vectors is not None:
        for i in range(min(n_neighbors, len(sample_data.neighbor_vectors))):
            ax = fig.add_subplot(gs[3, i])
            ax.imshow(vec_to_img(sample_data.neighbor_vectors[i]))
            label_str = ""
            if sample_data.neighbor_labels is not None and i < len(sample_data.neighbor_labels):
                label_str = f"\nLabel: {sample_data.neighbor_labels[i]}"
            if i == 0:
                ax.set_title(f"Nearest Neighbors{label_str}", fontsize=10)
            else:
                ax.set_title(label_str, fontsize=9)
            ax.axis("off")
    
    # Main title
    cert_status = "ABSTAIN" if sample_data.abstained else ("✓" if sample_data.predicted_label == sample_data.true_label else "✗")
    default_title = (
        f"Sample {sample_data.sample_idx} | "
        f"True: {sample_data.true_label} | "
        f"Pred: {sample_data.predicted_label} {cert_status} | "
        f"Radius: {sample_data.certified_radius:.4f}"
    )
    fig.suptitle(title or default_title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def generate_image_sample_data(
    original_vector: np.ndarray,
    true_label: int,
    predicted_label: int,
    certified_radius: float,
    abstained: bool,
    sample_idx: int,
    index: NeighborIndex,
    sigma: float,
    knn_k: int = 32,
    n_whitened_samples: int = 20,
    n_isotropic_samples: int = 20,
    eps_eig: float = 1e-6,
) -> ImageSampleData:
    """Generate all data needed for visualization.
    
    Args:
        original_vector: Flattened image vector
        true_label: Ground truth label
        predicted_label: Certified prediction
        certified_radius: Certified radius
        abstained: Whether certification abstained
        sample_idx: Sample index
        index: kNN index for neighbor lookup
        sigma: Noise standard deviation
        knn_k: Number of neighbors for PCA
        n_whitened_samples: Number of whitened noise samples
        n_isotropic_samples: Number of isotropic noise samples
        eps_eig: Minimum eigenvalue for PCA
        
    Returns:
        ImageSampleData with all vectors populated
    """
    original = np.asarray(original_vector, dtype=np.float32).reshape(-1)
    
    # Get neighbors
    neighbors = neighbor_vectors(index, k=knn_k + 1, vector=original)
    neighbors = neighbors[1:]  # Skip self-match
    
    # Fit local PCA
    pca = fit_local_pca(neighbors, eps_eig=eps_eig)
    
    # Reconstruct through PCA
    reconstructed = unwhiten(whiten(original, pca), pca)
    
    # Generate whitened samples (manifold-aware)
    whitened_samples = []
    for _ in range(n_whitened_samples):
        w = whiten(original, pca)
        w_noisy = w + np.random.randn(len(w)).astype(np.float32) * sigma
        whitened_samples.append(unwhiten(w_noisy, pca))
    whitened_samples = np.stack(whitened_samples)
    
    # Generate isotropic samples
    isotropic_samples = []
    for _ in range(n_isotropic_samples):
        noisy = original + np.random.randn(len(original)).astype(np.float32) * sigma
        isotropic_samples.append(noisy)
    isotropic_samples = np.stack(isotropic_samples)
    
    return ImageSampleData(
        sample_idx=sample_idx,
        true_label=true_label,
        predicted_label=predicted_label,
        certified_radius=certified_radius,
        abstained=abstained,
        original_vector=original,
        reconstructed_vector=reconstructed,
        whitened_samples=whitened_samples,
        isotropic_samples=isotropic_samples,
        neighbor_vectors=neighbors[:5],  # Store top 5 for viz
        neighbor_labels=None,  # Can be populated if available
        pca_mean=pca.mean,
        pca_evals=pca.evals,
        pca_evecs=pca.evecs,
    )


def visualize_certification_batch(
    samples: List[ImageSampleData],
    image_shape: Tuple[int, int, int],
    unnormalize_fn: callable,
    output_dir: Path,
    n_samples_to_visualize: int = 20,
) -> None:
    """Visualize a batch of certification samples.
    
    Args:
        samples: List of ImageSampleData
        image_shape: (C, H, W) shape
        unnormalize_fn: Unnormalization function
        output_dir: Directory to save visualizations
        n_samples_to_visualize: Max samples to visualize
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples[:n_samples_to_visualize]):
        # Save visualization
        create_jonas_grid(
            sample_data=sample,
            image_shape=image_shape,
            unnormalize_fn=unnormalize_fn,
            output_path=output_dir / f"sample_{sample.sample_idx:04d}.png",
        )
        
        # Save embeddings for later PCA visualization
        save_sample_data(sample, output_dir.parent / "embeddings" / f"sample_{sample.sample_idx:04d}.npz")


def plot_radius_distribution(
    radii: List[float],
    sigma: float,
    output_path: Optional[Path] = None,
    title: str = "Certified Radius Distribution",
) -> None:
    """Plot histogram of certified radii."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out zeros (abstained)
    valid_radii = [r for r in radii if r > 0]
    
    ax.hist(valid_radii, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(np.mean(valid_radii), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(valid_radii):.4f}")
    ax.axvline(np.median(valid_radii), color="green", linestyle="--", linewidth=2,
               label=f"Median: {np.median(valid_radii):.4f}")
    
    ax.set_xlabel("Certified Radius", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{title} (σ = {sigma})", fontsize=14)
    ax.legend(fontsize=10)
    
    # Add summary stats
    stats_text = (
        f"N = {len(valid_radii)}\n"
        f"Mean = {np.mean(valid_radii):.4f}\n"
        f"Std = {np.std(valid_radii):.4f}\n"
        f"Min = {np.min(valid_radii):.4f}\n"
        f"Max = {np.max(valid_radii):.4f}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_accuracy_by_class(
    class_metrics: dict,
    output_path: Optional[Path] = None,
    title: str = "Certified Accuracy by Class",
) -> None:
    """Plot certified accuracy breakdown by class."""
    if not HAS_MATPLOTLIB:
        return
    
    classes = list(class_metrics.keys())
    accuracies = [m.get("certified_accuracy", 0) * 100 for m in class_metrics.values()]
    counts = [m.get("total_samples", 0) for m in class_metrics.values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy bar chart
    bars = axes[0].bar(classes, accuracies, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Class", fontsize=12)
    axes[0].set_ylabel("Certified Accuracy (%)", fontsize=12)
    axes[0].set_title("Certified Accuracy by Class", fontsize=14)
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{acc:.1f}%", ha="center", fontsize=9)
    
    # Sample count bar chart
    bars = axes[1].bar(classes, counts, color="orange", edgecolor="black")
    axes[1].set_xlabel("Class", fontsize=12)
    axes[1].set_ylabel("Sample Count", fontsize=12)
    axes[1].set_title("Sample Distribution by Class", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
