"""Visualization utilities for token classification certification."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.smoothing import Smoother
from .spaces import TokenHiddenStateSpace


def visualize_token_samples(
    token_text: str,
    anchor_vector: np.ndarray,
    smoother: Smoother,
    neighbor_texts: list[str],
    neighbor_labels: list[int],
    n_samples: int = 5,
    label_names: Optional[dict[int, str]] = None,
) -> None:
    """Visualize smoothing samples for a single token.
    
    Shows:
    - Original token and its neighbors
    - Noisy sample distances
    - Neighbor label distribution
    
    Args:
        token_text: Text of the original token
        anchor_vector: Hidden state vector
        smoother: Smoother for generating samples
        neighbor_texts: Text of k nearest neighbors
        neighbor_labels: Labels of k nearest neighbors
        n_samples: Number of noisy samples to show
        label_names: Optional mapping from label ID to name
    """
    print(f"Token: '{token_text}'")
    print(f"Anchor norm: {np.linalg.norm(anchor_vector):.4f}")
    print()
    
    # Show neighbors
    print("Nearest neighbors:")
    for i, (text, label) in enumerate(zip(neighbor_texts[:10], neighbor_labels[:10])):
        label_str = label_names.get(label, str(label)) if label_names else str(label)
        print(f"  {i+1}. '{text}' [{label_str}]")
    print()
    
    # Show noisy samples
    print(f"Noisy samples (σ={smoother.sigma}):")
    for i in range(n_samples):
        noisy = smoother.sample(anchor_vector)
        dist = np.linalg.norm(noisy - anchor_vector)
        print(f"  Sample {i+1}: L2 distance = {dist:.4f}")
    print()
    
    # Label distribution in neighbors
    label_counts = {}
    for label in neighbor_labels:
        label_str = label_names.get(label, str(label)) if label_names else str(label)
        label_counts[label_str] = label_counts.get(label_str, 0) + 1
    
    print("Neighbor label distribution:")
    for label_str, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(neighbor_labels) * 100
        print(f"  {label_str}: {count} ({pct:.1f}%)")


def plot_certification_by_label(
    label_metrics: dict[str, dict],
    output_path: Optional[Path] = None,
    title: str = "Certification by Label",
) -> None:
    """Plot certification metrics by label.
    
    Args:
        label_metrics: Dict mapping label name to metrics dict
                      Each metrics dict has: count, certified_acc, mean_radius
        output_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    labels = list(label_metrics.keys())
    counts = [m.get("count", 0) for m in label_metrics.values()]
    cert_accs = [m.get("certified_accuracy", 0) * 100 for m in label_metrics.values()]
    radii = [m.get("mean_radius", 0) for m in label_metrics.values()]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Token counts
    axes[0].bar(labels, counts, color="steelblue")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Token Count")
    axes[0].set_title("Token Distribution")
    axes[0].tick_params(axis="x", rotation=45)
    
    # Certified accuracy
    axes[1].bar(labels, cert_accs, color="green")
    axes[1].set_xlabel("Label")
    axes[1].set_ylabel("Certified Accuracy (%)")
    axes[1].set_title("Certified Accuracy by Label")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(0, 100)
    
    # Mean radius
    axes[2].bar(labels, radii, color="orange")
    axes[2].set_xlabel("Label")
    axes[2].set_ylabel("Mean Certified Radius")
    axes[2].set_title("Mean Radius by Label")
    axes[2].tick_params(axis="x", rotation=45)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
