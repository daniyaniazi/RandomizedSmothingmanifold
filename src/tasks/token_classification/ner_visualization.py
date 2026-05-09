"""Token visualization for NER certification.

Creates visualizations showing:
    - Token with context window
    - Whitened noise samples (manifold-aware)
    - Isotropic noise samples
    - Nearest neighbor tokens
    - PCA embedding plots

Stores embeddings for later cross-experiment PCA comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..output_schema import TokenSampleData, save_sample_data


# Standard NER tag colors (CoNLL scheme)
NER_COLORS = {
    "O": "#CCCCCC",       # Grey
    "B-PER": "#E41A1C",   # Red
    "I-PER": "#E41A1C",
    "B-LOC": "#377EB8",   # Blue
    "I-LOC": "#377EB8",
    "B-ORG": "#4DAF4A",   # Green
    "I-ORG": "#4DAF4A",
    "B-MISC": "#984EA3",  # Purple
    "I-MISC": "#984EA3",
}


def create_token_context_figure(
    sample_data: TokenSampleData,
    output_path: Optional[Path] = None,
    context_window: int = 5,
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """Create visualization for a single token certification.
    
    Args:
        sample_data: TokenSampleData with embeddings and metadata
        output_path: Path to save figure
        context_window: Tokens on each side to show
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for visualization")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 2, 2]})
    
    # Row 0: Token in sentence context
    ax = axes[0]
    ax.axis("off")
    
    # Build context string with highlighted target
    if sample_data.context_tokens:
        context = sample_data.context_tokens
        target_idx = len(context) // 2 if context_window > 0 else 0
        
        text = ""
        for i, tok in enumerate(context):
            if i == target_idx:
                text += f" [{tok}] "
            else:
                text += f" {tok} "
        
        ax.text(0.5, 0.5, text.strip(), fontsize=14, ha="center", va="center",
                transform=ax.transAxes, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"))
    else:
        ax.text(0.5, 0.5, f"Token: '{sample_data.token_text}'", fontsize=14,
                ha="center", va="center", transform=ax.transAxes)
    
    # Add label info
    cert_status = "ABSTAIN" if sample_data.abstained else (
        "✓ Certified" if sample_data.predicted_label == sample_data.true_label else "✗ Wrong"
    )
    true_color = NER_COLORS.get(sample_data.true_label, "#333333")
    pred_color = NER_COLORS.get(sample_data.predicted_label, "#333333")
    
    info_text = (
        f"True: {sample_data.true_label}  |  "
        f"Pred: {sample_data.predicted_label}  |  "
        f"{cert_status}  |  "
        f"Radius: {sample_data.certified_radius:.4f}"
    )
    ax.set_title(info_text, fontsize=12, pad=10)
    
    # Row 1: Neighbor tokens
    ax = axes[1]
    ax.axis("off")
    
    if sample_data.neighbor_texts:
        ax.set_title("Nearest Neighbor Tokens", fontsize=12, fontweight="bold")
        n_neighbors = min(10, len(sample_data.neighbor_texts))
        
        for i in range(n_neighbors):
            neighbor_text = sample_data.neighbor_texts[i]
            neighbor_label = sample_data.neighbor_labels[i] if sample_data.neighbor_labels else "?"
            color = NER_COLORS.get(neighbor_label, "#CCCCCC")
            
            x = (i % 5) * 0.18 + 0.1
            y = 0.7 - (i // 5) * 0.35
            
            ax.text(x, y, f"'{neighbor_text}'", fontsize=11, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=to_rgba(color, 0.3),
                             edgecolor=color))
            ax.text(x, y - 0.12, neighbor_label, fontsize=9, ha="center", va="center",
                    color=color, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No neighbors available", fontsize=12, ha="center", va="center",
                transform=ax.transAxes, color="gray")
    
    # Row 2: Embedding distance summary
    ax = axes[2]
    
    if sample_data.original_hidden is not None and sample_data.neighbor_hiddens is not None:
        # Compute distances to neighbors
        original = sample_data.original_hidden.reshape(1, -1)
        neighbors = sample_data.neighbor_hiddens
        distances = np.linalg.norm(neighbors - original, axis=1)
        
        ax.bar(range(len(distances)), distances, color="steelblue", edgecolor="black")
        ax.set_xlabel("Neighbor Index", fontsize=10)
        ax.set_ylabel("L2 Distance to Target", fontsize=10)
        ax.set_title("Distance to Nearest Neighbors", fontsize=12, fontweight="bold")
        
        # Add threshold line if using manifold smoothing
        if sample_data.certified_radius > 0:
            ax.axhline(sample_data.certified_radius, color="red", linestyle="--",
                      label=f"Certified Radius: {sample_data.certified_radius:.4f}")
            ax.legend(fontsize=9)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Embedding distances not available", fontsize=12,
                ha="center", va="center", transform=ax.transAxes, color="gray")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_token_embedding_pca(
    samples: List[TokenSampleData],
    output_path: Optional[Path] = None,
    title: str = "Token Embeddings PCA",
    color_by: str = "true_label",  # or "predicted_label", "certified"
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Create 2D PCA plot of token embeddings.
    
    Args:
        samples: List of TokenSampleData
        output_path: Path to save figure
        title: Plot title
        color_by: How to color points
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Collect embeddings
    embeddings = []
    labels = []
    texts = []
    
    for sample in samples:
        if sample.original_hidden is not None:
            embeddings.append(sample.original_hidden)
            if color_by == "true_label":
                labels.append(sample.true_label)
            elif color_by == "predicted_label":
                labels.append(sample.predicted_label)
            elif color_by == "certified":
                labels.append("Certified" if not sample.abstained and sample.predicted_label == sample.true_label else "Not Certified")
            texts.append(sample.token_text)
    
    if len(embeddings) < 2:
        print("Not enough embeddings for PCA")
        return
    
    embeddings = np.stack(embeddings)
    
    # Fit PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = list(set(labels))
    colors = {label: NER_COLORS.get(label, plt.cm.tab10(i / len(unique_labels)))
              for i, label in enumerate(unique_labels)}
    
    for label in unique_labels:
        mask = [l == label for l in labels]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[colors[label]], label=label, alpha=0.6, s=50)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def create_noise_comparison_figure(
    sample_data: TokenSampleData,
    output_path: Optional[Path] = None,
    n_samples: int = 10,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Compare whitened vs isotropic noise in embedding space.
    
    Shows how the two noise types differ in terms of:
    - Direction of perturbation
    - Distance from original
    - Alignment with neighbor directions
    """
    if not HAS_MATPLOTLIB:
        return
    
    if sample_data.whitened_samples is None and sample_data.isotropic_samples is None:
        print("No noise samples available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    original = sample_data.original_hidden
    
    # PCA on all samples for visualization
    all_samples = [original]
    sample_labels = ["Original"]
    sample_colors = ["black"]
    
    if sample_data.whitened_samples is not None:
        for i, s in enumerate(sample_data.whitened_samples[:n_samples]):
            all_samples.append(s)
            sample_labels.append(f"Whitened {i}")
            sample_colors.append("blue")
    
    if sample_data.isotropic_samples is not None:
        for i, s in enumerate(sample_data.isotropic_samples[:n_samples]):
            all_samples.append(s)
            sample_labels.append(f"Isotropic {i}")
            sample_colors.append("red")
    
    if sample_data.neighbor_hiddens is not None:
        for i, n in enumerate(sample_data.neighbor_hiddens[:5]):
            all_samples.append(n)
            sample_labels.append(f"Neighbor {i}")
            sample_colors.append("green")
    
    all_samples = np.stack(all_samples)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_samples)
    
    # Left plot: 2D PCA
    ax = axes[0]
    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, c=sample_colors[i], s=100 if i == 0 else 50, 
                  marker="*" if i == 0 else "o", alpha=0.7)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    ax.set_title("Noise Samples in PCA Space", fontsize=12, fontweight="bold")
    
    # Legend
    handles = [
        mpatches.Patch(color="black", label="Original"),
        mpatches.Patch(color="blue", label="Whitened (Manifold)"),
        mpatches.Patch(color="red", label="Isotropic (Gaussian)"),
        mpatches.Patch(color="green", label="Neighbors"),
    ]
    ax.legend(handles=handles, fontsize=9)
    
    # Right plot: Distance distribution
    ax = axes[1]
    
    distances = {"Whitened": [], "Isotropic": []}
    
    if sample_data.whitened_samples is not None:
        for s in sample_data.whitened_samples[:n_samples]:
            distances["Whitened"].append(np.linalg.norm(s - original))
    
    if sample_data.isotropic_samples is not None:
        for s in sample_data.isotropic_samples[:n_samples]:
            distances["Isotropic"].append(np.linalg.norm(s - original))
    
    positions = []
    labels = []
    data = []
    
    if distances["Whitened"]:
        positions.append(1)
        labels.append("Whitened")
        data.append(distances["Whitened"])
    
    if distances["Isotropic"]:
        positions.append(2)
        labels.append("Isotropic")
        data.append(distances["Isotropic"])
    
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    colors_box = ["steelblue", "coral"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("L2 Distance from Original", fontsize=10)
    ax.set_title("Noise Magnitude Comparison", fontsize=12, fontweight="bold")
    
    fig.suptitle(f"Token: '{sample_data.token_text}'", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_ner_certification_batch(
    samples: List[TokenSampleData],
    output_dir: Path,
    n_samples_to_visualize: int = 20,
) -> None:
    """Visualize a batch of NER certification samples.
    
    Args:
        samples: List of TokenSampleData
        output_dir: Directory to save visualizations
        n_samples_to_visualize: Max samples to visualize
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir.parent / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples[:n_samples_to_visualize]):
        # Save token context figure
        create_token_context_figure(
            sample_data=sample,
            output_path=output_dir / f"token_{sample.sample_idx:04d}.png",
        )
        
        # Save noise comparison if available
        if sample.whitened_samples is not None or sample.isotropic_samples is not None:
            create_noise_comparison_figure(
                sample_data=sample,
                output_path=output_dir / f"token_{sample.sample_idx:04d}_noise.png",
            )
        
        # Save embeddings for later cross-experiment PCA
        save_sample_data(sample, embeddings_dir / f"token_{sample.sample_idx:04d}.npz")
    
    # Create overall PCA plot
    plot_token_embedding_pca(
        samples=samples,
        output_path=output_dir / "token_embeddings_pca.png",
        title="Token Embeddings by True Label",
    )


def generate_token_sample_data(
    token_text: str,
    original_hidden: np.ndarray,
    true_label: str,
    predicted_label: str,
    certified_radius: float,
    abstained: bool,
    sample_idx: int,
    context_tokens: Optional[List[str]] = None,
    neighbor_hiddens: Optional[np.ndarray] = None,
    neighbor_texts: Optional[List[str]] = None,
    neighbor_labels: Optional[List[str]] = None,
    sigma: float = 1.0,
    n_whitened_samples: int = 20,
    n_isotropic_samples: int = 20,
    pca_components: Optional[np.ndarray] = None,
    pca_mean: Optional[np.ndarray] = None,
    pca_evals: Optional[np.ndarray] = None,
) -> TokenSampleData:
    """Generate TokenSampleData with noise samples.
    
    Args:
        token_text: The token text
        original_hidden: Hidden state embedding
        true_label: Ground truth label
        predicted_label: Predicted label
        certified_radius: Certification radius
        abstained: Whether abstained
        sample_idx: Sample index
        context_tokens: Tokens in context window
        neighbor_hiddens: Neighbor embeddings
        neighbor_texts: Neighbor token texts
        neighbor_labels: Neighbor labels
        sigma: Noise standard deviation
        n_whitened_samples: Number of whitened samples
        n_isotropic_samples: Number of isotropic samples
        pca_components: PCA eigenvectors (for whitened noise)
        pca_mean: PCA mean
        pca_evals: PCA eigenvalues
        
    Returns:
        TokenSampleData with all fields populated
    """
    original = np.asarray(original_hidden, dtype=np.float32).reshape(-1)
    dim = len(original)
    
    # Generate isotropic samples
    isotropic_samples = original + np.random.randn(n_isotropic_samples, dim).astype(np.float32) * sigma
    
    # Generate whitened samples if PCA is provided
    whitened_samples = None
    if pca_components is not None and pca_mean is not None and pca_evals is not None:
        from src.smoothing.pca import LocalPCA, whiten, unwhiten
        pca = LocalPCA(mean=pca_mean, evecs=pca_components, evals=pca_evals)
        
        whitened_samples = []
        for _ in range(n_whitened_samples):
            w = whiten(original, pca)
            w_noisy = w + np.random.randn(len(w)).astype(np.float32) * sigma
            whitened_samples.append(unwhiten(w_noisy, pca))
        whitened_samples = np.stack(whitened_samples)
    
    return TokenSampleData(
        sample_idx=sample_idx,
        token_text=token_text,
        true_label=true_label,
        predicted_label=predicted_label,
        certified_radius=certified_radius,
        abstained=abstained,
        original_hidden=original,
        reconstructed_hidden=None,  # Can add if needed
        whitened_samples=whitened_samples,
        isotropic_samples=isotropic_samples,
        neighbor_hiddens=neighbor_hiddens,
        neighbor_texts=neighbor_texts,
        neighbor_labels=neighbor_labels,
        context_tokens=context_tokens,
    )
