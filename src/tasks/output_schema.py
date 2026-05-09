"""Standardized output schema for certification experiments.

Defines directory structure, file formats, and metadata for:
- Image classification (CelebA, ImageNet, etc.)
- Token classification (NER, POS, etc.)

Output Structure:
    output/{task}/{dataset}/
    ├── experiments/
    │   └── {experiment_name}/
    │       ├── config.yaml                    # Experiment configuration
    │       ├── metrics_summary.json           # Quick overview metrics
    │       ├── certification_results.json     # Full certification results
    │       ├── per_class_metrics.json         # Breakdown by class/entity
    │       ├── samples/                       # Sample visualizations
    │       │   ├── sample_0000.png           # Jonas-style grid per sample
    │       │   ├── sample_0000.npz           # Stored embeddings for PCA
    │       │   └── ...
    │       └── comparisons/                   # Cross-experiment comparisons
    │           └── radius_by_class.png
    │
    └── index/                                 # Shared kNN indices
        └── {space}/{metric}/
            └── index.ann
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


# =============================================================================
# Output Directory Structure
# =============================================================================

@dataclass
class ExperimentPaths:
    """Standardized paths for a single experiment."""
    
    root: Path
    
    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"
    
    @property
    def metrics_summary_path(self) -> Path:
        return self.root / "metrics_summary.json"
    
    @property
    def certification_results_path(self) -> Path:
        return self.root / "certification_results.json"
    
    @property
    def per_class_metrics_path(self) -> Path:
        return self.root / "per_class_metrics.json"
    
    @property
    def samples_dir(self) -> Path:
        return self.root / "samples"
    
    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings"
    
    @property
    def comparisons_dir(self) -> Path:
        return self.root / "comparisons"
    
    def ensure_dirs(self) -> None:
        """Create all directories."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        self.comparisons_dir.mkdir(exist_ok=True)
    
    def sample_image_path(self, idx: int) -> Path:
        return self.samples_dir / f"sample_{idx:04d}.png"
    
    def sample_embedding_path(self, idx: int) -> Path:
        return self.embeddings_dir / f"sample_{idx:04d}.npz"


def build_experiment_path(
    output_root: Path,
    task: str,
    dataset: str,
    smoothing_mode: str,
    space: str,
    sigma: float,
    layer: Optional[str] = None,
    masking_mode: Optional[str] = None,
) -> ExperimentPaths:
    """Build standardized experiment path.
    
    Args:
        output_root: Base output directory
        task: "image_classification" or "token_classification"
        dataset: "celeba", "conll2003", etc.
        smoothing_mode: "manifold" or "isotropic"
        space: "pixel", "latent", "hidden_state"
        sigma: Noise standard deviation
        layer: Layer identifier (e.g., "last", "layer_6")
        masking_mode: Optional masking mode (e.g., "context", "entity")
        
    Returns:
        ExperimentPaths object
    """
    sigma_str = f"sigma_{sigma:.2f}".replace(".", "_")
    
    parts = [smoothing_mode, space]
    if layer:
        parts.append(layer)
    if masking_mode:
        parts.append(f"mask_{masking_mode}")
    parts.append(sigma_str)
    
    experiment_name = "_".join(parts)
    
    path = output_root / task / dataset / "experiments" / experiment_name
    return ExperimentPaths(root=path)


# =============================================================================
# Metrics Data Classes
# =============================================================================

@dataclass
class CertificationMetricsSummary:
    """Quick overview metrics for an experiment."""
    
    # Experiment info
    experiment_name: str
    task: str
    dataset: str
    smoothing_mode: str
    space: str
    sigma: float
    n_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Overall metrics
    total_samples: int = 0
    certified_samples: int = 0
    abstained_samples: int = 0
    
    certified_accuracy: float = 0.0
    clean_accuracy: float = 0.0
    abstention_rate: float = 0.0
    
    mean_radius: float = 0.0
    median_radius: float = 0.0
    std_radius: float = 0.0
    min_radius: float = 0.0
    max_radius: float = 0.0
    
    # Optional: layer/masking info
    layer: Optional[str] = None
    masking_mode: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "CertificationMetricsSummary":
        data = json.loads(path.read_text())
        return cls(**data)


@dataclass 
class PerClassMetrics:
    """Metrics breakdown by class/entity/label."""
    
    class_name: str
    class_id: int
    
    total_samples: int = 0
    certified_samples: int = 0
    abstained_samples: int = 0
    certified_correct: int = 0
    
    certified_accuracy: float = 0.0
    abstention_rate: float = 0.0
    mean_radius: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerClassMetricsCollection:
    """Collection of per-class metrics."""
    
    experiment_name: str
    by_class: Dict[str, PerClassMetrics] = field(default_factory=dict)
    
    # For NER: also by entity type (O, PER, ORG, LOC, MISC)
    by_entity: Optional[Dict[str, PerClassMetrics]] = None
    
    def add_class(self, metrics: PerClassMetrics) -> None:
        self.by_class[metrics.class_name] = metrics
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "experiment_name": self.experiment_name,
            "by_class": {k: v.to_dict() for k, v in self.by_class.items()},
        }
        if self.by_entity:
            result["by_entity"] = {k: v.to_dict() for k, v in self.by_entity.items()}
        return result
    
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


# =============================================================================
# Sample Storage for Visualization
# =============================================================================

@dataclass
class ImageSampleData:
    """Data for a single image sample visualization.
    
    Stores everything needed for Jonas-style visualization:
    - Original image
    - PCA reconstruction
    - Samples with noise in whitened space
    - Samples with noise in original space (isotropic)
    - Nearest neighbors
    """
    
    sample_idx: int
    true_label: int
    predicted_label: int
    certified_radius: float
    abstained: bool
    
    # For storage in .npz
    original_vector: Any = None          # Flattened image or latent
    reconstructed_vector: Any = None     # PCA reconstruction
    whitened_samples: Any = None         # (n_samples, dim)
    isotropic_samples: Any = None        # (n_samples, dim)
    neighbor_vectors: Any = None         # (k, dim)
    neighbor_labels: Any = None          # (k,)
    
    # PCA info for later visualization
    pca_mean: Any = None
    pca_evals: Any = None
    pca_evecs: Any = None


@dataclass
class TokenSampleData:
    """Data for a single token sample visualization.
    
    Stores everything needed for token PCA visualization:
    - Original token text and hidden state
    - Reconstructed hidden state
    - Samples in whitened space
    - Samples in original hidden space
    - Neighbor tokens and their hidden states
    """
    
    sample_idx: int
    token_idx: int
    token_text: str
    true_label: int
    true_label_name: str
    predicted_label: int
    predicted_label_name: str
    certified_radius: float
    abstained: bool
    
    # Embeddings for visualization
    original_hidden: Any = None          # (hidden_dim,)
    reconstructed_hidden: Any = None     # (hidden_dim,)
    whitened_samples: Any = None         # (n_samples, hidden_dim)
    isotropic_samples: Any = None        # (n_samples, hidden_dim)
    neighbor_hiddens: Any = None         # (k, hidden_dim)
    neighbor_texts: Any = None           # List of k strings
    neighbor_labels: Any = None          # (k,)
    
    # PCA info
    pca_mean: Any = None
    pca_evals: Any = None
    pca_evecs: Any = None


def save_sample_data(data: ImageSampleData | TokenSampleData, path: Path) -> None:
    """Save sample data to .npz file for later visualization."""
    import numpy as np
    
    # Convert dataclass to dict, handling None values
    save_dict = {}
    for key, value in asdict(data).items():
        if value is not None:
            if isinstance(value, (list, str, int, float, bool)):
                save_dict[key] = np.array(value) if isinstance(value, list) else value
            else:
                save_dict[key] = value
    
    np.savez(path, **save_dict)


def load_sample_data(path: Path) -> Dict[str, Any]:
    """Load sample data from .npz file."""
    import numpy as np
    
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}
