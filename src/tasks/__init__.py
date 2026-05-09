"""Task-specific certification workflows.

This module provides task-specific wrappers that combine:
- A Space (pixel, latent, hidden_state)
- A Smoother (isotropic, manifold)
- A Certifier
- Task-specific data loading and visualization

Supported tasks:
- image_classification: CelebA smile detection, ImageNet, etc.
- token_classification: NER, POS tagging, etc.
- (future) segmentation: Semantic segmentation certification
"""

from . import image_classification
from . import token_classification

from .output_schema import (
    ExperimentPaths,
    build_experiment_path,
    CertificationMetricsSummary,
    PerClassMetrics,
    PerClassMetricsCollection,
    ImageSampleData,
    TokenSampleData,
    save_sample_data,
    load_sample_data,
)

from .metrics_output import (
    compute_metrics_summary,
    compute_per_class_metrics,
    save_metrics_json,
    save_metrics_csv,
    format_metrics_markdown,
    format_metrics_console,
    load_experiment_metrics,
    compare_experiments,
)

from .experiment_comparison import (
    ExperimentConfig,
    parse_experiment_name,
    discover_experiments,
    compare_smoothing_modes,
    compare_layers,
    compare_sigma_sensitivity,
    generate_comparison_report,
)

__all__ = [
    "image_classification",
    "token_classification",
    # Output schema
    "ExperimentPaths",
    "build_experiment_path",
    "CertificationMetricsSummary",
    "PerClassMetrics",
    "PerClassMetricsCollection",
    "ImageSampleData",
    "TokenSampleData",
    "save_sample_data",
    "load_sample_data",
    # Metrics output
    "compute_metrics_summary",
    "compute_per_class_metrics",
    "save_metrics_json",
    "save_metrics_csv",
    "format_metrics_markdown",
    "format_metrics_console",
    "load_experiment_metrics",
    "compare_experiments",
    # Experiment comparison
    "ExperimentConfig",
    "parse_experiment_name",
    "discover_experiments",
    "compare_smoothing_modes",
    "compare_layers",
    "compare_sigma_sensitivity",
    "generate_comparison_report",
]
