"""Token classification certification task (NER, POS tagging, etc.).

Supports certification of transformer-based token classifiers
using smoothing in the hidden state space.
"""

from .spaces import TokenHiddenStateSpace
from .certify import TokenCertifier, run_token_certification
from .visualize import visualize_token_samples
from .ner_visualization import (
    create_token_context_figure,
    plot_token_embedding_pca,
    create_noise_comparison_figure,
    visualize_ner_certification_batch,
    generate_token_sample_data,
    NER_COLORS,
)

__all__ = [
    "TokenHiddenStateSpace",
    "TokenCertifier",
    "run_token_certification",
    "visualize_token_samples",
    # NER visualization
    "create_token_context_figure",
    "plot_token_embedding_pca",
    "create_noise_comparison_figure",
    "visualize_ner_certification_batch",
    "generate_token_sample_data",
    "NER_COLORS",
]
