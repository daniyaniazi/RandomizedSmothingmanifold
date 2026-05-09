"""Representation spaces for smoothing.

A Space defines how to encode/decode data to/from a vector representation
where smoothing operations (isotropic or manifold) can be applied.

Supported spaces:
- PixelSpace: Flattened image pixels (H*W*C)
- LatentSpace: VAE/autoencoder latent vectors
- HiddenStateSpace: Transformer layer embeddings
- (future) ConceptSpace: Concept bottleneck representations
"""

from .base import Space
from .pixel import PixelSpace
from .latent import LatentSpace
from .hidden_state import HiddenStateSpace

__all__ = [
    "Space",
    "PixelSpace",
    "LatentSpace",
    "HiddenStateSpace",
]
