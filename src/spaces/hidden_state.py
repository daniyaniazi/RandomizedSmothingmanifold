"""Hidden state space for transformer models.

Encodes/decodes token embeddings from transformer hidden layers.
Used for smoothing in the hidden state space of BERT, GPT, etc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import Space, to_numpy, to_tensor


class HiddenStateSpace(Space[torch.Tensor]):
    """Hidden state space for transformer token embeddings.
    
    Represents individual token hidden states as vectors.
    Unlike PixelSpace/LatentSpace, this operates on single tokens
    rather than entire sequences.
    
    Args:
        hidden_dim: Dimensionality of hidden states (e.g., 768 for BERT-base)
        layer_index: Which transformer layer to use (None = last layer)
        device: Torch device for computations
        
    Example:
        space = HiddenStateSpace(hidden_dim=768, layer_index=None)
        vec = space.encode(hidden_state)  # (768,) -> (768,)
        h = space.decode(vec)              # (768,) -> (768,)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        layer_index: Optional[int] = None,
        device: torch.device | str = "cpu",
    ):
        self._hidden_dim = hidden_dim
        self._layer_index = layer_index
        self._device = torch.device(device) if isinstance(device, str) else device
    
    @property
    def dim(self) -> int:
        return self._hidden_dim
    
    @property
    def layer_index(self) -> Optional[int]:
        """Layer index (None = last layer)."""
        return self._layer_index
    
    def encode(self, x: torch.Tensor) -> np.ndarray:
        """Encode hidden state tensor to vector.
        
        Args:
            x: Hidden state tensor of shape (hidden_dim,)
            
        Returns:
            Numpy vector of shape (hidden_dim,)
        """
        return to_numpy(x).reshape(-1)
    
    def decode(self, z: np.ndarray) -> torch.Tensor:
        """Decode vector to hidden state tensor.
        
        Args:
            z: Numpy vector of shape (hidden_dim,)
            
        Returns:
            Hidden state tensor of shape (hidden_dim,)
        """
        return to_tensor(z.reshape(-1), device=self._device)
    
    @staticmethod
    def extract_layer(
        hidden_states: tuple[torch.Tensor, ...],
        layer_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract hidden states from a specific layer.
        
        Args:
            hidden_states: Tuple of hidden states from all layers
                          Shape of each: (batch, seq_len, hidden_dim)
            layer_index: Layer to extract (None = last layer, -1 = last, etc.)
            
        Returns:
            Hidden states tensor (batch, seq_len, hidden_dim)
        """
        if layer_index is None:
            return hidden_states[-1]
        return hidden_states[layer_index]
    
    def encode_sequence(self, hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Encode all tokens in a sequence.
        
        Args:
            hidden: Hidden states (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
            mask: Optional mask for valid tokens (seq_len,) or (batch, seq_len)
            
        Returns:
            Encoded vectors (num_valid_tokens, hidden_dim)
        """
        h = to_numpy(hidden)
        
        if h.ndim == 2:
            # Single sequence (seq_len, hidden_dim)
            if mask is not None:
                m = to_numpy(mask).astype(bool)
                return h[m]
            return h
        
        elif h.ndim == 3:
            # Batch (batch, seq_len, hidden_dim)
            if mask is not None:
                m = to_numpy(mask).astype(bool)
                # Flatten batch and filter by mask
                flat_h = h.reshape(-1, h.shape[-1])
                flat_m = m.reshape(-1)
                return flat_h[flat_m]
            return h.reshape(-1, h.shape[-1])
        
        raise ValueError(f"Expected 2D or 3D tensor, got shape {hidden.shape}")
    
    def inject_smoothed(
        self,
        original: torch.Tensor,
        smoothed: np.ndarray,
        batch_idx: int,
        token_idx: int,
    ) -> torch.Tensor:
        """Inject a smoothed token back into the hidden state tensor.
        
        Args:
            original: Original hidden states (batch, seq_len, hidden_dim)
            smoothed: Smoothed vector (hidden_dim,)
            batch_idx: Batch index to inject into
            token_idx: Token index to inject into
            
        Returns:
            Modified hidden states tensor (same shape as original)
        """
        result = original.clone()
        result[batch_idx, token_idx] = to_tensor(smoothed, device=original.device, dtype=original.dtype)
        return result
