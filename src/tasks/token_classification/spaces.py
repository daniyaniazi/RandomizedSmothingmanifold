"""Space adapters for token classification.

Wraps HiddenStateSpace with token-specific utilities for NER, POS, etc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.spaces import HiddenStateSpace


class TokenHiddenStateSpace(HiddenStateSpace):
    """Hidden state space for token classification.
    
    Extends HiddenStateSpace with:
    - Token metadata (text, position, sentence)
    - Batch processing utilities
    - Integration with transformer models
    
    Args:
        hidden_dim: Dimensionality of hidden states
        layer_index: Which transformer layer to use (None = last)
        device: Torch device
    """
    
    def __init__(
        self,
        hidden_dim: int,
        layer_index: Optional[int] = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__(hidden_dim=hidden_dim, layer_index=layer_index, device=device)
    
    def extract_token_vectors(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        valid_mask: torch.Tensor,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Extract valid token vectors from hidden states.
        
        Args:
            hidden_states: Tuple of hidden states from transformer
            valid_mask: Boolean mask (batch, seq_len) for valid tokens
            
        Returns:
            Tuple of:
            - vectors: (num_valid, hidden_dim) token vectors
            - positions: List of (batch_idx, token_idx) for each vector
        """
        # Get target layer
        hidden = self.extract_layer(hidden_states, self._layer_index)
        
        vectors = []
        positions = []
        
        batch_size, seq_len, _ = hidden.shape
        for b in range(batch_size):
            for t in range(seq_len):
                if valid_mask[b, t]:
                    vec = self.encode(hidden[b, t])
                    vectors.append(vec)
                    positions.append((b, t))
        
        if len(vectors) == 0:
            return np.zeros((0, self.dim), dtype=np.float32), []
        
        return np.stack(vectors), positions
    
    def inject_smoothed_vectors(
        self,
        original_hidden: torch.Tensor,
        smoothed_vectors: np.ndarray,
        positions: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Inject smoothed vectors back into hidden state tensor.
        
        Args:
            original_hidden: Original hidden states (batch, seq_len, hidden_dim)
            smoothed_vectors: Smoothed vectors (num_valid, hidden_dim)
            positions: List of (batch_idx, token_idx) for each vector
            
        Returns:
            Modified hidden states tensor
        """
        result = original_hidden.clone()
        
        for i, (b, t) in enumerate(positions):
            result[b, t] = self.decode(smoothed_vectors[i])
        
        return result
    
    def compute_delta(
        self,
        original_hidden: torch.Tensor,
        smoothed_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compute delta between original and smoothed hidden states.
        
        This delta can be added to the hidden states during forward pass
        via a layer_noise_fn.
        
        Args:
            original_hidden: Original hidden states
            smoothed_hidden: Smoothed hidden states
            
        Returns:
            Delta tensor (smoothed - original)
        """
        return smoothed_hidden - original_hidden
