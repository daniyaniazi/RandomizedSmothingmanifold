"""Voting utilities for randomized smoothing.

Provides helpers for counting votes and computing majority predictions
from multiple noisy samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class VotingResult:
    """Result of voting over multiple predictions.
    
    Attributes:
        majority_pred: The class with the most votes
        vote_counts: Per-class vote counts
        majority_count: Number of votes for the majority class
        total_votes: Total number of votes
        confidence: Fraction of votes for the majority class
    """
    majority_pred: int
    vote_counts: np.ndarray
    majority_count: int
    total_votes: int
    confidence: float


def count_votes(predictions: Sequence[int], num_classes: Optional[int] = None) -> np.ndarray:
    """Count votes for each class.
    
    Args:
        predictions: Sequence of class predictions
        num_classes: Number of classes (inferred if None)
        
    Returns:
        Array of vote counts (num_classes,)
    """
    preds = np.asarray(predictions, dtype=np.int64)
    
    if num_classes is None:
        num_classes = int(preds.max()) + 1 if len(preds) > 0 else 2
    
    vote_counts = np.zeros(num_classes, dtype=np.int64)
    for p in preds:
        if 0 <= p < num_classes:
            vote_counts[p] += 1
    
    return vote_counts


def majority_vote(predictions: Sequence[int], num_classes: Optional[int] = None) -> VotingResult:
    """Compute majority vote from a sequence of predictions.
    
    Args:
        predictions: Sequence of class predictions
        num_classes: Number of classes (inferred if None)
        
    Returns:
        VotingResult with majority class and vote statistics
    """
    vote_counts = count_votes(predictions, num_classes)
    total_votes = int(vote_counts.sum())
    majority_pred = int(vote_counts.argmax())
    majority_count = int(vote_counts[majority_pred])
    confidence = majority_count / total_votes if total_votes > 0 else 0.0
    
    return VotingResult(
        majority_pred=majority_pred,
        vote_counts=vote_counts,
        majority_count=majority_count,
        total_votes=total_votes,
        confidence=confidence,
    )


def sample_and_vote(
    sample_fn: Callable[[], int],
    n_samples: int,
    num_classes: Optional[int] = None,
) -> VotingResult:
    """Sample predictions and compute majority vote.
    
    Args:
        sample_fn: Callable that returns a class prediction
        n_samples: Number of samples to draw
        num_classes: Number of classes (inferred if None)
        
    Returns:
        VotingResult with majority class and vote statistics
    """
    predictions = [sample_fn() for _ in range(n_samples)]
    return majority_vote(predictions, num_classes)


def top2_counts(vote_counts: np.ndarray) -> tuple[int, int, int, int]:
    """Get counts for top-2 classes.
    
    Args:
        vote_counts: Per-class vote counts
        
    Returns:
        (top1_class, top2_class, top1_count, top2_count)
    """
    top2_idx = np.argsort(vote_counts)[-2:]
    a = int(top2_idx[-1])
    b = int(top2_idx[-2]) if len(top2_idx) > 1 else a
    return a, b, int(vote_counts[a]), int(vote_counts[b])


class BatchVoter:
    """Accumulates votes across multiple samples for a batch of inputs.
    
    Useful when processing samples in chunks or when predictions
    come from multiple forward passes.
    
    Example:
        voter = BatchVoter(batch_size=32, num_classes=10)
        for _ in range(n_samples):
            preds = model(batch)  # (32, 10) logits
            voter.add_predictions(preds.argmax(dim=-1))
        results = voter.finalize()
    """
    
    def __init__(self, batch_size: int, num_classes: int):
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._vote_counts = np.zeros((batch_size, num_classes), dtype=np.int64)
        self._n_samples = 0
    
    def add_predictions(self, predictions: np.ndarray | Sequence[int]) -> None:
        """Add a batch of predictions.
        
        Args:
            predictions: Array of predictions (batch_size,)
        """
        preds = np.asarray(predictions, dtype=np.int64)
        
        if len(preds) != self._batch_size:
            raise ValueError(f"Expected {self._batch_size} predictions, got {len(preds)}")
        
        for i, p in enumerate(preds):
            if 0 <= p < self._num_classes:
                self._vote_counts[i, p] += 1
        
        self._n_samples += 1
    
    def get_vote_counts(self) -> np.ndarray:
        """Get vote counts array (batch_size, num_classes)."""
        return self._vote_counts.copy()
    
    def get_majority_predictions(self) -> np.ndarray:
        """Get majority predictions for each sample in batch."""
        return self._vote_counts.argmax(axis=-1)
    
    def finalize(self) -> list[VotingResult]:
        """Compute VotingResult for each sample in batch."""
        results = []
        for i in range(self._batch_size):
            counts = self._vote_counts[i]
            total = int(counts.sum())
            majority = int(counts.argmax())
            majority_count = int(counts[majority])
            confidence = majority_count / total if total > 0 else 0.0
            
            results.append(VotingResult(
                majority_pred=majority,
                vote_counts=counts,
                majority_count=majority_count,
                total_votes=total,
                confidence=confidence,
            ))
        
        return results
