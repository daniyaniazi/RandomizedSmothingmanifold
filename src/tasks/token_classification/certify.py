"""Token classification certification workflow.

Provides certification for token-level predictions (NER, POS, etc.)
using smoothing in transformer hidden state space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.certify import TokenCertificate, certify_token_from_counts_two_stage_paper
from src.certify.voting import BatchVoter
from src.indexing.base import NeighborIndex, load_index
from src.smoothing import create_smoother, Smoother, ManifoldSmoother
from src.smoothing.manifold import CachedPCA

from .spaces import TokenHiddenStateSpace


@dataclass
class TokenCertificationConfig:
    """Configuration for token certification.
    
    Attributes:
        sigma: Noise standard deviation
        n_samples: Number of samples per token
        alpha: Confidence level for Clopper-Pearson
        smoothing_mode: "isotropic" or "manifold"
        knn_k: Number of neighbors for manifold smoothing
        layer_index: Transformer layer to smooth (None = last)
        index_path: Path to token embedding index
    """
    sigma: float = 0.25
    n0_samples: int = 64
    n_samples: int = 100
    alpha: float = 0.001
    smoothing_mode: str = "manifold"
    knn_k: int = 32
    layer_index: Optional[int] = None
    index_path: Optional[str] = None
    device: str = "cuda"


@dataclass
class TokenCertificationResult:
    """Result for a single token."""
    token_idx: int
    pred: int
    certificate: TokenCertificate
    vote_counts: np.ndarray
    clean_pred: int
    token_text: Optional[str] = None
    true_label: Optional[int] = None


@dataclass
class SentenceCertificationResult:
    """Result for a full sentence."""
    tokens: list[TokenCertificationResult]
    predictions: np.ndarray
    radii: np.ndarray
    abstained: np.ndarray


class TokenCertifier:
    """Certifier for token classification tasks.
    
    Uses an optimized workflow:
    1. Compute hidden states once (clean forward pass)
    2. For each valid token, precompute PCA (manifold mode)
    3. For each sample:
       - Sample noise using cached PCAs
       - Inject smoothed hidden states
       - Get predictions
       - Accumulate votes
    4. Compute certificates from vote counts
    
    This avoids redundant PCA computation across samples.
    """
    
    def __init__(
        self,
        space: TokenHiddenStateSpace,
        smoother: Smoother,
        model: Any,
        n0_samples: int = 64,
        n_samples: int = 100,
        alpha: float = 0.001,
        num_classes: int = 9,
    ):
        if int(n0_samples) <= 0:
            raise ValueError("Paper-aligned CERTIFY requires n0_samples > 0.")
        if int(n_samples) <= 0:
            raise ValueError("Paper-aligned CERTIFY requires n_samples > 0.")
        self._space = space
        self._smoother = smoother
        self._model = model
        self._n0_samples = n0_samples
        self._n_samples = n_samples
        self._alpha = alpha
        self._num_classes = num_classes
    
    @classmethod
    def from_config(
        cls,
        config: TokenCertificationConfig,
        model: Any,
        hidden_dim: int,
        num_classes: int,
        index: Optional[NeighborIndex] = None,
    ) -> "TokenCertifier":
        """Create certifier from config."""
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        space = TokenHiddenStateSpace(
            hidden_dim=hidden_dim,
            layer_index=config.layer_index,
            device=device,
        )
        
        # Load index if needed
        if config.smoothing_mode == "manifold" and index is None:
            if config.index_path is None:
                raise ValueError("Index path required for manifold smoothing")
            index = load_index(dim=hidden_dim, index_path=config.index_path, backend="annoy")
        
        smoother = create_smoother(
            mode=config.smoothing_mode,
            sigma=config.sigma,
            index=index,
            knn_k=config.knn_k,
        )
        
        return cls(
            space=space,
            smoother=smoother,
            model=model,
            n0_samples=config.n0_samples,
            n_samples=config.n_samples,
            alpha=config.alpha,
            num_classes=num_classes,
        )
    
    @property
    def sigma(self) -> float:
        return self._smoother.sigma
    
    @torch.no_grad()
    def certify_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: Optional[Any] = None,
    ) -> SentenceCertificationResult:
        """Certify a batch of sentences.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: True labels (batch, seq_len), -100 for ignored
            tokenizer: Optional tokenizer for debug info
            
        Returns:
            SentenceCertificationResult with per-token certificates
        """
        self._model.eval()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Valid tokens: not padding, not special, not ignored
        valid_mask = (labels != -100) & attention_mask.bool()
        
        # Get clean hidden states
        clean_out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        clean_hidden = self._space.extract_layer(
            clean_out.hidden_states,
            self._space.layer_index,
        )
        
        # Extract valid token vectors and positions
        vectors, positions = self._space.extract_token_vectors(
            clean_out.hidden_states,
            valid_mask,
        )
        
        if len(positions) == 0:
            return SentenceCertificationResult(
                tokens=[],
                predictions=np.array([]),
                radii=np.array([]),
                abstained=np.array([]),
            )
        
        # Precompute PCAs for manifold smoothing
        cached_pcas: list[CachedPCA] = []
        if isinstance(self._smoother, ManifoldSmoother):
            for vec in vectors:
                cached = self._smoother.compute_pca(vec)
                cached_pcas.append(cached)
        
        # Initialize two-stage vote counts (paper CERTIFY)
        vote_counts_n0 = np.zeros((len(positions), self._num_classes), dtype=np.int64)
        vote_counts_n = np.zeros((len(positions), self._num_classes), dtype=np.int64)
        
        # Get target layer for injection
        target_layer = self._model.num_layers() - 1 if self._space.layer_index is None else self._space.layer_index
        
        # Sampling loop (first n0 for class selection, then n for certification)
        total_samples = int(max(0, self._n0_samples) + max(0, self._n_samples))
        for sample_idx in range(total_samples):
            # Generate smoothed vectors
            if cached_pcas:
                smoothed_vectors = np.stack([
                    self._smoother.sample_from_cached(cached)
                    for cached in cached_pcas
                ])
            else:
                smoothed_vectors = np.stack([
                    self._smoother.sample(vec) for vec in vectors
                ])
            
            # Inject smoothed vectors
            smoothed_hidden = self._space.inject_smoothed_vectors(
                clean_hidden, smoothed_vectors, positions
            )
            delta = smoothed_hidden - clean_hidden
            
            # Layer noise function for injection
            def _layer_noise_fn(h: torch.Tensor, idx: int) -> torch.Tensor:
                if idx != target_layer:
                    return torch.zeros_like(h)
                return delta
            
            # Forward pass with smoothed hidden states
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                layer_noise_fn=_layer_noise_fn,
            )
            
            # Get predictions and accumulate votes
            pred_ids = torch.argmax(out.logits, dim=-1).cpu().numpy()
            stage_n0 = sample_idx < int(max(0, self._n0_samples))
            
            for i, (b, t) in enumerate(positions):
                pred = pred_ids[b, t]
                if 0 <= pred < self._num_classes:
                    if stage_n0:
                        vote_counts_n0[i, pred] += 1
                    else:
                        vote_counts_n[i, pred] += 1
        
        # Compute certificates
        token_results = []
        for i, (b, t) in enumerate(positions):
            counts_n = vote_counts_n[i]
            counts_n0 = vote_counts_n0[i]
            certificate = certify_token_from_counts_two_stage_paper(
                class_counts_n0=counts_n0,
                class_counts_n=counts_n,
                alpha_noise=self.sigma,
                alpha_conf=self._alpha,
            )
            
            # Get clean prediction
            clean_pred = int(torch.argmax(clean_out.logits[b, t]).item())
            
            # Get token text if tokenizer provided
            token_text = None
            if tokenizer:
                token_text = tokenizer.convert_ids_to_tokens([int(input_ids[b, t].item())])[0]
            
            token_results.append(TokenCertificationResult(
                token_idx=t,
                pred=certificate.pred,
                certificate=certificate,
                vote_counts=counts_n,
                clean_pred=clean_pred,
                token_text=token_text,
                true_label=int(labels[b, t].item()),
            ))
        
        # Aggregate results
        predictions = np.array([r.pred for r in token_results])
        radii = np.array([r.certificate.radius for r in token_results])
        abstained = np.array([r.certificate.abstained for r in token_results])
        
        return SentenceCertificationResult(
            tokens=token_results,
            predictions=predictions,
            radii=radii,
            abstained=abstained,
        )


def run_token_certification(
    config: TokenCertificationConfig,
    model: Any,
    dataloader: Any,
    hidden_dim: int,
    num_classes: int,
    index: Optional[NeighborIndex] = None,
    tokenizer: Optional[Any] = None,
    max_batches: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run full token certification pipeline.
    
    Args:
        config: Certification configuration
        model: Transformer model with layer_noise_fn support
        dataloader: DataLoader yielding batches
        hidden_dim: Hidden state dimensionality
        num_classes: Number of output classes
        index: kNN index (optional, loaded from config if not provided)
        tokenizer: Tokenizer for debug info
        max_batches: Maximum batches to process
        output_dir: Directory to save results
        
    Returns:
        Dictionary with certification metrics
    """
    certifier = TokenCertifier.from_config(
        config=config,
        model=model,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        index=index,
    )
    
    all_results = []
    total_tokens = 0
    total_certified = 0
    total_correct = 0
    all_radii = []
    
    iterator = tqdm(dataloader, desc="Certifying")
    for batch_idx, batch in enumerate(iterator):
        if max_batches and batch_idx >= max_batches:
            break
        
        result = certifier.certify_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            tokenizer=tokenizer,
        )
        
        all_results.append(result)
        
        # Accumulate metrics
        total_tokens += len(result.tokens)
        total_certified += (~result.abstained).sum()
        
        for tok in result.tokens:
            if not tok.certificate.abstained:
                all_radii.append(tok.certificate.radius)
                if tok.pred == tok.true_label:
                    total_correct += 1
    
    # Compute metrics
    metrics = {
        "total_tokens": total_tokens,
        "certified_tokens": int(total_certified),
        "abstained_tokens": total_tokens - int(total_certified),
        "certified_accuracy": total_correct / total_certified if total_certified > 0 else 0.0,
        "abstention_rate": (total_tokens - total_certified) / total_tokens if total_tokens > 0 else 0.0,
        "mean_radius": float(np.mean(all_radii)) if all_radii else 0.0,
        "sigma": config.sigma,
        "n_samples": config.n_samples,
        "smoothing_mode": config.smoothing_mode,
    }
    
    # Save results
    if output_dir:
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics
