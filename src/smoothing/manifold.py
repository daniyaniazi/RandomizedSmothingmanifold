"""Manifold-aware smoothing using local PCA.

Adds Gaussian noise in the whitened (PCA) space, then transforms back.
This respects the local geometry of the data manifold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.indexing.base import NeighborIndex, neighbor_vectors

from .base import Smoother, SmoothingResult, gaussian_noise
from .pca import LocalPCA, fit_local_pca, whiten, unwhiten


@dataclass
class CachedPCA:
    """Cached PCA for a single anchor - reused across samples.
    
    Attributes:
        anchor: Original anchor vector
        pca: Fitted local PCA
        neighbors: Neighbor vectors used to fit PCA
    """
    anchor: np.ndarray
    pca: LocalPCA
    neighbors: np.ndarray


class ManifoldSmoother(Smoother):
    """Manifold-aware smoothing using local PCA.
    
    For each anchor:
    1. Find k nearest neighbors from the index
    2. Fit local PCA on the neighborhood
    3. Whiten the anchor: w = (x @ V) / sqrt(λ)
    4. Add isotropic noise in whitened space: w' = w + N(0, σ²I)
    5. Unwhiten back: x' = (w' * sqrt(λ)) @ V.T + mean
    
    This ensures noise is scaled according to local variance directions,
    respecting the data manifold geometry.
    
    Args:
        sigma: Standard deviation in whitened space
        index: kNN index for neighbor lookup
        knn_k: Number of neighbors for local PCA
        eps_eig: Minimum eigenvalue clamp
        
    Example:
        index = load_index(...)
        smoother = ManifoldSmoother(sigma=0.25, index=index, knn_k=32)
        noisy = smoother.sample(anchor)
    """
    
    def __init__(
        self,
        sigma: float,
        index: NeighborIndex,
        knn_k: int = 32,
        eps_eig: float = 1e-6,
    ):
        super().__init__(sigma)
        self._index = index
        self._knn_k = knn_k
        self._eps_eig = eps_eig
    
    @property
    def index(self) -> NeighborIndex:
        """kNN index for neighbor lookup."""
        return self._index
    
    @property
    def knn_k(self) -> int:
        """Number of neighbors for local PCA."""
        return self._knn_k
    
    def _get_neighbors(self, anchor: np.ndarray) -> np.ndarray:
        """Get k nearest neighbors for the anchor.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            Neighbor vectors (k, D)
        """
        # Request k+1 and drop self-match only when it is actually present.
        # For certification, the index is often built on TRAIN while anchors come
        # from TEST, so there is typically no exact self in the index.
        neighbors = neighbor_vectors(self._index, k=self._knn_k + 1, vector=anchor)
        if neighbors.shape[0] == 0:
            raise ValueError("Neighbor lookup returned no neighbors.")

        first_is_self = np.allclose(neighbors[0], anchor, rtol=1e-5, atol=1e-7)
        if first_is_self and neighbors.shape[0] > 1:
            return neighbors[1 : self._knn_k + 1]
        return neighbors[: self._knn_k]
    
    def _fit_pca(self, neighbors: np.ndarray) -> LocalPCA:
        """Fit local PCA on neighbors.
        
        Args:
            neighbors: Neighbor vectors (k, D)
            
        Returns:
            LocalPCA object
        """
        return fit_local_pca(neighbors, eps_eig=self._eps_eig)
    
    def compute_pca(self, anchor: np.ndarray) -> CachedPCA:
        """Compute and cache PCA for an anchor.
        
        Use this when you need to sample multiple times from the same anchor
        without recomputing the PCA.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            CachedPCA object for reuse
        """
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        neighbors = self._get_neighbors(anchor)
        pca = self._fit_pca(neighbors)
        return CachedPCA(anchor=anchor, pca=pca, neighbors=neighbors)
    
    def sample(self, anchor: np.ndarray) -> np.ndarray:
        """Sample by adding noise in whitened space.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            Noisy vector (D,)
        """
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        neighbors = self._get_neighbors(anchor)
        pca = self._fit_pca(neighbors)
        return self._sample_from_pca(anchor, pca)
    
    def sample_from_cached(self, cached: CachedPCA) -> np.ndarray:
        """Sample using a cached PCA (fast - no kNN/PCA recomputation).
        
        Args:
            cached: CachedPCA from compute_pca()
            
        Returns:
            Noisy vector (D,)
        """
        return self._sample_from_pca(cached.anchor, cached.pca)
    
    def _sample_from_pca(self, anchor: np.ndarray, pca: LocalPCA) -> np.ndarray:
        """Sample manifold noise and add to the ORIGINAL anchor.

        The noise direction and scale come from PCA of the local neighbourhood,
        but the noise is added to the original anchor — NOT to the PCA reconstruction.

        Steps:
          1. Sample noise in whitened space: n ~ N(0, alpha^2 I)
          2. Map noise back to original space: noise_orig = (n * sqrt(λ)) @ V.T
             (no mean added — this is pure noise, not a reconstruction)
          3. Return: anchor + noise_orig
        """
        lambda_max = float(pca.evals[0])
        alpha = self._sigma / np.sqrt(max(lambda_max, 1e-12))
        # Sample noise in whitened space
        noise_w = gaussian_noise(shape=len(pca.evals), std=alpha)
        # Map noise to original space — no mean shift, just rotate+scale
        noise_orig = (noise_w * np.sqrt(pca.evals)) @ pca.evecs.T
        return anchor + noise_orig
    
    def sample_n(self, anchor: np.ndarray, n: int) -> np.ndarray:
        """Sample n noisy points with single PCA computation.
        
        Args:
            anchor: Anchor vector (D,)
            n: Number of samples
            
        Returns:
            Noisy vectors (n, D)
        """
        cached = self.compute_pca(anchor)
        return np.stack([self.sample_from_cached(cached) for _ in range(n)])
    
    def sample_with_details(self, anchor: np.ndarray) -> SmoothingResult:
        """Sample with full details.
        
        Args:
            anchor: Anchor vector (D,)
            
        Returns:
            SmoothingResult with noise in original space
        """
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        cached = self.compute_pca(anchor)
        noisy = self.sample_from_cached(cached)
        diff = noisy - anchor
        
        return SmoothingResult(
            noisy=noisy,
            anchor=anchor,
            noise=diff,
            radius=float(np.linalg.norm(diff)),
        )

    # ------------------------------------------------------------------
    # GPU batch API (training and certification)
    # ------------------------------------------------------------------

    def build_gpu_cache(self, dataset, device) -> dict:
        """Pre-compute kNN+SVD for every image in dataset, return GPU tensors.

        Call once before training or certification.  After this, every call to
        sample_batch_gpu() does only cheap batched matmuls on GPU — no kNN, no SVD.

        Args:
            dataset: any object with .samples list of (img_path, label) tuples
                     OR a plain list of flat numpy vectors.
            device:  torch.device

        Returns:
            dict with GPU float32 tensors:
                'mean'  (N, D)
                'evals' (N, K)
                'evecs' (N, D, K)
        """
        import torch
        from pathlib import Path
        from PIL import Image
        from tqdm import tqdm

        samples = dataset.samples if hasattr(dataset, 'samples') else dataset
        N = len(samples)

        # Probe first entry to get D and K
        if isinstance(samples[0], (list, tuple)):
            img_path = samples[0][0]
            # load image the same way the dataloader does
            from torchvision import transforms as T
            _to_flat = lambda p: (
                np.asarray(
                    Image.open(p).convert('RGB').resize(
                        (int(np.sqrt(self._index.dim // 3)),
                         int(np.sqrt(self._index.dim // 3))),
                        Image.BILINEAR),
                    dtype=np.float32) / 255.0
            ).flatten()
            flat0 = _to_flat(img_path)
        else:
            flat0 = np.asarray(samples[0], dtype=np.float32).flatten()
            _to_flat = lambda x: np.asarray(x, dtype=np.float32).flatten()

        cached0 = self.compute_pca(flat0)
        D = int(cached0.pca.mean.shape[0])
        K = int(cached0.pca.evals.shape[0])

        means  = np.zeros((N, D), dtype=np.float32)
        evals_ = np.zeros((N, K), dtype=np.float32)
        evecs_ = np.zeros((N, D, K), dtype=np.float32)
        means[0]  = cached0.pca.mean
        evals_[0] = cached0.pca.evals[:K]
        evecs_[0] = cached0.pca.evecs[:, :K]

        for i in tqdm(range(1, N), desc='build_gpu_cache', leave=True):
            entry = samples[i]
            flat = _to_flat(entry[0] if isinstance(entry, (list, tuple)) else entry)
            c = self.compute_pca(flat)
            means[i]  = c.pca.mean
            evals_[i] = c.pca.evals[:K]
            evecs_[i] = c.pca.evecs[:, :K]

        print(f'GPU cache ready: N={N} D={D} K={K} device={device}')
        return {
            'mean':  torch.from_numpy(means).to(device),
            'evals': torch.from_numpy(evals_).to(device),
            'evecs': torch.from_numpy(evecs_).to(device),
        }

    def build_gpu_cache_from_npz(self, dataset, npz_path: str, device) -> dict:
        """Load pre-computed PCA .npz and align to dataset order as GPU tensors.

        Args:
            dataset:  object with .samples list of (img_path, label)
            npz_path: path to .npz saved by precompute_pca_cache.py
            device:   torch.device

        Returns:
            dict with GPU float32 tensors  mean (N,D), evals (N,K), evecs (N,D,K)
        """
        import torch
        from pathlib import Path

        print(f'Loading PCA cache: {npz_path}')
        raw = np.load(npz_path)
        stems: set = {k[:-5] for k in raw.files if k.endswith('_mean')}
        first = next(iter(stems))
        D = int(raw[f'{first}_mean'].shape[0])
        K = int(raw[f'{first}_evals'].shape[0])

        N = len(dataset.samples)
        means  = np.zeros((N, D), dtype=np.float32)
        evals_ = np.zeros((N, K), dtype=np.float32)
        evecs_ = np.zeros((N, D, K), dtype=np.float32)

        missing = 0
        for i, (img_path, _) in enumerate(dataset.samples):
            stem = Path(img_path).stem
            if stem in stems:
                means[i]  = raw[f'{stem}_mean']
                evals_[i] = raw[f'{stem}_evals'][:K]
                evecs_[i] = raw[f'{stem}_evecs'][:, :K]
            else:
                missing += 1
        if missing:
            print(f'  WARNING: {missing}/{N} images missing from cache.')
        print(f'  GPU tensors: N={N} D={D} K={K} -> {device}')
        return {
            'mean':  torch.from_numpy(means).to(device),
            'evals': torch.from_numpy(evals_).to(device),
            'evecs': torch.from_numpy(evecs_).to(device),
        }

    def sample_batch_gpu(
        self,
        images: 'torch.Tensor',
        gpu_cache: dict,
        indices: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Sample manifold noise for a batch — fully on GPU.

        Implements whiten -> N(0, alpha^2 I) -> unwhiten in batched torch ops.
        No CPU involved; noise is fresh every call.

        Args:
            images:    (B, C, H, W) float32 on GPU  -- training batch
                       OR (B, D) flat vectors on GPU -- certification
            gpu_cache: dict from build_gpu_cache / build_gpu_cache_from_npz
                         'mean'  (N, D)
                         'evals' (N, K)
                         'evecs' (N, D, K)
            indices:   (B,) long tensor -- row indices into gpu_cache tensors

        Returns:
            Noisy tensor same shape as images, on GPU.
        """
        import torch
        shape = images.shape
        if images.dim() == 4:
            B, C, H, W = shape
            x = images.view(B, C * H * W)
        else:
            B = shape[0]
            x = images  # already (B, D)

        mean  = gpu_cache['mean'][indices]    # (B, D)
        evecs = gpu_cache['evecs'][indices]   # (B, D, K)
        evals = gpu_cache['evals'][indices]   # (B, K)

        # Whiten: w = (x - mean) @ evecs / sqrt(evals)
        x_c = (x - mean).unsqueeze(1)                          # (B, 1, D)
        w   = torch.bmm(x_c, evecs).squeeze(1)                 # (B, K)
        w   = w / torch.sqrt(evals.clamp(min=1e-12))           # (B, K)

        # Noise: alpha = sigma / sqrt(lambda_max)
        alpha = self._sigma / torch.sqrt(evals[:, 0].clamp(min=1e-12))  # (B,)
        noise = torch.randn_like(w) * alpha.unsqueeze(1)                # (B, K)
        w_noisy = w + noise

        # Unwhiten: x' = (w_noisy * sqrt(evals)) @ evecs.T + mean
        w_s    = w_noisy * torch.sqrt(evals.clamp(min=1e-12))  # (B, K)
        x_out  = torch.bmm(w_s.unsqueeze(1),
                            evecs.transpose(1, 2)).squeeze(1) + mean  # (B, D)

        return x_out.view(shape)
