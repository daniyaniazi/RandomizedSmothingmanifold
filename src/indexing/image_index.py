"""Image embedding index utilities for manifold smoothing.

Generic utilities for extracting image embeddings (pixel or latent space)
and building neighbor indexes. Works with any image dataset.

This module keeps indexing modular and separate from smoothing:
1. Run through image dataset
2. Extract embeddings (pixel-space or VAE latent-space)
3. Store vectors + metadata
4. Build a reusable neighbor index

Usage:
    from src.indexing.image_index import (
        extract_pixel_vectors,
        extract_latent_vectors,
        save_image_index_artifacts,
        build_or_load_image_index,
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import NeighborIndex, build_index, load_index


@dataclass
class ImageIndexArtifacts:
    """Artifacts from building an image embedding index.
    
    Attributes:
        vectors: Embedding vectors (N, D)
        index: NeighborIndex for kNN queries
        metadata: Optional metadata dict (e.g., labels, paths)
    """
    vectors: np.ndarray
    index: NeighborIndex
    metadata: dict | None = None


@torch.no_grad()
def extract_pixel_vectors(
    dataloader: DataLoader,
    image_key: str = "image",
    device: torch.device | None = None,
    max_samples: int | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract flattened pixel vectors from images.
    
    Args:
        dataloader: DataLoader yielding batches with images
        image_key: Key to access images in batch dict (default: "image")
        device: Device for processing (default: CPU)
        max_samples: Maximum samples to extract (default: all)
        show_progress: Show tqdm progress bar
        
    Returns:
        Pixel vectors array (N, C*H*W)
    """
    device = device or torch.device("cpu")
    vectors: list[np.ndarray] = []
    count = 0
    
    iterator = tqdm(dataloader, desc="Extracting pixel vectors") if show_progress else dataloader
    
    for batch in iterator:
        # Handle both dict batches and tuple batches
        if isinstance(batch, dict):
            images = batch[image_key]
        elif isinstance(batch, (list, tuple)):
            images = batch[0]  # Assume images are first element
        else:
            images = batch
            
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Flatten to (B, C*H*W)
        flat = images.view(batch_size, -1).cpu().numpy().astype(np.float32)
        vectors.append(flat)
        
        count += batch_size
        if max_samples and count >= max_samples:
            break
    
    if not vectors:
        raise RuntimeError("No pixel vectors extracted from dataloader.")
    
    return np.concatenate(vectors, axis=0).astype(np.float32)


@torch.no_grad()
def extract_latent_vectors(
    dataloader: DataLoader,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    image_key: str = "image",
    device: torch.device | None = None,
    max_samples: int | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract latent vectors using an encoder (VAE, autoencoder, etc.).
    
    Args:
        dataloader: DataLoader yielding batches with images
        encoder: Function that takes images (B,C,H,W) and returns latents (B,D)
        image_key: Key to access images in batch dict (default: "image")
        device: Device for processing (default: CPU)
        max_samples: Maximum samples to extract (default: all)
        show_progress: Show tqdm progress bar
        
    Returns:
        Latent vectors array (N, D)
        
    Example:
        # With VAE
        vae = load_vae(...)
        encoder = lambda x: vae.encode(x)[0]  # Return mu only
        vectors = extract_latent_vectors(loader, encoder, device=device)
    """
    device = device or torch.device("cpu")
    vectors: list[np.ndarray] = []
    count = 0
    
    iterator = tqdm(dataloader, desc="Extracting latent vectors") if show_progress else dataloader
    
    for batch in iterator:
        # Handle both dict batches and tuple batches
        if isinstance(batch, dict):
            images = batch[image_key]
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
            
        images = images.to(device)
        
        # Encode to latent space
        latents = encoder(images)
        if isinstance(latents, tuple):
            latents = latents[0]  # Handle VAE returning (mu, logvar)
        
        vectors.append(latents.cpu().numpy().astype(np.float32))
        
        count += latents.shape[0]
        if max_samples and count >= max_samples:
            break
    
    if not vectors:
        raise RuntimeError("No latent vectors extracted from dataloader.")
    
    return np.concatenate(vectors, axis=0).astype(np.float32)


def save_image_index_artifacts(
    out_dir: str | Path,
    vectors: np.ndarray,
    metadata: dict | None = None,
) -> tuple[Path, Path | None]:
    """Save image embedding vectors and optional metadata.
    
    Args:
        out_dir: Output directory
        vectors: Embedding vectors (N, D)
        metadata: Optional metadata dict
        
    Returns:
        Tuple of (vectors_path, metadata_path or None)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vectors_path = out_dir / "image_vectors.npz"
    np.savez(vectors_path, vectors=vectors)
    
    meta_path = None
    if metadata is not None:
        meta_path = out_dir / "image_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
    
    return vectors_path, meta_path


def load_image_index_artifacts(out_dir: str | Path) -> tuple[np.ndarray, dict | None]:
    """Load image embedding vectors and metadata.
    
    Args:
        out_dir: Directory containing saved artifacts
        
    Returns:
        Tuple of (vectors, metadata or None)
    """
    out_dir = Path(out_dir)
    vectors_path = out_dir / "image_vectors.npz"
    meta_path = out_dir / "image_metadata.json"
    
    payload = np.load(vectors_path, allow_pickle=True)
    vectors = payload["vectors"].astype(np.float32)
    
    metadata = None
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    
    return vectors, metadata


def build_image_neighbor_index(
    vectors: np.ndarray,
    backend: str = "faiss",
    metric: str = "euclidean",
    index_path: str | None = None,
    n_trees: int = 20,
) -> NeighborIndex:
    """Build a neighbor index from image embedding vectors.
    
    Args:
        vectors: Embedding vectors (N, D)
        backend: Index backend ("faiss", "annoy", "torch")
        metric: Distance metric ("euclidean", "cosine")
        index_path: Path to save/load index (optional)
        n_trees: Number of trees for Annoy (ignored for other backends)
        
    Returns:
        NeighborIndex for kNN queries
    """
    return build_index(
        vectors=vectors,
        backend=backend,
        metric=metric,
        index_path=index_path,
        n_trees=n_trees,
    )


def build_or_load_image_index(
    out_dir: str | Path,
    dataloader: DataLoader,
    space: str = "pixel",
    encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
    image_key: str = "image",
    device: torch.device | None = None,
    backend: str = "faiss",
    metric: str = "euclidean",
    index_path: str | None = None,
    n_trees: int = 20,
    max_samples: int | None = None,
    rebuild: bool = False,
    metadata: dict | None = None,
) -> ImageIndexArtifacts:
    """Build or load image embedding index.
    
    Args:
        out_dir: Directory for saving/loading artifacts
        dataloader: DataLoader yielding image batches
        space: "pixel" or "latent"
        encoder: Encoder function (required if space="latent")
        image_key: Key to access images in batch dict
        device: Device for processing
        backend: Index backend ("faiss", "annoy", "torch")
        metric: Distance metric
        index_path: Path for index file
        n_trees: Number of trees for Annoy
        max_samples: Maximum samples to extract
        rebuild: Force rebuild even if artifacts exist
        metadata: Optional metadata to save
        
    Returns:
        ImageIndexArtifacts with vectors and index
    """
    out_dir = Path(out_dir)
    vectors_path = out_dir / "image_vectors.npz"
    
    # Load or extract vectors
    if not rebuild and vectors_path.exists():
        vectors, saved_metadata = load_image_index_artifacts(out_dir)
        metadata = saved_metadata or metadata
    else:
        if space == "pixel":
            vectors = extract_pixel_vectors(
                dataloader=dataloader,
                image_key=image_key,
                device=device,
                max_samples=max_samples,
            )
        elif space == "latent":
            if encoder is None:
                raise ValueError("encoder required for latent space indexing")
            vectors = extract_latent_vectors(
                dataloader=dataloader,
                encoder=encoder,
                image_key=image_key,
                device=device,
                max_samples=max_samples,
            )
        else:
            raise ValueError(f"Unknown space: {space}. Use 'pixel' or 'latent'.")
        
        save_image_index_artifacts(out_dir, vectors, metadata)
    
    # Build or load index
    backend_norm = str(backend).strip().lower()
    metric_norm = str(metric).strip().lower()
    index_file = Path(index_path) if index_path else None
    
    if (
        not rebuild
        and index_file is not None
        and index_file.exists()
        and backend_norm in {"annoy", "faiss"}
    ):
        index = load_index(
            dim=int(vectors.shape[1]),
            index_path=str(index_file),
            backend=backend_norm,
            metric=metric_norm,
        )
        index.vectors = vectors
    else:
        index = build_image_neighbor_index(
            vectors=vectors,
            backend=backend_norm,
            metric=metric_norm,
            index_path=index_path,
            n_trees=n_trees,
        )
    
    return ImageIndexArtifacts(vectors=vectors, index=index, metadata=metadata)
