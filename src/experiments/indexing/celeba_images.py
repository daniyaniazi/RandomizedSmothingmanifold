"""Build image embedding index for CelebA/CelebA-HQ.

This is a CLI script that uses generic utilities from src/indexing/image_index.
Works with CelebA, CelebA-HQ, or any image dataset.

Usage:
    python -m src.experiments.indexing.celeba_images \\
        --config CONFIG \\
        --space pixel|latent

Example:
    # Build pixel-space index
    python -m src.experiments.indexing.celeba_images \\
        --config src/configs/experiments/certify_celeba_pixel.yaml \\
        --space pixel

    # Build latent-space index (uses VAE encoder)
    python -m src.experiments.indexing.celeba_images \\
        --config src/configs/experiments/certify_celeba_latent_128.yaml \\
        --space latent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch

from src.configs.certify_celeba_io import load_certify_config
from src.dataloaders.celeba_smile import build_smile_dataloaders
from src.configs.train_smile_schema import SmileDatasetConfig, SmileDataloaderConfig, SmileModelConfig
from src.models.VAE import ConvVAE, load_checkpoint as load_vae_checkpoint
from src.indexing.image_index import (
    extract_pixel_vectors,
    extract_latent_vectors,
    build_image_neighbor_index,
    build_or_load_pixel_index_streaming,
    save_image_index_artifacts,
)
from src.indexing.annoy_indexing import save_annoy_index
from src.indexing.faiss_indexing import save_faiss_index


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    """Build image embedding index."""
    parser = argparse.ArgumentParser(
        description="Build image embedding index for CelebA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build pixel-space index for CelebA
  python -m src.experiments.indexing.celeba_images \\
      --config src/configs/experiments/certify_celeba_pixel.yaml \\
      --space pixel

  # Build latent-space index using VAE
  python -m src.experiments.indexing.celeba_images \\
      --config src/configs/experiments/certify_celeba_latent_128.yaml \\
      --space latent
        """
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to certification config YAML")
    parser.add_argument("--space", type=str, required=True,
                        choices=["pixel", "latent"],
                        help="Embedding space: pixel or latent (VAE)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for debugging")
    parser.add_argument("--backend", type=str, default="annoy",
                        choices=["annoy", "faiss", "torch"],
                        help="Index backend (default: annoy)")
    parser.add_argument("--metric", type=str, default=None,
                        choices=["euclidean", "angular"],
                        help="Override index metric from config (euclidean or angular)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_certify_config(args.config)
    if args.metric is not None:
        cfg.index.metric = args.metric  # CLI override takes precedence
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    _log(f"Loading dataset: {cfg.dataset.name}")
    
    # Build dataloader for TRAIN split
    dataset_cfg = SmileDatasetConfig(
        root_dir=cfg.dataset.root_dir,
        image_dir=cfg.dataset.image_dir,
        annotation_file=cfg.dataset.annotation_file,
        annotation_format=cfg.dataset.annotation_format,
        file_extension=getattr(cfg.dataset, "file_extension", ""),
        # Explicitly match certification split — must be identical to celeba.py get_train_test_samples()
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio,
        split_seed=cfg.seed,  # same seed as run_certification()
    )
    loader_cfg = SmileDataloaderConfig(
        batch_size=32,
        shuffle_train=False,
    )
    model_cfg = SmileModelConfig(
        input_size=cfg.model.input_size if args.space == "pixel" else getattr(cfg.vae, 'image_size', 128),
    )
    
    data = build_smile_dataloaders(
        dataset_cfg=dataset_cfg,
        loader_cfg=loader_cfg,
        model_cfg=model_cfg,
    )
    
    # Set output directory
    out_dir = Path(cfg.output.output_dir) / "smile_classification" / cfg.dataset.name.lower().replace("-", "").replace("_", "") / "index" / args.space
    index_dir = out_dir / args.backend / cfg.index.metric
    index_dir.mkdir(parents=True, exist_ok=True)
    
    if args.space == "pixel" and args.backend == "annoy":
        # ── Streaming mode for pixel+annoy: avoids OOM for high-dim vectors ──
        first_batch = next(iter(data.train_loader))
        img = first_batch["image"] if isinstance(first_batch, dict) else first_batch[0]
        dim = img.view(img.size(0), -1).shape[1]
        _log(f"Streaming pixel vectors into Annoy index (dim={dim})...")

        artifacts = build_or_load_pixel_index_streaming(
            out_dir=index_dir,
            dataloader=data.train_loader,
            dim=dim,
            image_key="image",
            metric=cfg.index.metric,
            n_trees=cfg.index.n_trees,
            max_samples=args.max_samples,
            metadata={
                "space": args.space,
                "backend": args.backend,
                "metric": cfg.index.metric,
                "dataset": cfg.dataset.name,
                "image_size": cfg.model.input_size,
            },
            log_fn=_log,
        )

        n = artifacts.metadata.get("n_vectors", "?") if artifacts.metadata else "?"
        _log(f"Done! {n} vectors, dim={dim}")
    
    else:
        # ── Original batch mode for latent space (small vectors, fits in memory) ──
        if args.space == "pixel":
            _log("Extracting pixel vectors...")
            vectors = extract_pixel_vectors(
                dataloader=data.train_loader,
                image_key="image",
                device=device,
                max_samples=args.max_samples,
            )
        else:
            _log("Loading VAE...")
            vae = ConvVAE(
                image_size=cfg.vae.image_size,
                latent_dim=cfg.vae.latent_dim,
                in_channels=cfg.vae.in_channels,
            ).to(device)
            load_vae_checkpoint(vae, cfg.vae.checkpoint_path, device=device)
            vae.eval()
            
            def vae_encoder(images: torch.Tensor) -> torch.Tensor:
                if images.shape[-1] != vae.image_size:
                    images = torch.nn.functional.interpolate(
                        images, size=vae.image_size, mode="bilinear", align_corners=False
                    )
                mu, _ = vae.encode(images)
                return mu
            
            _log("Extracting latent vectors...")
            vectors = extract_latent_vectors(
                dataloader=data.train_loader,
                encoder=vae_encoder,
                image_key="image",
                device=device,
                max_samples=args.max_samples,
            )
        
        _log(f"Building {args.backend} index with {len(vectors)} vectors of dim {vectors.shape[1]}...")
        
        index = build_image_neighbor_index(
            vectors=vectors,
            backend=args.backend,
            metric=cfg.index.metric,
            n_trees=cfg.index.n_trees if args.backend == "annoy" else 20,
        )
        
        index_path = str(index_dir / "index.ann")
        if args.backend == "annoy":
            save_annoy_index(index, index_path)
        elif args.backend == "faiss":
            save_faiss_index(index, index_path)
        
        metadata = {
            "space": args.space,
            "backend": args.backend,
            "metric": cfg.index.metric,
            "n_vectors": len(vectors),
            "embedding_dim": int(vectors.shape[1]),
            "dataset": cfg.dataset.name,
        }
        save_image_index_artifacts(index_dir, vectors, metadata)
        
        _log(f"Index saved to: {index_dir}")
        _log(f"Vectors: {len(vectors)}, Dim: {vectors.shape[1]}")


if __name__ == "__main__":
    main()
