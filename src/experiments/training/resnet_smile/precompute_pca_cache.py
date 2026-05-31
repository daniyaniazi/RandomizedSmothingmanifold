"""Pre-compute and save local PCA for every image in a dataset.

Runs kNN lookup + SVD once per image, saves results to a single .npz file.
Subsequent dataset generation or training loads this file and only needs
a cheap matrix-multiply to sample noise -- same cost as isotropic.

Output:
    output/pca_cache/{dataset_name}/knn{k}/{image_dir}_pca_cache.npz

    The .npz contains three arrays per image, keyed by filename stem:
        {name}_mean   (D,)    neighborhood mean
        {name}_evals  (K,)    eigenvalues
        {name}_evecs  (D, K)  eigenvectors

Usage:
    python -m src.experiments.training.resnet_smile.precompute_pca_cache \\
        --image_dir /BS/databases08/CelebA/img_align_celeba \\
        --index_path output/smile_classification/celeba/index/pixel/annoy/euclidean/index.ann \\
        --image_size 224 \\
        --knn_k 500 \\
        --output_dir output/pca_cache/celeba

Re-run safely -- already-cached images are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image_flat(path: Path, size: int) -> np.ndarray:
    """Load, resize -> float32 flat vector in [0,1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0).flatten()


def _worker_chunk(
    image_paths: List[str],
    image_size: int,
    index_path: str,
    knn_k: int,
    eps_eig: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Process a chunk of images in a subprocess.

    Loads the Annoy index and ManifoldSmoother independently per worker
    (Annoy indexes are not fork-safe, so we always construct inside the worker).
    Returns dict: stem -> (mean, evals, evecs).
    """
    # Import here so the module path is available inside subprocess
    from src.indexing.base import load_index
    from src.smoothing.manifold import ManifoldSmoother

    dim = image_size * image_size * 3
    index = load_index(dim=dim, index_path=index_path, backend="annoy")
    smoother = ManifoldSmoother(sigma=0.25, index=index, knn_k=knn_k, eps_eig=eps_eig)

    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        try:
            flat = load_image_flat(img_path, image_size)
            cached = smoother.compute_pca(flat)
            results[img_path.stem] = (cached.pca.mean, cached.pca.evals, cached.pca.evecs)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: failed {img_path.name}: {exc}", flush=True)
    return results


def save_cache(cache: dict, path: Path) -> None:
    """Atomically write cache to .npz via a temp file to avoid corruption."""
    tmp = path.with_suffix(".tmp.npz")
    npz_dict: dict = {}
    for stem, (mean, evals, evecs) in cache.items():
        npz_dict[f"{stem}_mean"]  = mean
        npz_dict[f"{stem}_evals"] = evals
        npz_dict[f"{stem}_evecs"] = evecs
    np.savez_compressed(tmp, **npz_dict)
    tmp.replace(path)  # atomic rename — safe even if killed mid-write


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute local PCA cache for every image (saves kNN+SVD to .npz)"
    )
    parser.add_argument("--image_dir", required=True,
                        help="Raw image folder (e.g. img_align_celeba)")
    parser.add_argument("--index_path", required=True,
                        help="Annoy pixel index (.ann)")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--knn_k", type=int, default=500,
                        help="Number of neighbours for local PCA (default: 500)")
    parser.add_argument("--eps_eig", type=float, default=1e-6)
    parser.add_argument("--output_dir", default="output/pca_cache",
                        help="Directory to save the .npz cache file")
    parser.add_argument("--extensions", nargs="+",
                        default=[".jpg", ".jpeg", ".png"])
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Images per work unit submitted to the pool (default: 64)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"ERROR: image_dir not found: {image_dir}"); sys.exit(1)

    index_path = Path(args.index_path)
    if not index_path.exists():
        print(f"ERROR: index not found: {index_path}"); sys.exit(1)

    all_images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in args.extensions
    ])
    if not all_images:
        print(f"ERROR: no images in {image_dir}"); sys.exit(1)
    print(f"Found {len(all_images)} images")

    # Output path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"knn{args.knn_k}_{image_dir.name}_pca_cache.npz"

    # Load existing cache if present
    cache: dict = {}
    if cache_file.exists():
        print(f"Loading existing cache: {cache_file}")
        data = np.load(cache_file)
        stems: set = set()
        for key in data.files:
            stems.add(key.rsplit("_", 1)[0])
        for stem in stems:
            cache[stem] = (data[f"{stem}_mean"], data[f"{stem}_evals"], data[f"{stem}_evecs"])
        print(f"  {len(cache)} images already cached")

    # Filter to only uncached images
    todo = [p for p in all_images if p.stem not in cache]
    if not todo:
        print("All images already cached — nothing to do.")
        return
    print(f"Images to process: {len(todo)}  workers: {args.workers}  chunk_size: {args.chunk_size}")

    # Split into chunks
    chunks = [
        [str(p) for p in todo[i : i + args.chunk_size]]
        for i in range(0, len(todo), args.chunk_size)
    ]

    CHECKPOINT_EVERY = 5  # checkpoint after every N completed chunks

    new_count = 0
    chunks_since_save = 0

    worker_kwargs = dict(
        image_size=args.image_size,
        index_path=str(index_path),
        knn_k=args.knn_k,
        eps_eig=args.eps_eig,
    )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_worker_chunk, chunk, **worker_kwargs): len(chunk)
            for chunk in chunks
        }
        pbar = tqdm(total=len(todo), desc="Computing PCA", unit="img")
        for future in as_completed(futures):
            n_imgs = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"\nWARNING: chunk failed: {exc}", flush=True)
                pbar.update(n_imgs)
                continue
            cache.update(result)
            new_count += len(result)
            chunks_since_save += 1
            pbar.update(n_imgs)

            if chunks_since_save >= CHECKPOINT_EVERY:
                save_cache(cache, cache_file)
                chunks_since_save = 0
        pbar.close()

    if new_count == 0:
        print("No new entries were computed.")
        return

    # Final save
    save_cache(cache, cache_file)
    print(f"Done. New entries: {new_count}  Total: {len(cache)}")
    print(f"Cache file: {cache_file}")


if __name__ == "__main__":
    main()
