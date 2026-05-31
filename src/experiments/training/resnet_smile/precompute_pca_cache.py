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
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image_flat(path: Path, size: int) -> np.ndarray:
    """Load, resize -> float32 flat vector in [0,1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0).flatten()


def sigma_tag(sigma: float) -> str:
    return f"sigma_{sigma:.2f}".replace(".", "_")


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
    existing: dict = {}
    if cache_file.exists():
        print(f"Loading existing cache: {cache_file}")
        data = np.load(cache_file)
        # Reconstruct dict of stem -> (mean, evals, evecs)
        stems = set()
        for key in data.files:
            stems.add(key.rsplit("_", 1)[0])
        for stem in stems:
            existing[stem] = (data[f"{stem}_mean"], data[f"{stem}_evals"], data[f"{stem}_evecs"])
        print(f"  {len(existing)} images already cached")

    # Load index + smoother
    dim = args.image_size * args.image_size * 3
    from src.indexing.base import load_index
    from src.smoothing.manifold import ManifoldSmoother
    index = load_index(dim=dim, index_path=str(index_path), backend="annoy")
    smoother = ManifoldSmoother(sigma=0.25, index=index,  # sigma irrelevant for PCA
                                knn_k=args.knn_k, eps_eig=args.eps_eig)

    CHECKPOINT_EVERY = 5  # save to disk after every N new images

    def save_cache(cache: dict, path: Path) -> None:
        """Atomically write cache to .npz via a temp file to avoid corruption."""
        tmp = path.with_suffix(".tmp.npz")
        npz_dict = {}
        for stem, (mean, evals, evecs) in cache.items():
            npz_dict[f"{stem}_mean"]  = mean
            npz_dict[f"{stem}_evals"] = evals
            npz_dict[f"{stem}_evecs"] = evecs
        np.savez_compressed(tmp, **npz_dict)
        tmp.replace(path)  # atomic rename — safe even if killed mid-write

    # Compute missing entries with incremental checkpointing
    cache = dict(existing)
    new_count = 0
    for img_path in tqdm(all_images, desc="Computing PCA"):
        stem = img_path.stem
        if stem in cache:
            continue
        flat = load_image_flat(img_path, args.image_size)
        cached = smoother.compute_pca(flat)
        cache[stem] = (cached.pca.mean, cached.pca.evals, cached.pca.evecs)
        new_count += 1

        # Checkpoint every CHECKPOINT_EVERY new images
        if new_count % CHECKPOINT_EVERY == 0:
            save_cache(cache, cache_file)

    if new_count == 0:
        print("All images already cached — nothing to do.")
        return

    # Final save
    save_cache(cache, cache_file)
    print(f"Done. New entries: {new_count}  Total: {len(cache)}")
    print(f"Cache file: {cache_file}")


if __name__ == "__main__":
    main()
