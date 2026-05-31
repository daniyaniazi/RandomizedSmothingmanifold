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

Strategy (memory-safe):
    Each thread writes its result immediately to a tiny per-image .npz file
    inside a scratch directory.  The main thread only tracks stem names
    (strings) -- never holds large arrays in RAM.  After all images are done,
    a single merge pass reads the per-image files and writes the final .npz.
    Re-runs skip any stem whose per-image scratch file already exists.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image_flat(path: Path, size: int) -> np.ndarray:
    """Load, resize -> float32 flat vector in [0,1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0).flatten()


def _process_one(
    img_path: Path,
    image_size: int,
    smoother,
    scratch_dir: Path,
) -> str:
    """Compute PCA for one image and write immediately to scratch_dir/{stem}.npz.

    Returns the stem on success.  Never keeps large arrays in the calling
    thread after the file is written -- RAM per thread is O(one image).
    numpy SVD releases the GIL so threads truly run in parallel.
    """
    out_file = scratch_dir / f"{img_path.stem}.npz"
    if out_file.exists():
        return img_path.stem  # already done from a previous run

    flat = load_image_flat(img_path, image_size)
    cached = smoother.compute_pca(flat)
    np.savez_compressed(
        out_file,
        mean=cached.pca.mean,
        evals=cached.pca.evals,
        evecs=cached.pca.evecs,
    )
    return img_path.stem


def merge_scratch_to_npz(scratch_dir: Path, cache_file: Path, stems: list) -> None:
    """Read per-image scratch files and write a single merged .npz.

    Streams one image at a time -- RAM during merge is O(one image).
    """
    print(f"Merging {len(stems)} per-image files -> {cache_file}", flush=True)
    tmp = cache_file.with_suffix(".tmp.npz")
    npz_dict: dict = {}
    for stem in tqdm(stems, desc="Merging", unit="img"):
        f = scratch_dir / f"{stem}.npz"
        if not f.exists():
            print(f"  WARNING: scratch file missing for {stem}, skipping", flush=True)
            continue
        d = np.load(f)
        npz_dict[f"{stem}_mean"]  = d["mean"]
        npz_dict[f"{stem}_evals"] = d["evals"]
        npz_dict[f"{stem}_evecs"] = d["evecs"]
    np.savez_compressed(tmp, **npz_dict)
    tmp.replace(cache_file)
    print("Merge complete.", flush=True)


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
                        help="Number of parallel threads (numpy SVD releases GIL)")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip computation, only merge existing scratch files into final .npz")
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
    print(f"Found {len(all_images)} images", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"knn{args.knn_k}_{image_dir.name}_pca_cache.npz"

    # Scratch dir: one small .npz per image, survives restarts
    scratch_dir = out_dir / f"knn{args.knn_k}_{image_dir.name}_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    all_stems = [p.stem for p in all_images]

    if args.merge_only:
        merge_scratch_to_npz(scratch_dir, cache_file, all_stems)
        return

    todo = [p for p in all_images if not (scratch_dir / f"{p.stem}.npz").exists()]
    already_done = len(all_images) - len(todo)
    if already_done:
        print(f"  {already_done} images already in scratch — skipping", flush=True)
    if not todo:
        print("All images computed. Running merge step...", flush=True)
        merge_scratch_to_npz(scratch_dir, cache_file, all_stems)
        return

    print(f"Images to process: {len(todo)}  workers: {args.workers}", flush=True)

    # Load index + smoother once — shared read-only across all threads
    from src.indexing.base import load_index
    from src.smoothing.manifold import ManifoldSmoother
    dim = args.image_size * args.image_size * 3
    index = load_index(dim=dim, index_path=str(index_path), backend="annoy")
    smoother = ManifoldSmoother(sigma=0.25, index=index, knn_k=args.knn_k, eps_eig=args.eps_eig)
    print("Index loaded — starting threads.", flush=True)

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_one, p, args.image_size, smoother, scratch_dir): p
            for p in todo
        }
        pbar = tqdm(total=len(todo), desc="Computing PCA", unit="img")
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                img_path = futures[future]
                print(f"\nWARNING: failed {img_path.name}: {exc}", flush=True)
                failed += 1
            pbar.update(1)
        pbar.close()

    print(f"Computation done. Failed: {failed}", flush=True)

    # Merge scratch -> final .npz  (streams one image at a time, O(1) RAM)
    merge_scratch_to_npz(scratch_dir, cache_file, all_stems)
    print(f"Cache file: {cache_file}", flush=True)


if __name__ == "__main__":
    main()
