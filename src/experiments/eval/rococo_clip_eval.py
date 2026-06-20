"""RoCOCO CLIP evaluation with randomized smoothing.

Three modes (set via config smoothing.mode):
  baseline  — standard CLIP retrieval, no smoothing
  isotropic — isotropic Gaussian noise on image embeddings
  manifold  — manifold noise via kNN PCA on image embeddings
  dropout   — random dropout ablation on image embeddings

For each mode evaluates on all 5 annotation files:
  coco_karpathy_test → R@1 (baseline, no RSMS/DropRate)
  danger / same_concept / diff_concept / rand_voca → R@1, RSMS, DropRate

Outputs:
  output/rococo/{mode}/{ann_stem}_metrics.json

Usage:
    # Single annotation file (used by Slurm array job)
    python -m src.experiments.eval.rococo_clip_eval \\
        --config src/configs/experiments/rococo_clip_isotropic.yaml \\
        --ann-file danger.json

    # All annotation files at once
    python -m src.experiments.eval.rococo_clip_eval \\
        --config src/configs/experiments/rococo_clip_manifold.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.rococo_clip_schema import RoCoCoConfig, load_rococo_config, save_rococo_config
from src.dataloaders.rococo import RoCoCoDataset
from src.smoothing.isotropic import IsotropicSmoother
from src.smoothing.manifold import ManifoldSmoother


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def compute_retrieval(
    image_embs: np.ndarray,
    text_embs:  np.ndarray,
    img2txt:    Dict[int, List[int]],
    adv_text_indices: Optional[Set[int]] = None,
    checkpoint_every: int = 10,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Compute R@1, R@5, R@10, RSMS, DropRate.

    Saves partial metrics every checkpoint_every images to checkpoint_path.
    """
    N      = image_embs.shape[0]
    scores = image_embs @ text_embs.T   # (N, M)

    r1 = r5 = r10 = rsms = 0
    n_valid = 0

    for i in range(N):
        gt_indices = set(img2txt.get(i, []))
        if not gt_indices:
            continue
        n_valid += 1
        ranked  = np.argsort(-scores[i])

        gt_rank = min(np.where(np.isin(ranked, list(gt_indices)))[0])
        if gt_rank < 1:  r1  += 1
        if gt_rank < 5:  r5  += 1
        if gt_rank < 10: r10 += 1

        if adv_text_indices is not None:
            top1 = ranked[0]
            if top1 in adv_text_indices: rsms += 1

        # Save partial every checkpoint_every images
        if checkpoint_path is not None and checkpoint_every > 0:
            if (i + 1) % checkpoint_every == 0 or (i + 1) == N:
                partial = {
                    "processed": i + 1, "total": N,
                    "R@1":  round(r1  / n_valid * 100, 2),
                    "R@5":  round(r5  / n_valid * 100, 2),
                    "R@10": round(r10 / n_valid * 100, 2),
                }
                if adv_text_indices is not None:
                    partial["RSMS"] = round(rsms / n_valid * 100, 2)
                checkpoint_path.write_text(json.dumps(partial, indent=2))

    total = n_valid if n_valid > 0 else N
    metrics = {
        "R@1":  round(r1  / total * 100, 2),
        "R@5":  round(r5  / total * 100, 2),
        "R@10": round(r10 / total * 100, 2),
    }
    if adv_text_indices is not None:
        # RSMS: fraction of images where adversarial caption ranked top-1
        metrics["RSMS"] = round(rsms / total * 100, 2)
        # DropRate is computed LATER in the notebook/analysis:
        # DropRate = (R@1_coco_karpathy_test - R@1_this_ann) / R@1_coco_karpathy_test
        # It requires comparing two separate eval runs, not computable per-image.
    return metrics


# ── Smoothing helpers ─────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-12)


def smooth_iso(
    emb: np.ndarray,
    smoother: IsotropicSmoother,
    n_samples: int = 1,
) -> np.ndarray:
    """Smooth once (n_samples=1) or average N samples for a more stable query.
    For retrieval eval, n_samples=1 is standard — just adds noise once.
    Averaging N>1 samples reduces variance but costs N× more compute.
    """
    if n_samples == 1:
        return _normalize(smoother.sample(emb))
    samples = np.stack([smoother.sample(emb) for _ in range(n_samples)])
    return _normalize(samples.mean(axis=0))


def smooth_manifold(
    emb: np.ndarray,
    smoother: ManifoldSmoother,
    n_samples: int = 1,
) -> np.ndarray:
    """PCA computed once, then sample once (or average N samples)."""
    cached = smoother.compute_pca(emb)
    if n_samples == 1:
        return _normalize(smoother.sample_from_cached(cached))
    samples = np.stack([smoother.sample_from_cached(cached) for _ in range(n_samples)])
    return _normalize(samples.mean(axis=0))


def smooth_dropout(
    emb: np.ndarray,
    dropout_rate: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    mask    = rng.binomial(1, 1.0 - dropout_rate, size=emb.shape).astype(np.float32)
    dropped = emb * mask
    return _normalize(dropped)


# ── Smooth all image embeddings ───────────────────────────────────────────────

def get_smoothed_embeddings(
    image_embs: np.ndarray,
    cfg: RoCoCoConfig,
    pixel_index=None,
) -> np.ndarray:
    """Return smoothed image embeddings based on cfg.smoothing.mode.

    n_samples: number of noisy embeddings averaged to produce one stable
    smoothed query — lower than certification (10 is sufficient for retrieval).
    """
    mode = cfg.smoothing.mode

    if mode == "baseline":
        return image_embs

    N, D = image_embs.shape
    smoothed = np.zeros_like(image_embs)

    if mode == "isotropic":
        smoother = IsotropicSmoother(sigma=cfg.smoothing.sigma)
        for i in tqdm(range(N), desc=f"ISO smoothing (n={cfg.smoothing.n_samples})"):
            smoothed[i] = smooth_iso(image_embs[i], smoother, cfg.smoothing.n_samples)

    elif mode == "manifold":
        if pixel_index is None:
            raise ValueError("Manifold mode requires kNN index on CLIP embeddings")
        smoother = ManifoldSmoother(
            sigma=cfg.smoothing.sigma,
            index=pixel_index,
            knn_k=cfg.smoothing.knn_k,
            eps_eig=cfg.smoothing.eps_eig,
        )
        for i in tqdm(range(N), desc=f"Manifold smoothing (n={cfg.smoothing.n_samples})"):
            smoothed[i] = smooth_manifold(image_embs[i], smoother, cfg.smoothing.n_samples)

    else:
        raise ValueError(f"Unknown smoothing mode: {mode}  (valid: baseline | isotropic | manifold)")

    return smoothed


# ── Per-annotation evaluation ─────────────────────────────────────────────────

PARTIAL_FILE = "results.partial.json"


def _save_partial(out_dir: Path, ann_stem: str, processed: int,
                  total: int, running: Dict) -> None:
    p = out_dir / f"{ann_stem}_partial.json"
    p.write_text(json.dumps({
        "processed": processed, "total": total, "running": running
    }, indent=2))


def _load_partial(out_dir: Path, ann_stem: str, total: int) -> Optional[Dict]:
    p = out_dir / f"{ann_stem}_partial.json"
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text())
        if s.get("total") != total:
            return None
        return s
    except Exception:
        return None


def evaluate_on_annotation(
    smoothed_image_embs: np.ndarray,
    dataset: RoCoCoDataset,
    ann_stem: str,
    cache_dir: Path,
    out_dir: Optional[Path] = None,
    checkpoint_every: int = 10,
) -> Dict:
    """Load cached caption embeddings and compute retrieval metrics."""
    cap_path = cache_dir / f"{ann_stem}_captions.pt"
    if not cap_path.exists():
        _log(f"  SKIP {ann_stem} — caption embeddings not found: {cap_path}")
        return {}

    cap_cache = torch.load(cap_path, map_location="cpu")
    text_embs = cap_cache["embeddings"].numpy().astype(np.float32)

    # New format: img2txt and wrongtext saved directly
    if "img2txt" in cap_cache:
        img2txt   = cap_cache["img2txt"]          # {img_idx: [cap_idx, ...]}
        wrongtext = cap_cache.get("wrongtext", [])
    else:
        # Legacy format: reconstruct from image_indices
        img_indices = cap_cache["image_indices"]
        img2txt = {}
        for cap_idx, img_idx in enumerate(img_indices):
            img2txt.setdefault(img_idx, []).append(cap_idx)
        wrongtext = [] if ann_stem == "coco_karpathy_test" else list(range(len(text_embs)))

    adv_set: Optional[Set[int]] = set(wrongtext) if wrongtext else None

    # Compute metrics with checkpointing every checkpoint_every images
    ckpt_path = (out_dir / f"{ann_stem}_partial.json") if out_dir else None
    metrics   = compute_retrieval(smoothed_image_embs, text_embs, img2txt, adv_set,
                                  checkpoint_every=checkpoint_every,
                                  checkpoint_path=ckpt_path)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def _sigma_tag(s: float) -> str:
    return f"sigma_{s:.2f}".replace(".", "_")


def run_evaluation(cfg: RoCoCoConfig, ann_files: Optional[List[str]] = None,
                   n_eval: Optional[int] = None) -> None:
    """Run evaluation for all sigmas in cfg.smoothing.sigma_values (or single sigma).

    Output structure:
      output/rococo/{mode}/sigma_{s}/{ann_stem}_metrics.json
    """
    mode = cfg.smoothing.mode

    # Resolve sigma list
    if cfg.smoothing.sigma_values:
        sigma_values = [float(s) for s in cfg.smoothing.sigma_values]
    else:
        sigma_values = [float(cfg.smoothing.sigma)]

    _log(f"RoCOCO eval  mode={mode}  sigmas={sigma_values}  "
         f"knn_k={cfg.smoothing.knn_k}  n_samples={cfg.smoothing.n_samples}")

    cache_dir = Path(cfg.embedding_cache_dir)

    # Load image embeddings once — shared across all sigmas and all annotations
    emb_path = cache_dir / "image_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Image embeddings not found: {emb_path}. "
                                f"Run generate_embeddings.py first.")
    cache         = torch.load(emb_path, map_location="cpu")
    all_image_embs = cache["embeddings"].numpy().astype(np.float32)
    all_image_ids  = cache["image_ids"]   # full list — needed for kNN index lookup

    if n_eval is not None and n_eval < len(all_image_embs):
        image_embs = all_image_embs[:n_eval]
        image_ids  = all_image_ids[:n_eval]
        _log(f"Eval subset: {n_eval} images (index still has {len(all_image_ids)})")
    else:
        image_embs = all_image_embs
        image_ids  = all_image_ids
    _log(f"Image embeddings: {image_embs.shape}")

    # Load kNN index once (manifold only)
    pixel_index = None
    if mode == "manifold":
        from src.experiments.indexing.rococo_clip_images import build_rococo_clip_index
        pixel_index = build_rococo_clip_index(cfg, rebuild=False)
        _log(f"kNN index: {pixel_index.index.get_n_items():,} vectors")

    dataset = RoCoCoDataset(
        image_dir=cfg.image_dir,
        annotation_dir=cfg.annotation_dir,
        annotation_files=cfg.annotation_files,
    )

    ann_to_eval = ann_files or cfg.annotation_files

    # ── Loop over sigmas — smooth once per sigma, evaluate all annotations ────
    for sigma in sigma_values:
        cfg.smoothing.sigma = sigma
        sigma_dir = _ROOT / cfg.output_dir / mode / _sigma_tag(sigma)
        sigma_dir.mkdir(parents=True, exist_ok=True)

        # Check if all annotations already done for this sigma
        pending = [Path(af).stem for af in ann_to_eval
                   if not (sigma_dir / f"{Path(af).stem}_metrics.json").exists()]
        if not pending:
            _log(f"SKIP sigma={sigma} — all annotations done")
            continue

        _log(f"\n── sigma={sigma} ──────────────────────────────────────")
        t0 = time.time()
        smoothed = get_smoothed_embeddings(image_embs, cfg, pixel_index)
        _log(f"  Smoothing done in {time.time()-t0:.1f}s")

        for ann_file in ann_to_eval:
            ann_stem     = Path(ann_file).stem
            metrics_path = sigma_dir / f"{ann_stem}_metrics.json"
            if metrics_path.exists():
                _log(f"  SKIP {ann_stem} — exists")
                continue

            _log(f"  Evaluating {ann_stem} …")
            metrics = evaluate_on_annotation(smoothed, dataset, ann_stem, cache_dir,
                                             out_dir=sigma_dir, checkpoint_every=10)
            if not metrics:
                continue

            output = {
                "experiment": cfg.experiment_name,
                "mode":       mode,
                "ann_stem":   ann_stem,
                "clip_model": cfg.clip_model,
                "smoothing": {
                    "sigma":     sigma,
                    "n_samples": cfg.smoothing.n_samples,
                    "knn_k":     cfg.smoothing.knn_k,
                },
                "n_images": len(image_ids),
                **metrics,
            }
            metrics_path.write_text(json.dumps(output, indent=2))
            _log(f"    {ann_stem}: R@1={metrics.get('R@1')}%  RSMS={metrics.get('RSMS','–')}%")

            # Save visualizations into sigma_dir/visualizations/
            if cfg.output.save_visualizations if hasattr(cfg, 'output') else True:
                try:
                    from src.experiments.eval.rococo_viz import run_viz
                    run_viz(cfg, ann_file,
                            n_show=10,
                            smoothed_image_embs=smoothed,
                            image_ids=image_ids,        # eval subset (100)
                            all_image_ids=all_image_ids, # full 5k for kNN lookup
                            viz_dir=sigma_dir / "visualizations",
                            pixel_index=pixel_index)
                except Exception as e:
                    _log(f"    Viz skipped: {e}")

    _log(f"Done. Results in output/rococo/{mode}/")


def parse_args():
    p = argparse.ArgumentParser(description="RoCOCO CLIP retrieval evaluation")
    p.add_argument("--config",   required=True)
    p.add_argument("--ann-file", default=None,
                   help="Single annotation file (e.g. danger.json).")
    p.add_argument("--sigma",    type=float, default=None,
                   help="Override sigma for this run (used by Slurm array job).")
    p.add_argument("--n-eval",   type=int, default=None,
                   help="Evaluate on first N images only. Use 100 for fast testing.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_rococo_config(args.config)
    # CLI --sigma overrides config sigma_values → single sigma run
    if args.sigma is not None:
        cfg.smoothing.sigma_values = [args.sigma]
        cfg.smoothing.sigma        = args.sigma
    ann_f = [args.ann_file] if args.ann_file else None
    run_evaluation(cfg, ann_files=ann_f, n_eval=args.n_eval)
