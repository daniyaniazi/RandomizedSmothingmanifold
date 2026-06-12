"""CelebAMask-HQ Segmentation Certification Experiment.

Implements SEGCERTIFY (Fischer et al. 2021, Algorithm 2) with:
  - Pixel-space isotropic smoothing
  - Pixel-space manifold smoothing (PCA via kNN index)
  - Holm-Bonferroni FWER correction per image
  - Metrics: per-pixel accuracy, certified accuracy, mIoU, certified mIoU
  - Visualizations: noisy images, segmentation masks, geometry plots

Output Directory Structure:
    output/segmentation/celebahq/
    ├── index/
    │   └── pixel/annoy/euclidean/
    └── certify/
        ├── pixel_isotropic/sigma_0_25/
        └── pixel_manifold/sigma_0_25/
            ├── metrics.json
            ├── results.csv
            ├── config.yaml
            └── visualizations/
                ├── sample_0000.png
                └── sample_0000_geometry.png   (manifold only)

Usage:
    python -m src.experiments.certify.celebahq_segmentation \\
        --config src/configs/experiments/certify_celebahq_seg_isotropic.yaml

    python -m src.experiments.certify.celebahq_segmentation \\
        --config src/configs/experiments/certify_celebahq_seg_manifold.yaml \\
        --sigmas 0.10 0.25 0.50
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.certify_seg_io import load_seg_certify_config, save_seg_certify_config
from src.configs.certify_seg_schema import SegCertifyConfig, SEG_CLASS_NAMES
from src.certify.segmentation import (
    segcertify, SegCertResult,
    pixel_accuracy, mean_iou, per_class_iou,
    certified_radius_seg,
)
from src.dataloaders.celebahq_seg import (
    build_seg_dataloaders, _load_mask, CLASS_NAMES, N_CLASSES,
)
from src.indexing.base import load_index, NeighborIndex
from src.indexing.image_index import build_or_load_image_index
from src.smoothing.isotropic import IsotropicSmoother
from src.smoothing.manifold import ManifoldSmoother


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Paths ─────────────────────────────────────────────────────────────────────

PARTIAL_FILE = "results.partial.json"


@dataclass
class SegPaths:
    base_dir: Path
    index_dir: Path
    pixel_index_dir: Path
    certify_dir: Path
    experiment_dir: Path

    @classmethod
    def from_config(cls, cfg: SegCertifyConfig, sigma: float) -> "SegPaths":
        dataset_tag = cfg.dataset.name.lower().replace("-", "").replace("_", "")
        base_dir = Path(cfg.output.output_dir) / "segmentation" / dataset_tag
        index_dir = base_dir / "index"
        pixel_index_dir = index_dir / "pixel" / cfg.index.backend / cfg.index.metric
        certify_dir = base_dir / "certify"
        sigma_tag = f"sigma_{sigma:.2f}".replace(".", "_")
        mode_tag = f"pixel_{'manifold' if cfg.smoothing.use_manifold else 'isotropic'}"
        experiment_dir = certify_dir / mode_tag / sigma_tag
        return cls(base_dir, index_dir, pixel_index_dir, certify_dir, experiment_dir)

    def ensure_dirs(self):
        for d in [self.pixel_index_dir, self.experiment_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ── Partial checkpoint ────────────────────────────────────────────────────────

def _save_partial(exp_dir: Path, next_idx: int, results: List[Dict],
                  num_samples: int) -> None:
    state = {"next_idx": next_idx, "results": results,
             "num_samples": num_samples, "timestamp": datetime.now().isoformat()}
    (exp_dir / PARTIAL_FILE).write_text(json.dumps(state, indent=2))


def _load_partial(exp_dir: Path, num_samples: int) -> Optional[Dict]:
    path = exp_dir / PARTIAL_FILE
    if not path.exists():
        return None
    try:
        s = json.loads(path.read_text())
        if s.get("num_samples") != num_samples:
            return None
        return s
    except Exception:
        return None


def _remove_partial(exp_dir: Path) -> None:
    p = exp_dir / PARTIAL_FILE
    if p.exists():
        p.unlink()


# ── Index helpers (reuse image_index pipeline) ────────────────────────────────

def _build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_or_build_pixel_index(
    train_samples: List[Tuple[str, int]],
    image_size: int,
    index_dir: Path,
    n_trees: int,
    metric: str,
) -> NeighborIndex:
    index_path = index_dir / "index.ann"
    fnames = [str(p) for p, _ in train_samples]
    dim = 3 * image_size * image_size

    if index_path.exists():
        _log(f"Loading pixel index: {index_path}")
        idx = load_index(dim=dim, index_path=str(index_path), backend="annoy", metric=metric)
        idx.filenames = fnames
        return idx

    _log(f"Building pixel index from {len(train_samples)} images …")
    index_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.data import Dataset, DataLoader
    tf = _build_transform(image_size)

    class _ImgDs(Dataset):
        def __init__(self, samples):
            self.s = samples
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            p, _ = self.s[i]
            return {"image": tf(Image.open(p).convert("RGB")), "label": 0, "path": p}

    loader = DataLoader(_ImgDs(train_samples), batch_size=32, shuffle=False, num_workers=4)
    artifacts = build_or_load_image_index(
        out_dir=index_dir, dataloader=loader, space="pixel",
        backend="annoy", metric=metric, index_path=str(index_path),
        n_trees=n_trees, rebuild=False,
        metadata={"image_size": image_size, "split": "train"},
    )
    artifacts.index.filenames = fnames
    return artifacts.index


# ── BiSeNet loader ────────────────────────────────────────────────────────────

def load_bisenet(cfg: SegCertifyConfig, device: torch.device) -> torch.nn.Module:
    """Load BiSeNet from checkpoint.

    The model file uses a relative import `from resnet import Resnet18`.
    We add the model directory to sys.path to handle this.
    """
    bisenet_dir = str(_ROOT / "src" / "models" / "BiseNet")
    if bisenet_dir not in sys.path:
        sys.path.insert(0, bisenet_dir)

    from model import BiSeNet  # type: ignore

    net = BiSeNet(n_classes=cfg.model.n_classes)
    ckpt_path = Path(cfg.model.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"BiSeNet checkpoint not found: {ckpt_path}\n"
            f"Download from https://github.com/zllrunning/face-parsing.PyTorch"
        )
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    _log(f"BiSeNet loaded: {ckpt_path}  ({cfg.model.n_classes} classes)")
    return net


# ── Noisy image sampling ──────────────────────────────────────────────────────

def _sample_noisy(
    img_tensor: torch.Tensor,    # (C, H, W) in [0,1]
    smoother,
    pixel_index: Optional[NeighborIndex],
) -> torch.Tensor:
    """Return one noisy image tensor (C, H, W)."""
    flat = img_tensor.numpy().flatten().astype(np.float32)
    if isinstance(smoother, ManifoldSmoother) and pixel_index is not None:
        noisy_flat = smoother.sample(flat)
    else:
        noisy_flat = smoother.sample(flat)
    return torch.from_numpy(noisy_flat.reshape(img_tensor.shape)).float().clamp(0, 1)


def sample_counts(
    net: torch.nn.Module,
    img_tensor: torch.Tensor,      # (C, H, W) [0,1]
    smoother,
    n_samples: int,
    n_classes: int,
    image_size: int,
    device: torch.device,
    cached_pca=None,               # pre-computed PCA for manifold smoother
) -> np.ndarray:
    """Generate n_samples noisy predictions and accumulate per-pixel class counts.

    Returns (H, W, C) int32 array of class vote counts.
    """
    H = W = image_size
    counts = np.zeros((H, W, n_classes), dtype=np.int32)
    img_np = img_tensor.numpy().flatten().astype(np.float32)

    # BiSeNet expects ImageNet-normalised input
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for _ in range(n_samples):
        # Sample noisy image
        if cached_pca is not None and isinstance(smoother, ManifoldSmoother):
            noisy_flat = smoother.sample_from_cached(cached_pca)
        else:
            noisy_flat = smoother.sample(img_np)
        noisy = torch.from_numpy(noisy_flat.reshape(img_tensor.shape)).float().clamp(0, 1)

        # Normalise for BiSeNet
        noisy_norm = (noisy - _mean) / _std
        x = noisy_norm.unsqueeze(0).to(device)

        with torch.no_grad():
            out, _, _ = net(x)          # (1, C, H, W)
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)  # (H, W)

        # Accumulate counts
        for c in range(n_classes):
            counts[:, :, c] += (pred == c).astype(np.int32)

    return counts


# ── Visualisation ─────────────────────────────────────────────────────────────

# CelebAMask-HQ colour palette (19 classes)
SEG_PALETTE = np.array([
    [0,   0,   0],    # 0 background
    [255, 0,   0],    # 1 skin
    [255, 85,  0],    # 2 l_brow
    [255, 170, 0],    # 3 r_brow
    [255, 0,  85],    # 4 l_eye
    [255, 0,  170],   # 5 r_eye
    [0,   255, 0],    # 6 eye_g
    [85,  255, 0],    # 7 l_ear
    [170, 255, 0],    # 8 r_ear
    [0,   255, 85],   # 9 ear_r
    [0,   255, 170],  # 10 nose
    [0,   0,   255],  # 11 mouth
    [85,  0,   255],  # 12 u_lip
    [170, 0,   255],  # 13 l_lip
    [0,   85,  255],  # 14 neck
    [0,   170, 255],  # 15 neck_l
    [255, 255, 0],    # 16 cloth
    [255, 0,   255],  # 17 hair
    [0,   255, 255],  # 18 hat
], dtype=np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) class index mask → (H, W, 3) RGB."""
    rgb = SEG_PALETTE[mask.clip(0, len(SEG_PALETTE) - 1)]
    return rgb


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_seg_visualization(
    viz_dir: Path,
    sample_idx: int,
    img_tensor: torch.Tensor,       # (C, H, W)
    gt_mask: torch.Tensor,          # (H, W)
    pred_mask: np.ndarray,          # (H, W) majority vote
    cert_result: SegCertResult,
    nn_imgs: List[torch.Tensor],    # neighbour raw images
    nn_masks: List[np.ndarray],     # neighbour predicted masks
    noisy_imgs: List[torch.Tensor], # a few noisy samples for display
    noisy_masks: List[np.ndarray],  # masks for noisy samples
    is_manifold: bool,
    sigma: float,
    pca_cached=None,                # for geometry plot
    pixel_index=None,
    smoother=None,
    cfg=None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    viz_dir.mkdir(parents=True, exist_ok=True)
    n_noisy = len(noisy_imgs)

    # ── Layout:
    # Row 0: Original | GT mask
    # Row 1: PCA recon (manifold) | Certified mask
    # Row 2..2+n_noisy-1: Noisy image | Noisy mask
    # Row last: NN images | NN masks

    n_nn = len(nn_imgs)
    n_rows = 2 + n_noisy + (1 if n_nn > 0 else 0)
    n_cols = max(2, n_nn, n_noisy)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")

    def _show(ax, img_arr, title="", border=None):
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=7, pad=2)
        ax.axis("off")
        if border:
            for sp in ax.spines.values():
                sp.set_edgecolor(border); sp.set_linewidth(2); sp.set_visible(True)

    # Row 0: original + GT
    _show(axes[0, 0], _tensor_to_pil(img_tensor), "Original", "gold")
    _show(axes[0, 1], Image.fromarray(_mask_to_rgb(gt_mask.numpy())), "GT mask")

    # Row 1: certified mask (col 0) + abstain overlay (col 1)
    cert_rgb = _mask_to_rgb(pred_mask)
    cert_rgb[~cert_result.certified] = [128, 128, 128]   # grey = abstain
    _show(axes[1, 0], Image.fromarray(cert_rgb),
          f"Certified mask  R={cert_result.radius:.3f}")
    # Abstain overlay on original
    overlay = np.array(_tensor_to_pil(img_tensor)).copy()
    overlay[~cert_result.certified] = [200, 200, 200]
    _show(axes[1, 1], Image.fromarray(overlay),
          f"Abstain: {cert_result.abstain_rate*100:.1f}%")

    # Rows 2+: noisy images + their masks
    for i, (ni, nm) in enumerate(zip(noisy_imgs, noisy_masks)):
        r = 2 + i
        if r >= n_rows - (1 if n_nn > 0 else 0):
            break
        _show(axes[r, 0], _tensor_to_pil(ni), f"Noisy {i+1}  σ={sigma}", "#4c78a8")
        _show(axes[r, 1], Image.fromarray(_mask_to_rgb(nm)), f"Seg {i+1}")

    # Last row: neighbours
    if n_nn > 0:
        r = n_rows - 1
        for j, (ni, nm) in enumerate(zip(nn_imgs, nn_masks)):
            if j >= n_cols:
                break
            _show(axes[r, j], _tensor_to_pil(ni), f"NN-{j+1}", "#e07b54")

    cert_px = cert_result.n_certified
    total_px = cert_result.n_pixels
    fig.suptitle(
        f"Sample {sample_idx}  |  σ={sigma}  |  {'Manifold' if is_manifold else 'Isotropic'}\n"
        f"Certified: {cert_px}/{total_px} px ({100*(1-cert_result.abstain_rate):.1f}%)  "
        f"R={cert_result.radius:.4f}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(viz_dir / f"sample_{sample_idx:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # ── Geometry plot (manifold only) ─────────────────────────────────────────
    if is_manifold and pca_cached is not None and smoother is not None:
        try:
            from src.experiments.certify.celeba import _draw_geometry_figure
            _draw_geometry_figure(
                pca_obj=pca_cached.pca,
                nbrs=pca_cached.neighbors,
                query_vec=img_tensor.numpy().flatten().astype(np.float32),
                smoother=smoother,
                cached_pca=pca_cached,
                sigma=sigma,
                n_mc=cfg.smoothing.n_samples if cfg else 20,
                save_path=viz_dir / f"sample_{sample_idx:04d}_geometry.png",
                sample_idx=sample_idx,
                space_label="Pixel",
                ood_attr_name="",
                ood_attr_map=None,
                index=pixel_index,
            )
        except Exception as e:
            _log(f"Geometry plot skipped for sample {sample_idx}: {e}")


# ── Main certification pipeline ───────────────────────────────────────────────

def run_seg_certification(cfg: SegCertifyConfig, sigma: float) -> Dict:
    """Run SEGCERTIFY for one sigma value."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}  sigma={sigma}")

    paths = SegPaths.from_config(cfg, sigma)
    paths.ensure_dirs()

    # Skip if already done
    if (paths.experiment_dir / "metrics.json").exists():
        _log(f"SKIP sigma={sigma} — metrics.json exists: {paths.experiment_dir}")
        return {}

    save_seg_certify_config(cfg, paths.experiment_dir / "config.yaml")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data = build_seg_dataloaders(cfg)
    train_samples = data["train_samples"]
    test_samples  = data["test_samples"]
    mask_dir      = data["mask_dir"]
    image_size    = data["image_size"]
    _log(f"Train: {len(train_samples)}  Test: {len(test_samples)}")

    # ── BiSeNet ───────────────────────────────────────────────────────────────
    net = load_bisenet(cfg, device)

    # ── Index (manifold only) ─────────────────────────────────────────────────
    pixel_index = None
    if cfg.smoothing.use_manifold:
        pixel_index = load_or_build_pixel_index(
            train_samples, image_size, paths.pixel_index_dir,
            cfg.index.n_trees, cfg.index.metric,
        )

    # ── Smoother ──────────────────────────────────────────────────────────────
    if cfg.smoothing.use_manifold and pixel_index is not None:
        smoother = ManifoldSmoother(
            sigma=sigma, index=pixel_index,
            knn_k=cfg.smoothing.knn_k, eps_eig=cfg.smoothing.eps_eig,
        )
    else:
        smoother = IsotropicSmoother(sigma=sigma)

    tf = _build_transform(image_size)

    # ── Resume ────────────────────────────────────────────────────────────────
    results: List[Dict] = []
    start_idx = 0
    if cfg.checkpoint.resume:
        partial = _load_partial(paths.experiment_dir, len(test_samples))
        if partial:
            start_idx = partial["next_idx"]
            results   = partial["results"]
            _log(f"Resuming from sample {start_idx}/{len(test_samples)}")

    checkpoint_every = cfg.checkpoint.checkpoint_every if cfg.checkpoint.enabled else 0

    # ── Certification loop ────────────────────────────────────────────────────
    t_start = time.time()

    for idx in tqdm(range(start_idx, len(test_samples)),
                    desc=f"SegCertify σ={sigma}", initial=start_idx, total=len(test_samples)):
        img_path, image_id = test_samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = tf(img)                       # (C, H, W) [0,1]
        gt_mask = _load_mask(mask_dir, image_id, image_size)  # (H, W)

        # Pre-compute PCA once for manifold
        cached_pca = None
        if isinstance(smoother, ManifoldSmoother):
            flat = img_tensor.numpy().flatten().astype(np.float32)
            cached_pca = smoother.compute_pca(flat)

        # ── Pilot samples (n0) ────────────────────────────────────────────────
        counts_n0 = sample_counts(
            net, img_tensor, smoother, cfg.smoothing.n0_samples,
            cfg.dataset.n_classes, image_size, device, cached_pca,
        )
        # ── Main MC samples (n) ───────────────────────────────────────────────
        counts_n = sample_counts(
            net, img_tensor, smoother, cfg.smoothing.n_samples,
            cfg.dataset.n_classes, image_size, device, cached_pca,
        )

        # ── SEGCERTIFY (Holm correction) ──────────────────────────────────────
        cert = segcertify(
            counts_n0=counts_n0,
            counts_n=counts_n,
            sigma=sigma,
            tau=cfg.smoothing.tau,
            alpha=cfg.alpha_conf,
            correction="holm",
        )

        # ── Metrics for this image ────────────────────────────────────────────
        gt_np = gt_mask.numpy()
        acc_all   = pixel_accuracy(cert.pred_mask, gt_np)
        acc_cert  = pixel_accuracy(cert.pred_mask, gt_np, cert.certified)
        miou_all  = mean_iou(cert.pred_mask, gt_np, cfg.dataset.n_classes)
        miou_cert = mean_iou(cert.pred_mask, gt_np, cfg.dataset.n_classes, cert.certified)
        iou_per_class = per_class_iou(cert.pred_mask, gt_np, cfg.dataset.n_classes).tolist()

        result = {
            "idx":            idx,
            "image_id":       image_id,
            "image_path":     img_path,
            "radius":         cert.radius,
            "n_pixels":       cert.n_pixels,
            "n_certified":    cert.n_certified,
            "n_abstained":    cert.n_abstained,
            "abstain_rate":   cert.abstain_rate,
            "pixel_acc":      acc_all,
            "certified_pixel_acc":  acc_cert,
            "miou":           miou_all,
            "certified_miou": miou_cert,
            "iou_per_class":  iou_per_class,
        }
        results.append(result)

        # ── Visualisation (first N samples) ───────────────────────────────────
        if cfg.output.save_visualizations and idx < cfg.output.num_viz_samples:
            # Generate a few noisy samples for display
            noisy_imgs, noisy_masks = [], []
            _mean_n = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            _std_n  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            for _ in range(3):
                ni = _sample_noisy(img_tensor, smoother, pixel_index)
                noisy_imgs.append(ni)
                x_n = ((ni - _mean_n) / _std_n).unsqueeze(0).to(device)
                with torch.no_grad():
                    nm_out, _, _ = net(x_n)
                    nm = nm_out.argmax(1).squeeze(0).cpu().numpy()
                noisy_masks.append(nm)

            # Nearest neighbours from index
            nn_imgs, nn_masks = [], []
            if pixel_index is not None and hasattr(pixel_index, "index"):
                qvec = img_tensor.numpy().flatten().astype(np.float32)
                nn_ids = pixel_index.index.get_nns_by_vector(qvec.tolist(), 4, include_distances=False)
                for nid in nn_ids[:3]:
                    if hasattr(pixel_index, "filenames"):
                        nn_path = pixel_index.filenames[nid]
                        nn_img_t = tf(Image.open(nn_path).convert("RGB"))
                        nn_imgs.append(nn_img_t)
                        x_nn = (((nn_img_t - _mean_n) / _std_n).unsqueeze(0).to(device))
                        with torch.no_grad():
                            nn_out, _, _ = net(x_nn)
                            nn_masks.append(nn_out.argmax(1).squeeze(0).cpu().numpy())

            save_seg_visualization(
                viz_dir=paths.experiment_dir / "visualizations",
                sample_idx=idx,
                img_tensor=img_tensor,
                gt_mask=gt_mask,
                pred_mask=cert.pred_mask,
                cert_result=cert,
                nn_imgs=nn_imgs,
                nn_masks=nn_masks,
                noisy_imgs=noisy_imgs,
                noisy_masks=noisy_masks,
                is_manifold=cfg.smoothing.use_manifold,
                sigma=sigma,
                pca_cached=cached_pca,
                pixel_index=pixel_index,
                smoother=smoother,
                cfg=cfg,
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
            _save_partial(paths.experiment_dir, idx + 1, results, len(test_samples))

    elapsed = time.time() - t_start
    n_proc = len(test_samples) - start_idx

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total = len(results)
    mean_acc       = float(np.mean([r["pixel_acc"]           for r in results])) if results else 0.0
    mean_cert_acc  = float(np.mean([r["certified_pixel_acc"] for r in results])) if results else 0.0
    mean_miou      = float(np.mean([r["miou"]                for r in results])) if results else 0.0
    mean_cert_miou = float(np.mean([r["certified_miou"]      for r in results])) if results else 0.0
    mean_abstain   = float(np.mean([r["abstain_rate"]        for r in results])) if results else 0.0

    # Per-class mIoU averaged over images
    iou_stack = np.array([r["iou_per_class"] for r in results], dtype=np.float64)  # (N, C)
    mean_iou_per_class = np.nanmean(iou_stack, axis=0).tolist()

    radius = certified_radius_seg(sigma, cfg.smoothing.tau)

    metrics = {
        # ── Identity ──────────────────────────────────────────────────────────
        "experiment":   cfg.experiment_name,
        "dataset":      cfg.dataset.name,
        "output_dir":   str(paths.experiment_dir),

        # ── Config ────────────────────────────────────────────────────────────
        "model": {
            "n_classes":       cfg.model.n_classes,
            "checkpoint":      cfg.model.checkpoint_path,
            "input_size":      cfg.model.input_size,
        },
        "smoothing": {
            "mode":            cfg.smoothing.mode,
            "use_manifold":    cfg.smoothing.use_manifold,
            "sigma":           sigma,
            "tau":             cfg.smoothing.tau,
            "n0_samples":      cfg.smoothing.n0_samples,
            "n_samples":       cfg.smoothing.n_samples,
            "knn_k":           cfg.smoothing.knn_k,
            "correction":      "holm",
            "alpha_conf":      cfg.alpha_conf,
        },
        "index": {
            "backend":         cfg.index.backend,
            "metric":          cfg.index.metric,
            "n_train_samples": len(train_samples),
        },
        "dataset_info": {
            "train_samples":   len(train_samples),
            "test_samples":    len(test_samples),
            "split_seed":      cfg.dataset.split_seed,
            "image_size":      image_size,
            "n_classes":       cfg.dataset.n_classes,
        },

        # ── Runtime ───────────────────────────────────────────────────────────
        "runtime": {
            "certify_seconds":    round(elapsed, 2),
            "certify_minutes":    round(elapsed / 60, 2),
            "samples_processed":  n_proc,
            "seconds_per_sample": round(elapsed / max(n_proc, 1), 4),
            "device":             str(device),
        },

        # ── Results ───────────────────────────────────────────────────────────
        "certified_radius":       radius,
        "total_test_samples":     total,
        "mean_pixel_acc":         mean_acc,
        "mean_certified_pixel_acc": mean_cert_acc,
        "mean_miou":              mean_miou,
        "mean_certified_miou":    mean_cert_miou,
        "mean_abstain_rate":      mean_abstain,
        "mean_iou_per_class":     {CLASS_NAMES[i]: v for i, v in enumerate(mean_iou_per_class)},
    }

    _log("=" * 70)
    _log(f"SEGCERTIFY RESULTS: {cfg.experiment_name}  sigma={sigma}")
    _log(f"  Pixel acc:          {mean_acc*100:.2f}%  (certified: {mean_cert_acc*100:.2f}%)")
    _log(f"  mIoU:               {mean_miou*100:.2f}%  (certified: {mean_cert_miou*100:.2f}%)")
    _log(f"  Mean abstain rate:  {mean_abstain*100:.2f}%")
    _log(f"  Certified radius:   {radius:.4f}")
    _log("=" * 70)

    if cfg.output.save_results:
        (paths.experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        _log(f"Metrics saved: {paths.experiment_dir / 'metrics.json'}")

        if cfg.output.save_per_sample:
            csv_path = paths.experiment_dir / "results.csv"
            fieldnames = ["idx", "image_id", "image_path", "radius",
                          "n_pixels", "n_certified", "n_abstained", "abstain_rate",
                          "pixel_acc", "certified_pixel_acc", "miou", "certified_miou"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            _log(f"Results CSV: {csv_path}")

    _remove_partial(paths.experiment_dir)
    return metrics


def run_seg_certification_multi_sigma(cfg: SegCertifyConfig, sigma_values: List[float]) -> List[Dict]:
    """Run SEGCERTIFY across multiple sigmas — one-time setup, PCA cached per sample."""
    sigma_values = sorted(set(float(s) for s in sigma_values))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}  sigmas={sigma_values}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data = build_seg_dataloaders(cfg)
    train_samples = data["train_samples"]
    test_samples  = data["test_samples"]
    mask_dir      = data["mask_dir"]
    image_size    = data["image_size"]
    tf = _build_transform(image_size)

    # ── BiSeNet (loaded once) ─────────────────────────────────────────────────
    net = load_bisenet(cfg, device)

    # ── Index (manifold only — loaded once) ──────────────────────────────────
    # Use paths from first sigma (index is shared)
    paths0 = SegPaths.from_config(cfg, sigma_values[0])
    paths0.pixel_index_dir.mkdir(parents=True, exist_ok=True)
    pixel_index = None
    if cfg.smoothing.use_manifold:
        pixel_index = load_or_build_pixel_index(
            train_samples, image_size, paths0.pixel_index_dir,
            cfg.index.n_trees, cfg.index.metric,
        )

    # ── Per-sigma state ───────────────────────────────────────────────────────
    sigma_states: Dict[float, Dict] = {}
    active_sigmas: List[float] = []
    for sigma in sigma_values:
        paths = SegPaths.from_config(cfg, sigma)
        paths.ensure_dirs()
        if (paths.experiment_dir / "metrics.json").exists():
            _log(f"SKIP sigma={sigma} — already done")
            continue
        state: Dict = {"paths": paths, "results": [], "start_idx": 0}
        if cfg.checkpoint.resume:
            partial = _load_partial(paths.experiment_dir, len(test_samples))
            if partial:
                state["start_idx"] = partial["next_idx"]
                state["results"]   = partial["results"]
                _log(f"sigma={sigma}: resuming from {state['start_idx']}/{len(test_samples)}")
        save_seg_certify_config(cfg, paths.experiment_dir / "config.yaml")
        sigma_states[sigma] = state
        active_sigmas.append(sigma)

    if not active_sigmas:
        _log("All sigmas done.")
        return []

    global_start = min(s["start_idx"] for s in sigma_states.values())
    checkpoint_every = cfg.checkpoint.checkpoint_every if cfg.checkpoint.enabled else 0
    t_start = time.time()

    _mean_n = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _std_n  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx in tqdm(range(global_start, len(test_samples)),
                    desc="SegCertify multi-sigma", initial=global_start, total=len(test_samples)):
        img_path, image_id = test_samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = tf(img)
        gt_mask = _load_mask(mask_dir, image_id, image_size)
        gt_np   = gt_mask.numpy()

        # ── Pre-compute PCA once (manifold) — shared across all sigmas ────────
        flat = img_tensor.numpy().flatten().astype(np.float32)
        # Use sigma=1.0 as placeholder for PCA (only kNN structure matters)
        cached_pca = None
        if cfg.smoothing.use_manifold and pixel_index is not None:
            _tmp_sm = ManifoldSmoother(sigma=1.0, index=pixel_index,
                                       knn_k=cfg.smoothing.knn_k, eps_eig=cfg.smoothing.eps_eig)
            cached_pca = _tmp_sm.compute_pca(flat)

        for sigma in active_sigmas:
            state = sigma_states[sigma]
            if idx < state["start_idx"]:
                continue

            # Build sigma-specific smoother (cheap)
            if cfg.smoothing.use_manifold and pixel_index is not None:
                smoother = ManifoldSmoother(sigma=sigma, index=pixel_index,
                                            knn_k=cfg.smoothing.knn_k, eps_eig=cfg.smoothing.eps_eig)
            else:
                smoother = IsotropicSmoother(sigma=sigma)

            counts_n0 = sample_counts(net, img_tensor, smoother, cfg.smoothing.n0_samples,
                                      cfg.dataset.n_classes, image_size, device, cached_pca)
            counts_n  = sample_counts(net, img_tensor, smoother, cfg.smoothing.n_samples,
                                      cfg.dataset.n_classes, image_size, device, cached_pca)

            cert = segcertify(counts_n0, counts_n, sigma, cfg.smoothing.tau,
                              cfg.alpha_conf, correction="holm")

            acc_cert  = pixel_accuracy(cert.pred_mask, gt_np, cert.certified)
            miou_cert = mean_iou(cert.pred_mask, gt_np, cfg.dataset.n_classes, cert.certified)

            state["results"].append({
                "idx":                 idx,
                "image_id":            image_id,
                "image_path":          img_path,
                "radius":              cert.radius,
                "n_pixels":            cert.n_pixels,
                "n_certified":         cert.n_certified,
                "n_abstained":         cert.n_abstained,
                "abstain_rate":        cert.abstain_rate,
                "pixel_acc":           pixel_accuracy(cert.pred_mask, gt_np),
                "certified_pixel_acc": acc_cert,
                "miou":                mean_iou(cert.pred_mask, gt_np, cfg.dataset.n_classes),
                "certified_miou":      miou_cert,
                "iou_per_class":       per_class_iou(cert.pred_mask, gt_np,
                                                     cfg.dataset.n_classes).tolist(),
            })

            # Visualisation (first sigma only for first N samples)
            if (cfg.output.save_visualizations and idx < cfg.output.num_viz_samples
                    and sigma == active_sigmas[0]):
                noisy_imgs, noisy_masks = [], []
                for _ in range(3):
                    ni = _sample_noisy(img_tensor, smoother, pixel_index)
                    noisy_imgs.append(ni)
                    x_n = ((ni - _mean_n) / _std_n).unsqueeze(0).to(device)
                    with torch.no_grad():
                        nm_out, _, _ = net(x_n)
                        noisy_masks.append(nm_out.argmax(1).squeeze(0).cpu().numpy())
                save_seg_visualization(
                    viz_dir=state["paths"].experiment_dir / "visualizations",
                    sample_idx=idx, img_tensor=img_tensor, gt_mask=gt_mask,
                    pred_mask=cert.pred_mask, cert_result=cert,
                    nn_imgs=[], nn_masks=[],
                    noisy_imgs=noisy_imgs, noisy_masks=noisy_masks,
                    is_manifold=cfg.smoothing.use_manifold, sigma=sigma,
                    pca_cached=cached_pca, pixel_index=pixel_index,
                    smoother=smoother, cfg=cfg,
                )

        # Checkpoint all active sigmas
        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
            for sigma in active_sigmas:
                st = sigma_states[sigma]
                _save_partial(st["paths"].experiment_dir, idx + 1,
                              st["results"], len(test_samples))

    # ── Finalise each sigma ───────────────────────────────────────────────────
    elapsed = time.time() - t_start
    all_outputs = []

    for sigma in active_sigmas:
        state   = sigma_states[sigma]
        results = state["results"]
        paths   = state["paths"]
        total   = len(results)
        if total == 0:
            continue

        mean_cert_acc  = float(np.mean([r["certified_pixel_acc"] for r in results]))
        mean_cert_miou = float(np.mean([r["certified_miou"]      for r in results]))
        mean_abstain   = float(np.mean([r["abstain_rate"]        for r in results]))
        radius = certified_radius_seg(sigma, cfg.smoothing.tau)
        n_proc = total - state["start_idx"]

        metrics = {
            "experiment":   cfg.experiment_name,
            "dataset":      cfg.dataset.name,
            "output_dir":   str(paths.experiment_dir),
            "smoothing": {
                "mode": cfg.smoothing.mode, "use_manifold": cfg.smoothing.use_manifold,
                "sigma": sigma, "tau": cfg.smoothing.tau,
                "n0_samples": cfg.smoothing.n0_samples, "n_samples": cfg.smoothing.n_samples,
                "knn_k": cfg.smoothing.knn_k, "correction": "holm", "alpha_conf": cfg.alpha_conf,
            },
            "runtime": {
                "certify_seconds": round(elapsed, 2),
                "certify_minutes": round(elapsed / 60, 2),
                "samples_processed": n_proc,
                "seconds_per_sample": round(elapsed / max(n_proc, 1), 4),
                "device": str(device),
                "n_active_sigmas": len(active_sigmas),
            },
            "certified_radius":           radius,
            "total_test_samples":         total,
            "mean_pixel_acc":             float(np.mean([r["pixel_acc"] for r in results])),
            "mean_certified_pixel_acc":   mean_cert_acc,
            "mean_miou":                  float(np.mean([r["miou"] for r in results])),
            "mean_certified_miou":        mean_cert_miou,
            "mean_abstain_rate":          mean_abstain,
        }

        _log(f"sigma={sigma}  cert_acc={mean_cert_acc*100:.2f}%  "
             f"cert_miou={mean_cert_miou*100:.2f}%  abstain={mean_abstain*100:.2f}%")

        if cfg.output.save_results:
            (paths.experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            if cfg.output.save_per_sample:
                csv_path = paths.experiment_dir / "results.csv"
                fieldnames = ["idx", "image_id", "image_path", "radius",
                              "n_pixels", "n_certified", "n_abstained", "abstain_rate",
                              "pixel_acc", "certified_pixel_acc", "miou", "certified_miou"]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for r in results:
                        writer.writerow(r)

        _remove_partial(paths.experiment_dir)
        all_outputs.append({"sigma": sigma, "metrics": metrics})

    return all_outputs


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CelebAMask-HQ Segmentation Certification")
    p.add_argument("--config", required=True)
    p.add_argument("--sigmas", type=float, nargs="+", default=None,
                   help="Override sigma_values from config. E.g. --sigmas 0.10 0.25 0.50")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_seg_certify_config(args.config)

    if args.sigmas:
        sigma_values = [float(s) for s in args.sigmas]
    elif cfg.smoothing.sigma_values:
        sigma_values = [float(s) for s in cfg.smoothing.sigma_values]
    else:
        sigma_values = [cfg.smoothing.sigma]

    if len(sigma_values) == 1:
        run_seg_certification(cfg, sigma_values[0])
    else:
        run_seg_certification_multi_sigma(cfg, sigma_values)
