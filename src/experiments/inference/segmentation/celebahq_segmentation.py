"""CelebAMask-HQ BiSeNet baseline evaluation (clean images, no noise).

Runs the pretrained BiSeNet on the official test split (28000–29999) and reports:
  - Per-pixel accuracy, mean class accuracy
  - Per-class IoU and mIoU

Note on normalization:
  The dataloader returns raw [0,1] tensors (no normalization) — correct for
  the kNN index and smoothing pipeline.  BiSeNet however was trained with
  ImageNet normalization, so we apply it only here, just before the forward
  pass — same as the certify script does with `(noisy - _mean) / _std`.

Usage:
    python -m src.experiments.inference.segmentation.celebahq_segmentation \
        --checkpoint output/pretrained_model/bisenet_celebahq/bisenet.pth \
        --data-root /BS/dniazi_thesis/static00/CelebAMask-HQ/CelebAMask-HQ \
        --subset 100

    # Full 2000-image test set
    python -m src.experiments.inference.segmentation.celebahq_segmentation \
        --checkpoint output/pretrained_model/bisenet_celebahq/bisenet.pth \
        --data-root /BS/dniazi_thesis/static00/CelebAMask-HQ/CelebAMask-HQ
"""

# Code Adapted from : https://github.com/zllrunning/face-parsing.PyTorch/blob/master/test.py
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# BiSeNet uses `from resnet import Resnet18` (relative), so its dir must be on sys.path
_BISENET_DIR = str(_ROOT / "src" / "models" / "BiseNet")
if _BISENET_DIR not in sys.path:
    sys.path.insert(0, _BISENET_DIR)

from PIL import Image
from src.dataloaders.celebahq_seg import (
    CelebAHQSegDataset, build_seg_transform,
    OFFICIAL_TEST_IDS, CLASS_NAMES,
    _load_mask,
)
from src.experiments.certify.celebahq_segmentation import (
    load_bisenet, SEG_PALETTE, _mask_to_rgb, _tensor_to_pil,
)
from torch.utils.data import DataLoader

# BiSeNet was trained with ImageNet normalisation — apply only before forward pass.
# The dataloader stores raw [0,1] tensors so the kNN index and smoother see
# unnormalised pixels (consistent with the rest of the pipeline).
_BISENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_BISENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def save_viz_samples(net, test_samples, mask_dir, image_size, device, viz_dir, n_viz,
                     pixel_index=None, knn_k=500, eps_eig=1e-6):
    """Save visualization for first n_viz images.

    Row 1: GT mask | Original | Predicted mask (clean)
    Row 2: GT mask | Original | Predicted mask (clean) | PCA recon | Predicted mask on recon
           (row 2 only when pixel_index provided — same PCA as certification pipeline)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _log("matplotlib not available — skipping visualizations")
        return

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    viz_dir.mkdir(parents=True, exist_ok=True)
    tf = build_seg_transform(image_size)

    # PCA modules — same as certification pipeline
    use_pca = pixel_index is not None
    if use_pca:
        try:
            from src.smoothing.manifold import ManifoldSmoother
            from src.smoothing.pca import whiten, unwhiten
            _pca_smoother = ManifoldSmoother(sigma=1.0, index=pixel_index,
                                             knn_k=knn_k, eps_eig=eps_eig)
        except Exception as e:
            _log(f"PCA smoother init failed: {e}")
            use_pca = False

    _log(f"Saving {n_viz} visualizations (PCA={'yes' if use_pca else 'no'}) …")

    for img_path, image_id in test_samples[:n_viz]:
        img   = Image.open(img_path).convert("RGB")
        img_t = tf(img)
        gt    = _load_mask(mask_dir, image_id, image_size).numpy().astype("int32")

        # Clean prediction
        inp = ((img_t - _BISENET_MEAN) / _BISENET_STD).unsqueeze(0).to(device)
        with torch.no_grad():
            out, _, _ = net(inp)
            pred_clean = out.argmax(1).squeeze(0).cpu().numpy().astype("int32")

        # PCA reconstruction + prediction
        recon_t      = None
        pred_recon   = None
        if use_pca:
            try:
                flat = img_t.numpy().flatten().astype("float32")
                cached = _pca_smoother.compute_pca(flat)
                rv = unwhiten(whiten(flat, cached.pca), cached.pca)
                recon_t = torch.from_numpy(rv.reshape(img_t.shape)).float()
                x_r = ((recon_t - _BISENET_MEAN) / _BISENET_STD).unsqueeze(0).to(device)
                with torch.no_grad():
                    r_out, _, _ = net(x_r)
                    pred_recon = r_out.argmax(1).squeeze(0).cpu().numpy().astype("int32")
            except Exception as e:
                _log(f"  PCA failed for {image_id}: {e}")
                use_pca_this = False
            else:
                use_pca_this = True
        else:
            use_pca_this = False

        # ── Layout ───────────────────────────────────────────────────────────
        n_cols = 5 if use_pca_this else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
        for ax in axes.flat:
            ax.axis("off")

        def _show(ax, img_arr, title):
            ax.imshow(img_arr)
            ax.set_title(title, fontsize=9, pad=3)
            ax.axis("off")

        _show(axes[0], Image.fromarray(_mask_to_rgb(gt)),   "GT mask")
        _show(axes[1], _tensor_to_pil(img_t),               "Original")
        _show(axes[2], Image.fromarray(_mask_to_rgb(pred_clean)), "Predicted mask")

        if use_pca_this:
            _show(axes[3], _tensor_to_pil(recon_t),
                  f"PCA recon\n(k={knn_k} neighbours)")
            _show(axes[4], Image.fromarray(_mask_to_rgb(pred_recon)),
                  "Predicted mask\non PCA recon")

        fig.suptitle(f"Image ID: {image_id}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(viz_dir / f"sample_{image_id}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    _log(f"Saved {n_viz} visualizations.")


def evaluate(net, test_samples, mask_dir, image_size, n_classes, device):
    """Run inference on all test samples, return confusion matrix (n_classes × n_classes)."""
    tf  = build_seg_transform(image_size)
    ds  = CelebAHQSegDataset(test_samples, mask_dir, image_size, img_transform=tf)
    dl  = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    conf = np.zeros((n_classes, n_classes), dtype=np.int64)

    for imgs, masks in tqdm(dl, desc="Evaluating"):
        # imgs: (B, C, H, W) in [0,1] — normalise for BiSeNet
        inp = ((imgs - _BISENET_MEAN) / _BISENET_STD).to(device)
        with torch.no_grad():
            out, _, _ = net(inp)                                 # (B, C, H, W)
            preds = out.argmax(dim=1).cpu().numpy().astype(np.int32)  # (B, H, W)

        gt = masks.numpy().astype(np.int32)                      # (B, H, W)
        for pred, gt_i in zip(preds, gt):
            valid = (gt_i >= 0) & (gt_i < n_classes)
            np.add.at(conf, (gt_i[valid], pred[valid]), 1)

    return conf


def compute_metrics(conf: np.ndarray) -> dict:
    """Pixel accuracy, mean class accuracy, per-class IoU, mIoU from confusion matrix."""
    diag     = np.diag(conf)
    row_sums = conf.sum(axis=1)   # GT pixels per class
    col_sums = conf.sum(axis=0)   # predicted pixels per class

    pixel_acc      = float(diag.sum() / conf.sum()) if conf.sum() > 0 else 0.0
    per_cls_acc    = np.where(row_sums > 0, diag / row_sums, np.nan)
    iou_denom      = row_sums + col_sums - diag
    per_cls_iou    = np.where(iou_denom > 0, diag / iou_denom, np.nan)
    present        = row_sums > 0
    miou           = float(np.nanmean(per_cls_iou[present])) if present.any() else 0.0
    mean_cls_acc   = float(np.nanmean(per_cls_acc[present]))  if present.any() else 0.0

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "iou":      None if np.isnan(per_cls_iou[i]) else round(float(per_cls_iou[i]), 4),
            "accuracy": None if np.isnan(per_cls_acc[i]) else round(float(per_cls_acc[i]), 4),
            "gt_pixels": int(row_sums[i]),
        }

    return {
        "pixel_accuracy":      round(pixel_acc, 4),
        "mean_class_accuracy": round(mean_cls_acc, 4),
        "miou":                round(miou, 4),
        "per_class":           per_class,
    }


def main():
    p = argparse.ArgumentParser(description="CelebAMask-HQ BiSeNet baseline evaluation")
    p.add_argument("--checkpoint",  required=True, help="Path to pretrained BiSeNet .pth")
    p.add_argument("--data-root",   required=True, help="CelebAMask-HQ root dir")
    p.add_argument("--image-dir",   default="CelebA-HQ-img")
    p.add_argument("--mask-dir",    default="CelebAMask-HQ-mask-anno")
    p.add_argument("--image-size",  type=int, default=512)
    p.add_argument("--n-classes",   type=int, default=19)
    p.add_argument("--subset",      type=int, default=None,
                   help="First N test images (default: all 2000)")
    p.add_argument("--output-dir",  default="output/segmentation/celebahq/baseline")
    p.add_argument("--num-viz",     type=int, default=10,
                   help="Save this many sample visualizations (0 to skip)")
    p.add_argument("--pixel-index", type=str, default=None,
                   help="Path to pixel index .ann — enables PCA recon row in visualizations")
    p.add_argument("--knn-k",       type=int, default=500,
                   help="kNN neighbours for PCA (default: 500, same as certification)")
    p.add_argument("--viz-only",    action="store_true",
                   help="Regenerate visualizations only — skip inference, metrics already exist")
    p.add_argument("--device",      default="cuda")
    args = p.parse_args()

    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root     = Path(args.data_root)
    img_dir  = root / args.image_dir
    mask_dir = root / args.mask_dir
    out_dir  = _ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Device: {device}")

    # Official test split — first N if subset requested
    test_ids = [i for i in OFFICIAL_TEST_IDS
                if (img_dir / f"{i}.jpg").exists()]
    if args.subset:
        test_ids = test_ids[:args.subset]
    test_samples = [(str(img_dir / f"{i}.jpg"), i) for i in test_ids]
    _log(f"Test images: {len(test_samples)}  (IDs {test_ids[0]}–{test_ids[-1]})")

    # Load model
    class _FakeCfg:
        class model:
            n_classes = args.n_classes
            checkpoint_path = args.checkpoint
    net = load_bisenet(_FakeCfg(), device)

    # Load pixel index for PCA recon row (optional)
    pixel_index = None
    if args.pixel_index:
        try:
            from src.indexing.base import load_index
            dim = 3 * args.image_size * args.image_size
            pixel_index = load_index(dim=dim, index_path=args.pixel_index,
                                     backend="annoy", metric="euclidean")
            _log(f"Pixel index loaded: {pixel_index.index.get_n_items():,} vectors  "
                 f"→ PCA recon enabled (knn_k={args.knn_k})")
        except Exception as e:
            _log(f"Could not load pixel index: {e}  (skipping PCA recon)")

    # Visualizations
    if args.num_viz > 0:
        save_viz_samples(net, test_samples, mask_dir, args.image_size, device,
                         out_dir / "visualizations", args.num_viz,
                         pixel_index=pixel_index, knn_k=args.knn_k)

    # --viz-only: skip inference and metrics, just regenerate visualizations
    if args.viz_only:
        _log("--viz-only: skipping inference. Visualizations saved.")
        return

    _log("Running inference …")
    conf = evaluate(net, test_samples, mask_dir, args.image_size, args.n_classes, device)

    _log("Computing metrics …")
    metrics = compute_metrics(conf)

    # ── Print ─────────────────────────────────────────────────────────────────
    _log("=" * 62)
    _log("BASELINE  (clean images, official test split 28000–29999)")
    _log("=" * 62)
    _log(f"  Pixel accuracy:       {metrics['pixel_accuracy']*100:.2f}%")
    _log(f"  Mean class accuracy:  {metrics['mean_class_accuracy']*100:.2f}%")
    _log(f"  mIoU:                 {metrics['miou']*100:.2f}%")
    _log(f"  {'Class':<15}  {'IoU':>7}  {'Acc':>7}  {'GT pixels':>12}")
    _log(f"  {'-'*15}  {'-'*7}  {'-'*7}  {'-'*12}")
    for name, v in metrics["per_class"].items():
        iou = f"{v['iou']*100:.1f}%" if v["iou"] is not None else "  N/A"
        acc = f"{v['accuracy']*100:.1f}%" if v["accuracy"] is not None else "  N/A"
        _log(f"  {name:<15}  {iou:>7}  {acc:>7}  {v['gt_pixels']:>12,}")
    _log("=" * 62)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "split": "official_test",
        "test_ids": f"{test_ids[0]}–{test_ids[-1]}",
        "n_images": len(test_ids),
        "checkpoint": args.checkpoint,
        "normalization": "ImageNet (0.485/0.456/0.406, 0.229/0.224/0.225) — applied before BiSeNet only",
        **metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(output, indent=2))
    np.save(out_dir / "confusion_matrix.npy", conf)
    _log(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
