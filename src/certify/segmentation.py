"""SEGCERTIFY — Algorithm 2 from Fischer et al. (2021).

Scalable Certified Segmentation via Randomized Smoothing.

Key components:
  - Per-pixel binomial p-value (one-sided, H0: p ≤ τ)
  - Holm-Bonferroni FWER correction across all pixels
  - Certified radius R = σ · Φ⁻¹(τ)
  - Metrics: per-pixel accuracy, mIoU, abstention rate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.stats import norm as _norm


# ── Certified radius ──────────────────────────────────────────────────────────

def certified_radius_seg(sigma: float, tau: float) -> float:
    """R = σ · Φ⁻¹(τ)  (Theorem 5.1)."""
    return float(sigma * _norm.ppf(tau))


# ── Binomial p-value ──────────────────────────────────────────────────────────

def binom_pvalue_one_sided(n_successes: int, n_total: int, tau: float) -> float:
    """One-sided binomial p-value for H0: p ≤ τ  vs  HA: p > τ.

    P(X ≥ n_successes | X ~ Binomial(n_total, tau))
    = 1 - CDF(n_successes - 1, n_total, tau)
    """
    if n_total == 0:
        return 1.0
    return float(stats.binom.sf(n_successes - 1, n_total, tau))


# ── Holm-Bonferroni FWER correction ─────────────────────────────────────────

def holm_correction(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Holm-Bonferroni FWER control at level alpha.

    Returns boolean array of shape (N,): True = rejected (certified), False = abstain.

    Algorithm:
      1. Sort p-values ascending, track original indices.
      2. Compare p[i] against α / (N - i).
      3. Reject while p[i] < threshold; stop at first failure.
      4. All subsequent tests fail (not rejected).
    """
    n = len(pvalues)
    rejected = np.zeros(n, dtype=bool)
    if n == 0:
        return rejected

    order = np.argsort(pvalues)
    sorted_pv = pvalues[order]

    for i in range(n):
        threshold = alpha / (n - i)
        if sorted_pv[i] < threshold:
            rejected[order[i]] = True
        else:
            break  # Holm stops at first failure

    return rejected


def bonferroni_correction(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Bonferroni FWER control — simpler but more conservative than Holm."""
    threshold = alpha / len(pvalues)
    return pvalues < threshold


# ── SEGCERTIFY ────────────────────────────────────────────────────────────────

@dataclass
class SegCertResult:
    """Result for one image."""
    pred_mask: np.ndarray        # (H, W) int — majority class per pixel
    certified: np.ndarray        # (H, W) bool — True = certified pixel
    pvalues: np.ndarray          # (H, W) float — per-pixel p-value
    radius: float                # single global radius R = σΦ⁻¹(τ)
    n_pixels: int
    n_certified: int
    n_abstained: int
    abstain_rate: float


def segcertify(
    counts_n0: np.ndarray,    # (H, W, C) — pilot counts for class selection
    counts_n: np.ndarray,     # (H, W, C) — main MC counts for hypothesis test
    sigma: float,
    tau: float,
    alpha: float,
    correction: str = "holm",  # "holm" | "bonferroni"
) -> SegCertResult:
    """Algorithm 2: SEGCERTIFY.

    Args:
        counts_n0: (H, W, C) pilot sample class counts — used to pick majority class.
        counts_n:  (H, W, C) main MC class counts — used for binomial test.
        sigma:     noise level.
        tau:       stability threshold τ ∈ [0.5, 1).
        alpha:     FWER confidence level.
        correction: "holm" (default) or "bonferroni".

    Returns:
        SegCertResult with certified mask, p-values, radius.
    """
    H, W, C = counts_n0.shape
    N = H * W

    # Step 1: pick majority class from pilot samples
    pred_mask = counts_n0.argmax(axis=-1)  # (H, W)

    # Step 2: get count for majority class in main samples
    n_total = counts_n.sum(axis=-1)        # (H, W) — should equal n_samples everywhere
    n_i = counts_n[np.arange(H)[:, None], np.arange(W)[None, :], pred_mask]  # (H, W)

    # Step 3: compute p-values (H0: p ≤ τ) for all pixels
    n_flat = n_total.flatten()
    ni_flat = n_i.flatten()
    pvalues_flat = np.array([
        binom_pvalue_one_sided(int(ni_flat[k]), int(n_flat[k]), tau)
        for k in range(N)
    ], dtype=np.float64)
    pvalues = pvalues_flat.reshape(H, W)

    # Step 4: Holm (or Bonferroni) FWER correction
    if correction == "holm":
        certified_flat = holm_correction(pvalues_flat, alpha)
    else:
        certified_flat = bonferroni_correction(pvalues_flat, alpha)
    certified = certified_flat.reshape(H, W)

    # Step 5: certified radius (same for all certified pixels, Theorem 5.1)
    radius = certified_radius_seg(sigma, tau)

    n_certified = int(certified.sum())
    n_abstained = N - n_certified

    return SegCertResult(
        pred_mask=pred_mask,
        certified=certified,
        pvalues=pvalues,
        radius=radius,
        n_pixels=N,
        n_certified=n_certified,
        n_abstained=n_abstained,
        abstain_rate=n_abstained / N,
    )


# ── Evaluation metrics ────────────────────────────────────────────────────────

def pixel_accuracy(pred: np.ndarray, gt: np.ndarray, certified: Optional[np.ndarray] = None) -> float:
    """Per-pixel accuracy.

    Without certified mask: correct / all pixels.
    With certified mask:    (certified AND correct) / all pixels.
      Abstained pixels count as wrong — denominator is always total pixels.
    """
    total = pred.size
    if total == 0:
        return 0.0
    if certified is not None:
        return float(int(((pred == gt) & certified).sum()) / total)
    return float((pred == gt).mean())


def mean_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    n_classes: int,
    certified: Optional[np.ndarray] = None,
    ignore_abstain: bool = True,
) -> float:
    """Compute mIoU across all classes.

    Without certified mask: standard mIoU over all pixels.
    With certified mask:    certified mIoU — abstained pixels are treated as a
      wrong prediction (class -1), so they hurt both precision and recall.
      Denominator is always all pixels (via union).
    """
    if certified is not None and ignore_abstain:
        # Abstained pixels get prediction = -1 (no valid class) → never match gt
        p = np.where(certified, pred, -1).flatten()
        g = gt.flatten()
    else:
        p = pred.flatten()
        g = gt.flatten()

    ious = []
    for c in range(n_classes):
        inter = int(((p == c) & (g == c)).sum())
        union = int(((p == c) | (g == c)).sum())
        if union == 0:
            continue  # class absent — skip
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def per_class_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    n_classes: int,
    certified: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns IoU per class (NaN if class absent)."""
    if certified is not None:
        mask = certified
        p = pred[mask] if mask.sum() > 0 else np.array([], dtype=pred.dtype)
        g = gt[mask]   if mask.sum() > 0 else np.array([], dtype=gt.dtype)
    else:
        p = pred.flatten()
        g = gt.flatten()

    iou = np.full(n_classes, np.nan)
    for c in range(n_classes):
        inter = int(((p == c) & (g == c)).sum())
        union = int(((p == c) | (g == c)).sum())
        if union > 0:
            iou[c] = inter / union
    return iou
