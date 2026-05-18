"""Certification helpers for randomized and manifold smoothing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, norm
from scipy.special import gammaln


# ═══════════════════════════════════════════════════════════════════════════════
# Core certification
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TokenCertificate:
    pred: int
    p_a_lower: float
    p_b_upper: float
    radius: float
    abstained: bool


def clopper_pearson_lower(successes: int, total: int, alpha: float) -> float:
    if successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def clopper_pearson_upper(successes: int, total: int, alpha: float) -> float:
    if successes >= total:
        return 1.0
    return float(beta.ppf(1 - alpha, successes + 1, total - successes))


def certified_radius(alpha_noise: float, p_a_lower: float, p_b_upper: float) -> float:
    if p_a_lower <= p_b_upper:
        return 0.0
    return 0.5 * alpha_noise * (norm.ppf(p_a_lower) - norm.ppf(p_b_upper))


def certify_token_from_counts(
    class_counts: np.ndarray,
    alpha_noise: float,
    alpha_conf: float,
    abstain_label: int = -1,
) -> TokenCertificate:
    top2 = np.argsort(class_counts)[-2:]
    a = int(top2[-1])
    b = int(top2[-2]) if len(top2) > 1 else int(top2[-1])
    total = int(class_counts.sum())
    n_a = int(class_counts[a])
    n_b = int(class_counts[b])

    p_a_lower = clopper_pearson_lower(n_a, total, alpha_conf)
    p_b_upper = clopper_pearson_upper(n_b, total, alpha_conf)

    abstained = p_a_lower <= p_b_upper
    pred = abstain_label if abstained else a
    radius = 0.0 if abstained else certified_radius(alpha_noise, p_a_lower, p_b_upper)

    return TokenCertificate(
        pred=pred,
        p_a_lower=p_a_lower,
        p_b_upper=p_b_upper,
        radius=radius,
        abstained=abstained,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Volume computation
# ═══════════════════════════════════════════════════════════════════════════════


def log_volume_isotropic(radius: float, k: int) -> float:
    """Log-volume of isotropic certified region (k-ball of radius r).

    V_k(r) = (π^(k/2) / Γ(k/2 + 1)) * r^k

    Uses log-space to avoid overflow for large k.
    """
    if radius <= 0.0 or k <= 0:
        return -np.inf
    # log(C_k) = (k/2)*log(π) - log(Γ(k/2 + 1))
    log_ck = (k / 2.0) * np.log(np.pi) - gammaln(k / 2.0 + 1.0)
    return log_ck + k * np.log(radius)


def log_volume_manifold(radius: float, eigenvalues: np.ndarray) -> float:
    """Log-volume of manifold certified region (ellipsoid).

    V_mani = C_k * r^k * sqrt(det(Λ))

    In log-space:
        log V = log(C_k) + k*log(r) + (1/2)*Σ log(λ_i)

    Args:
        radius: Certified radius in whitened space (r_mani from Cohen formula with σ=alpha).
        eigenvalues: PCA eigenvalues λ_1...λ_k (from local neighborhood).

    Returns:
        Log-volume of the manifold-shaped ellipsoid.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    k = len(eigenvalues)
    if radius <= 0.0 or k <= 0:
        return -np.inf
    log_ball = log_volume_isotropic(radius, k)
    # sqrt(det(Λ)) = exp(0.5 * sum(log(λ)))
    log_det_half = 0.5 * np.sum(np.log(np.maximum(eigenvalues, 1e-30)))
    return log_ball + log_det_half


def log_volume_manifold_from_iso_radius(
    radius_iso: float,
    eigenvalues: np.ndarray,
) -> float:
    """Predicted manifold volume if we assume r_mani = r_iso (same noise, same radius).

    This is the 'geometric-only' effect: V_predicted = C_k * r_iso^k * sqrt(det(Λ))
    Compare this with actual manifold volume to isolate the radius improvement effect.
    """
    return log_volume_manifold(radius_iso, eigenvalues)


def log_volume_ratio(
    radius_mani: float,
    radius_iso: float,
    eigenvalues: np.ndarray,
) -> float:
    """Log(V_mani / V_iso) = k*log(r_mani/r_iso) + 0.5*Σlog(λ_i).

    Two effects:
        - Radius effect: k*log(r_mani/r_iso)  — manifold noise may preserve prediction better
        - Geometry effect: 0.5*Σlog(λ_i)      — eigenvalue stretching
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    k = len(eigenvalues)
    if radius_iso <= 0.0 or radius_mani <= 0.0:
        return 0.0
    radius_effect = k * np.log(radius_mani / radius_iso)
    geometry_effect = 0.5 * np.sum(np.log(np.maximum(eigenvalues, 1e-30)))
    return radius_effect + geometry_effect


# ═══════════════════════════════════════════════════════════════════════════════
# Eigenvalue diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EigenDiagnostics:
    """Diagnostic stats for a set of PCA eigenvalues."""
    k: int
    lambda_min: float
    lambda_max: float
    condition_number: float
    eigenvalue_sum: float
    effective_rank: float  # exp(entropy of normalized eigenvalues)
    log_det_half: float    # 0.5 * Σ log(λ_i) — the geometry factor

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "condition_number": self.condition_number,
            "eigenvalue_sum": self.eigenvalue_sum,
            "effective_rank": self.effective_rank,
            "log_det_half": self.log_det_half,
        }


def eigenvalue_diagnostics(eigenvalues: np.ndarray) -> EigenDiagnostics:
    """Compute diagnostic statistics for eigenvalues.

    Args:
        eigenvalues: Array of PCA eigenvalues (k,).

    Returns:
        EigenDiagnostics dataclass.
    """
    evals = np.asarray(eigenvalues, dtype=np.float64)
    evals = np.maximum(evals, 1e-30)
    k = len(evals)

    # Effective rank: exp(Shannon entropy of normalized eigenvalues)
    p = evals / evals.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    effective_rank = float(np.exp(entropy))

    return EigenDiagnostics(
        k=k,
        lambda_min=float(evals.min()),
        lambda_max=float(evals.max()),
        condition_number=float(evals.max() / evals.min()),
        eigenvalue_sum=float(evals.sum()),
        effective_rank=effective_rank,
        log_det_half=float(0.5 * np.sum(np.log(evals))),
    )


@dataclass
class VolumeResult:
    """Per-sample volume computation result."""
    radius_iso: float
    radius_mani: float
    log_vol_iso: float
    log_vol_mani: float
    log_vol_mani_predicted: float  # V_mani assuming r_mani = r_iso (geometry-only)
    log_vol_ratio: float           # log(V_mani / V_iso)
    eigen_diagnostics: EigenDiagnostics

    def to_dict(self) -> dict:
        return {
            "radius_iso": self.radius_iso,
            "radius_mani": self.radius_mani,
            "log_vol_iso": self.log_vol_iso,
            "log_vol_mani": self.log_vol_mani,
            "log_vol_mani_predicted": self.log_vol_mani_predicted,
            "log_vol_ratio": self.log_vol_ratio,
            **{f"eigen_{k}": v for k, v in self.eigen_diagnostics.to_dict().items()},
        }


def compute_volume_result(
    radius_iso: float,
    radius_mani: float,
    eigenvalues: np.ndarray,
) -> VolumeResult:
    """Compute full volume comparison for a single sample.

    Args:
        radius_iso: Certified radius from isotropic smoothing.
        radius_mani: Certified radius from manifold smoothing.
        eigenvalues: PCA eigenvalues from the local neighborhood.

    Returns:
        VolumeResult with all volume metrics.
    """
    evals = np.asarray(eigenvalues, dtype=np.float64)
    k = len(evals)

    lv_iso = log_volume_isotropic(radius_iso, k)
    lv_mani = log_volume_manifold(radius_mani, evals)
    lv_mani_pred = log_volume_manifold_from_iso_radius(radius_iso, evals)
    lv_ratio = log_volume_ratio(radius_mani, radius_iso, evals)
    diag = eigenvalue_diagnostics(evals)

    return VolumeResult(
        radius_iso=radius_iso,
        radius_mani=radius_mani,
        log_vol_iso=lv_iso,
        log_vol_mani=lv_mani,
        log_vol_mani_predicted=lv_mani_pred,
        log_vol_ratio=lv_ratio,
        eigen_diagnostics=diag,
    )
