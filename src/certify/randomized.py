"""Certification helpers for randomized and manifold smoothing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, norm, binomtest
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


def binom_pvalue_two_sided(n_a: int, n_total: int, p: float = 0.5) -> float:
    """Two-sided binomial test p-value used by paper's PREDICT routine."""
    if n_total <= 0:
        return 1.0
    return float(binomtest(k=int(n_a), n=int(n_total), p=float(p), alternative="two-sided").pvalue)


def predict_from_counts_paper(
    class_counts: np.ndarray,
    alpha_pred: float,
    abstain_label: int = -1,
) -> int:
    """Paper PREDICT routine (top-2 + two-sided binomial test at p=0.5).

    - Let cA, cB be top-2 classes by count.
    - If BINOMPVALUE(nA, nA+nB, 0.5) <= alpha_pred: return cA else ABSTAIN.
    """
    counts = np.asarray(class_counts, dtype=np.int64)
    if counts.size == 0:
        return abstain_label

    top2 = np.argsort(counts)[-2:]
    c_a = int(top2[-1])
    c_b = int(top2[-2]) if len(top2) > 1 else int(top2[-1])
    n_a = int(counts[c_a])
    n_b = int(counts[c_b])
    p_val = binom_pvalue_two_sided(n_a=n_a, n_total=(n_a + n_b), p=0.5)
    return c_a if p_val <= float(alpha_pred) else abstain_label


def certified_radius_paper(alpha_noise: float, p_a_lower: float) -> float:
    """Paper CERTIFY radius: R = σ * Φ^{-1}(p_A_lower), valid when p_A_lower > 0.5."""
    if p_a_lower <= 0.5:
        return 0.0
    return float(alpha_noise) * float(norm.ppf(p_a_lower))


def certify_token_from_counts_two_stage_paper(
    class_counts_n0: np.ndarray,
    class_counts_n: np.ndarray,
    alpha_noise: float,
    alpha_conf: float,
    abstain_label: int = -1,
) -> TokenCertificate:
    """Paper-exact two-stage CERTIFY routine.

    CERTIFY(f, σ, x, n0, n, α):
      1) counts0 <- SAMPLEUNDERNOISE(..., n0, σ)
      2) ĉA <- top index in counts0
      3) counts <- SAMPLEUNDERNOISE(..., n, σ)
      4) pA <- LOWERCONFBOUND(counts[ĉA], n, 1-α)
      5) if pA > 1/2: return ĉA, R = σ * Φ^{-1}(pA) else ABSTAIN

    Notes:
      - This intentionally does NOT use top-2 competitor bound for certification.
      - p_b_upper is reported as (1 - p_a_lower) for bookkeeping compatibility.
    """
    counts_n0 = np.asarray(class_counts_n0, dtype=np.int64)
    counts_n = np.asarray(class_counts_n, dtype=np.int64)

    if counts_n0.size == 0 or counts_n.size == 0:
        return TokenCertificate(
            pred=abstain_label,
            p_a_lower=0.0,
            p_b_upper=1.0,
            radius=0.0,
            abstained=True,
        )

    c_a = int(np.argmax(counts_n0))
    total_n = int(counts_n.sum())
    if total_n <= 0:
        return TokenCertificate(
            pred=abstain_label,
            p_a_lower=0.0,
            p_b_upper=1.0,
            radius=0.0,
            abstained=True,
        )

    n_a = int(counts_n[c_a])
    p_a_lower = clopper_pearson_lower(n_a, total_n, alpha_conf)
    p_b_upper = 1.0 - p_a_lower

    abstained = not (p_a_lower > 0.5)
    pred = abstain_label if abstained else c_a
    radius = 0.0 if abstained else certified_radius_paper(alpha_noise, p_a_lower)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry-first volume metrics (supervisor's formula — sigma-based, not radius-based)
# ═══════════════════════════════════════════════════════════════════════════════


def normalize_eigenvalues(eigenvalues: np.ndarray, mode: str = "max") -> np.ndarray:
    """Normalize eigenvalues for geometry-first comparison at the SAME sigma.

    Two modes
    ---------
    'max'  — λ̃_i = λ_i / λ_max   (principal direction = 1, shape only)
    'mean' — λ̃_i = λ_i / mean(λ) (unit mean energy)

    After normalization det(Λ̃) reflects SHAPE, not absolute scale.
    Use 'max' for axis-length plots; 'mean' for log-volume ratios.
    """
    evals = np.asarray(eigenvalues, dtype=np.float64)
    evals = np.maximum(evals, 1e-30)
    if mode == "max":
        return evals / evals.max()
    elif mode == "mean":
        return evals / evals.mean()
    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}. Use 'max' or 'mean'.")


def log_volume_geo_iso(sigma: float, k: int) -> float:
    """Supervisor geometry-first iso volume: V_iso,geo = C_k · σ^k.

    Uses sigma directly — NOT the certified radius from voting.
    This is the k-ball of radius σ: the isotropic perturbation region.
    """
    return log_volume_isotropic(sigma, k)


def log_volume_geo_mani(sigma: float, eigenvalues_norm: np.ndarray) -> float:
    """Supervisor geometry-first manifold volume: V_mani,geo = C_k · σ^k · √det(Λ̃).

    Λ̃ is the NORMALIZED eigenvalue matrix (pass output of normalize_eigenvalues).
    Uses sigma directly — NOT the certified radius from voting.

    In log-space:
        log V = log(C_k) + k·log(σ) + 0.5·Σ log(λ̃_i)

    The key comparison:
        log V_mani,geo - log V_iso,geo = 0.5·Σ log(λ̃_i)  (pure shape gain)

    Args:
        sigma: Noise level (same for iso and mani — the fair comparison point).
        eigenvalues_norm: Normalized PCA eigenvalues λ̃_1...λ̃_k (use normalize_eigenvalues first).
    """
    evals_norm = np.asarray(eigenvalues_norm, dtype=np.float64)
    k = len(evals_norm)
    log_det_half_norm = 0.5 * np.sum(np.log(np.maximum(evals_norm, 1e-30)))
    return log_volume_geo_iso(sigma, k) + log_det_half_norm


def axis_lengths(sigma: float, eigenvalues_norm: np.ndarray) -> np.ndarray:
    """Ellipsoid axis lengths in the PCA perturbation plane.

    a_i = σ · √λ̃_i

    Interpretation
    --------------
    Isotropic:  all a_i = σ  → circle / sphere in every direction
    Manifold:   a_i = σ·√λ̃_i → ellipsoid; large λ̃ stretches noise along that axis

    This is the TRUE manifold stretch visible in PCA space.
    Plot the spectrum a_1 ≥ a_2 ≥ ... ≥ a_k to show directional geometry.

    Args:
        sigma: Noise level σ.
        eigenvalues_norm: Normalized eigenvalues (use normalize_eigenvalues first).
    """
    return sigma * np.sqrt(np.maximum(eigenvalues_norm, 0.0))


def anisotropy_ratio(eigenvalues_norm: np.ndarray) -> float:
    """Ratio of largest to smallest axis length.

    anisotropy = a_1 / a_k = √(λ̃_max / λ̃_min)

    Interpretation
    --------------
    = 1.0   →  isotropic (sphere): noise equally distributed in every direction
    > 1.0   →  anisotropic ellipsoid: first PCA axis stretched MORE than last
    >> 1    →  highly elongated: noise is concentrated in a few key directions

    For 'max'-normalized eigenvalues λ̃_1 = 1 by definition, so:
        anisotropy = 1 / √λ̃_k = √(λ_1/λ_k)
    which equals the square root of the condition number.

    High anisotropy → the manifold is genuinely low-dimensional / elongated.
    Low anisotropy  → the local PCA ball is nearly spherical.
    """
    evals = np.asarray(eigenvalues_norm, dtype=np.float64)
    evals = np.maximum(evals, 1e-30)
    return float(np.sqrt(evals.max() / evals.min()))


def cumulative_stretch_energy(eigenvalues_norm: np.ndarray, m: int | None = None) -> np.ndarray:
    """Cumulative fraction of total eigenvalue energy in the top-m PCA directions.

    cumulative_energy[i] = Σ_{j=0}^{i} λ̃_j / Σ_j λ̃_j

    Interpretation
    --------------
    Shows how many principal axes carry most of the perturbation energy.
    - cumulative_energy[2] = 0.90 → 90% of noise energy in 3 directions.
    - Fast decay (few components dominate) → low-dimensional manifold.
    - Slow decay (energy spread across many) → near-isotropic local geometry.

    This is the KEY diagnostic for your thesis:
        plot cumulative_stretch_energy vs component index for many samples
        → distribution shows heterogeneity of local manifold structure.

    Args:
        eigenvalues_norm: Normalized eigenvalues (sorted largest first).
        m: Include only top-m components (None = all).

    Returns:
        Array of cumulative fractions, length min(m, k).
    """
    evals = np.asarray(eigenvalues_norm, dtype=np.float64)
    evals = np.maximum(evals, 0.0)
    total = evals.sum()
    if total <= 0:
        n = len(evals) if m is None else min(m, len(evals))
        return np.zeros(n)
    cumsum = np.cumsum(evals)
    if m is not None:
        cumsum = cumsum[:m]
    return cumsum / total


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
