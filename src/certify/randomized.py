"""Certification helpers for randomized and manifold smoothing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, norm


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
