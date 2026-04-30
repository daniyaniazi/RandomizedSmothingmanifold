"""Reusable certification workflow helpers.

Task code should only provide a callable that returns one prediction sample.
This module handles repeated sampling, vote counting, abstention, and metrics.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .randomized import certify_token_from_counts


def certify_prediction_set(
    sample_predictions_fn: Callable[[], np.ndarray],
    labels: np.ndarray,
    n_samples: int,
    num_classes: int,
    alpha_noise: float,
    alpha_conf: float,
    abstain_label: int = -1,
    ignore_index: int = -100,
) -> dict:
    """Run smoothing samples, vote, and compute aggregate certification metrics.

    Parameters
    ----------
    sample_predictions_fn:
        Returns one array of predicted class ids with shape matching ``labels``.
    labels:
        Ground-truth integer labels with ignored positions set to ``ignore_index``.
    """
    pred_samples = np.stack([sample_predictions_fn() for _ in range(n_samples)], axis=0)

    total = 0
    certified_correct = 0
    abstained = 0
    radii: list[float] = []

    for idx in np.ndindex(labels.shape):
        gt = int(labels[idx])
        if gt == ignore_index:
            continue

        counts = np.bincount(pred_samples[(slice(None),) + idx], minlength=num_classes)
        cert = certify_token_from_counts(
            counts,
            alpha_noise=alpha_noise,
            alpha_conf=alpha_conf,
            abstain_label=abstain_label,
        )

        total += 1
        if cert.abstained:
            abstained += 1
            continue

        radii.append(cert.radius)
        if cert.pred == gt:
            certified_correct += 1

    return {
        "certified_accuracy": (certified_correct / total) if total else 0.0,
        "abstain_rate": (abstained / total) if total else 0.0,
        "mean_radius": float(np.mean(radii)) if radii else 0.0,
    }