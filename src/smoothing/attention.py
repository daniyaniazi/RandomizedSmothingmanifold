"""Attention-map stability analysis under manifold / isotropic noise.

Study C: Does smoothing destabilise attention patterns?
         Which layers break first under noise?

Usage
-----
    report = attention_stability_report(model, batch, cfg, device, n_samples=50)
    # report["mean_cosine_sim"]  shape: [num_layers, num_heads]  (higher = more stable)
    # report["mean_entropy"]     shape: [num_layers, num_heads]  (lower  = more peaked)
    # report["topk_overlap"]     shape: [num_layers, num_heads]  (higher = stable top-k)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .noise import isotropic_noise_like, manifold_noise_from_batch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_attentions(
    model,
    batch: dict,
    noise_fn=None,
) -> list[torch.Tensor]:
    """Run one noisy forward pass and return per-layer attention tensors.

    Each element: [B, num_heads, T, T]
    If noise_fn is provided it is applied to input embeddings before encoding.
    """
    input_ids = batch["input_ids"]
    mask = batch["attention_mask"]

    if noise_fn is not None:
        embeds = model.encoder.get_input_embeddings()(input_ids)
        embeds = embeds + noise_fn(embeds)
        enc = model.encoder(
            inputs_embeds=embeds,
            attention_mask=mask,
            output_attentions=True,
            return_dict=True,
        )
    else:
        enc = model.encoder(
            input_ids=input_ids,
            attention_mask=mask,
            output_attentions=True,
            return_dict=True,
        )

    return list(enc.attentions)  # list of [B, H, T, T] per layer


def _attn_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of attention distribution.

    attn: [B, num_heads, T, T] — already softmaxed rows.
    Returns: [B, num_heads, T]
    """
    eps = 1e-12
    return -(attn * (attn + eps).log()).sum(dim=-1)


def _attn_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two attention maps (over the key dimension).

    a, b: [B, num_heads, T, T]
    Returns: [B, num_heads, T]  — per query-token cosine sim
    """
    a_flat = a  # [B, H, T, T]
    b_flat = b
    num = (a_flat * b_flat).sum(dim=-1)          # [B, H, T]
    denom = a_flat.norm(dim=-1) * b_flat.norm(dim=-1) + 1e-12
    return num / denom


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Fraction of top-k attended tokens that overlap between two maps.

    a, b: [B, num_heads, T, T]
    Returns: [B, num_heads, T]
    """
    k_eff = min(k, a.shape[-1])
    top_a = torch.topk(a, k_eff, dim=-1).indices  # [B, H, T, k]
    top_b = torch.topk(b, k_eff, dim=-1).indices

    # count overlap per query token
    overlap = torch.zeros(a.shape[:-1], device=a.device)
    for ki in range(k_eff):
        idx_a = top_a[..., ki].unsqueeze(-1)           # [B, H, T, 1]
        match = (top_b == idx_a).any(dim=-1).float()   # [B, H, T]
        overlap += match
    return overlap / k_eff


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def attention_stability_report(
    model,
    batch: dict,
    cfg,
    device: torch.device,
    n_samples: int = 50,
    topk: int = 5,
) -> dict:
    """Measure attention-map stability under noise for a single batch.

    Returns a dict with numpy arrays of shape [num_layers, num_heads]:
        mean_cosine_sim   — average cosine similarity vs clean map (↑ = stable)
        mean_entropy      — average entropy of noisy attention (↑ = diffuse/broken)
        mean_topk_overlap — average top-k overlap with clean map (↑ = stable)
        clean_entropy     — entropy of clean attention map
    """
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}

    sigma = cfg.smoothing.sigma

    def _noise_fn(embeds):
        if cfg.smoothing.mode == "manifold":
            return manifold_noise_from_batch(
                embeds,
                sigma=sigma,
                knn_k=cfg.smoothing.knn_k,
                eps_eig=cfg.smoothing.eps_eig,
                attention_mask=batch["attention_mask"],
            )
        return torch.randn_like(embeds) * sigma

    # ── Clean reference ───────────────────────────────────────────────────
    clean_attns = _collect_attentions(model, batch, noise_fn=None)
    # clean_attns: list[L] of [B, H, T, T]

    num_layers = len(clean_attns)
    num_heads = clean_attns[0].shape[1]

    # accumulators: [L, H]
    cosim_acc = np.zeros((num_layers, num_heads))
    overlap_acc = np.zeros((num_layers, num_heads))
    entropy_acc = np.zeros((num_layers, num_heads))
    clean_entropy_acc = np.zeros((num_layers, num_heads))

    # clean entropy
    for l_idx, ca in enumerate(clean_attns):
        ent = _attn_entropy(ca).mean(dim=(0, 2)).cpu().numpy()  # [H]
        clean_entropy_acc[l_idx] = ent

    # ── N noisy samples ───────────────────────────────────────────────────
    for _ in range(n_samples):
        noisy_attns = _collect_attentions(model, batch, noise_fn=_noise_fn)

        for l_idx, (ca, na) in enumerate(zip(clean_attns, noisy_attns)):
            # cosine sim: [B, H, T] → mean over batch+tokens → [H]
            cs = _cosine_sim_mean(ca, na)   # [H]
            cosim_acc[l_idx] += cs

            ol = _topk_overlap(ca, na, k=topk).mean(dim=(0, 2)).cpu().numpy()
            overlap_acc[l_idx] += ol

            ent = _attn_entropy(na).mean(dim=(0, 2)).cpu().numpy()
            entropy_acc[l_idx] += ent

    cosim_acc /= n_samples
    overlap_acc /= n_samples
    entropy_acc /= n_samples

    return {
        "mean_cosine_sim": cosim_acc,         # [L, H]
        "mean_topk_overlap": overlap_acc,     # [L, H]
        "mean_entropy": entropy_acc,          # [L, H]
        "clean_entropy": clean_entropy_acc,   # [L, H]
        "num_layers": num_layers,
        "num_heads": num_heads,
    }


def _cosine_sim_mean(ca: torch.Tensor, na: torch.Tensor) -> np.ndarray:
    """[B, H, T, T] → cosine sim → mean over B and T → [H] numpy"""
    cs = _attn_cosine_sim(ca, na)          # [B, H, T]
    return cs.mean(dim=(0, 2)).cpu().numpy()


@torch.no_grad()
def layer_attention_breakdown_summary(
    model,
    data_loader,
    cfg,
    device: torch.device,
    n_samples: int = 50,
    max_batches: Optional[int] = 5,
    topk: int = 5,
) -> dict:
    """Run attention stability over multiple batches and average results.

    Returns same structure as attention_stability_report but averaged over batches.
    Useful for running as part of the layer-sweep experiment.
    """
    from itertools import islice

    reports = []
    batches = data_loader if max_batches is None else islice(data_loader, max_batches)

    for batch in batches:
        r = attention_stability_report(model, batch, cfg, device, n_samples=n_samples, topk=topk)
        reports.append(r)

    if not reports:
        return {}

    avg = {
        "mean_cosine_sim": np.mean([r["mean_cosine_sim"] for r in reports], axis=0),
        "mean_topk_overlap": np.mean([r["mean_topk_overlap"] for r in reports], axis=0),
        "mean_entropy": np.mean([r["mean_entropy"] for r in reports], axis=0),
        "clean_entropy": np.mean([r["clean_entropy"] for r in reports], axis=0),
        "num_layers": reports[0]["num_layers"],
        "num_heads": reports[0]["num_heads"],
    }
    return avg
