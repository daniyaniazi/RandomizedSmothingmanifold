"""Layer-wise smoothing sweep experiment.

For each transformer layer l = 0 … L-1:
    1. Apply manifold / isotropic noise at the hidden states of layer l
    2. Run certification  → certified_accuracy, abstain_rate, mean_radius, task_f1
    3. Run attention stability analysis → cosine_sim, topk_overlap, entropy per head

Produces:
    <output_dir>/layer_sweep/layer_<l>/metrics.json
    <output_dir>/layer_sweep/cert_layer_sweep_summary.json   ← one row per layer

Usage
-----
    python3 -m src.experiments.cert_layer_sweep \\
        --config src/configs/experiments/ner_conll2003_distilbert.yaml \\
        --checkpoint outputs/ner/model.pt \\
        [--layers 0 2 4 6]          # default: all layers
        [--n-attn-samples 30]       # samples for attention stability
        [--max-batches 20]          # limit batches for quick runs
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.configs import load_experiment_config
from src.smoothing.attention import layer_attention_breakdown_summary
from src.smoothing.noise import isotropic_noise_like, manifold_noise_from_batch
from src.certify import certify_token_from_counts


def parse_args():
    p = argparse.ArgumentParser(description="Layer-wise smoothing and certification sweep")
    p.add_argument("--config", required=True, help="Experiment YAML config path")
    p.add_argument("--checkpoint", default=None, help="Path to model.pt (optional)")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Which layers to sweep (default: all)")
    p.add_argument("--n-attn-samples", type=int, default=30,
                   help="Number of noise samples for attention stability")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Limit test batches for fast iteration")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-layer certification
# ---------------------------------------------------------------------------

@torch.no_grad()
def certify_at_layer(model, data, cfg, device, layer_index: int) -> dict:
    """Run certification with noise injected at a specific transformer layer.

    Pipeline:
        encoder processes input up to layer_index-1 normally.
        At layer_index the hidden states are perturbed N times.
        Remaining encoder layers process the perturbed states.
        Classifier head produces the final label.

    The encoder still runs N times here (one per sample), but only a single
    forward pass per sample is needed — there is no redundant full-BERT work
    compared to the embedding-level baseline for layer-l injection.
    """
    if not cfg.certification.enabled:
        return {}

    model.eval()
    n = cfg.certification.n
    sigma = cfg.smoothing.sigma

    def layer_noise_fn(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Inject noise only at the target layer."""
        if layer_idx != layer_index:
            return torch.zeros_like(h)
        mask = None  # hook does not have access to original mask; shape [B,T,D] is fine
        if cfg.smoothing.mode == "manifold":
            return manifold_noise_from_batch(h, sigma=sigma,
                                             knn_k=cfg.smoothing.knn_k,
                                             eps_eig=cfg.smoothing.eps_eig)
        return isotropic_noise_like(h, sigma=sigma)

    total = certified_correct = abstained = 0
    radii = []
    n_classes = model.classifier.out_features
    max_b = cfg.eval.max_batches

    for b_idx, batch in enumerate(tqdm(data.test_loader,
                                        desc=f"certify layer {layer_index}",
                                        leave=False)):
        if max_b is not None and b_idx >= max_b:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels_np = batch["labels"].cpu().numpy()

        # N samples — each runs the encoder with a hook at the target layer
        pred_samples = []
        for _ in range(n):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                layer_noise_fn=layer_noise_fn,
            )
            pred_samples.append(out.logits.argmax(-1).cpu().numpy())

        pred_samples = np.stack(pred_samples, axis=0)  # [n, B, T]

        for bi in range(pred_samples.shape[1]):
            for ti in range(pred_samples.shape[2]):
                gt = int(labels_np[bi, ti])
                if gt == -100:
                    continue
                counts = np.bincount(pred_samples[:, bi, ti], minlength=n_classes)
                cert = certify_token_from_counts(
                    counts,
                    alpha_noise=cfg.smoothing.sigma,
                    alpha_conf=cfg.certification.alpha,
                    abstain_label=cfg.certification.abstain_label,
                )
                total += 1
                if cert.abstained:
                    abstained += 1
                    continue
                radii.append(cert.radius)
                if cert.pred == gt:
                    certified_correct += 1

    return {
        "layer": layer_index,
        "certified_accuracy": (certified_correct / total) if total else 0.0,
        "abstain_rate": (abstained / total) if total else 0.0,
        "mean_radius": float(np.mean(radii)) if radii else 0.0,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override max_batches from CLI for quick runs
    if args.max_batches is not None:
        cfg.eval.max_batches = args.max_batches

    # ── Load plugin and build model + data ──────────────────────────────────
    import importlib
    plugin = importlib.import_module(cfg.task.module or cfg.task.eval_plugin)
    model, data = plugin.build_model_and_data(cfg, device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")

    model.eval()

    num_layers = model.num_layers()
    layers_to_sweep = args.layers if args.layers else list(range(num_layers))
    print(f"Sweeping {len(layers_to_sweep)} layers out of {num_layers} total")

    out_dir = Path(cfg.output_dir if hasattr(cfg, "output_dir") else "outputs") / "layer_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for layer_idx in layers_to_sweep:
        print(f"\n=== Layer {layer_idx}/{num_layers - 1} ===")
        layer_dir = out_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        # ── Certification at this layer ─────────────────────────────────────
        cert_metrics = certify_at_layer(model, data, cfg, device, layer_index=layer_idx)

        # ── Attention stability at this layer ───────────────────────────────
        # Build a per-layer cfg override so noise applies at this layer
        layer_cfg = copy.deepcopy(cfg)
        layer_cfg.smoothing.layer_index = layer_idx

        attn_report = layer_attention_breakdown_summary(
            model=model,
            data_loader=data.test_loader,
            cfg=layer_cfg,
            device=device,
            n_samples=args.n_attn_samples,
            max_batches=min(5, args.max_batches or 5),
        )

        # Summarise attention report (mean over heads for each layer)
        attn_summary = {}
        if attn_report:
            l = layer_idx
            if l < attn_report["num_layers"]:
                attn_summary = {
                    "attn_cosine_sim_mean": float(attn_report["mean_cosine_sim"][l].mean()),
                    "attn_topk_overlap_mean": float(attn_report["mean_topk_overlap"][l].mean()),
                    "attn_entropy_noisy_mean": float(attn_report["mean_entropy"][l].mean()),
                    "attn_entropy_clean_mean": float(attn_report["clean_entropy"][l].mean()),
                }
                # Per-head arrays for detailed analysis
                attn_summary["per_head"] = {
                    "cosine_sim": attn_report["mean_cosine_sim"][l].tolist(),
                    "topk_overlap": attn_report["mean_topk_overlap"][l].tolist(),
                    "entropy_noisy": attn_report["mean_entropy"][l].tolist(),
                    "entropy_clean": attn_report["clean_entropy"][l].tolist(),
                }

        row = {**cert_metrics, **attn_summary}
        summary.append(row)

        (layer_dir / "metrics.json").write_text(json.dumps(row, indent=2))
        print(f"  certified_accuracy={cert_metrics.get('certified_accuracy', 0):.4f}  "
              f"mean_radius={cert_metrics.get('mean_radius', 0):.4f}  "
              f"abstain_rate={cert_metrics.get('abstain_rate', 0):.4f}")
        if attn_summary:
            print(f"  attn_cosine_sim={attn_summary.get('attn_cosine_sim_mean', 0):.4f}  "
                  f"attn_topk_overlap={attn_summary.get('attn_topk_overlap_mean', 0):.4f}")

    summary_path = out_dir / "cert_layer_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")
    print(json.dumps(
        [{k: v for k, v in r.items() if k != "per_head"} for r in summary],
        indent=2,
    ))


if __name__ == "__main__":
    main()
