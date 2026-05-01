"""Run clean vs token-wise manifold-smoothed NER evaluation.

This script keeps the workflow simple and modular:
1. load trained NER checkpoint
2. build or load a train-token index for a chosen hidden layer
3. evaluate clean baseline on val/test split
4. evaluate token-wise manifold smoothing with voting/certification
5. save metrics and a small debug artifact for visualization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from src.configs import load_experiment_config, save_resolved_config
from src.dataloaders import build_conll_dataloaders
from src.indexing import build_or_load_token_index
from src.models.transformer.ner.model import TransformerNER
from src.models.transformer.ner.train import device_from_cfg, move_batch, set_seed
from src.smoothing.ner_token_manifold import sample_smoothed_token_predictions


@torch.no_grad()
def evaluate_clean(model, loader, device, id2label: dict[int, str], max_batches: int | None = None):
    model.eval()
    y_true_all = []
    y_pred_all = []
    losses = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        losses.append(float(out.loss.detach().cpu()))
        preds = torch.argmax(out.logits, dim=-1).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        for p_row, l_row in zip(preds, labels):
            p_seq = []
            l_seq = []
            for p, l in zip(p_row, l_row):
                if l == -100:
                    continue
                p_seq.append(id2label[int(p)])
                l_seq.append(id2label[int(l)])
            if l_seq:
                y_pred_all.append(p_seq)
                y_true_all.append(l_seq)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "precision": float(precision_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "recall": float(recall_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "f1": float(f1_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "token_acc": float(accuracy_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
    }


@torch.no_grad()
def evaluate_smoothed(model, loader, device, tokenizer_name: str, id2label: dict[int, str], token_index_artifacts, cfg, max_batches: int | None = None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    y_true_all = []
    y_pred_all = []
    radii = []
    abstentions = 0
    total_tokens = 0
    debug_payload = None

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        out = sample_smoothed_token_predictions(
            model=model,
            batch=batch,
            neighbor_index=token_index_artifacts.index,
            token_texts=token_index_artifacts.token_texts,
            label_ids=token_index_artifacts.label_ids,
            tokenizer=tokenizer,
            num_samples=cfg.certification.n,
            sigma=cfg.smoothing.sigma,
            knn_k=cfg.smoothing.knn_k,
            eps_eig=cfg.smoothing.eps_eig,
            layer_index=cfg.smoothing.layer_index,
            alpha_conf=cfg.certification.alpha,
            abstain_label=cfg.certification.abstain_label,
            collect_debug=batch_idx == 0,
        )

        pred_ids = out.pred_ids.detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()

        for row_idx, (p_row, l_row) in enumerate(zip(pred_ids, labels)):
            p_seq = []
            l_seq = []
            for token_idx, (p, l) in enumerate(zip(p_row, l_row)):
                if l == -100:
                    continue
                cert = out.certificates[row_idx][token_idx]
                if cert is not None:
                    radii.append(float(cert.radius))
                    abstentions += int(cert.abstained)
                    total_tokens += 1
                p_seq.append(id2label[int(p)])
                l_seq.append(id2label[int(l)])
            if l_seq:
                y_pred_all.append(p_seq)
                y_true_all.append(l_seq)

        if batch_idx == 0 and out.debug is not None:
            debug_rows = []
            for row in out.debug:
                debug_row = []
                for item in row:
                    if item is None:
                        debug_row.append(None)
                    else:
                        debug_row.append(
                            {
                                "token": item.token,
                                "true_label": item.true_label,
                                "neighbor_tokens": item.neighbor_tokens,
                                "neighbor_labels": item.neighbor_labels,
                                "reconstruction_l2": item.reconstruction_l2,
                                "noisy_l2": item.noisy_l2,
                                "explained_variance": item.explained_variance,
                            }
                        )
                debug_rows.append(debug_row)
            debug_payload = debug_rows

    metrics = {
        "precision": float(precision_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "recall": float(recall_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "f1": float(f1_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "token_acc": float(accuracy_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "mean_certified_radius": float(np.mean(radii)) if radii else 0.0,
        "abstention_rate": float(abstentions / total_tokens) if total_tokens else 0.0,
        "total_certified_tokens": int(total_tokens),
    }
    return metrics, debug_payload


def run(cfg_path: str, checkpoint_path: str, split: str = "test", rebuild_index: bool = False):
    cfg = load_experiment_config(cfg_path)
    set_seed(cfg.train.seed)
    device = device_from_cfg(cfg.train.device)

    out_dir = Path(cfg.output_dir) / f"{cfg.experiment_name}_smoothing_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, out_dir / "resolved_config.yaml")

    data = build_conll_dataloaders(cfg.dataset, cfg.dataloader, cfg.model.encoder_name)
    model = TransformerNER(
        encoder_name=cfg.model.encoder_name,
        num_labels=len(data.id2label),
        id2label=data.id2label,
        label2id=data.label2id,
        dropout=cfg.model.dropout,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    token_index_dir = out_dir / "token_index"
    token_index_artifacts = build_or_load_token_index(
        out_dir=token_index_dir,
        model=model,
        loader=data.train_loader,
        device=device,
        tokenizer_name=data.tokenizer_name,
        layer_index=cfg.smoothing.layer_index,
        backend=cfg.smoothing.index_backend,
        metric=cfg.smoothing.index_metric,
        index_path=cfg.smoothing.index_path,
        n_trees=cfg.smoothing.index_n_trees,
        rebuild=rebuild_index,
    )

    target_loader = data.val_loader if split == "val" else data.test_loader
    clean_metrics = evaluate_clean(model, target_loader, device, data.id2label, cfg.eval.max_batches)
    smooth_metrics, debug_payload = evaluate_smoothed(
        model=model,
        loader=target_loader,
        device=device,
        tokenizer_name=data.tokenizer_name,
        id2label=data.id2label,
        token_index_artifacts=token_index_artifacts,
        cfg=cfg,
        max_batches=cfg.eval.max_batches,
    )

    summary = {
        "checkpoint": checkpoint_path,
        "split": split,
        "clean": clean_metrics,
        "smoothed": smooth_metrics,
        "smoothing": {
            "sigma": cfg.smoothing.sigma,
            "knn_k": cfg.smoothing.knn_k,
            "layer_index": cfg.smoothing.layer_index,
            "num_samples": cfg.certification.n,
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    if debug_payload is not None:
        (out_dir / "debug_neighbors.json").write_text(json.dumps(debug_payload, indent=2))

    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run NER smoothing experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained NER model.pt")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--rebuild-index", action="store_true", help="Recompute token index from train split")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.checkpoint, split=args.split, rebuild_index=args.rebuild_index)
