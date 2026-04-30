"""NER task plugin for generic evaluation and certification runners.

Certification pipeline (correct, efficient):
    1. Run BERT encoder ONCE  → last_hidden_state H  [B, T, D]
    2. Sample noise on H N times (manifold or isotropic)
    3. Pass H + noise through classifier head only  (NO second BERT run)
    4. Vote per token → Clopper-Pearson certification

Training/plain-eval forward still runs the full model (optionally with
embedding-level smoothing for noise-augmented training).
"""

from __future__ import annotations

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.certify import certify_prediction_set
from src.dataloaders import build_conll_dataloaders
from src.models.specific.ner.model import TransformerNER
from src.smoothing import (
    smooth_tensor,
    smooth_input_embeddings,
)


def move_batch(batch, device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def extract_seqeval_lists(logits: torch.Tensor, labels: torch.Tensor, id2label: dict[int, str]):
    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    y_pred = []
    y_true = []
    for p_row, l_row in zip(preds, labels_np):
        p_seq = []
        l_seq = []
        for p, l in zip(p_row, l_row):
            if l == -100:
                continue
            p_seq.append(id2label[int(p)])
            l_seq.append(id2label[int(l)])
        if l_seq:
            y_pred.append(p_seq)
            y_true.append(l_seq)
    return y_true, y_pred


def smooth_hidden_states(hidden_states: torch.Tensor, mask: torch.Tensor, cfg) -> torch.Tensor:
    """Reusable hidden-state smoothing entrypoint for NER certification."""
    return smooth_tensor(
        hidden_states,
        sigma=cfg.smoothing.sigma,
        mode=cfg.smoothing.mode,
        knn_k=cfg.smoothing.knn_k,
        eps_eig=cfg.smoothing.eps_eig,
        attention_mask=mask,
    )


def forward_task_prediction(model, batch, cfg, use_smoothing: bool):
    """Standard single-sample forward for training / plain evaluation.

    For noise-augmented training (use_smoothing=True), applies smoothing at the
    embedding level so gradients still flow through the encoder.
    For clean evaluation, runs the plain model.
    """
    smoothing_target = cfg.smoothing.target or cfg.smoothing.space
    if use_smoothing and cfg.smoothing.enabled and smoothing_target == "input_embeddings":
        smoothed_embeds = smooth_input_embeddings(
            embedding_layer=model.encoder.get_input_embeddings(),
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            mode=cfg.smoothing.mode,
            sigma=cfg.smoothing.sigma,
            knn_k=cfg.smoothing.knn_k,
            eps_eig=cfg.smoothing.eps_eig,
        )
        return model(
            inputs_embeds=smoothed_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )

    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch.get("labels"),
    )


def build_model_and_data(cfg, device: torch.device):
    data = build_conll_dataloaders(cfg.dataset, cfg.dataloader, cfg.model.encoder_name)
    model = TransformerNER(
        encoder_name=cfg.model.encoder_name,
        num_labels=len(data.id2label),
        id2label=data.id2label,
        label2id=data.label2id,
        dropout=cfg.model.dropout,
    ).to(device)
    return model, data


@torch.no_grad()
def evaluate_task_accuracy(model, data, cfg, device: torch.device):
    model.eval()
    y_true_all = []
    y_pred_all = []
    losses = []

    max_batches = cfg.eval.max_batches
    for b_idx, batch in enumerate(tqdm(data.test_loader, desc="eval", leave=False)):
        if max_batches is not None and b_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        losses.append(float(out.loss.detach().cpu()))
        y_true, y_pred = extract_seqeval_lists(out.logits, batch["labels"], data.id2label)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    return {
        "task_loss": float(np.mean(losses)) if losses else 0.0,
        "task_f1": float(f1_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "task_accuracy": float(accuracy_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
    }


@torch.no_grad()
def certify_task(model, data, cfg, device: torch.device):
    """Certified prediction via randomized smoothing on BERT hidden states.

    Efficient pipeline per batch:
        1. Run BERT encoder ONCE  → H = last_hidden_state  [B, T, D]
        2. For n samples:
               noise  = manifold / isotropic noise on H
               logits = classifier(dropout(H + noise))    ← NO second encoder run
               pred   = argmax(logits)
        3. Per token: count votes over n samples → Clopper-Pearson → radius / abstain
    """
    if not cfg.certification.enabled:
        return {"certified_accuracy": 0.0, "abstain_rate": 0.0, "mean_radius": 0.0}

    model.eval()
    total = 0
    certified_correct = 0
    abstained = 0
    radii = []

    max_batches = cfg.eval.max_batches
    n = cfg.certification.n

    for b_idx, batch in enumerate(tqdm(data.test_loader, desc="certify", leave=False)):
        if max_batches is not None and b_idx >= max_batches:
            break

        batch = move_batch(batch, device)
        labels = batch["labels"]
        mask = batch["attention_mask"]

        enc = model.encoder(
            input_ids=batch["input_ids"],
            attention_mask=mask,
            return_dict=True,
        )
        hidden_states = enc.last_hidden_state
        labels_np = labels.detach().cpu().numpy()

        batch_metrics = certify_prediction_set(
            sample_predictions_fn=lambda: model.classifier(
                model.dropout(smooth_hidden_states(hidden_states, mask, cfg))
            ).argmax(dim=-1).detach().cpu().numpy(),
            labels=labels_np,
            n_samples=n,
            num_classes=model.classifier.out_features,
            alpha_noise=cfg.smoothing.sigma,
            alpha_conf=cfg.certification.alpha,
            abstain_label=cfg.certification.abstain_label,
        )

        valid = int((labels_np != -100).sum())
        total += valid
        certified_correct += batch_metrics["certified_accuracy"] * valid
        abstained += batch_metrics["abstain_rate"] * valid
        if batch_metrics["mean_radius"] > 0:
            radii.append(batch_metrics["mean_radius"])

    return {
        "certified_accuracy": (certified_correct / total) if total else 0.0,
        "abstain_rate": (abstained / total) if total else 0.0,
        "mean_radius": float(np.mean(radii)) if radii else 0.0,
    }
