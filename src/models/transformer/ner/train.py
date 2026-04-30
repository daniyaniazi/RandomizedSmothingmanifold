"""Training entrypoint for NER token classification models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from tqdm import tqdm

from src.configs import ExperimentConfig, load_experiment_config, save_resolved_config
from src.dataloaders import build_conll_dataloaders
from src.models.transformer.ner.model import TransformerNER
from src.smoothing import smooth_input_embeddings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_cfg(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


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


def forward_task_prediction(model, batch, cfg: ExperimentConfig, use_smoothing: bool):
    if use_smoothing and cfg.smoothing.enabled:
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
            labels=batch["labels"],
        )

    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )


def train_one_epoch(model, loader, optimizer, device, cfg: ExperimentConfig):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        out = forward_task_prediction(model, batch, cfg, use_smoothing=cfg.smoothing.enabled)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        optimizer.step()
        losses.append(float(out.loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_plain(model, loader, device, id2label: dict[int, str], max_batches: int | None = None):
    model.eval()
    y_true_all = []
    y_pred_all = []
    losses = []
    for b_idx, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches is not None and b_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        losses.append(float(out.loss.detach().cpu()))
        y_true, y_pred = extract_seqeval_lists(out.logits, batch["labels"], id2label)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "f1": float(f1_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "token_acc": float(accuracy_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
    }


def run(cfg_path: str):
    cfg = load_experiment_config(cfg_path)
    set_seed(cfg.train.seed)
    device = device_from_cfg(cfg.train.device)

    out_dir = Path(cfg.output_dir)
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

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    history = []
    for epoch in range(cfg.train.epochs):
        train_loss = train_one_epoch(model, data.train_loader, optimizer, device, cfg)
        val_metrics = evaluate_plain(model, data.val_loader, device, data.id2label, cfg.eval.max_batches)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_token_acc": val_metrics["token_acc"],
        }
        history.append(row)
        print(json.dumps(row))

    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/experiments/ner_conll2003_distilbert.yaml",
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
