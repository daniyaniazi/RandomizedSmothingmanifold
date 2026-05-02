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
from datetime import datetime
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


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _load_partial_state(partial_path: Path) -> dict | None:
    if not partial_path.exists():
        return None
    try:
        return json.loads(partial_path.read_text())
    except Exception:
        return None


def _save_partial_state(partial_path: Path, state: dict) -> None:
    partial_path.write_text(json.dumps(state, indent=2))


def _compute_running_metrics(
    y_true_all,
    y_pred_all,
    radii,
    abstentions: int,
    total_tokens: int,
    certified_correct: int,
    sentence_count: int,
    sentence_token_counts,
    num_samples: int,
) -> dict:
    return {
        "precision": float(precision_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "recall": float(recall_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "f1": float(f1_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "token_acc": float(accuracy_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "certified_token_acc": float(certified_correct / total_tokens) if total_tokens else 0.0,
        "certified_correct_tokens": int(certified_correct),
        "mean_certified_radius": float(np.mean(radii)) if radii else 0.0,
        "abstention_rate": float(abstentions / total_tokens) if total_tokens else 0.0,
        "total_certified_tokens": int(total_tokens),
        "sentences_evaluated": int(sentence_count),
        "avg_tokens_per_sentence": float(np.mean(sentence_token_counts)) if sentence_token_counts else 0.0,
        "max_tokens_in_sentence": int(np.max(sentence_token_counts)) if sentence_token_counts else 0,
        "noisy_samples_per_token": int(num_samples),
        "total_noisy_samples": int(total_tokens * num_samples),
    }


def _write_running_artifacts(out_dir: Path, batch_idx: int, metrics: dict, debug_payload: dict | None) -> None:
    running_metrics_path = out_dir / "running_metrics.json"
    running_metrics_path.write_text(
        json.dumps(
            {
                "last_completed_batch": int(batch_idx + 1),
                "running": metrics,
            },
            indent=2,
        )
    )

    if debug_payload is not None:
        running_examples_path = out_dir / "running_examples.json"
        running_examples_path.write_text(json.dumps(debug_payload, indent=2))
        fig_path = _save_debug_vote_plot(debug_payload.get("examples", []), out_dir)
        if fig_path is not None:
            (out_dir / "running_artifacts.json").write_text(
                json.dumps(
                    {
                        "running_examples": str(running_examples_path),
                        "running_vote_distribution": str(fig_path),
                    },
                    indent=2,
                )
            )


def _masked_batch_and_stats(batch, tokenizer, masking_cfg, rng: np.random.Generator):
    if masking_cfg is None or not bool(getattr(masking_cfg, "enabled", False)):
        return batch, {"masked_tokens": 0, "masked_sentences": 0}
    if getattr(masking_cfg, "mode", "none") == "none":
        return batch, {"masked_tokens": 0, "masked_sentences": 0}

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.unk_token_id
    if mask_token_id is None:
        return batch, {"masked_tokens": 0, "masked_sentences": 0}

    input_ids = batch["input_ids"].clone()
    labels = batch["labels"]
    attn = batch["attention_mask"]

    valid = (labels != -100) & attn.bool()
    lengths = valid.sum(dim=1).detach().cpu().numpy().astype(np.int64)
    batch_avg_tokens = int(np.round(float(np.mean(lengths)))) if len(lengths) else 1

    entity_set = {int(x) for x in getattr(masking_cfg, "entity_label_ids", [1])}
    mode = getattr(masking_cfg, "mode", "none")
    ratio = float(getattr(masking_cfg, "mask_ratio", 0.15))
    max_masks_cfg = getattr(masking_cfg, "max_masks_per_sentence", None)
    cap_by_avg = bool(getattr(masking_cfg, "cap_by_batch_avg_tokens", True))

    masked_tokens = 0
    masked_sentences = 0

    for row_idx in range(input_ids.shape[0]):
        valid_positions = [int(i) for i in torch.where(valid[row_idx])[0].detach().cpu().numpy().tolist()]
        if not valid_positions:
            continue

        labels_row = labels[row_idx].detach().cpu().numpy()
        entity_positions = [i for i in valid_positions if int(labels_row[i]) in entity_set]
        context_positions = [i for i in valid_positions if int(labels_row[i]) not in entity_set]

        if mode == "context":
            candidates = context_positions
        elif mode == "entity":
            candidates = entity_positions
        elif mode == "hybrid":
            candidates = list(dict.fromkeys(context_positions + entity_positions))
        else:
            candidates = []

        if not candidates:
            continue

        requested = max(1, int(round(ratio * len(valid_positions))))
        if max_masks_cfg is not None:
            requested = min(requested, int(max_masks_cfg))
        if cap_by_avg:
            requested = min(requested, max(1, int(batch_avg_tokens)))
        requested = min(requested, len(candidates))
        if requested <= 0:
            continue

        selected = rng.choice(np.asarray(candidates, dtype=np.int64), size=requested, replace=False)
        input_ids[row_idx, torch.as_tensor(selected, dtype=torch.long)] = int(mask_token_id)
        masked_tokens += int(requested)
        masked_sentences += 1

    masked_batch = dict(batch)
    masked_batch["input_ids"] = input_ids
    return masked_batch, {"masked_tokens": int(masked_tokens), "masked_sentences": int(masked_sentences)}


def _top_vote_labels(counts: np.ndarray, id2label: dict[int, str], k: int = 5) -> list[dict[str, float]]:
    k = max(1, min(k, int(len(counts))))
    idx = np.argsort(counts)[::-1][:k]
    total = int(np.sum(counts))
    rows = []
    for i in idx:
        c = int(counts[int(i)])
        rows.append(
            {
                "label_id": int(i),
                "label": id2label[int(i)],
                "count": c,
                "fraction": float(c / total) if total else 0.0,
            }
        )
    return rows


def _build_debug_examples(batch, out, tokenizer, id2label: dict[int, str], max_sentences: int = 3, max_tokens: int = 8):
    input_ids = batch["input_ids"].detach().cpu().numpy()
    labels = batch["labels"].detach().cpu().numpy()
    pred_ids = out.pred_ids.detach().cpu().numpy()
    vote_counts = out.vote_counts

    examples = []
    for row_idx in range(min(input_ids.shape[0], max_sentences)):
        valid_positions = [i for i, l in enumerate(labels[row_idx]) if int(l) != -100]
        if not valid_positions:
            continue

        sentence_tokens = tokenizer.convert_ids_to_tokens([int(input_ids[row_idx, i]) for i in valid_positions])
        sentence_true_labels = [id2label[int(labels[row_idx, i])] for i in valid_positions]
        sentence_pred_labels = [id2label[int(pred_ids[row_idx, i])] for i in valid_positions]

        token_debug = []
        for token_idx in valid_positions[:max_tokens]:
            cert = out.certificates[row_idx][token_idx]
            dbg = None
            if out.debug is not None and row_idx < len(out.debug):
                dbg = out.debug[row_idx][token_idx]

            token_debug.append(
                {
                    "token_position": int(token_idx),
                    "token": tokenizer.convert_ids_to_tokens([int(input_ids[row_idx, token_idx])])[0],
                    "true_label": id2label[int(labels[row_idx, token_idx])],
                    "pred_label_majority_vote": id2label[int(pred_ids[row_idx, token_idx])],
                    "certified": bool(cert is not None and not cert.abstained),
                    "certified_radius": float(cert.radius) if cert is not None else 0.0,
                    "vote_top5": _top_vote_labels(vote_counts[row_idx, token_idx], id2label, k=5),
                    "top10_nearest_neighbor_tokens": [] if dbg is None else dbg.neighbor_tokens,
                    "top10_nearest_neighbor_labels": [] if dbg is None else [id2label[int(x)] for x in dbg.neighbor_labels],
                    "reconstruction_l2": None if dbg is None else float(dbg.reconstruction_l2),
                    "noisy_l2": None if dbg is None else float(dbg.noisy_l2),
                }
            )

        examples.append(
            {
                "sentence_index_in_batch": int(row_idx),
                "sentence_tokens": sentence_tokens,
                "sentence_true_labels": sentence_true_labels,
                "sentence_pred_labels_majority_vote": sentence_pred_labels,
                "token_debug": token_debug,
            }
        )

    return examples


def _save_debug_vote_plot(debug_examples: list[dict], out_dir: Path) -> str | None:
    if not debug_examples:
        return None
    first = debug_examples[0]
    if not first.get("token_debug"):
        return None
    token_row = first["token_debug"][0]
    vote_rows = token_row.get("vote_top5", [])
    if not vote_rows:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = [r["label"] for r in vote_rows]
    counts = [int(r["count"]) for r in vote_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, counts)
    ax.set_title(f"Vote Distribution | token={token_row['token']}")
    ax.set_ylabel("votes")
    fig.tight_layout()

    fig_path = out_dir / "debug_vote_distribution.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    return str(fig_path)


@torch.no_grad()
def evaluate_clean(model, loader, device, id2label: dict[int, str], max_batches: int | None = None, tokenizer=None, masking_cfg=None):
    _log(f"Starting clean evaluation (max_batches={max_batches})")
    model.eval()
    y_true_all = []
    y_pred_all = []
    losses = []
    rng = np.random.default_rng(int(getattr(masking_cfg, "seed", 73)))
    total_masked_tokens = 0
    total_masked_sentences = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch, mask_stats = _masked_batch_and_stats(batch, tokenizer=tokenizer, masking_cfg=masking_cfg, rng=rng)
        total_masked_tokens += int(mask_stats["masked_tokens"])
        total_masked_sentences += int(mask_stats["masked_sentences"])
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

        if (batch_idx + 1) % 10 == 0:
            _log(f"Clean eval progress: processed {batch_idx + 1} batches")

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "precision": float(precision_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "recall": float(recall_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "f1": float(f1_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "token_acc": float(accuracy_score(y_true_all, y_pred_all)) if y_true_all else 0.0,
        "masked_tokens": int(total_masked_tokens),
        "masked_sentences": int(total_masked_sentences),
    }


@torch.no_grad()
def evaluate_smoothed(
    model,
    loader,
    device,
    tokenizer,
    id2label: dict[int, str],
    token_index_artifacts,
    cfg,
    max_batches: int | None = None,
    out_dir: Path | None = None,
    resume: bool = False,
    save_every_batches: int = 5,
    log_every_batches: int = 1,
):
    _log(
        "Starting smoothed evaluation "
        f"(max_batches={max_batches}, n={cfg.certification.n}, knn_k={cfg.smoothing.knn_k}, "
        f"sigma={cfg.smoothing.sigma}, resume={resume})"
    )
    y_true_all = []
    y_pred_all = []
    radii = []
    abstentions = 0
    total_tokens = 0
    certified_correct = 0
    sentence_count = 0
    sentence_token_counts = []
    debug_payload = None
    rng = np.random.default_rng(int(getattr(cfg.masking, "seed", 73)))
    total_masked_tokens = 0
    total_masked_sentences = 0
    start_batch = 0
    partial_path = (out_dir / "smoothed.partial.json") if out_dir is not None else None

    if resume and partial_path is not None:
        state = _load_partial_state(partial_path)
        if state is not None:
            start_batch = int(state.get("next_batch_idx", 0))
            y_true_all = state.get("y_true_all", [])
            y_pred_all = state.get("y_pred_all", [])
            radii = [float(x) for x in state.get("radii", [])]
            abstentions = int(state.get("abstentions", 0))
            total_tokens = int(state.get("total_tokens", 0))
            certified_correct = int(state.get("certified_correct", 0))
            sentence_count = int(state.get("sentence_count", 0))
            sentence_token_counts = [int(x) for x in state.get("sentence_token_counts", [])]
            debug_payload = state.get("debug_payload", None)
            _log(
                "Loaded partial state "
                f"from batch {start_batch} (tokens={total_tokens}, certified_correct={certified_correct})"
            )
        else:
            _log("Resume requested but no valid partial state found; starting from batch 0")

    for batch_idx, batch in enumerate(loader):
        if batch_idx < start_batch:
            continue
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch, mask_stats = _masked_batch_and_stats(batch, tokenizer=tokenizer, masking_cfg=cfg.masking, rng=rng)
        total_masked_tokens += int(mask_stats["masked_tokens"])
        total_masked_sentences += int(mask_stats["masked_sentences"])
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
            collect_debug=batch_idx == start_batch,
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
                    if (not cert.abstained) and int(p) == int(l):
                        certified_correct += 1
                if cert is not None and cert.abstained:
                    p_seq.append("ABSTAIN")
                else:
                    p_seq.append(id2label[int(p)])
                l_seq.append(id2label[int(l)])
            if l_seq:
                y_pred_all.append(p_seq)
                y_true_all.append(l_seq)
                sentence_count += 1
                sentence_token_counts.append(len(l_seq))

        if batch_idx == start_batch and out.debug is not None:
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
            debug_payload = {
                "neighbors": debug_rows,
                "examples": _build_debug_examples(batch=batch, out=out, tokenizer=tokenizer, id2label=id2label),
            }

        if partial_path is not None and ((batch_idx + 1) % max(1, save_every_batches) == 0):
            running_metrics = _compute_running_metrics(
                y_true_all=y_true_all,
                y_pred_all=y_pred_all,
                radii=radii,
                abstentions=abstentions,
                total_tokens=total_tokens,
                certified_correct=certified_correct,
                sentence_count=sentence_count,
                sentence_token_counts=sentence_token_counts,
                num_samples=cfg.certification.n,
            )

            _save_partial_state(
                partial_path,
                {
                    "next_batch_idx": int(batch_idx + 1),
                    "y_true_all": y_true_all,
                    "y_pred_all": y_pred_all,
                    "radii": radii,
                    "abstentions": int(abstentions),
                    "total_tokens": int(total_tokens),
                    "certified_correct": int(certified_correct),
                    "sentence_count": int(sentence_count),
                    "sentence_token_counts": sentence_token_counts,
                    "debug_payload": debug_payload,
                    "running_metrics": running_metrics,
                },
            )
            if out_dir is not None:
                _write_running_artifacts(out_dir=out_dir, batch_idx=batch_idx, metrics=running_metrics, debug_payload=debug_payload)
            _log(
                "Saved partial state "
                f"at batch {batch_idx + 1}: tokens={total_tokens}, cert_acc="
                f"{(certified_correct / total_tokens) if total_tokens else 0.0:.4f}, "
                f"abstain_rate={(abstentions / total_tokens) if total_tokens else 0.0:.4f}; "
                f"running_metrics={out_dir / 'running_metrics.json' if out_dir is not None else 'n/a'}"
            )

        if (batch_idx + 1) % max(1, log_every_batches) == 0:
            running_metrics = _compute_running_metrics(
                y_true_all=y_true_all,
                y_pred_all=y_pred_all,
                radii=radii,
                abstentions=abstentions,
                total_tokens=total_tokens,
                certified_correct=certified_correct,
                sentence_count=sentence_count,
                sentence_token_counts=sentence_token_counts,
                num_samples=cfg.certification.n,
            )
            if out_dir is not None:
                _write_running_artifacts(out_dir=out_dir, batch_idx=batch_idx, metrics=running_metrics, debug_payload=debug_payload)
            _log(
                "Smoothed eval progress: "
                f"batch={batch_idx + 1}, tokens={total_tokens}, cert_acc="
                f"{(certified_correct / total_tokens) if total_tokens else 0.0:.4f}, "
                f"abstain_rate={(abstentions / total_tokens) if total_tokens else 0.0:.4f}, "
                f"mean_radius={float(np.mean(radii)) if radii else 0.0:.4f}"
            )

    metrics = _compute_running_metrics(
        y_true_all=y_true_all,
        y_pred_all=y_pred_all,
        radii=radii,
        abstentions=abstentions,
        total_tokens=total_tokens,
        certified_correct=certified_correct,
        sentence_count=sentence_count,
        sentence_token_counts=sentence_token_counts,
        num_samples=cfg.certification.n,
    )
    metrics["masked_tokens"] = int(total_masked_tokens)
    metrics["masked_sentences"] = int(total_masked_sentences)

    if partial_path is not None and partial_path.exists():
        partial_path.unlink()
        _log(f"Removed partial state after successful completion: {partial_path}")

    _log(
        "Completed smoothed evaluation: "
        f"tokens={total_tokens}, certified_correct={certified_correct}, "
        f"cert_acc={(certified_correct / total_tokens) if total_tokens else 0.0:.4f}, "
        f"abstain_rate={(abstentions / total_tokens) if total_tokens else 0.0:.4f}"
    )

    return metrics, debug_payload


def run(
    cfg_path: str,
    checkpoint_path: str,
    split: str = "test",
    rebuild_index: bool = False,
    resume: bool = False,
    save_every_batches: int = 5,
    log_every_batches: int = 1,
):
    _log(
        "Run started with "
        f"cfg={cfg_path}, checkpoint={checkpoint_path}, split={split}, "
        f"rebuild_index={rebuild_index}, resume={resume}"
    )
    cfg = load_experiment_config(cfg_path)
    set_seed(cfg.train.seed)
    device = device_from_cfg(cfg.train.device)

    out_dir = Path(cfg.output_dir) / f"{cfg.experiment_name}_smoothing_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, out_dir / "resolved_config.yaml")

    data = build_conll_dataloaders(cfg.dataset, cfg.dataloader, cfg.model.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(data.tokenizer_name)
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
    _log(f"Loaded checkpoint and model on device={device}")

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
    _log(
        "Token index ready "
        f"(backend={cfg.smoothing.index_backend}, metric={cfg.smoothing.index_metric}, "
        f"vectors={len(token_index_artifacts.token_texts)})"
    )
    _log(
        "Masking config "
        f"(enabled={cfg.masking.enabled}, mode={cfg.masking.mode}, ratio={cfg.masking.mask_ratio}, "
        f"entity_label_ids={cfg.masking.entity_label_ids})"
    )

    target_loader = data.val_loader if split == "val" else data.test_loader
    clean_metrics = evaluate_clean(
        model,
        target_loader,
        device,
        data.id2label,
        cfg.eval.max_batches,
        tokenizer=tokenizer,
        masking_cfg=cfg.masking,
    )
    smooth_metrics, debug_payload = evaluate_smoothed(
        model=model,
        loader=target_loader,
        device=device,
        tokenizer=tokenizer,
        id2label=data.id2label,
        token_index_artifacts=token_index_artifacts,
        cfg=cfg,
        max_batches=cfg.eval.max_batches,
        out_dir=out_dir,
        resume=resume,
        save_every_batches=save_every_batches,
        log_every_batches=log_every_batches,
    )

    summary = {
        "checkpoint": checkpoint_path,
        "split": split,
        "clean": clean_metrics,
        "smoothed": smooth_metrics,
        "comparison": {
            "clean_token_acc": clean_metrics["token_acc"],
            "smoothed_token_acc": smooth_metrics["token_acc"],
            "certified_token_acc": smooth_metrics["certified_token_acc"],
        },
        "smoothing": {
            "sigma": cfg.smoothing.sigma,
            "knn_k": cfg.smoothing.knn_k,
            "layer_index": cfg.smoothing.layer_index,
            "num_samples": cfg.certification.n,
        },
        "masking": {
            "enabled": cfg.masking.enabled,
            "mode": cfg.masking.mode,
            "mask_ratio": cfg.masking.mask_ratio,
            "entity_label_ids": cfg.masking.entity_label_ids,
            "cap_by_batch_avg_tokens": cfg.masking.cap_by_batch_avg_tokens,
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    _log(f"Wrote metrics to {(out_dir / 'metrics.json')}" )
    if debug_payload is not None:
        (out_dir / "debug_neighbors.json").write_text(json.dumps(debug_payload, indent=2))
        _log(f"Wrote debug neighbors to {(out_dir / 'debug_neighbors.json')}")
        fig_path = _save_debug_vote_plot(debug_payload.get("examples", []), out_dir)
        if fig_path is not None:
            summary["artifacts"] = {"debug_vote_distribution": fig_path}
            (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
            _log(f"Wrote debug figure to {fig_path}")

    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run NER smoothing experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained NER model.pt")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--rebuild-index", action="store_true", help="Recompute token index from train split")
    parser.add_argument("--resume", action="store_true", help="Resume smoothed eval from out_dir/smoothed.partial.json if present")
    parser.add_argument("--save-every-batches", type=int, default=5, help="Persist partial smoothed state every N batches")
    parser.add_argument("--log-every-batches", type=int, default=1, help="Log smoothed progress every N batches")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.config,
        args.checkpoint,
        split=args.split,
        rebuild_index=args.rebuild_index,
        resume=args.resume,
        save_every_batches=args.save_every_batches,
        log_every_batches=args.log_every_batches,
    )
