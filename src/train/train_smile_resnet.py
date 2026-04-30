"""Training loop for ResNet smile classification on CelebA/CelebA-HQ."""

from __future__ import annotations

import csv
import json
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.configs.train_smile_io import save_smile_resolved_config
from src.configs.train_smile_schema import SmileTrainingConfig
from src.dataloaders.celeba_smile import SmileDataBundle, build_smile_dataloaders
from src.models.resnet_smile import build_resnet_smile_classifier


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


class SmileTrainer:
    def __init__(self, cfg: SmileTrainingConfig):
        self.cfg = cfg
        self.device = device_from_cfg(cfg.train.device)
        self._stop_requested = False
        self._wandb_run = None

        set_seed(cfg.train.seed)

        self.data: SmileDataBundle = build_smile_dataloaders(cfg.dataset, cfg.dataloader, cfg.model)
        self.model = build_resnet_smile_classifier(
            name=cfg.model.name,
            pretrained=cfg.model.pretrained,
            dropout=cfg.model.dropout,
        ).to(self.device)

        pos_weight = torch.tensor([self.data.pos_weight], dtype=torch.float32, device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        self.checkpoint_dir = Path(cfg.checkpoint.base_dir) / cfg.dataset.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = Path(cfg.output_dir) / "training_history.csv"
        self.summary_path = Path(cfg.output_dir) / "training_summary.json"
        self.resolved_config_path = Path(cfg.output_dir) / "resolved_smile_config.yaml"
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        save_smile_resolved_config(cfg, self.resolved_config_path)

        self._configure_signal_handlers()
        self._init_wandb()

    def _configure_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            print(f"Received signal {signum}. Stopping after current step and saving checkpoint.")
            self._stop_requested = True

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def _init_wandb(self) -> None:
        if not self.cfg.wandb.enabled:
            return
        try:
            import wandb
        except ImportError:
            print("W&B is enabled in config but package is not installed. Continuing without W&B logging.")
            return

        self._wandb_run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            name=self.cfg.wandb.run_name,
            config=asdict(self.cfg),
        )

    def _log_wandb(self, payload: Dict) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(payload)

    @staticmethod
    def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct = (preds == labels).float().mean()
        return float(correct.detach().cpu())

    def _run_epoch(self, epoch_idx: int) -> Dict[str, float]:
        self.model.train()
        losses: List[float] = []
        accuracies: List[float] = []

        pbar = tqdm(self.data.train_loader, desc=f"train epoch {epoch_idx}", leave=False)
        for step, (images, labels) in enumerate(pbar, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).unsqueeze(1)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
            self.optimizer.step()

            batch_loss = float(loss.detach().cpu())
            batch_acc = self._batch_accuracy(logits, labels)

            losses.append(batch_loss)
            accuracies.append(batch_acc)

            if step % self.cfg.logging.log_every_n_steps == 0:
                pbar.set_postfix({"loss": f"{np.mean(losses):.4f}", "acc": f"{np.mean(accuracies):.4f}"})
                self._log_wandb(
                    {
                        "step_train_loss": batch_loss,
                        "step_train_acc": batch_acc,
                        "epoch": epoch_idx,
                    }
                )

            if self._stop_requested:
                break

        return {
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_acc": float(np.mean(accuracies)) if accuracies else 0.0,
        }

    @torch.no_grad()
    def _evaluate(self, split: str = "val") -> Dict[str, float]:
        loader = self.data.val_loader if split == "val" else self.data.test_loader
        self.model.eval()
        losses: List[float] = []
        accuracies: List[float] = []

        for images, labels in tqdm(loader, desc=f"{split} eval", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).unsqueeze(1)
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            losses.append(float(loss.detach().cpu()))
            accuracies.append(self._batch_accuracy(logits, labels))

        return {
            f"{split}_loss": float(np.mean(losses)) if losses else 0.0,
            f"{split}_acc": float(np.mean(accuracies)) if accuracies else 0.0,
        }

    def _checkpoint_path(self, epoch_idx: int) -> Path:
        model_name = self.cfg.model.name
        return self.checkpoint_dir / f"{model_name}_checkpoint_epoch_{epoch_idx}.pt"

    def _save_checkpoint(self, epoch_idx: int, interrupted: bool = False) -> Path:
        checkpoint_path = self._checkpoint_path(epoch_idx)
        payload = {
            "epoch": epoch_idx,
            "model_name": self.cfg.model.name,
            "state_dict": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "interrupted": interrupted,
        }
        torch.save(payload, checkpoint_path)

        # Keep only the latest K epoch checkpoints to improve resiliency on interrupted jobs.
        keep_k = max(1, self.cfg.checkpoint.keep_last_k)
        all_ckpts = sorted(self.checkpoint_dir.glob(f"{self.cfg.model.name}_checkpoint_epoch_*.pt"))
        for old_path in all_ckpts[:-keep_k]:
            old_path.unlink(missing_ok=True)

        latest_path = self.checkpoint_dir / f"{self.cfg.model.name}_latest.pt"
        torch.save(payload, latest_path)
        return checkpoint_path

    def _append_history_row(self, row: Dict) -> None:
        file_exists = self.history_path.exists()
        with self.history_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def train(self) -> Dict:
        start_time = time.perf_counter()
        rows: List[Dict] = []
        last_checkpoint: Optional[str] = None

        for epoch in range(1, self.cfg.train.epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = self._run_epoch(epoch)
            val_metrics = self._evaluate(split="val")
            epoch_time_sec = time.perf_counter() - epoch_start

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "train_acc": train_metrics["train_acc"],
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["val_acc"],
                "epoch_time_sec": epoch_time_sec,
            }
            self._append_history_row(row)
            rows.append(row)

            self._log_wandb({**row, "event": "epoch_end"})
            print(json.dumps(row))

            if epoch % self.cfg.train.save_every_n_epochs == 0 or self._stop_requested:
                ckpt_path = self._save_checkpoint(epoch_idx=epoch, interrupted=self._stop_requested)
                last_checkpoint = str(ckpt_path)

            if self._stop_requested:
                break

        test_metrics = self._evaluate(split="test")
        total_time_sec = time.perf_counter() - start_time

        summary = {
            "experiment_name": self.cfg.experiment_name,
            "dataset": self.cfg.dataset.name,
            "model": self.cfg.model.name,
            "device": str(self.device),
            "class_counts": self.data.class_counts,
            "train_pos_weight": self.data.pos_weight,
            "epochs_completed": len(rows),
            "total_training_time_sec": total_time_sec,
            "history_file": str(self.history_path),
            "last_checkpoint": last_checkpoint,
            "interrupted": self._stop_requested,
            **test_metrics,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2))

        if self._wandb_run is not None:
            self._wandb_run.log({**test_metrics, "total_training_time_sec": total_time_sec})
            self._wandb_run.finish()

        print(json.dumps(summary, indent=2))
        return summary


def run_smile_training(cfg: SmileTrainingConfig) -> Dict:
    trainer = SmileTrainer(cfg)
    return trainer.train()