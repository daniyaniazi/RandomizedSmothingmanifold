"""
Smoke test: 1 epoch, max 3 batches per split.
Verifies dataset loading, model forward pass, loss, and checkpoint saving.
Run: python test.py
"""

from pathlib import Path

from src.configs.train_smile_io import load_smile_training_config
from src.configs.train_smile_schema import SmileTrainingConfig
from src.dataloaders.celeba_smile import build_smile_dataloaders
from src.models.resnet_smile import build_resnet_smile_classifier

import torch
from torch import nn
from torch.optim import AdamW

CONFIG = "src/configs/training/smile_resnet_celeba.yaml"
MAX_BATCHES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run():
	print(f"[smoke] device = {DEVICE}")
	print(f"[smoke] loading config: {CONFIG}")

	cfg: SmileTrainingConfig = load_smile_training_config(CONFIG)
	cfg.train.epochs = 1
	cfg.dataloader.batch_size = 8
	cfg.dataset.num_workers = 0

	print("[smoke] building dataloaders ...")
	data = build_smile_dataloaders(cfg.dataset, cfg.dataloader, cfg.model)
	print(f"[smoke] class_counts: {data.class_counts}")
	print(f"[smoke] pos_weight:   {data.pos_weight:.4f}")

	print(f"[smoke] building model: {cfg.model.name}")
	model = build_resnet_smile_classifier(
		name=cfg.model.name,
		pretrained=cfg.model.pretrained,
		dropout=cfg.model.dropout,
	).to(DEVICE)

	pos_weight = torch.tensor([data.pos_weight], dtype=torch.float32, device=DEVICE)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	optimizer = AdamW(model.parameters(), lr=cfg.train.lr)

	print(f"[smoke] train loop ({MAX_BATCHES} batches) ...")
	model.train()
	for i, (imgs, labels) in enumerate(data.train_loader):
		if i >= MAX_BATCHES:
			break
		imgs = imgs.to(DEVICE)
		labels = labels.to(DEVICE).unsqueeze(1)
		logits = model(imgs)
		loss = criterion(logits, labels)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()
		acc = ((torch.sigmoid(logits) >= 0.5).float() == labels).float().mean().item()
		print(f"  train batch {i+1}: loss={loss.item():.4f}  acc={acc:.4f}")

	print(f"[smoke] val loop ({MAX_BATCHES} batches) ...")
	model.eval()
	with torch.no_grad():
		for i, (imgs, labels) in enumerate(data.val_loader):
			if i >= MAX_BATCHES:
				break
			imgs = imgs.to(DEVICE)
			labels = labels.to(DEVICE).unsqueeze(1)
			logits = model(imgs)
			loss = criterion(logits, labels)
			acc = ((torch.sigmoid(logits) >= 0.5).float() == labels).float().mean().item()
			print(f"  val   batch {i+1}: loss={loss.item():.4f}  acc={acc:.4f}")

	ckpt_dir = Path(cfg.checkpoint.base_dir) / cfg.dataset.name
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	ckpt_path = ckpt_dir / f"{cfg.model.name}_smoke_test.pt"
	torch.save({"epoch": 1, "state_dict": model.state_dict()}, ckpt_path)
	print(f"[smoke] checkpoint saved: {ckpt_path}")
	print("\n[smoke] ALL CHECKS PASSED — safe to submit SLURM job.")


if __name__ == "__main__":
	run()