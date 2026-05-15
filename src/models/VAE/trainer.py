"""
Training loop for ConvVAE.
"""

from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import ConvVAE, vae_loss


def train_vae(
    model: ConvVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
    beta: float = 1e-4,
    ckpt_dir: str | Path | None = None,
    save_every: int = 1,
    start_epoch: int = 1,
    history: list | None = None,
) -> List[Dict]:
    """
    Train the VAE for a given number of epochs.

    Returns
    -------
    history : list of dicts with keys epoch, loss, recon, kld.
    """
    model.train()
    if history is None:
        history = []

    for epoch in range(start_epoch, epochs + 1):
        total_loss = total_recon = total_kld = 0.0
        pbar = tqdm(train_loader, desc=f"VAE epoch {epoch}/{epochs}")

        for x, _ in pbar:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kld = vae_loss(x, x_hat, mu, logvar, beta=beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kld=f"{kld.item():.4f}",
            )

        n = max(1, len(train_loader))
        stats = {
            "epoch": epoch,
            "loss": total_loss / n,
            "recon": total_recon / n,
            "kld": total_kld / n,
        }
        history.append(stats)
        print(
            f"Epoch {epoch:02d} | loss={stats['loss']:.5f} "
            f"recon={stats['recon']:.5f} kld={stats['kld']:.5f}"
        )

        # Save per-epoch checkpoint for resume
        if ckpt_dir is not None and epoch % save_every == 0:
            ckpt_path = Path(ckpt_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            latest = ckpt_path / "latest.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
            }, latest)
            print(f"  Checkpoint saved → {latest}")

    return history


def save_checkpoint(model: ConvVAE, path: str | Path, extra: dict | None = None):
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(model: ConvVAE, path: str | Path, device: torch.device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    print(f"Checkpoint loaded ← {path}")
    return state
