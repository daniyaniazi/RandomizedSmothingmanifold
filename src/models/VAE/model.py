"""
VAE model definition.
ConvVAE: convolutional encoder-decoder that works for any image_size divisible by 16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder.

    Parameters
    ----------
    in_channels : int
        Number of image channels (3 for RGB, 1 for greyscale).
    image_size : int
        Spatial size of the (square) input image. Must be divisible by 16.
    latent_dim : int
        Dimensionality of the latent space z.
    """

    def __init__(self, in_channels: int = 3, image_size: int = 64, latent_dim: int = 128):
        super().__init__()

        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16.")

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.enc_spatial = image_size // 16
        self.enc_feat_dim = 256 * self.enc_spatial * self.enc_spatial

        # --- Encoder -------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(self.enc_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_feat_dim, latent_dim)

        # --- Decoder -------------------------------------------------------
        self.fc_dec = nn.Linear(latent_dim, self.enc_feat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        h = self.encoder(x).view(x.shape[0], -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.shape[0], 256, self.enc_spatial, self.enc_spatial)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-4,
):
    """
    Beta-VAE ELBO loss.

    Returns (total_loss, recon_loss, kld_loss) — the last two are detached scalars
    for logging purposes.
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kld
    return total, recon.detach(), kld.detach()
