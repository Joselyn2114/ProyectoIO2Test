"""Modelo C: Autoencoder tipo U-Net.

El modelo aprende a reconstruir imágenes normales y genera un embedding latente
para la evaluación basada en distancia de Mahalanobis como indica el enunciado.
"""
from __future__ import annotations

from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetAutoencoder(nn.Module):
    """Arquitectura U-Net simplificada para reconstrucción."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, latent_dim: int = 128):
        super().__init__()
        self.enc1 = double_conv(in_channels, base_channels)
        self.enc2 = double_conv(base_channels, base_channels * 2)
        self.enc3 = double_conv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(latent_dim, base_channels * 4, kernel_size=2, stride=2)
        self.dec1 = double_conv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = double_conv(base_channels * 4, base_channels * 2)
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec3 = double_conv(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))

        d1 = self.up1(bottleneck)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)
        out = self.out_conv(d3)
        return out, bottleneck


class UNetAutoencoderModule(pl.LightningModule):
    """LightningModule para entrenamiento del autoencoder."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 128,
        lr: float = 1e-3,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = UNetAutoencoder(in_channels, base_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.model(x)
        return recon

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str):
        x, _ = batch
        recon, _ = self.model(x)
        loss = F.l1_loss(recon, x)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el embedding latente del bottleneck."""
        _, bottleneck = self.model(x)
        return torch.flatten(bottleneck, 1)


__all__ = ["UNetAutoencoder", "UNetAutoencoderModule"]
