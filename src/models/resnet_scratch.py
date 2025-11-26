"""Modelo A: ResNet-18 entrenada desde cero.

Incluye un LightningModule con pasos training/validation/test y extracción de
embeddings para evaluación con distancia de Mahalanobis como exige el enunciado.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class ResNetScratchModule(pl.LightningModule):
    """LightningModule para entrenamiento desde cero de ResNet-18."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False, embedding_dim: int = 512, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=pretrained)
        # Reemplaza la última capa fully connected por una cabeza ajustada al número de clases
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
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
        """Obtiene embeddings antes de la capa final para evaluación de anomalías."""
        features = self.model.forward_features(x) if hasattr(self.model, "forward_features") else self.model.avgpool(
            self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))))))
        )
        emb = torch.flatten(features, 1)
        return emb


__all__ = ["ResNetScratchModule"]
