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

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        embedding_dim: int = 256,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        # Guarda TODOS los hiperparámetros (incluye lr y weight_decay)
        self.save_hyperparameters()

        # ResNet-18 desde cero (o con pretrained si se indica)
        self.model = models.resnet18(pretrained=pretrained)

        # Reemplaza la última capa fully connected por una cabeza ajustada al número de clases
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log por época (mejor para EarlyStopping y para lectura de métricas)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Usa lr y weight_decay que vienen desde los YAML (cfg.optim)
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Obtiene embeddings antes de la capa final para evaluación de anomalías."""
        # Pasamos por la red hasta antes de la fc
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        emb = torch.flatten(x, 1)
        return emb


__all__ = ["ResNetScratchModule"]
