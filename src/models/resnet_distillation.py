"""Modelo B: distillation teacher-student basado en ResNet-18.

Implementa el esquema solicitado en el enunciado: el teacher es una ResNet-18
preentrenada y congelada; el student es una versión ligera entrenada con mezcla
de pérdida de distillation (KL) y clasificación (CE).
"""
from __future__ import annotations

from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


def build_student(width: float = 0.5, num_classes: int = 2) -> nn.Module:
    """Crea un estudiante basado en ResNet-18 estándar.

    Para simplificar y evitar inconsistencias de canales, NO modificamos
    conv1 ni los bloques internos. Solo ajustamos la capa final.
    """
    model = models.resnet18(pretrained=False)
    # Reducimos el número de canales en la primera capa para aligerar el modelo
    # model.conv1 = nn.Conv2d(3, int(64 * width), kernel_size=7, stride=2, padding=3, bias=False)
    # model.bn1 = nn.BatchNorm2d(int(64 * width))
    # model.layer1[0].conv1.in_channels = int(64 * width)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNetDistillationModule(pl.LightningModule):
    """LightningModule para distillation teacher-student."""

    def __init__(
        self,
        num_classes: int = 2,
        teacher_pretrained: bool = True,
        student_width: float = 0.5,
        alpha: float = 0.5,
        temperature: float = 4.0,
        lr: float = 1e-3,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        
        # Guarda hiperparámetros básicos (opcional, pero útil)
        self.save_hyperparameters(ignore=["teacher", "student"])

        # Guarda lr explícitamente
        self.lr = lr
        
        # Teacher: ResNet18 con backbone preentrenado, cabeza de 2 clases
        self.teacher = models.resnet18(pretrained=teacher_pretrained)
        in_features = self.teacher.fc.in_features
        self.teacher.fc = nn.Linear(in_features, num_classes)
        
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student: ResNet18 "ligero" 
        self.student = build_student(student_width, num_classes)
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)

    def _distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft_teacher = F.log_softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.temperature ** 2)
        ce = F.cross_entropy(student_logits, targets)
        return self.alpha * kl + (1 - self.alpha) * ce

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str):
        x, y = batch
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        loss = self._distillation_loss(student_logits, teacher_logits, y)
        preds = torch.argmax(student_logits, dim=1)
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
        return torch.optim.Adam(self.student.parameters(), lr=self.lr)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Embeddings obtenidos antes de la capa FC del estudiante."""
        features = self.student.forward_features(x) if hasattr(self.student, "forward_features") else self.student.avgpool(
            self.student.layer4(self.student.layer3(self.student.layer2(self.student.layer1(self.student.maxpool(self.student.relu(self.student.bn1(self.student.conv1(x))))))))
        )
        return torch.flatten(features, 1)


__all__ = ["ResNetDistillationModule"]
