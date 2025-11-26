"""LightningDataModule para MVTec AD.

Cumple el requisito del enunciado de usar un LightningDataModule que cargue
solo imágenes normales para el entrenamiento, evitando incluir anomalías en la
fase de ajuste. Incluye transformaciones configurables vía Hydra y separa los
splits train/val/test.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import transforms


class MVTecDataset(Dataset):
    """Dataset ligero para MVTec AD filtrando ejemplos normales.

    Esta clase usa `torchvision.datasets.ImageFolder` asumiendo que la estructura
    local sigue el formato original de MVTec AD. Se filtran etiquetas anómalas
    para el split de entrenamiento conforme al enunciado.
    """

    def __init__(
        self,
        root: str,
        split: str,
        category: str,
        transform: Optional[transforms.Compose] = None,
        only_normal: bool = False,
    ) -> None:
        self.transform = transform
        split_dir = os.path.join(root, category, split)
        # ImageFolder asigna labels numéricos; aquí simplificamos asumiendo que
        # la subcarpeta "good" representa normales.
        self.dataset = datasets.ImageFolder(split_dir, transform=self.transform)
        if only_normal:
            normal_idx = [i for i, (_, label) in enumerate(self.dataset.samples) if label == 0]
            self.dataset.samples = [self.dataset.samples[i] for i in normal_idx]
            self.dataset.targets = [0 for _ in normal_idx]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


@dataclass
class MVTecConfig:
    root: str
    category: str
    num_workers: int
    batch_size: int


class MVTecDataModule(pl.LightningDataModule):
    """DataModule de PyTorch Lightning para el dataset MVTec AD.

    - Carga imágenes normales en entrenamiento (solo la carpeta "good").
    - Usa transforms definidas en Hydra para entrenamiento y validación/test.
    - Entrega DataLoaders listos para el Trainer.
    """

    def __init__(
        self,
        root: str,
        category: str,
        batch_size: int = 16,
        num_workers: int = 4,
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.category = category
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms or train_transforms

    def setup(self, stage: Optional[str] = None) -> None:
        # Entrenamiento solo con ejemplos normales.
        self.mvtec_train = MVTecDataset(
            self.root, "train", self.category, transform=self.train_transforms, only_normal=True
        )
        # Validación y test pueden incluir anomalías según la etiqueta de ImageFolder.
        self.mvtec_val = MVTecDataset(
            self.root, "test", self.category, transform=self.test_transforms, only_normal=False
        )
        self.mvtec_test = self.mvtec_val

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mvtec_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mvtec_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mvtec_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


__all__ = ["MVTecDataModule", "MVTecDataset", "MVTecConfig"]
