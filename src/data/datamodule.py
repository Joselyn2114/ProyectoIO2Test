"""LightningDataModule para MVTec AD.

Cumple el requisito del enunciado de usar un LightningDataModule que cargue
solo imágenes normales para el entrenamiento, evitando incluir anomalías en la
fase de ajuste. Incluye transformaciones configurables vía Hydra y separa los
splits train/val/test.

Ahora soporta múltiples categorías (lista de clases) para entrenar con imágenes
normales de 10 clases distintas, como pide el enunciado.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import transforms


class MVTecDataset(Dataset):
    """Dataset ligero para MVTec AD filtrando ejemplos normales.

    - only_normal=True  -> solo imágenes 'good', label binario = 0
    - only_normal=False -> good = 0, cualquier defecto = 1
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

        # ImageFolder con todas las subcarpetas (good + defectos)
        self.dataset = datasets.ImageFolder(split_dir, transform=self.transform)

        # índice real de la clase "good"
        self.good_idx = self.dataset.class_to_idx["good"]

        # índices que vamos a usar según only_normal
        if only_normal:
            # solo ejemplos normales (good)
            self.indices = [
                i for i, (_, label) in enumerate(self.dataset.samples)
                if label == self.good_idx
            ]
        else:
            # todos los ejemplos (good + defectos)
            self.indices = list(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        """Devuelve img y label binario: 0 = good, 1 = defecto."""
        real_idx = self.indices[index]
        img, y_raw = self.dataset[real_idx]

        # Binarizar etiqueta: good -> 0, resto -> 1
        y_bin = 0 if y_raw == self.good_idx else 1

        return img, y_bin


@dataclass
class MVTecConfig:
    root: str
    # Ahora soportamos múltiples categorías (lista de strings)
    categories: List[str]
    num_workers: int
    batch_size: int


class MVTecDataModule(pl.LightningDataModule):
    """DataModule de PyTorch Lightning para el dataset MVTec AD.

    - Carga imágenes normales en entrenamiento (solo la carpeta "good")
      de TODAS las categorías indicadas.
    - Usa transforms definidas en Hydra para entrenamiento y validación/test.
    - Entrega DataLoaders listos para el Trainer.
    """

    def __init__(
        self,
        root: str,
        categories: List[str],
        batch_size: int = 8,
        num_workers: int = 2,
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.categories = categories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms or train_transforms

    def setup(self, stage: Optional[str] = None) -> None:
        """Crea los datasets para train/val/test usando TODAS las categorías.

        - Train: solo imágenes normales (good) de todas las clases.
        - Val/Test: normales + anomalías de todas las clases.
        """

        # -------- TRAIN: solo normales (good) de todas las categorías --------
        train_datasets = []
        for cat in self.categories:
            ds = MVTecDataset(
                root=self.root,
                split="train",
                category=cat,
                transform=self.train_transforms,
                only_normal=True,  # ⬅ solo good
            )
            train_datasets.append(ds)

        # Un solo dataset que concatena todas las clases normales
        self.mvtec_train = ConcatDataset(train_datasets)

        # -------- VAL / TEST: normales + anomalías de todas las categorías ----
        val_datasets = []
        for cat in self.categories:
            ds = MVTecDataset(
                root=self.root,
                split="test",
                category=cat,
                transform=self.test_transforms,
                only_normal=False,  # ⬅ good + defectos
            )
            val_datasets.append(ds)

        self.mvtec_val = ConcatDataset(val_datasets)
        self.mvtec_test = self.mvtec_val  # usamos el mismo split para test

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
