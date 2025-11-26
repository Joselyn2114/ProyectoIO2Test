"""Clustering de outliers con DBSCAN sobre embeddings reducidos.

Cumple la sección del enunciado que solicita identificar puntos de ruido como
posibles anomalías.
"""
from __future__ import annotations

from typing import Dict

import torch
from sklearn.cluster import DBSCAN


def apply_dbscan(embeddings: torch.Tensor, eps: float = 0.5, min_samples: int = 5) -> Dict[str, torch.Tensor]:
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(embeddings.cpu().numpy())
    labels_tensor = torch.from_numpy(labels)
    noise_mask = labels_tensor == -1
    return {"labels": labels_tensor, "noise_mask": noise_mask}


__all__ = ["apply_dbscan"]
