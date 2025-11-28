"""Clustering de outliers con DBSCAN sobre embeddings reducidos.

Cumple la sección del enunciado que solicita identificar puntos de ruido como
posibles anomalías.
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import torch
from sklearn.cluster import DBSCAN


def _to_numpy(x):
    """Convierte tensor → numpy y deja numpy igual."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x  # ya es numpy array


def apply_dbscan(embeddings, eps: float = 0.5, min_samples: int = 5) -> Dict[str, torch.Tensor]:
    """
    Aplica DBSCAN sobre embeddings reducidos (por ejemplo PCA o t-SNE),
    permitiendo que `embeddings` sea torch.Tensor o numpy.ndarray.

    Devuelve:
      - labels: tensor con etiquetas de cluster (-1 = ruido)
      - noise_mask: tensor booleano indicando cuáles son outliers
      - n_clusters: número de clusters encontrados (sin contar -1)
    """
    # 1) Aseguramos que trabajamos en numpy para DBSCAN
    emb_np = _to_numpy(embeddings)

    # 2) Ejecutamos DBSCAN en el espacio reducido
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_np = dbscan.fit_predict(emb_np)  # numpy array de shape [N]

    # 3) Convertimos a tensor para mantener consistencia con el resto del código
    labels = torch.from_numpy(labels_np)

    # 4) Máscaras y número de clusters (ignorando -1)
    noise_mask = labels == -1
    n_clusters = len(set(labels_np)) - (1 if -1 in labels_np else 0)

    return {
        "labels": labels,
        "noise_mask": noise_mask,
        "n_clusters": n_clusters,
    }


__all__ = ["apply_dbscan"]
