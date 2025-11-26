"""Utilidades para cálculo de embeddings y distancia de Mahalanobis.

Responde a la sección de evaluación del enunciado: se obtienen embeddings de los
modelos, se calcula la media y covarianza de los datos normales y se aplica la
distancia de Mahalanobis para detectar anomalías.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def compute_gaussian_stats(embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula media y covarianza regularizada de un conjunto de embeddings."""
    mu = embeddings.mean(dim=0)
    centered = embeddings - mu
    cov = centered.t() @ centered / (embeddings.shape[0] - 1)
    # Añadimos una pequeña regularización para estabilidad numérica
    cov += 1e-6 * torch.eye(cov.shape[0], device=cov.device)
    return mu, cov


def mahalanobis_distance(embeddings: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Computa la distancia de Mahalanobis para cada embedding."""
    centered = embeddings - mu
    inv_cov = torch.linalg.inv(cov)
    left = centered @ inv_cov
    dists = torch.sqrt(torch.sum(left * centered, dim=1))
    return dists


def score_samples(
    model, dataloader, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Extrae embeddings y calcula estadísticos de referencia.

    El modelo debe exponer un método `extract_embeddings` según se define en los
    LightningModule de los modelos A, B y C.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            emb = model.extract_embeddings(x)
            all_embeddings.append(emb.cpu())
            all_labels.append(y)
    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    mu, cov = compute_gaussian_stats(embeddings)
    return {"embeddings": embeddings, "labels": labels, "mu": mu, "cov": cov}


__all__ = ["compute_gaussian_stats", "mahalanobis_distance", "score_samples"]
