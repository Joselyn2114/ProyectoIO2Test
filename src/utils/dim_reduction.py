"""Reducción de dimensionalidad con PCA y t-SNE para visualización.

Sigue lo indicado en el enunciado para facilitar la interpretación visual de los
embeddings antes de aplicar clustering.
"""
from __future__ import annotations

from typing import Tuple

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def apply_pca(embeddings: torch.Tensor, n_components: int = 50) -> torch.Tensor:
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings.cpu().numpy())
    return torch.from_numpy(reduced)


def apply_tsne(embeddings: torch.Tensor, n_components: int = 2, perplexity: float = 30.0) -> torch.Tensor:
    tsne = TSNE(n_components=n_components, perplexity=perplexity, init="random", learning_rate="auto")
    reduced = tsne.fit_transform(embeddings.cpu().numpy())
    return torch.from_numpy(reduced)


__all__ = ["apply_pca", "apply_tsne"]
