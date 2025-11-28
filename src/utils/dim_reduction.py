"""
Reducción de dimensionalidad con PCA y t-SNE para visualización.
Acepta tanto tensores como arrays de numpy.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _to_numpy(x):
    """
    Convierte tensores a numpy y deja arrays tal cual.
    - Si viene de embeddings_calculator → será numpy.
    - Si un estudiante usa torch directamente → también funciona.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x  # ya es numpy, no se toca


def apply_pca(embeddings, n_components=2):
    """
    Aplica PCA a los embeddings.
    Retorna un array numpy 2D [N, n_components].
    """
    emb_np = _to_numpy(embeddings)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(emb_np)
    return reduced


def apply_tsne(embeddings, n_components=2, perplexity=30):
    """
    Aplica t-SNE a los embeddings.
    Retorna un array numpy 2D [N, n_components].
    """
    emb_np = _to_numpy(embeddings)
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                init="random", learning_rate="auto")
    reduced = tsne.fit_transform(emb_np)
    return reduced


__all__ = ["apply_pca", "apply_tsne"]
