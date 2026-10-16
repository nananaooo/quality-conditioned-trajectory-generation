from __future__ import annotations

import numpy as np


def wasserstein_1d(a: np.ndarray, b: np.ndarray, n_quantiles: int = 200) -> float:
    """1-D Wasserstein-1 distance between two empirical samples via quantile matching."""
    qs = np.linspace(0.0, 1.0, n_quantiles)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def per_joint_wasserstein(gen_chunks: np.ndarray, real_chunks: np.ndarray) -> float:
    """Mean per-joint marginal Wasserstein-1 over action channels.

    gen_chunks / real_chunks: [N, k, 6] in raw units. Pools all timesteps per joint and
    measures how closely the generated marginal value distribution matches the real one.
    Lower = better marginal fidelity.
    """
    d = gen_chunks.shape[-1]
    return float(np.mean([wasserstein_1d(gen_chunks[..., j].ravel(), real_chunks[..., j].ravel()) for j in range(d)]))


def _pairwise_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix [Nx, Ny] via the matmul identity (memory-friendly)."""
    x2 = (x * x).sum(1)[:, None]
    y2 = (y * y).sum(1)[None, :]
    d2 = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)
    return np.sqrt(d2)


def _knn_radius(x: np.ndarray, k: int) -> np.ndarray:
    """Distance from each point to its k-th nearest neighbor within the same set."""
    d = _pairwise_dist(x, x)
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, k - 1]


def precision_recall(real: np.ndarray, gen: np.ndarray, k: int = 3) -> tuple[float, float]:
    """Improved precision/recall (Kynkaanniemi et al.) on flattened chunk vectors.

    precision = fraction of GENERATED chunks lying inside the real-data manifold
      (within the k-NN radius of some real chunk) -> realism / on-manifold.
    recall = fraction of REAL chunks covered by the generated manifold -> coverage / diversity.
    real / gen: [N, D] (flatten chunks to D = k*6 before calling).
    """
    r_radius = _knn_radius(real, k)
    g_radius = _knn_radius(gen, k)
    d = _pairwise_dist(gen, real)  # [Ngen, Nreal]
    precision = float(np.mean((d <= r_radius[None, :]).any(axis=1)))
    recall = float(np.mean((d <= g_radius[:, None]).any(axis=0)))
    return precision, recall
