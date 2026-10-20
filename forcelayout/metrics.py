"""Layout quality and convergence metrics."""
from __future__ import annotations

import numpy as np


def _orient(p, q, r):
    return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])


def crossing_pairs(z, edges) -> np.ndarray:
    """Index pairs (e1, e2) of edges that properly cross.

    Edges sharing an endpoint are never counted.  Returns shape (k, 2).
    """
    X = np.asarray(z, float).reshape(-1, 2)
    E = np.asarray(edges, int)
    i, j = np.triu_indices(len(E), 1)
    a, b, c, d = E[i, 0], E[i, 1], E[j, 0], E[j, 1]
    keep = ~((a == c) | (a == d) | (b == c) | (b == d))
    i, j, a, b, c, d = i[keep], j[keep], a[keep], b[keep], c[keep], d[keep]
    o1 = _orient(X[a], X[b], X[c])
    o2 = _orient(X[a], X[b], X[d])
    o3 = _orient(X[c], X[d], X[a])
    o4 = _orient(X[c], X[d], X[b])
    hit = (o1 * o2 < 0) & (o3 * o4 < 0)
    return np.stack([i[hit], j[hit]], axis=1)


def count_crossings(z, edges) -> int:
    return int(len(crossing_pairs(z, edges)))


def crossing_edge_mask(z, edges) -> np.ndarray:
    """Boolean mask over edges: True if the edge takes part in any crossing."""
    mask = np.zeros(len(edges), dtype=bool)
    p = crossing_pairs(z, edges)
    mask[p.ravel()] = True
    return mask


def edge_length_cv(z, edges) -> float:
    """Coefficient of variation (std / mean) of edge lengths.  0 = all equal."""
    X = np.asarray(z, float).reshape(-1, 2)
    E = np.asarray(edges, int)
    d = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    return float(d.std() / d.mean())


def min_node_distance(z) -> float:
    X = np.asarray(z, float).reshape(-1, 2)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return float(D[np.triu_indices(len(X), 1)].min())


def tail_rate(gnorm, frac: float = 0.3, min_points: int = 5) -> float:
    """Empirical convergence ratio  q = geometric mean of ||g_{k+1}|| / ||g_k||
    over the last ``frac`` of the run.  q < 1 means (at least) linear
    convergence; the closer to 0, the faster.  NaN if too few points."""
    g = np.asarray(gnorm, float)
    g = g[g > 0]
    k = max(int(len(g) * frac), min_points)
    if len(g) < k + 1:
        return float("nan")
    t = g[-(k + 1):]
    return float(np.exp(np.mean(np.diff(np.log(t)))))
