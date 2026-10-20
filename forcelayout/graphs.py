"""Test graphs and random starting layouts.

Every generator returns a ``Graph`` with a unique, undirected edge list
(each edge stored once as ``(i, j)`` with ``i < j``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    name: str
    n: int
    edges: np.ndarray  # shape (m, 2), int

    @property
    def m(self) -> int:
        return len(self.edges)


def _make(name: str, n: int, edge_iter) -> Graph:
    uniq = sorted({(min(a, b), max(a, b)) for a, b in edge_iter if a != b})
    return Graph(name, n, np.array(uniq, dtype=int))


def grid(rows: int = 6, cols: int = 6) -> Graph:
    """rows x cols lattice. Planar, a perfect layout is the lattice itself."""
    idx = lambda r, c: r * cols + c
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return _make(f"grid{rows}x{cols}", rows * cols, edges)


def binary_tree(depth: int = 4) -> Graph:
    """Complete binary tree with ``depth`` levels below the root."""
    n = 2 ** (depth + 1) - 1
    edges = [(i, 2 * i + 1) for i in range(n) if 2 * i + 1 < n]
    edges += [(i, 2 * i + 2) for i in range(n) if 2 * i + 2 < n]
    return _make(f"tree{depth}", n, edges)


def cycle(n: int = 20) -> Graph:
    return _make(f"cycle{n}", n, [(i, (i + 1) % n) for i in range(n)])


def clusters(k: int = 4, size: int = 6) -> Graph:
    """k cliques joined in a ring by single bridge edges (a 'community graph',
    like a social network or an Obsidian vault with topic clusters)."""
    edges = []
    for c in range(k):
        base = c * size
        edges += [(base + i, base + j) for i in range(size) for j in range(i + 1, size)]
        nxt = ((c + 1) % k) * size
        edges.append((base + size - 1, nxt))
    return _make(f"clusters{k}x{size}", k * size, edges)


def _delaunay_edges(P: np.ndarray):
    from scipy.spatial import Delaunay

    edges = []
    for a, b, c in Delaunay(P).simplices:
        edges += [(a, b), (b, c), (a, c)]
    return edges


def delaunay_mesh(n: int = 30, seed: int = 0) -> Graph:
    """Delaunay triangulation of n random points (dense planar graph, ~3n edges).

    NOTE: not used in the benchmark.  A random triangulation cannot have all
    edges the same length, so with uniform rest lengths the spring energy is
    frustrated and its minimum is a *folded* drawing, even when the optimiser
    starts from the true planar drawing.  See README ("Design notes").
    """
    rng = np.random.default_rng(1000 + seed)
    return _make(f"mesh{n}_s{seed}", n, _delaunay_edges(rng.random((n, 2))))


def gabriel_network(n: int = 30, seed: int = 0) -> Graph:
    """Sparse random *planar* network: the Gabriel graph of n random points
    (a Delaunay edge is kept only if no other point lies inside the circle
    having that edge as diameter).  Connected, ~1.7n edges, irregular like a
    real network / mind-map, and a crossing-free drawing is known to exist, so
    'number of edge crossings' is a meaningful quality measure."""
    rng = np.random.default_rng(1000 + seed)
    P = rng.random((n, 2))
    keep = []
    for u, v in {(min(a, b), max(a, b)) for a, b in _delaunay_edges(P)}:
        mid, r2 = (P[u] + P[v]) / 2, np.sum((P[u] - P[v]) ** 2) / 4
        others = np.delete(np.arange(n), [u, v])
        if not np.any(np.sum((P[others] - mid) ** 2, axis=1) < r2):
            keep.append((u, v))
    return _make(f"net{n}_s{seed}", n, keep)


def random_layout(n: int, seed: int, spread: float | None = None) -> np.ndarray:
    """Tangled starting point: node positions uniform in a square.

    ``spread`` is the half-width; by default ``0.6*sqrt(n)`` so the initial
    density is comparable to the final layout (a pure 'knot' with lots of
    crossings, but not absurdly far from the answer).
    """
    if spread is None:
        spread = 0.6 * np.sqrt(n)
    rng = np.random.default_rng(seed)
    return rng.uniform(-spread, spread, size=2 * n)  # flat [x0,y0,x1,y1,...]


REGISTRY = {
    "grid": lambda: grid(6, 6),
    "tree": lambda: binary_tree(4),
    "cycle": lambda: cycle(24),
    "clusters": lambda: clusters(4, 6),
    "net": lambda: gabriel_network(30, 0),
}

# graphs that admit a crossing-free drawing (crossings are meaningful for these)
PLANAR = {"grid", "tree", "cycle", "net"}
