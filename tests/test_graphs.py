import numpy as np

from forcelayout.graphs import REGISTRY, PLANAR, gabriel_network, grid, random_layout
from forcelayout.metrics import count_crossings


def _connected(G):
    adj = {i: set() for i in range(G.n)}
    for a, b in G.edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {0}, [0]
    while stack:
        for w in adj[stack.pop()] - seen:
            seen.add(w)
            stack.append(w)
    return len(seen) == G.n


def test_edges_unique_sorted_and_no_self_loops():
    for make in REGISTRY.values():
        G = make()
        assert np.all(G.edges[:, 0] < G.edges[:, 1])
        assert len({tuple(e) for e in G.edges}) == G.m


def test_all_benchmark_graphs_connected():
    assert all(_connected(make()) for make in REGISTRY.values())


def test_grid_edge_count():
    G = grid(4, 5)
    assert G.m == 4 * 4 + 3 * 5


def test_gabriel_network_is_planar_for_its_true_coordinates():
    n, seed = 30, 0
    G = gabriel_network(n, seed)
    P = np.random.default_rng(1000 + seed).random((n, 2))
    assert count_crossings(P.ravel(), G.edges) == 0


def test_random_layout_reproducible_and_shape():
    a, b = random_layout(10, 3), random_layout(10, 3)
    assert a.shape == (20,) and np.array_equal(a, b)
    assert not np.array_equal(a, random_layout(10, 4))


def test_planar_set_names_exist():
    assert PLANAR <= set(REGISTRY)
