import numpy as np
import pytest

from forcelayout.metrics import (
    count_crossings, crossing_edge_mask, edge_length_cv, min_node_distance, tail_rate,
)


def test_x_shape_has_one_crossing():
    z = np.array([0, 0, 1, 1, 0, 1, 1, 0], float)  # diagonals of a unit square
    edges = np.array([[0, 1], [2, 3]])
    assert count_crossings(z, edges) == 1
    assert crossing_edge_mask(z, edges).all()


def test_square_has_no_crossings():
    z = np.array([0, 0, 1, 0, 1, 1, 0, 1], float)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    assert count_crossings(z, edges) == 0


def test_edges_sharing_endpoint_not_counted():
    z = np.array([0, 0, 1, 0, 1, 1], float)
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    assert count_crossings(z, edges) == 0


def test_edge_length_cv_zero_for_equal_edges():
    z = np.array([0, 0, 1, 0, 2, 0], float)
    assert edge_length_cv(z, np.array([[0, 1], [1, 2]])) == pytest.approx(0.0)


def test_min_node_distance():
    z = np.array([0, 0, 3, 4, 10, 10], float)
    assert min_node_distance(z) == pytest.approx(5.0)


def test_tail_rate_recovers_geometric_ratio():
    g = 0.5 ** np.arange(60)
    assert tail_rate(g) == pytest.approx(0.5, rel=1e-9)
