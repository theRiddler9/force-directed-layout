import numpy as np
import pytest

from forcelayout.energy import LayoutEnergy
from forcelayout.graphs import grid, random_layout
from forcelayout.optimizers import steepest_descent, conjugate_gradient, levenberg_marquardt


@pytest.fixture
def problem():
    G = grid(3, 3)
    return LayoutEnergy(G.n, G.edges), random_layout(G.n, seed=2)


@pytest.mark.parametrize("method", [steepest_descent, conjugate_gradient, levenberg_marquardt])
def test_energy_never_increases_and_gradient_small_at_end(problem, method):
    E, z0 = problem
    res = method(E, z0, gtol=1e-5)
    e = np.array(res.energy)
    assert np.all(np.diff(e) <= 1e-10), "energy must be monotonically non-increasing"
    assert res.energy[-1] < res.energy[0]
    assert res.converged, res.stop_reason
    assert np.linalg.norm(E.gradient(res.z)) < 1e-5


def test_all_methods_reach_same_local_minimum_on_convex_like_case():
    # tiny path graph: unique minimum up to rotation, so energies must agree
    from forcelayout.graphs import cycle

    G = cycle(5)
    E = LayoutEnergy(G.n, G.edges)
    z0 = random_layout(G.n, seed=1)
    finals = [m(E, z0, gtol=1e-6).energy[-1] for m in
              (steepest_descent, conjugate_gradient, levenberg_marquardt)]
    assert max(finals) - min(finals) < 1e-6


def test_snapshots_recorded_every_iteration(problem):
    E, z0 = problem
    res = conjugate_gradient(E, z0)
    assert len(res.snapshots) == len(res.energy) == res.n_iter + 1
    assert np.allclose(res.snapshots[0], z0)
    assert np.allclose(res.snapshots[-1], res.z)
