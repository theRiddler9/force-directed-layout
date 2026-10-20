import numpy as np
import pytest

from forcelayout.energy import LayoutEnergy
from forcelayout.graphs import grid, delaunay_mesh, random_layout


def fd_gradient(f, z, h=1e-6):
    g = np.zeros_like(z)
    for i in range(len(z)):
        e = np.zeros_like(z)
        e[i] = h
        g[i] = (f(z + e) - f(z - e)) / (2 * h)
    return g


@pytest.fixture(params=[grid(3, 4), delaunay_mesh(12, 1)])
def setup(request):
    G = request.param
    return LayoutEnergy(G.n, G.edges), random_layout(G.n, seed=5)


def test_gradient_matches_finite_differences(setup):
    E, z = setup
    assert np.allclose(E.gradient(z), fd_gradient(E, z), atol=1e-6, rtol=1e-5)


def test_energy_equals_half_sum_of_squared_residuals(setup):
    E, z = setup
    r = E.residuals(z)
    assert E(z) == pytest.approx(0.5 * r @ r, rel=1e-12)


def test_gradient_equals_Jt_r(setup):
    E, z = setup
    assert np.allclose(E.gradient(z), E.jacobian(z).T @ E.residuals(z), atol=1e-10)


def test_jacobian_matches_finite_differences(setup):
    E, z = setup
    J = E.jacobian(z)
    h = 1e-6
    Jfd = np.zeros_like(J)
    for i in range(len(z)):
        e = np.zeros_like(z)
        e[i] = h
        Jfd[:, i] = (E.residuals(z + e) - E.residuals(z - e)) / (2 * h)
    assert np.allclose(J, Jfd, atol=1e-6, rtol=1e-5)


def test_counters(setup):
    E, z = setup
    E.reset_counters()
    E(z), E(z), E.gradient(z), E.jacobian(z)
    assert (E.n_f, E.n_g, E.n_J) == (2, 1, 1)
