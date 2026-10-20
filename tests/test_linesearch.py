import numpy as np
import pytest

from forcelayout.linesearch import bracket_minimum, golden_section, line_search


def test_golden_section_finds_parabola_minimum():
    phi = lambda a: (a - 1.7) ** 2 + 3
    alpha, f = golden_section(phi, 0.0, 5.0, rel_tol=1e-6)
    assert alpha == pytest.approx(1.7, abs=1e-4)
    assert f == pytest.approx(3.0, abs=1e-8)


def test_bracket_expands_until_function_rises():
    phi = lambda a: (a - 37.0) ** 2
    a, b = bracket_minimum(phi, phi(0.0), step=0.5)
    assert a < 37.0 < b


def test_bracket_when_first_step_overshoots():
    phi = lambda a: (a - 0.1) ** 2
    a, b = bracket_minimum(phi, phi(0.0), step=1.0)
    assert a <= 0.1 <= b
    assert b < 1.0  # step was shrunk


def test_line_search_decreases_on_nonunimodal_phi():
    # wiggly function: plain golden section on [0, 1] could land on a worse point
    f = lambda z: float(z[0] ** 2 + 0.3 * np.sin(40 * z[0]) ** 2)
    z = np.array([0.9])
    d = np.array([-1.0])
    alpha, fn = line_search(f, z, d, f(z), alpha_init=1.0)
    assert fn < f(z)


def test_line_search_on_quadratic():
    A = np.diag([1.0, 10.0])
    f = lambda z: 0.5 * z @ A @ z
    z = np.array([2.0, 1.0])
    d = -(A @ z)
    alpha, fn = line_search(f, z, d, f(z), alpha_init=0.01, rel_tol=1e-6)
    exact = (d @ d) / (d @ A @ d)  # exact minimiser of a quadratic along d
    assert alpha == pytest.approx(exact, rel=1e-3)
