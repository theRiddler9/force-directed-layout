"""Gradient-based multivariable optimisers (Module 2).

    steepest_descent         d_k = -g_k, exact-ish line search
    conjugate_gradient       Fletcher-Reeves: beta_k = (g_{k+1}.g_{k+1}) / (g_k.g_k)
    levenberg_marquardt      (J^T J + lambda diag(J^T J)) delta = -J^T r

All three return a ``Result`` that stores every iterate so the layout can be
animated, plus energy / gradient-norm / evaluation counts for the benchmark.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .linesearch import line_search


@dataclass
class Result:
    name: str
    z: np.ndarray                      # final layout (flat)
    snapshots: list = field(default_factory=list)   # z at every iteration (incl. start)
    energy: list = field(default_factory=list)
    gnorm: list = field(default_factory=list)
    n_f: list = field(default_factory=list)         # cumulative E evaluations
    n_g: list = field(default_factory=list)         # cumulative gradient evaluations
    n_J: list = field(default_factory=list)         # cumulative Jacobian evaluations
    seconds: list = field(default_factory=list)     # cumulative wall time
    lam: list = field(default_factory=list)         # LM damping (empty otherwise)
    converged: bool = False
    stop_reason: str = ""

    @property
    def n_iter(self) -> int:
        return len(self.energy) - 1


def _record(res, energy_fn, z, f, gn, t0, lam=None):
    res.snapshots.append(z.copy())
    res.energy.append(f)
    res.gnorm.append(gn)
    res.n_f.append(energy_fn.n_f)
    res.n_g.append(energy_fn.n_g)
    res.n_J.append(energy_fn.n_J)
    res.seconds.append(time.perf_counter() - t0)
    if lam is not None:
        res.lam.append(lam)


def _first_step(d, L=1.0, move=0.2):
    """Initial trial step: move the largest-moving node by ~`move` rest lengths."""
    return move * L / max(np.max(np.abs(d)), 1e-12)


def steepest_descent(energy, z0, max_iter=3000, gtol=1e-4, ls_tol=1e-3):
    energy.reset_counters()
    t0 = time.perf_counter()
    z = z0.astype(float).copy()
    f, g = energy(z), energy.gradient(z)
    res = Result("Steepest descent", z)
    _record(res, energy, z, f, np.linalg.norm(g), t0)
    alpha = None
    for _ in range(max_iter):
        gn = np.linalg.norm(g)
        if gn < gtol:
            res.converged, res.stop_reason = True, "gradient tolerance"
            break
        d = -g
        a0 = _first_step(d, energy.L) if alpha is None else 2.0 * alpha
        alpha, f_new = line_search(energy, z, d, f, a0, ls_tol)
        if not f_new < f:
            res.stop_reason = "line search stalled"
            break
        z = z + alpha * d
        f, g = f_new, energy.gradient(z)
        _record(res, energy, z, f, np.linalg.norm(g), t0)
    else:
        res.stop_reason = "max iterations"
    res.z = z
    return res


def conjugate_gradient(energy, z0, max_iter=3000, gtol=1e-4, ls_tol=1e-3, restart=None):
    """Fletcher-Reeves nonlinear conjugate gradient.

    Restarts with d = -g every ``restart`` iterations (default: n = dimension)
    and whenever d fails to be a descent direction.
    """
    energy.reset_counters()
    t0 = time.perf_counter()
    z = z0.astype(float).copy()
    restart = restart or energy.dim
    f, g = energy(z), energy.gradient(z)
    d = -g
    res = Result("Conjugate gradient (FR)", z)
    _record(res, energy, z, f, np.linalg.norm(g), t0)
    alpha = None
    for k in range(max_iter):
        gn = np.linalg.norm(g)
        if gn < gtol:
            res.converged, res.stop_reason = True, "gradient tolerance"
            break
        if g @ d >= 0:  # not a descent direction -> restart
            d = -g
        a0 = _first_step(d, energy.L) if alpha is None else 2.0 * alpha
        alpha, f_new = line_search(energy, z, d, f, a0, ls_tol)
        if not f_new < f:
            if np.allclose(d, -g):
                res.stop_reason = "line search stalled"
                break
            d = -g  # retry once with steepest descent
            alpha = None
            continue
        z = z + alpha * d
        g_new = energy.gradient(z)
        beta = (g_new @ g_new) / (g @ g)                   # Fletcher-Reeves
        d = -g_new + beta * d
        if (k + 1) % restart == 0:
            d = -g_new
        f, g = f_new, g_new
        _record(res, energy, z, f, np.linalg.norm(g), t0)
    else:
        res.stop_reason = "max iterations"
    res.z = z
    return res


def levenberg_marquardt(energy, z0, max_iter=300, gtol=1e-4, lam0=1e-2, max_lam=1e14):
    """Levenberg-Marquardt for E = 1/2 ||r||^2  (Marquardt diagonal scaling,
    Nielsen's rule for updating lambda from the gain ratio)."""
    energy.reset_counters()
    t0 = time.perf_counter()
    z = z0.astype(float).copy()
    f = energy(z)
    J = energy.jacobian(z)
    r = energy.residuals(z)
    g = J.T @ r
    lam, nu = lam0, 2.0
    res = Result("Levenberg-Marquardt", z)
    _record(res, energy, z, f, np.linalg.norm(g), t0, lam)
    for _ in range(max_iter):
        gn = np.linalg.norm(g)
        if gn < gtol:
            res.converged, res.stop_reason = True, "gradient tolerance"
            break
        JTJ = J.T @ J
        diag = np.diag(JTJ).copy()
        accepted = False
        while lam < max_lam:
            delta = np.linalg.solve(JTJ + lam * np.diag(diag), -g)
            z_new = z + delta
            f_new = energy(z_new)
            predicted = -(g @ delta + 0.5 * delta @ JTJ @ delta)   # model decrease
            rho = (f - f_new) / predicted if predicted > 0 else -1.0
            if rho > 0:                                     # good step: accept
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                accepted = True
                break
            lam *= nu                                        # bad step: more damping
            nu *= 2.0
        if not accepted:
            res.stop_reason = "damping exhausted"
            break
        z, f = z_new, f_new
        J = energy.jacobian(z)
        r = energy.residuals(z)
        g = J.T @ r
        _record(res, energy, z, f, np.linalg.norm(g), t0, lam)
    else:
        res.stop_reason = "max iterations"
    res.z = z
    return res


METHODS = {
    "sd": steepest_descent,
    "cg": conjugate_gradient,
    "lm": levenberg_marquardt,
}
