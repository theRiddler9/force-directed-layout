"""One-dimensional minimisation along a search direction (Module 1 tools).

phi(alpha) = E(z + alpha * d)

1. Bounding phase: find a bracket [a, b] that must contain a minimiser,
   i.e. a middle point whose value is lower than phi(0) and phi(b).
     * if the first trial step is already too long (phi(step) >= phi(0)),
       halve it until phi drops below phi(0);
     * otherwise double it until phi starts rising again.
2. Golden-section search on that bracket (elimination method, one new
   function evaluation per iteration, interval shrinks by 0.618 each time).

The energy is not unimodal along every direction (it has repulsion bumps), so
``line_search`` returns the best point *evaluated*, which guarantees a decrease
whenever the bounding phase succeeds.
"""
from __future__ import annotations

import math

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0  # 0.618...


def bracket_minimum(phi, f0, step, grow=2.0, max_iter=60):
    """Return (a, b) with a minimiser of phi inside [a, b]  (a = 0 or a > 0)."""
    f_mid = phi(step)
    if f_mid >= f0:  # step too long: shrink until phi dips below phi(0)
        hi = step
        for _ in range(max_iter):
            step *= 0.5
            f_mid = phi(step)
            if f_mid < f0:
                return 0.0, hi
            hi = step
        return 0.0, hi  # no descent found (direction is flat / not descent)
    lo, mid = 0.0, step  # step ok: expand until phi rises
    for _ in range(max_iter):
        nxt = mid * grow
        f_nxt = phi(nxt)
        if f_nxt >= f_mid:
            return lo, nxt
        lo, mid, f_mid = mid, nxt, f_nxt
    return lo, mid * grow


def golden_section(phi, a, b, rel_tol=1e-3, max_iter=60):
    """Golden-section search for the minimiser of a unimodal phi on [a, b].

    Stops when the interval is shorter than ``rel_tol`` times its initial
    length.  Returns (alpha, phi(alpha)).
    """
    width0 = b - a
    c = b - GOLDEN * (b - a)
    d = a + GOLDEN * (b - a)
    fc, fd = phi(c), phi(d)
    for _ in range(max_iter):
        if (b - a) < rel_tol * width0:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = phi(d)
    return (c, fc) if fc < fd else (d, fd)


def line_search(energy, z, d, f0, alpha_init, rel_tol=1e-3):
    """Bounding phase + golden section.  Returns (alpha, f_new).

    Keeps track of the best point seen so the result is never worse than any
    evaluated trial (and strictly better than f0 if any trial was).
    """
    best = [0.0, f0]

    def phi(a):
        v = energy(z + a * d)
        if v < best[1]:
            best[0], best[1] = a, v
        return v

    a, b = bracket_minimum(phi, f0, alpha_init)
    golden_section(phi, a, b, rel_tol=rel_tol)
    return best[0], best[1]
