"""Total system energy of a spring-electrical graph layout.

Variables: z = [x_0, y_0, x_1, y_1, ..., x_{n-1}, y_{n-1}]   (dimension 2n)

    E(z) = sum_{(i,j) in edges}  (k/2) (d_ij - L)^2                  springs, rest length L
         + sum_{i<j}   (c L^2 / 2) ln(1 + R^2 / (d_ij^2 + eps^2))    repulsion, all pairs
         + (g/2) |z|^2                                               weak gravity

with d_ij = |p_i - p_j|.

* Repulsion is a smooth form of the classic Fruchterman-Reingold law (force
  ~ 1/d, i.e. potential ~ -ln d): for d << R it behaves like -(c L^2) ln d, and
  it decays beyond the range R so the energy stays bounded below.  The long
  range matters: with a short-range 1/d^2 repulsion, folds in the drawing are
  cheap and the energy minimum is *not* a planar drawing even for planar graphs.
* eps softens the repulsion so E is smooth (C^infinity) even if two nodes coincide.
* Gravity removes the translation null-space and keeps pieces from drifting apart.

Key structural fact:  E is a **sum of squares**, E = 1/2 r(z)^T r(z), with

    spring residual     r_ij = sqrt(k) (d_ij - L)
    repulsion residual  r_ij = sqrt(c) L sqrt(ln(1 + R^2 / (d_ij^2 + eps^2)))
    gravity residual    r_i  = sqrt(g) z_i

so the gradient is  grad E = J^T r  and the Gauss-Newton Hessian  J^T J  is
available -- which is exactly what Levenberg-Marquardt needs.
"""
from __future__ import annotations

import numpy as np


class LayoutEnergy:
    def __init__(self, n, edges, L=1.0, k=1.0, c=0.3, R=8.0, eps=0.15, g=0.02):
        self.n = int(n)
        self.edges = np.asarray(edges, dtype=int).reshape(-1, 2)
        self.ea, self.eb = self.edges[:, 0], self.edges[:, 1]
        self.pi, self.pj = np.triu_indices(self.n, 1)  # all node pairs
        self.L, self.k, self.c, self.R, self.eps, self.g = L, k, c, R, eps, g
        self.reset_counters()

    # ---- bookkeeping: how many times each quantity was evaluated -------------
    def reset_counters(self):
        self.n_f = 0
        self.n_g = 0
        self.n_J = 0

    @property
    def dim(self) -> int:
        return 2 * self.n

    def _pos(self, z):
        return z.reshape(self.n, 2)

    # ---- E(z) -----------------------------------------------------------------
    def __call__(self, z) -> float:
        self.n_f += 1
        X = self._pos(z)
        d = np.linalg.norm(X[self.ea] - X[self.eb], axis=1)
        E_spring = 0.5 * self.k * np.sum((d - self.L) ** 2)
        D = X[self.pi] - X[self.pj]
        s = np.einsum("ij,ij->i", D, D) + self.eps**2
        E_rep = 0.5 * self.c * self.L**2 * np.sum(np.log1p(self.R**2 / s))
        E_grav = 0.5 * self.g * float(z @ z)
        return float(E_spring + E_rep + E_grav)

    # ---- grad E(z) (analytic) ---------------------------------------------------
    def gradient(self, z) -> np.ndarray:
        self.n_g += 1
        X = self._pos(z)
        G = np.zeros_like(X)

        Da = X[self.ea] - X[self.eb]
        d = np.maximum(np.linalg.norm(Da, axis=1), 1e-12)
        F = (self.k * (d - self.L) / d)[:, None] * Da
        np.add.at(G, self.ea, F)
        np.add.at(G, self.eb, -F)

        D = X[self.pi] - X[self.pj]
        s = np.einsum("ij,ij->i", D, D) + self.eps**2
        F = (-self.c * self.L**2 * self.R**2 / (s * (s + self.R**2)))[:, None] * D
        np.add.at(G, self.pi, F)
        np.add.at(G, self.pj, -F)

        return G.ravel() + self.g * z

    # ---- residual vector r(z) and Jacobian J(z) for least-squares methods -------
    def residuals(self, z) -> np.ndarray:
        X = self._pos(z)
        d = np.linalg.norm(X[self.ea] - X[self.eb], axis=1)
        D = X[self.pi] - X[self.pj]
        s = np.einsum("ij,ij->i", D, D) + self.eps**2
        return np.concatenate(
            [
                np.sqrt(self.k) * (d - self.L),
                np.sqrt(self.c) * self.L * np.sqrt(np.log1p(self.R**2 / s)),
                np.sqrt(self.g) * z,
            ]
        )

    def jacobian(self, z) -> np.ndarray:
        self.n_J += 1
        X = self._pos(z)
        m, P, N = len(self.ea), len(self.pi), 2 * self.n
        J = np.zeros((m + P + N, N))

        # springs
        Da = X[self.ea] - X[self.eb]
        d = np.maximum(np.linalg.norm(Da, axis=1), 1e-12)
        u = np.sqrt(self.k) * Da / d[:, None]
        r = np.arange(m)
        J[r, 2 * self.ea] = u[:, 0]
        J[r, 2 * self.ea + 1] = u[:, 1]
        J[r, 2 * self.eb] = -u[:, 0]
        J[r, 2 * self.eb + 1] = -u[:, 1]

        # repulsion
        D = X[self.pi] - X[self.pj]
        s = np.einsum("ij,ij->i", D, D) + self.eps**2
        w = -np.sqrt(self.c) * self.L * self.R**2 / (np.sqrt(np.log1p(self.R**2 / s)) * s * (s + self.R**2))
        r = m + np.arange(P)
        J[r, 2 * self.pi] = w * D[:, 0]
        J[r, 2 * self.pi + 1] = w * D[:, 1]
        J[r, 2 * self.pj] = -w * D[:, 0]
        J[r, 2 * self.pj + 1] = -w * D[:, 1]

        # gravity
        J[m + P + np.arange(N), np.arange(N)] = np.sqrt(self.g)
        return J
