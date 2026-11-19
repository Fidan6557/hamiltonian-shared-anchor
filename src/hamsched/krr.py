"""Hamiltonian gradient kernel ridge regression (and an unstructured vector-KRR reference).

Weighted problem for gradient targets g_i = J^T v_i at anchors x_i:
    min_h  sum_units r_u^T N_u^{-1} r_u + lam ||h||_K^2,   r_i = grad h(x_i) - g_i,
solved by c = (G + lam N)^{-1} g with N the (normalized) noise-covariance structure of the
velocities:  'ols' = diagonal (1/h_i^2, i.e. the endpoint-defect loss of Hu et al., which is the
noise-whitened loss for independent pairs), 'gls' = full within-unit covariance (shared readings).
lam = eta * n * dbar (per-sample normalized ridge relative to the mean Gram diagonal), so a
common eta gives (approximately) the same smoothing operator for fits of different size.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .kernels import gram_grad, gram_scalar, predict_grad, predict_scalar
from .linalg import factor_regularized
from .systems import field_to_grad, grad_to_field


def noise_structure(ds, idx: np.ndarray, kind: str = "ols"):
    """Within-unit velocity-noise covariance (per coordinate) for the pairs `idx`, normalized.

    Returns ('diag', w) or ('block', (rows, cols, vals)) with mean diagonal 1.
    """
    p = ds.pairs
    idx = np.asarray(idx)
    P = len(idx)
    R = len(ds.readings.ids)
    rows = np.concatenate([np.arange(P), np.arange(P)])
    cols = np.concatenate([p.ia[idx], p.ib[idx]])
    vals = np.concatenate([-1.0 / p.h[idx], 1.0 / p.h[idx]])
    V = sp.csr_matrix((vals, (rows, cols)), shape=(P, R))
    C = (V @ V.T).tocoo()
    diag = np.asarray((V.multiply(V)).sum(axis=1)).ravel()
    scale = diag.mean()
    if kind == "ols":
        return ("diag", diag / scale)
    if kind == "gls":
        return ("block", (C.row, C.col, C.data / scale))
    raise ValueError(kind)


class GradFit:
    def __init__(self, X: np.ndarray, kernel, L: np.ndarray, struct):
        self.X = np.asarray(X, float)
        self.kernel = kernel
        self.L = np.asarray(L, float)
        self.struct = struct
        self.n = len(self.X)
        self.G = gram_grad(self.X, self.L, kernel)
        self.dbar = float(np.mean(np.diag(self.G)))

    def _add_reg(self, lam: float):
        kind, data = self.struct

        def add(A):
            if kind == "diag":
                w = np.repeat(data, 4)
                A[np.diag_indices_from(A)] += lam * w
            else:
                r, c, v = data
                for a in range(4):
                    A[4 * r + a, 4 * c + a] += lam * v
        return add

    def factor(self, eta: float):
        lam = eta * self.n * self.dbar
        return factor_regularized(self.G, self._add_reg(lam), self.dbar)

    def solve(self, fac, g: np.ndarray) -> np.ndarray:
        """g: (n, 4) or (n, 4, K) gradient targets -> coefficients of same shape."""
        if g.ndim == 2:
            return fac.solve(g.reshape(-1)).reshape(self.n, 4)
        K = g.shape[2]
        return fac.solve(g.reshape(self.n * 4, K)).reshape(self.n, 4, K)

    def field(self, coef: np.ndarray, Xe: np.ndarray) -> np.ndarray:
        return grad_to_field(np.moveaxis(predict_grad(Xe, self.X, self.L, self.kernel, coef), 1, -1)
                             if coef.ndim == 3 else predict_grad(Xe, self.X, self.L, self.kernel, coef))

    def release(self):
        self.G = None


class ScalarFit:
    """Unstructured reference: independent scalar KRR per velocity component (no Hamiltonian structure)."""

    def __init__(self, X, kernel, L, struct):
        self.X = np.asarray(X, float)
        self.kernel = kernel
        self.L = np.asarray(L, float)
        self.struct = struct
        self.n = len(self.X)
        self.G = gram_scalar(self.X, self.L, kernel)
        self.dbar = 1.0

    def factor(self, eta: float):
        lam = eta * self.n * self.dbar
        kind, data = self.struct

        def add(A):
            if kind == "diag":
                A[np.diag_indices_from(A)] += lam * data
            else:
                r, c, v = data
                A[r, c] += lam * v
        return factor_regularized(self.G, add, self.dbar)

    def solve_v(self, fac, v):
        return fac.solve(v)

    def field_from(self, beta, Xe):
        return predict_scalar(Xe, self.X, self.L, self.kernel, beta)


def targets(vel: np.ndarray) -> np.ndarray:
    return field_to_grad(vel)
