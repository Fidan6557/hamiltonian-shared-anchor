"""Stationary kernels and their gradient (cross-Hessian) blocks for Hamiltonian kernel ridge regression.

For k(x, y) = phi(rho), rho^2 = sum_j ((x_j - y_j)/L_j)^2, define
    phi1(rho) = k'(rho)/rho,   phi2(rho) = (k''(rho) - k'(rho)/rho)/rho^2.
Then  d/dx_a d/dy_b k(x, y) = -phi1 delta_ab / L_a^2 - phi2 u_a u_b,  with u = (x - y)/L^2.
The model is grad h(x) = sum_i K2(x, x_i) c_i (representer theorem for gradient observations).
"""
from __future__ import annotations

import numpy as np

SQRT5 = np.sqrt(5.0)


class SEKernel:
    name = "se"

    @staticmethod
    def phis(rho2):
        k = np.exp(-0.5 * rho2)
        return k, -k, k


class Matern52Kernel:
    name = "matern52"

    @staticmethod
    def phis(rho2):
        rho = np.sqrt(rho2)
        e = np.exp(-SQRT5 * rho)
        k = (1.0 + SQRT5 * rho + (5.0 / 3.0) * rho2) * e
        phi1 = -(5.0 / 3.0) * (1.0 + SQRT5 * rho) * e
        phi2 = (25.0 / 3.0) * e
        return k, phi1, phi2


def make_kernel(name: str):
    if name == "se":
        return SEKernel()
    if name in ("matern52", "matern"):
        return Matern52Kernel()
    raise ValueError(name)


def _use_torch() -> bool:
    from .linalg import backend
    return backend() == "torch"


def _phis_torch(kernel, rho2):
    import torch
    if kernel.name == "se":
        k = torch.exp(-0.5 * rho2)
        return k, -k, k
    rho = torch.sqrt(rho2)
    e = torch.exp(-SQRT5 * rho)
    k = (1.0 + SQRT5 * rho + (5.0 / 3.0) * rho2) * e
    return k, -(5.0 / 3.0) * (1.0 + SQRT5 * rho) * e, (25.0 / 3.0) * e


def _predict_grad_torch(Xe, Xtr, L, kernel, C, chunk):
    import torch
    Lt = torch.from_numpy(np.asarray(L, float))
    invL2 = 1.0 / Lt**2
    Ct = torch.from_numpy(np.ascontiguousarray(C))
    CL = Ct * invL2[None, :, None]
    Xtr_s = torch.from_numpy(np.ascontiguousarray(Xtr / L))
    Xe_s = torch.from_numpy(np.ascontiguousarray(Xe / L))
    out = torch.empty((len(Xe), Xe.shape[1], C.shape[2]), dtype=torch.float64)
    for i0 in range(0, len(Xe), chunk):
        i1 = min(len(Xe), i0 + chunk)
        D = Xe_s[i0:i1, None, :] - Xtr_s[None, :, :]
        rho2 = (D * D).sum(-1)
        _, p1, p2 = _phis_torch(kernel, rho2)
        U = D / Lt
        t1 = -torch.einsum("pn,nak->pak", p1, CL)
        s = torch.einsum("pna,nak->pnk", U, Ct)
        t2 = -torch.einsum("pnk,pna->pak", p2[..., None] * s, U)
        out[i0:i1] = t1 + t2
    return out.numpy()


def gram_grad(X: np.ndarray, L: np.ndarray, kernel, chunk: int = 96) -> np.ndarray:
    """Gradient Gram matrix of shape (4n, 4n), row index i*4 + a."""
    X = np.asarray(X, float)
    n, d = X.shape
    L = np.asarray(L, float)
    G = np.empty((n, d, n, d))
    Xs = X / L
    invL2 = 1.0 / L**2
    ar = np.arange(d)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        D = Xs[i0:i1, None, :] - Xs[None, :, :]
        rho2 = np.einsum("ijk,ijk->ij", D, D)
        _, p1, p2 = kernel.phis(rho2)
        U = D / L
        blk = -p2[..., None, None] * U[..., :, None] * U[..., None, :]
        blk[..., ar, ar] += (-p1)[..., None] * invL2
        G[i0:i1] = blk.transpose(0, 2, 1, 3)
    return G.reshape(n * d, n * d)


def predict_grad(Xe: np.ndarray, Xtr: np.ndarray, L: np.ndarray, kernel, C: np.ndarray, chunk: int = 256) -> np.ndarray:
    """grad h at Xe for coefficients C of shape (n, 4) or (n, 4, K). Matrix-free, chunked."""
    Xe = np.atleast_2d(np.asarray(Xe, float))
    squeeze = C.ndim == 2
    if squeeze:
        C = C[..., None]
    if _use_torch():
        out = _predict_grad_torch(Xe, np.asarray(Xtr, float), np.asarray(L, float), kernel,
                                  np.asarray(C, float), chunk)
        return out[..., 0] if squeeze else out
    L = np.asarray(L, float)
    invL2 = 1.0 / L**2
    CL = C * invL2[None, :, None]
    out = np.empty((len(Xe), Xe.shape[1], C.shape[2]))
    Xtr_s = Xtr / L
    for i0 in range(0, len(Xe), chunk):
        i1 = min(len(Xe), i0 + chunk)
        D = Xe[i0:i1, None, :] / L - Xtr_s[None, :, :]
        rho2 = np.einsum("ijk,ijk->ij", D, D)
        _, p1, p2 = kernel.phis(rho2)
        U = D / L
        t1 = -np.einsum("pn,nak->pak", p1, CL)
        s = np.einsum("pna,nak->pnk", U, C)
        t2 = -np.einsum("pn,pnk,pna->pak", p2, s, U)
        out[i0:i1] = t1 + t2
    return out[..., 0] if squeeze else out


def gram_scalar(X: np.ndarray, L: np.ndarray, kernel) -> np.ndarray:
    Xs = np.asarray(X, float) / L
    D2 = np.sum(Xs**2, 1)[:, None] + np.sum(Xs**2, 1)[None, :] - 2 * Xs @ Xs.T
    k, _, _ = kernel.phis(np.maximum(D2, 0.0))
    return k


def predict_scalar(Xe: np.ndarray, Xtr: np.ndarray, L: np.ndarray, kernel, B: np.ndarray, chunk: int = 1024) -> np.ndarray:
    Xs_tr = np.asarray(Xtr, float) / L
    out = []
    for i0 in range(0, len(Xe), chunk):
        Xs = Xe[i0:i0 + chunk] / L
        D2 = np.sum(Xs**2, 1)[:, None] + np.sum(Xs_tr**2, 1)[None, :] - 2 * Xs @ Xs_tr.T
        k, _, _ = kernel.phis(np.maximum(D2, 0.0))
        out.append(np.tensordot(k, B, axes=(1, 0)))
    return np.concatenate(out)
