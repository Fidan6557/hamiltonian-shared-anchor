"""Cholesky backend (scipy/LAPACK or torch float64) with one jitter-escalation policy for all plans."""
from __future__ import annotations

import os

import numpy as np
import scipy.linalg as sl

JITTER_LEVELS = (0.0, 1e-10, 1e-8, 1e-6)   # multiples of the mean Gram diagonal

_BACKEND = None


def backend() -> str:
    global _BACKEND
    if _BACKEND is None:
        want = os.environ.get("HAMSCHED_LINALG", "auto").lower()
        if want in ("auto", "torch"):
            try:
                import torch  # noqa: F401
                _BACKEND = "torch"
            except Exception:
                if want == "torch":
                    raise
                _BACKEND = "scipy"
        else:
            _BACKEND = "scipy"
    return _BACKEND


def set_backend(name: str):
    global _BACKEND
    _BACKEND = name


def set_threads(n: int):
    if backend() == "torch":
        import torch
        torch.set_num_threads(max(1, int(n)))


class Factor:
    def __init__(self, Lc, kind: str, jitter: float):
        self.Lc = Lc
        self.kind = kind
        self.jitter = jitter

    def solve(self, b: np.ndarray) -> np.ndarray:
        if self.kind == "torch":
            import torch
            bt = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float64))
            squeeze = bt.ndim == 1
            if squeeze:
                bt = bt[:, None]
            x = torch.cholesky_solve(bt, self.Lc)
            x = x.numpy()
            return x[:, 0] if squeeze else x
        return sl.cho_solve(self.Lc, b, check_finite=False)


def _try_cholesky(A: np.ndarray):
    if backend() == "torch":
        import torch
        At = torch.from_numpy(A)
        Lc, info = torch.linalg.cholesky_ex(At)
        if int(info) != 0 or not bool(torch.isfinite(Lc.diagonal()).all()):
            return None
        return Factor(Lc, "torch", 0.0)
    try:
        c = sl.cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(np.diag(c[0]))):
        return None
    return Factor(c, "scipy", 0.0)


def factor_regularized(G: np.ndarray, add_reg, dbar: float):
    """Factor A = G + R (+ jitter*dbar*I).  add_reg(A) adds the regularizer in place.

    Returns (factor or None, status) with status 0 = ok, 1 = jitter needed, 2 = failed.
    """
    for lvl in JITTER_LEVELS:
        A = G.copy()
        add_reg(A)
        if lvl > 0:
            A[np.diag_indices_from(A)] += lvl * dbar
        f = _try_cholesky(A)
        del A
        if f is not None:
            f.jitter = lvl
            return f, (0 if lvl == 0 else 1)
    return None, 2
