"""Error metrics on clean held-out states."""
from __future__ import annotations

import numpy as np


def rel_err(pred: np.ndarray, true: np.ndarray) -> float:
    """Relative vector-field error sqrt(sum ||f_hat - f||^2 / sum ||f||^2) (canonical coordinates)."""
    if not np.all(np.isfinite(pred)):
        return float("nan")
    return float(np.sqrt(np.sum((pred - true) ** 2) / np.sum(true**2)))


def rel_err_path(preds: np.ndarray, true: np.ndarray) -> np.ndarray:
    """preds (K, P, 4) -> (K,) relative errors (NaN where a prediction is non-finite)."""
    num = np.sum((preds - true[None]) ** 2, axis=(1, 2))
    out = np.sqrt(num / np.sum(true**2))
    bad = ~np.all(np.isfinite(preds), axis=(1, 2))
    out[bad] = np.nan
    return out


def energy_strata(E: np.ndarray, n_bins: int = 3) -> np.ndarray:
    qs = np.quantile(E, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(qs, E)


def stratified_rel_err(pred: np.ndarray, true: np.ndarray, strata: np.ndarray) -> list:
    out = []
    for k in np.unique(strata):
        m = strata == k
        out.append(rel_err(pred[m], true[m]))
    return out
