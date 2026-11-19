"""Local-averaging variance accounting for measurement plans (see docs/theory.md).

Model: i.i.d. reading noise e_r with per-coordinate variance sigma^2.  For a unit (the readings of
one plan "atom"), each pair p has velocity noise W_p = sum_r V_pr e_r.  In the local-averaging limit
(smoother bandwidth >> within-unit anchor separation) an estimator reduces, per unit, to a fixed
linear combination c^T e of the unit's reading noise.  Total variance per unit budget:
    factor = |c|^2 * (readings per unit) * h1^2           [units of sigma^2 / (h1^2 B)]
"""
from __future__ import annotations

import numpy as np

from .pipelines import richardson_weights


def unit_structure(plan: str, h1: float = 1.0, r: float = 2.0):
    """Return (n_readings, pairs) for one unit; pairs = list of (a, b, h, group)."""
    plan = plan.upper()
    h2 = r * h1
    if plan == "SI":
        return 2, [(0, 1, h1, 0)]
    if plan == "SI2":           # single interval at h2
        return 2, [(0, 1, h2, 0)]
    if plan == "IP":
        return 4, [(0, 1, h1, 0), (2, 3, h2, 1)]
    if plan == "RA":
        return 4, [(0, 2, h1, 0), (1, 3, h2, 1)]
    if plan == "SA":
        return 3, [(0, 1, h1, 0), (0, 2, h2, 1)]
    if plan == "MS3":
        if abs(r - 2.0) > 1e-12:
            raise ValueError("MS3 requires r = 2")
        return 3, [(0, 1, h1, 0), (1, 2, h1, 0), (0, 2, h2, 1)]
    if plan == "MS3_INDEP":     # Hu Sec. 6.3 protocol: stride-2 pair measured with independent noise
        return 5, [(0, 1, h1, 0), (1, 2, h1, 0), (3, 4, h2, 1)]
    raise ValueError(plan)


def _vmat(nr, pairs):
    V = np.zeros((len(pairs), nr))
    for k, (a, b, h, _) in enumerate(pairs):
        V[k, a] -= 1.0 / h
        V[k, b] += 1.0 / h
    return V


def variance_factor(plan: str, estimator: str, h1: float = 1.0, r: float = 2.0) -> float:
    """Variance per unit budget in units of sigma^2/(h1^2 B) (local-averaging idealization)."""
    nr, pairs = unit_structure(plan, h1, r)
    V = _vmat(nr, pairs)
    hs = np.array([p[2] for p in pairs])
    grp = np.array([p[3] for p in pairs])
    est = estimator.upper()
    if est in ("MP", "POOL", "OLS"):
        # endpoint-defect weighted least squares through the origin on displacements D = h V
        c = (hs**2) @ V / np.sum(hs**2)
    elif est == "GLS":
        D = hs[:, None] * V
        C = D @ D.T
        Ci = np.linalg.pinv(C)
        c = (hs @ Ci @ D) / (hs @ Ci @ hs)
    elif est == "RICH":
        gvals = sorted(set(grp))
        hg = [hs[grp == g][0] for g in gvals]
        a1, a2 = richardson_weights(hg[0], hg[1])
        c = a1 * V[grp == gvals[0]].mean(0) + a2 * V[grp == gvals[1]].mean(0)
    else:
        raise ValueError(estimator)
    return float(c @ c * nr * h1**2)


def pair_noise_covariances(plan: str, h1: float = 1.0, r: float = 2.0):
    """Cov of anchor noise U and velocity noise W for every pair combination (per coordinate)."""
    nr, pairs = unit_structure(plan, h1, r)
    A = np.zeros((len(pairs), nr))
    for k, (a, b, h, _) in enumerate(pairs):
        A[k, a] += 0.5
        A[k, b] += 0.5
    V = _vmat(nr, pairs)
    return {"UU": A @ A.T, "WW": V @ V.T, "UW": A @ V.T}


def mse_optimum(beta2: float, var_coef: float, p_bias: int, sigma2_over_B: float):
    """min_h  beta2 * h^(2 p_bias) + var_coef * sigma2/B / h^2  -> (h*, M*)."""
    q = 2 * p_bias
    hstar = (var_coef * sigma2_over_B / (p_bias * beta2)) ** (1.0 / (q + 2))
    M = beta2 * hstar**q + var_coef * sigma2_over_B / hstar**2
    return hstar, M
