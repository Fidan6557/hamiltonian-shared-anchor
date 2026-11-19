"""Measurement-error correction: reading-level, Rao-Blackwellized SIMEX (Cook & Stefanski 1994).

Assumption (stated in the report): the reading noise covariance Sigma is known (here sigma_hat^2 I);
misspecification is tested by passing sigma_hat != sigma.  Pseudo-noise of covariance zeta*Sigma is
added to READINGS, so a reading shared by two pairs receives the same pseudo-noise in both.  The
pseudo velocity noise is replaced by its exact Gaussian conditional mean given the pseudo anchor
noise (see pipelines._fit_data), which is valid because the fit is linear in the velocities given
the anchors.  The extrapolated estimate is a fixed linear combination of Hamiltonian fits.
"""
from __future__ import annotations

import numpy as np

from .pipelines import evaluate


def extrapolation_weights(zetas, degree: int, target: float = -1.0) -> np.ndarray:
    z = np.asarray(zetas, float)
    X = np.vander(z, degree + 1, increasing=True)
    t = target ** np.arange(degree + 1)
    return t @ np.linalg.pinv(X)


def simex(ds, estimators, kernel, ell, eta, eval_X, sigma_hat, zetas=(0.5, 1.0, 1.5, 2.0), K=4, seed=0,
          L=None):
    """Returns preds[est][variant][set] with variants 'naive', 'simex_q' (quadratic), 'simex_l' (linear),
    and status[est] (max over all refits)."""
    if L is None:
        L = ell * ds.reading_std()
    base, st0 = evaluate(ds, estimators, kernel, ell, [eta], eval_X, L=L)
    rng = np.random.default_rng(seed)
    R = len(ds.readings.ids)
    means = {e: {s: [base[e][s][0]] for s in eval_X} for e in estimators}
    status = {e: int(st0[e][0]) for e in estimators}
    for zeta in zetas:
        acc = {e: {s: 0.0 for s in eval_X} for e in estimators}
        for _ in range(K):
            xi = rng.standard_normal((R, 4)) * (sigma_hat * np.sqrt(zeta))
            p, st = evaluate(ds, estimators, kernel, ell, [eta], eval_X, L=L, pseudo=xi)
            for e in estimators:
                status[e] = max(status[e], int(st[e][0]))
                for s in eval_X:
                    acc[e][s] = acc[e][s] + p[e][s][0] / K
        for e in estimators:
            for s in eval_X:
                means[e][s].append(acc[e][s])
    zs = np.concatenate([[0.0], np.asarray(zetas, float)])
    wq = extrapolation_weights(zs, 2)
    wl = extrapolation_weights(zs, 1)
    out = {}
    for e in estimators:
        out[e] = {"naive": {}, "simex_q": {}, "simex_l": {}}
        for s in eval_X:
            stack = np.stack(means[e][s])          # (n_zeta+1, P, 4)
            out[e]["naive"][s] = stack[0]
            out[e]["simex_q"][s] = np.tensordot(wq, stack, axes=(0, 0))
            out[e]["simex_l"][s] = np.tensordot(wl, stack, axes=(0, 0))
    return out, status
