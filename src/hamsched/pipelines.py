"""Estimator pipelines on a Dataset, sharing Gram matrices across estimators and the eta path.

Estimators
  MP     all pairs, one fit (single-interval plans: SI, FWD)
  DC     MP + model-based defect correction (single interval; cross-fitted, chord-midpoint defect)
  POOL   all pairs, one fit, endpoint-defect ('ols') weighting
  GLS    all pairs, one fit, full within-unit noise covariance (known Sigma up to scale)
  RICH   one fit per interval group with shared (ell, eta); Richardson weights for MP (even powers):
         alpha1 = r^2/(r^2-1), alpha2 = -1/(r^2-1), r = h2/h1
  LONG   only the longest-interval group (e.g. SA without its middle reading)
  U-MP, U-RICH   unstructured vector KRR analogues (robustness reference)

Optional `pseudo` (R,4): Rao-Blackwellized SIMEX pseudo-noise on readings.  Anchors get the
pseudo anchor noise U* = A xi; velocities get E[W* | U*] = M U* (exact Gaussian conditional mean,
zero for fits with one pair per unit), so the expectation over the pseudo velocity noise is exact.
"""
from __future__ import annotations

import numpy as np

from .krr import GradFit, ScalarFit, noise_structure, targets
from .systems import grad_to_field
from .kernels import predict_grad

GRAD_ESTS = {"MP", "DC", "POOL", "GLS", "RICH", "LONG", "SHORT"}


def richardson_weights(h1: float, h2: float, order: int = 2):
    r = h2 / h1
    a1 = r**order / (r**order - 1.0)
    return a1, 1.0 - a1


def _fit_data(ds, idx, values, pseudo, exact_fn=None):
    anchors, vel = ds.derive(values, idx)
    if exact_fn is not None:       # exact derivative targets at the (clean) anchors: approximation floor
        return anchors, exact_fn(anchors)
    if pseudo is not None:
        Ustar, _ = ds.derive(pseudo, idx)
        anchors = anchors + Ustar
        M = ds.unit_maps(idx)
        if np.any(M != 0):
            vel = vel + M @ Ustar
    return anchors, vel


def _rk4_field(fit: GradFit, coef: np.ndarray, z: np.ndarray, t: float, substeps: int) -> np.ndarray:
    dt = t / substeps

    def f(x):
        return grad_to_field(predict_grad(x, fit.X, fit.L, fit.kernel, coef))
    for _ in range(substeps):
        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def defect(fit: GradFit, coef: np.ndarray, x: np.ndarray, h: np.ndarray, substeps: int = 4) -> np.ndarray:
    """MP discretization defect of the model at anchors x, evaluated at the chord midpoint.

    z0' = phi^model_{-h/2}(x), z1' = phi^model_{+h/2}(x),  d = (z1'-z0')/h - f_model((z0'+z1')/2).
    Then v - d approximates f at x up to O(h^4 + h^2 * model error) (anchor shift m - x = O(h^2)
    multiplies a defect of O(h^2)).  Assumes a single interval value h.
    """
    hh = float(h[0])
    z0 = _rk4_field(fit, coef, x, -0.5 * hh, substeps)
    z1 = _rk4_field(fit, coef, x, 0.5 * hh, substeps)
    m = 0.5 * (z0 + z1)
    fm = grad_to_field(predict_grad(m, fit.X, fit.L, fit.kernel, coef))
    return (z1 - z0) / hh - fm


def evaluate(ds, estimators, kernel, ell: float, etas, eval_X: dict, L: np.ndarray | None = None,
             values: np.ndarray | None = None, pseudo: np.ndarray | None = None, dc_substeps: int = 4,
             dc_seed: int = 0, exact_fn=None):
    """Fit `estimators` over the eta path and predict the vector field on each eval set.

    Returns (preds, status): preds[est][set] -> array (n_eta, P, 4) (NaN on failure);
    status[est] -> int array (n_eta,), 0 ok, 1 jitter used, 2 failed.
    """
    etas = np.atleast_1d(np.asarray(etas, float))
    nE = len(etas)
    if L is None:
        L = ell * ds.reading_std()
    p = ds.pairs
    all_idx = np.arange(len(p.h))
    groups = sorted(ds.group_h)
    preds = {e: {s: np.full((nE, len(X), 4), np.nan) for s, X in eval_X.items()} for e in estimators}
    status = {e: np.zeros(nE, dtype=int) for e in estimators}

    unstr = [e for e in estimators if e.startswith("U-")]
    grad_est = [e for e in estimators if e in GRAD_ESTS]
    unknown = set(estimators) - set(unstr) - set(grad_est)
    if unknown:
        raise ValueError(f"unknown estimators {unknown}")

    # ---------- fits on all pairs (MP / POOL / DC share the 'ols' fit) ----------
    need_all_ols = any(e in ("MP", "POOL", "DC") for e in grad_est)
    if need_all_ols:
        X, v = _fit_data(ds, all_idx, values, pseudo, exact_fn)
        g = targets(v)
        fit = GradFit(X, kernel, L, noise_structure(ds, all_idx, "ols"))
        do_dc = "DC" in grad_est
        if do_dc:
            if len(groups) != 1:
                raise ValueError("DC is defined for single-interval data")
            rng = np.random.default_rng(dc_seed)
            units = np.unique(p.unit)
            fold_of_unit = rng.permutation(len(units)) % 2
            fold = fold_of_unit[np.searchsorted(units, p.unit)]
            ffits = [GradFit(X[fold == k], kernel, L, ("diag", np.ones((fold == k).sum()))) for k in (0, 1)]
        for j, eta in enumerate(etas):
            fac, st = fit.factor(eta)
            if fac is None:
                for e in ("MP", "POOL", "DC"):
                    if e in grad_est:
                        status[e][j] = 2
                continue
            coef = fit.solve(fac, g)
            for e in ("MP", "POOL"):
                if e in grad_est:
                    status[e][j] = st
                    for s, Xe in eval_X.items():
                        preds[e][s][j] = fit.field(coef, Xe)
            if do_dc:
                d = np.zeros_like(v)
                ok = True
                for k in (0, 1):
                    ff = ffits[k]
                    fk, stk = ff.factor(eta)
                    if fk is None:
                        ok = False
                        break
                    ck = ff.solve(fk, g[fold == k])
                    other = fold == (1 - k)
                    d[other] = defect(ff, ck, X[other], p.h[other], dc_substeps)
                    del fk
                if ok:
                    coef_c = fit.solve(fac, targets(v - d))
                    status["DC"][j] = st
                    for s, Xe in eval_X.items():
                        preds["DC"][s][j] = fit.field(coef_c, Xe)
                else:
                    status["DC"][j] = 2
            del fac
        fit.release()

    if "GLS" in grad_est:
        X, v = _fit_data(ds, all_idx, values, pseudo, exact_fn)
        g = targets(v)
        fit = GradFit(X, kernel, L, noise_structure(ds, all_idx, "gls"))
        for j, eta in enumerate(etas):
            fac, st = fit.factor(eta)
            status["GLS"][j] = st
            if fac is None:
                continue
            coef = fit.solve(fac, g)
            for s, Xe in eval_X.items():
                preds["GLS"][s][j] = fit.field(coef, Xe)
            del fac
        fit.release()

    # ---------- per-group fits (RICH, LONG) ----------
    if "RICH" in grad_est or "LONG" in grad_est or "SHORT" in grad_est:
        gp = {}
        for gi in groups:
            idx = np.where(p.group == gi)[0]
            X, v = _fit_data(ds, idx, values, pseudo, exact_fn)
            fit = GradFit(X, kernel, L, noise_structure(ds, idx, "ols"))
            gfit_preds = {s: np.full((nE, len(Xe), 4), np.nan) for s, Xe in eval_X.items()}
            gst = np.zeros(nE, dtype=int)
            g = targets(v)
            for j, eta in enumerate(etas):
                fac, st = fit.factor(eta)
                gst[j] = st
                if fac is None:
                    continue
                coef = fit.solve(fac, g)
                for s, Xe in eval_X.items():
                    gfit_preds[s][j] = fit.field(coef, Xe)
                del fac
            fit.release()
            gp[gi] = (gfit_preds, gst)
        if "RICH" in grad_est:
            if len(groups) != 2:
                raise ValueError("RICH needs exactly two interval groups")
            a1, a2 = richardson_weights(ds.group_h[groups[0]], ds.group_h[groups[1]])
            (p0, s0), (p1, s1) = gp[groups[0]], gp[groups[1]]
            status["RICH"] = np.maximum(s0, s1)
            for s in eval_X:
                preds["RICH"][s] = a1 * p0[s] + a2 * p1[s]
        for name, gi in (("LONG", groups[-1]), ("SHORT", groups[0])):
            if name in grad_est:
                pl, sl_ = gp[gi]
                status[name] = sl_
                for s in eval_X:
                    preds[name][s] = pl[s]

    # ---------- unstructured references ----------
    for e in unstr:
        base = e[2:]
        if base == "MP":
            parts = [(all_idx, 1.0)]
        elif base == "RICH":
            a1, a2 = richardson_weights(ds.group_h[groups[0]], ds.group_h[groups[1]])
            parts = [(np.where(p.group == groups[0])[0], a1), (np.where(p.group == groups[1])[0], a2)]
        else:
            raise ValueError(e)
        acc = {s: np.zeros((nE, len(Xe), 4)) for s, Xe in eval_X.items()}
        st_all = np.zeros(nE, dtype=int)
        for idx, w in parts:
            X, v = _fit_data(ds, idx, values, pseudo)
            fit = ScalarFit(X, kernel, L, noise_structure(ds, idx, "ols"))
            for j, eta in enumerate(etas):
                fac, st = fit.factor(eta)
                st_all[j] = max(st_all[j], st)
                if fac is None:
                    for s in eval_X:
                        acc[s][j] = np.nan
                    continue
                beta = fac.solve(v)
                for s, Xe in eval_X.items():
                    acc[s][j] += w * fit.field_from(beta, Xe)
            fit.G = None
        status[e] = st_all
        for s in eval_X:
            preds[e][s] = acc[s]
    return preds, status


class FieldModel:
    """Callable learned vector field: weighted sum of gradient-KRR fits (Richardson etc.)."""

    def __init__(self, parts):
        self.parts = parts            # list of (weight, GradFit, coef)

    def __call__(self, z):
        z = np.atleast_2d(z)
        out = np.zeros_like(z, dtype=float)
        for w, fit, coef in self.parts:
            out += w * grad_to_field(predict_grad(z, fit.X, fit.L, fit.kernel, coef))
        return out


def fit_field_model(ds, est, kernel, ell, eta, L=None, values=None):
    """Fit one estimator at a single (ell, eta) and return (FieldModel or None, status)."""
    if L is None:
        L = ell * ds.reading_std()
    p = ds.pairs
    groups = sorted(ds.group_h)
    if est in ("MP", "POOL", "DC"):
        idx = np.arange(len(p.h))
        X, v = _fit_data(ds, idx, values, None)
        fit = GradFit(X, kernel, L, noise_structure(ds, idx, "ols"))
        fac, st = fit.factor(eta)
        if fac is None:
            return None, 2
        coef = fit.solve(fac, targets(v))
        if est == "DC":
            # corrected coefficients (same cross-fitted procedure as in evaluate)
            rng = np.random.default_rng(0)
            units = np.unique(p.unit)
            fold = (rng.permutation(len(units)) % 2)[np.searchsorted(units, p.unit)]
            d = np.zeros_like(v)
            for k in (0, 1):
                ff = GradFit(X[fold == k], kernel, L, ("diag", np.ones((fold == k).sum())))
                fk, stk = ff.factor(eta)
                if fk is None:
                    return None, 2
                ck = ff.solve(fk, targets(v)[fold == k])
                other = fold == (1 - k)
                d[other] = defect(ff, ck, X[other], p.h[other])
            coef = fit.solve(fac, targets(v - d))
        fit.G = None
        return FieldModel([(1.0, fit, coef)]), st
    if est in ("RICH", "LONG", "SHORT"):
        if est == "RICH":
            a1, a2 = richardson_weights(ds.group_h[groups[0]], ds.group_h[groups[1]])
            use = [(groups[0], a1), (groups[1], a2)]
        else:
            use = [(groups[-1] if est == "LONG" else groups[0], 1.0)]
        parts, stmax = [], 0
        for gi, w in use:
            idx = np.where(p.group == gi)[0]
            X, v = _fit_data(ds, idx, values, None)
            fit = GradFit(X, kernel, L, noise_structure(ds, idx, "ols"))
            fac, st = fit.factor(eta)
            if fac is None:
                return None, 2
            stmax = max(stmax, st)
            coef = fit.solve(fac, targets(v))
            fit.G = None
            parts.append((w, fit, coef))
        return FieldModel(parts), stmax
    raise ValueError(est)


def rk4_flow(field, z, t, dt=0.01):
    """Integrate dz/dt = field(z) from 0 to t (t may be negative) with fixed-step RK4."""
    n = max(1, int(np.ceil(abs(t) / dt)))
    h = t / n
    for _ in range(n):
        k1 = field(z)
        k2 = field(z + 0.5 * h * k1)
        k3 = field(z + 0.5 * h * k2)
        k4 = field(z + h * k3)
        z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z
