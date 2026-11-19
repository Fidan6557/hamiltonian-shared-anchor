"""High-accuracy simulation (DOP853 on stacked trajectory batches), trajectory banks and eval sets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .systems import HamiltonianSystem

RTOL = 1e-12
ATOL = 1e-12


def integrate(system: HamiltonianSystem, z0: np.ndarray, times: np.ndarray, rtol: float = RTOL,
              atol: float = ATOL, chunk: int = 128) -> np.ndarray:
    """Exact-flow states at `times` (monotone, may be negative, need not start at 0) from z0 at t=0.

    Returns array (N, len(times), 4).  Trajectories are integrated in stacked batches; the
    per-trajectory accuracy is verified separately in tests (scipy's RMS error norm over stacked
    components can dilute a single trajectory's error, so tolerances are set very tight).
    """
    z0 = np.atleast_2d(np.asarray(z0, float))
    times = np.asarray(times, float)
    n = len(z0)
    out = np.empty((n, len(times), 4))

    for i in range(0, n, chunk):
        y0 = z0[i:i + chunk]
        m = len(y0)
        nonzero = times != 0.0
        res = np.repeat(y0[:, None, :], len(times), axis=1)
        if np.any(nonzero):
            def rhs(t, y, m=m):
                return system.field(y.reshape(m, 4)).ravel()
            # integrate separately for positive and negative times
            for sign in (1.0, -1.0):
                mask = (np.sign(times) == sign)
                if not np.any(mask):
                    continue
                ts = times[mask]
                order = np.argsort(sign * ts)
                ts_sorted = ts[order]
                sol = solve_ivp(rhs, (0.0, ts_sorted[-1]), y0.ravel(), method="DOP853",
                                t_eval=ts_sorted, rtol=rtol, atol=atol)
                if not sol.success:
                    raise RuntimeError(f"integration failed: {sol.message}")
                Y = sol.y.T.reshape(len(ts_sorted), m, 4).transpose(1, 0, 2)
                idx = np.where(mask)[0][order]
                res[:, idx, :] = Y
        out[i:i + m] = res

    return out


def flow(system: HamiltonianSystem, z: np.ndarray, t: float, **kw) -> np.ndarray:
    """phi_t(z) for a single scalar time t (positive or negative)."""
    return integrate(system, z, np.array([t]), **kw)[:, 0, :]


@dataclass
class Bank:
    """Clean states for a set of trajectory starts at a common time grid, plus standard noise draws.

    xi[start, time_index, copy, :] ~ N(0, I) (or a temporally correlated / alternative model); copy 1
    is only used for a second, independent reading at the same time (RA plan, t=0).
    """
    z0: np.ndarray        # (N, 4)
    times: np.ndarray     # (T,)
    X: np.ndarray         # (N, T, 4) clean states
    xi: np.ndarray        # (N, T, 2, 4) standard noise draws
    seed: tuple

    def tidx(self, t: float) -> int:
        j = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[j] - t) > 1e-9:
            raise KeyError(f"time {t} not in bank grid")
        return j


def standard_noise(rng: np.random.Generator, n: int, times: np.ndarray, model: str = "iid",
                   tau: float | None = None) -> np.ndarray:
    T = len(times)
    if model in ("iid", "hetero"):
        return rng.standard_normal((n, T, 2, 4))
    if model == "temporal":
        if tau is None or tau <= 0:
            raise ValueError("temporal noise needs tau > 0")
        C = np.exp(-np.abs(times[:, None] - times[None, :]) / tau)
        Lc = np.linalg.cholesky(C + 1e-12 * np.eye(T))
        xi = rng.standard_normal((n, T, 2, 4))
        # correlate along time for copy 0; copy 1 (repeated reading) stays independent
        xi[:, :, 0, :] = np.einsum("st,ntc->nsc", Lc, xi[:, :, 0, :])
        return xi
    raise ValueError(f"unknown noise model {model}")


def make_bank(system: HamiltonianSystem, n_starts: int, times, seed: tuple, noise_model: str = "iid",
              tau: float | None = None) -> Bank:
    ss = np.random.SeedSequence(list(seed))
    s_start, s_noise = ss.spawn(2)
    rng_s = np.random.default_rng(s_start)
    z0 = system.sample(n_starts, rng_s)
    times = np.unique(np.concatenate([[0.0], np.asarray(times, float)]))
    X = integrate(system, z0, times)
    xi = standard_noise(np.random.default_rng(s_noise), n_starts, times, noise_model, tau)
    return Bank(z0=z0, times=times, X=X, xi=xi, seed=tuple(seed))


@dataclass
class EvalSet:
    X: np.ndarray   # (P, 4) clean states along held-out trajectories
    F: np.ndarray   # (P, 4) true vector field
    E: np.ndarray   # (P,) energy
    traj: np.ndarray  # (P,) trajectory id


def make_eval_set(system: HamiltonianSystem, n_traj: int, n_states: int, t_max: float, seed: tuple) -> EvalSet:
    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    z0 = system.sample(n_traj, rng)
    times = np.linspace(0.0, t_max, n_states)
    Z = integrate(system, z0, times).reshape(-1, 4)
    traj = np.repeat(np.arange(n_traj), n_states)
    return EvalSet(X=Z, F=system.field(Z), E=system.H(Z), traj=traj)
