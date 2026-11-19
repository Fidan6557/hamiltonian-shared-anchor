"""Measurement schedules (plans) built from a trajectory bank, with unique-reading accounting.

A *reading* is one noisy observation of (q, p) with a unique integer id
    id = (start * n_times + time_index) * 2 + copy.
A *pair* references two readings (by index into the dataset's reading table).  A reading shared
by two pairs (shared-anchor plan) is literally the same row: same id, same noisy value.

Plans (B = number of unique readings):
  SI(h)        start s: readings (0, h);                    B/2 starts, pairs (0,h)
  IP(h1, h2)   unit u: start 2u reads (0,h1), start 2u+1 reads (0,h2);  B/2 starts
  RA(h1, h2)   start s: reads t=0 twice (independent copies), h1, h2; pairs (0a,h1), (0b,h2); B/4 starts
               (the noise protocol of Hu et al. Sec. 6.3 -- same start, independent noise per stride --
               expressed in unique-reading accounting)
  SA(h1, h2)   start s: readings (0, h1, h2); pairs (0,h1), (0,h2) sharing reading 0;  B/3 starts
  MS3(h)       start s: readings (0, h, 2h); pairs (0,h), (h,2h) [stride 1] and (0,2h) [stride 2]; B/3
  FWD(h)       SI readings but forward-difference anchoring (anchor = first reading): positive control
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simulate import Bank

READINGS_PER_UNIT = {"SI": 2, "FWD": 2, "IP": 4, "RA": 4, "SA": 3, "MS3": 3, "MS3I": 5}
STARTS_PER_UNIT = {"SI": 1, "FWD": 1, "IP": 2, "RA": 1, "SA": 1, "MS3": 1, "MS3I": 1}


@dataclass
class Readings:
    values: np.ndarray   # (R, 4) noisy
    clean: np.ndarray    # (R, 4)
    start: np.ndarray    # (R,)
    tidx: np.ndarray     # (R,)
    copy: np.ndarray     # (R,)
    t: np.ndarray        # (R,)
    ids: np.ndarray      # (R,) int64 unique ids
    unit: np.ndarray     # (R,)


@dataclass
class Pairs:
    ia: np.ndarray       # (P,) index of first reading
    ib: np.ndarray       # (P,) index of second reading
    h: np.ndarray        # (P,) interval
    group: np.ndarray    # (P,) interval/stride label (0 = shorter)
    unit: np.ndarray     # (P,)
    anchor: str = "mid"  # 'mid' (implicit midpoint) or 'fwd'


@dataclass
class Dataset:
    plan: str
    params: dict
    readings: Readings
    pairs: Pairs
    group_h: dict        # group -> interval length
    info: dict = field(default_factory=dict)

    def derive(self, values: np.ndarray | None = None, idx: np.ndarray | None = None):
        """Anchors and velocities of (a subset of) pairs from reading values (default: noisy)."""
        y = self.readings.values if values is None else values
        p = self.pairs
        sel = slice(None) if idx is None else idx
        ya, yb, h = y[p.ia[sel]], y[p.ib[sel]], p.h[sel]
        anchors = 0.5 * (ya + yb) if p.anchor == "mid" else ya.copy()
        vel = (yb - ya) / h[:, None]
        return anchors, vel

    def reading_std(self) -> np.ndarray:
        return self.readings.values.std(axis=0)

    def unit_maps(self, idx: np.ndarray):
        """Linear maps from a unit's readings to its pairs' anchor noise (A) and velocity noise (V).

        Returns (signature -> (A, V)) is not needed by callers; instead we return, for the subset
        `idx` (sorted by unit), the per-pair conditional-mean matrix M such that
        E[W* | U*] = M U* within each unit, for i.i.d. pseudo-noise on readings (any covariance
        Sigma shared by all readings, applied per coordinate).  M is block-diagonal by unit and is
        returned as a dense (P_sub, P_sub) array restricted to within-unit entries.
        """
        p = self.pairs
        idx = np.asarray(idx)
        units = p.unit[idx]
        M = np.zeros((len(idx), len(idx)))
        cache = {}
        for u in np.unique(units):
            loc = np.where(units == u)[0]
            pr = idx[loc]
            rds = np.unique(np.concatenate([p.ia[pr], p.ib[pr]]))
            key = (tuple(np.round(p.h[pr], 12)), tuple(np.searchsorted(rds, p.ia[pr])),
                   tuple(np.searchsorted(rds, p.ib[pr])), p.anchor)
            if key not in cache:
                A = np.zeros((len(pr), len(rds)))
                V = np.zeros((len(pr), len(rds)))
                for k, q in enumerate(pr):
                    a, b = np.searchsorted(rds, p.ia[q]), np.searchsorted(rds, p.ib[q])
                    if p.anchor == "mid":
                        A[k, a] += 0.5
                        A[k, b] += 0.5
                    else:
                        A[k, a] += 1.0
                    V[k, a] -= 1.0 / p.h[q]
                    V[k, b] += 1.0 / p.h[q]
                cache[key] = V @ A.T @ np.linalg.inv(A @ A.T)
            M[np.ix_(loc, loc)] = cache[key]
        return M

    def velocity_noise_cov(self, idx: np.ndarray) -> np.ndarray:
        """Within-unit covariance structure of velocity noise (per coordinate, unit reading variance)."""
        p = self.pairs
        idx = np.asarray(idx)
        C = np.zeros((len(idx), len(idx)))
        units = p.unit[idx]
        for u in np.unique(units):
            loc = np.where(units == u)[0]
            pr = idx[loc]
            for i, qi in enumerate(pr):
                for j, qj in enumerate(pr):
                    c = 0.0
                    for ra, sa in ((p.ia[qi], -1.0 / p.h[qi]), (p.ib[qi], 1.0 / p.h[qi])):
                        for rb, sb in ((p.ia[qj], -1.0 / p.h[qj]), (p.ib[qj], 1.0 / p.h[qj])):
                            if ra == rb:
                                c += sa * sb
                    C[loc[i], loc[j]] = c
        return C


def _noise_scale(clean: np.ndarray, sigma: float, scale_fn) -> np.ndarray:
    if scale_fn is None:
        return np.full((len(clean), 1), sigma)
    return sigma * scale_fn(clean)[:, None]


def build_dataset(plan: str, params: dict, B: int, bank: Bank, sigma: float, scale_fn=None,
                  start_offset: int = 0) -> Dataset:
    """Build the plan's readings and pairs from `bank` with i.i.d. (or bank-model) noise of size sigma.

    scale_fn(clean_states) -> per-reading multiplicative noise scale (heteroscedastic truth), or None.
    """
    plan = plan.upper()
    k = READINGS_PER_UNIT[plan]
    if B % k != 0:
        raise ValueError(f"B={B} not divisible by {k} readings per unit for {plan}")
    n_units = B // k
    n_starts_needed = n_units * STARTS_PER_UNIT[plan]
    if start_offset + n_starts_needed > len(bank.z0):
        raise ValueError("bank too small for this plan/budget")

    rows = []      # (start, tidx, copy, unit)
    pairs = []     # (local reading index a, b, h, group, unit)

    def add_reading(s, t, copy, u):
        rows.append((s, bank.tidx(t), copy, u))
        return len(rows) - 1

    anchor = "fwd" if plan == "FWD" else "mid"
    if plan in ("SI", "FWD"):
        h = float(params["h"])
        group_h = {0: h}
        for u in range(n_units):
            s = start_offset + u
            a, b = add_reading(s, 0.0, 0, u), add_reading(s, h, 0, u)
            pairs.append((a, b, h, 0, u))
    elif plan == "IP":
        h1, h2 = float(params["h1"]), float(params["h2"])
        group_h = {0: h1, 1: h2}
        for u in range(n_units):
            s1, s2 = start_offset + 2 * u, start_offset + 2 * u + 1
            a, b = add_reading(s1, 0.0, 0, u), add_reading(s1, h1, 0, u)
            pairs.append((a, b, h1, 0, u))
            a, b = add_reading(s2, 0.0, 0, u), add_reading(s2, h2, 0, u)
            pairs.append((a, b, h2, 1, u))
    elif plan == "RA":
        h1, h2 = float(params["h1"]), float(params["h2"])
        group_h = {0: h1, 1: h2}
        for u in range(n_units):
            s = start_offset + u
            a0, a1 = add_reading(s, 0.0, 0, u), add_reading(s, 0.0, 1, u)
            b1, b2 = add_reading(s, h1, 0, u), add_reading(s, h2, 0, u)
            pairs.append((a0, b1, h1, 0, u))
            pairs.append((a1, b2, h2, 1, u))
    elif plan == "SA":
        h1, h2 = float(params["h1"]), float(params["h2"])
        group_h = {0: h1, 1: h2}
        for u in range(n_units):
            s = start_offset + u
            a, b1, b2 = add_reading(s, 0.0, 0, u), add_reading(s, h1, 0, u), add_reading(s, h2, 0, u)
            pairs.append((a, b1, h1, 0, u))
            pairs.append((a, b2, h2, 1, u))
    elif plan == "MS3":
        h = float(params["h"])
        group_h = {0: h, 1: 2.0 * h}
        for u in range(n_units):
            s = start_offset + u
            r0, r1, r2 = add_reading(s, 0.0, 0, u), add_reading(s, h, 0, u), add_reading(s, 2 * h, 0, u)
            pairs.append((r0, r1, h, 0, u))
            pairs.append((r1, r2, h, 0, u))
            pairs.append((r0, r2, 2 * h, 1, u))
    elif plan == "MS3I":
        # mechanism check only (not budget matched): MS3 readings plus independent re-measurements of
        # t=0 and t=2h for the stride-2 pair (noise protocol of Hu et al. Sec. 6.3)
        h = float(params["h"])
        group_h = {0: h, 1: 2.0 * h}
        for u in range(n_units):
            s = start_offset + u
            r0, r1, r2 = add_reading(s, 0.0, 0, u), add_reading(s, h, 0, u), add_reading(s, 2 * h, 0, u)
            q0, q2 = add_reading(s, 0.0, 1, u), add_reading(s, 2 * h, 1, u)
            pairs.append((r0, r1, h, 0, u))
            pairs.append((r1, r2, h, 0, u))
            pairs.append((q0, q2, 2 * h, 1, u))
    else:
        raise ValueError(f"unknown plan {plan}")

    rows = np.array(rows, dtype=np.int64)
    s, ti, cp, un = rows.T
    clean = bank.X[s, ti]
    scale = _noise_scale(clean, sigma, scale_fn)
    values = clean + scale * bank.xi[s, ti, cp]
    T = len(bank.times)
    ids = (s * T + ti) * 2 + cp
    readings = Readings(values=values, clean=clean, start=s, tidx=ti, copy=cp, t=bank.times[ti], ids=ids, unit=un)
    pa = np.array(pairs, dtype=float)
    P = Pairs(ia=pa[:, 0].astype(int), ib=pa[:, 1].astype(int), h=pa[:, 2], group=pa[:, 3].astype(int),
              unit=pa[:, 4].astype(int), anchor=anchor)
    ds = Dataset(plan=plan, params=dict(params), readings=readings, pairs=P, group_h=group_h)
    ds.info = accounting(ds)
    return ds


def accounting(ds: Dataset) -> dict:
    r = ds.readings
    starts = np.unique(r.start)
    max_t = np.array([r.t[r.start == s].max() for s in starts]) if len(starts) < 5000 else None
    return {
        "n_readings": int(len(np.unique(r.ids))),
        "n_reading_rows": int(len(r.ids)),
        "n_starts": int(len(starts)),
        "n_pairs": int(len(ds.pairs.h)),
        "max_duration": float(max_t.max()) if max_t is not None else float("nan"),
        "total_duration": float(max_t.sum()) if max_t is not None else float("nan"),
    }


def plan_times(plan: str, params: dict) -> list:
    plan = plan.upper()
    if plan in ("SI", "FWD"):
        return [params["h"]]
    if plan in ("IP", "RA", "SA"):
        return [params["h1"], params["h2"]]
    if plan in ("MS3", "MS3I"):
        return [params["h"], 2 * params["h"]]
    raise ValueError(plan)
