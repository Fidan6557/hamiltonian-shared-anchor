"""Hamiltonian benchmark systems with analytic gradients and flow-invariant samplers.

State layout everywhere: z = (q1, q2, p1, p2).  Vector field f = J grad H with
J = [[0, I], [-I, 0]], i.e. dq/dt = dH/dp, dp/dt = -dH/dq.
"""
from __future__ import annotations

import numpy as np


def grad_to_field(g: np.ndarray) -> np.ndarray:
    """f = J g for g = grad H (last axis of length 4)."""
    return np.concatenate([g[..., 2:4], -g[..., 0:2]], axis=-1)


def field_to_grad(v: np.ndarray) -> np.ndarray:
    """g = J^T v (inverse of grad_to_field since J is orthogonal)."""
    return np.concatenate([-v[..., 2:4], v[..., 0:2]], axis=-1)


class HamiltonianSystem:
    name = "base"
    dim = 4

    def H(self, z):  # pragma: no cover - abstract
        raise NotImplementedError

    def grad(self, z):  # pragma: no cover - abstract
        raise NotImplementedError

    def field(self, z):
        return grad_to_field(self.grad(z))

    # --- sampling from a density rho(z) proportional to weight(H(z)) (invariant under the flow) ---
    def weight(self, E):  # pragma: no cover - abstract
        raise NotImplementedError

    def in_domain(self, z):
        return np.ones(z.shape[:-1], dtype=bool)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.box
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        out = []
        got = 0
        while got < n:
            m = max(4 * (n - got), 1024)
            z = lo + (hi - lo) * rng.random((m, self.dim))
            w = self.weight(self.H(z)) * self.in_domain(z)
            keep = rng.random(m) < w
            z = z[keep]
            out.append(z)
            got += len(z)
        return np.concatenate(out)[:n]

    def zrms(self, n: int = 200_000, seed: int = 12345) -> float:
        """RMS of all state coordinates under the sampling density (Hu et al. noise convention)."""
        z = self.sample(n, np.random.default_rng(seed))
        return float(np.sqrt(np.mean(z**2)))


class DoublePendulum(HamiltonianSystem):
    """Planar double pendulum, m1=m2=l1=l2=g=1 (nonseparable: mass matrix depends on q).

    H = [p1^2 + 2 p2^2 - 2 p1 p2 cos(q1-q2)] / [2 (1 + sin^2(q1-q2))] - 2 cos q1 - cos q2.
    States are drawn from rho ~ g(H) with g = 1 below a taper energy and a raised-cosine taper
    to 0 at e_max < -1 (no flips, bounded angles).  rho depends on z only through H, so it is
    invariant under the flow (Liouville + energy conservation).
    """

    name = "double_pendulum"

    def __init__(self, e_max: float = -1.5, taper_start: float = 0.6):
        if e_max >= -1.0:
            raise ValueError("e_max must be < -1 so that neither arm can flip over")
        self.e_min = -3.0
        self.e_max = float(e_max)
        self.taper_start = float(taper_start)
        de = self.e_max - self.e_min
        q1max = np.arccos(np.clip((-self.e_max - 1.0) / 2.0, -1, 1))
        q2max = np.arccos(np.clip(-self.e_max - 2.0, -1, 1))
        pmax = np.sqrt(2.0 * de * (3.0 + np.sqrt(5.0)) / 2.0)
        m = 1.02
        self.box = (np.array([-q1max, -q2max, -pmax, -pmax]) * m, np.array([q1max, q2max, pmax, pmax]) * m)

    def H(self, z):
        z = np.asarray(z, float)
        q1, q2, p1, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        d = q1 - q2
        c, s = np.cos(d), np.sin(d)
        D = 1.0 + s * s
        T = (p1**2 + 2.0 * p2**2 - 2.0 * p1 * p2 * c) / (2.0 * D)
        return T - 2.0 * np.cos(q1) - np.cos(q2)

    def grad(self, z):
        z = np.asarray(z, float)
        q1, q2, p1, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        d = q1 - q2
        c, s = np.cos(d), np.sin(d)
        D = 1.0 + s * s
        N = p1**2 + 2.0 * p2**2 - 2.0 * p1 * p2 * c
        dTdd = p1 * p2 * s / D - N * s * c / D**2
        g = np.empty(z.shape, float)
        g[..., 0] = dTdd + 2.0 * np.sin(q1)
        g[..., 1] = -dTdd + np.sin(q2)
        g[..., 2] = (p1 - p2 * c) / D
        g[..., 3] = (2.0 * p2 - p1 * c) / D
        return g

    def weight(self, E):
        s = (np.asarray(E) - self.e_min) / (self.e_max - self.e_min)
        s0 = self.taper_start
        w = np.where(s <= s0, 1.0, 0.5 * (1.0 + np.cos(np.pi * np.clip((s - s0) / (1.0 - s0), 0, 1))))
        return np.where(s > 1.0, 0.0, w)

    def linear_frequencies(self):
        # small oscillations: omega^2 = 2 +- sqrt(2)
        return np.sqrt([2 - np.sqrt(2), 2 + np.sqrt(2)])


class HenonHeiles(HamiltonianSystem):
    """Henon-Heiles (separable), the main benchmark of Hu et al. (arXiv:2610.17835).

    Sampling follows their description: uniform in a box, kept if H < e_max (here 0.14), restricted
    to the bounded potential well.  The box contains the whole well component of {H < 0.14}, so
    the resulting uniform (Liouville) measure is flow-invariant.
    """

    name = "henon_heiles"

    def __init__(self, e_max: float = 0.14):
        if e_max >= 1.0 / 6.0:
            raise ValueError("e_max must be below the escape energy 1/6")
        self.e_max = float(e_max)
        self.e_min = 0.0
        self.box = (np.array([-0.8, -0.8, -0.56, -0.56]), np.array([0.8, 0.8, 0.56, 0.56]))

    def H(self, z):
        z = np.asarray(z, float)
        q1, q2, p1, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        return 0.5 * (p1**2 + p2**2 + q1**2 + q2**2) + q1**2 * q2 - q2**3 / 3.0

    def grad(self, z):
        z = np.asarray(z, float)
        q1, q2, p1, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        g = np.empty(z.shape, float)
        g[..., 0] = q1 + 2.0 * q1 * q2
        g[..., 1] = q2 + q1**2 - q2**2
        g[..., 2] = p1
        g[..., 3] = p2
        return g

    def weight(self, E):
        return (np.asarray(E) < self.e_max).astype(float)

    def in_domain(self, z):
        q1, q2 = z[..., 0], z[..., 1]
        r3 = np.sqrt(3.0)
        return (q2 > -0.5) & (q2 < 1.0 - r3 * q1) & (q2 < 1.0 + r3 * q1)

    def linear_frequencies(self):
        return np.array([1.0, 1.0])


def make_system(cfg: dict) -> HamiltonianSystem:
    name = cfg["name"]
    if name == "double_pendulum":
        return DoublePendulum(e_max=cfg.get("e_max", -1.5), taper_start=cfg.get("taper_start", 0.6))
    if name == "henon_heiles":
        return HenonHeiles(e_max=cfg.get("e_max", 0.14))
    raise ValueError(f"unknown system {name}")
