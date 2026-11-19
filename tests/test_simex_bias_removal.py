"""Measurement-error (EIV) bias: SIMEX removes it where it exists; the Hamiltonian constraint removes the
forward-anchoring component by itself (code-correctness tests on a synthetic setting, not a benchmark).

Setting: anchors z0 ~ N(0, I), exact forward-Euler-consistent pairs y1 = y0 + h f(y0) for a linear Hamiltonian
field f = J S z, isotropic reading noise sigma, forward-difference anchoring (anchor = first noisy reading).
Then E[v | anchor] contains -(1/h) E[e0 | anchor] = -(1/h) sigma^2 grad log rho(anchor) (Tweedie), a field along
x for Gaussian anchors.  Bias is measured by projecting the noise-averaged estimate minus truth onto the two
bias directions (f: attenuation; x: forward-anchoring EIV), which averages out variance.

* Unstructured vector KRR: large x-component (linear theory: -(s^2/(1+s^2))/h = -0.41); RB-SIMEX removes it.
* Hamiltonian gradient KRR: x-component ~ 0, because sigma^2 grad log rho is L2(rho)-orthogonal to every
  divergence-free (Hamiltonian) field: int grad(rho) . J grad H = -int rho div(J grad H) = 0.
"""
import numpy as np

from hamsched.corrections import simex
from hamsched.kernels import SEKernel
from hamsched.schedules import build_dataset
from hamsched.simulate import Bank
from hamsched.systems import grad_to_field

S = np.diag([1.0, 0.7, 1.3, 0.9])


def f(z):
    return grad_to_field(z @ S)


def _projected_bias(est, n_draws=5, K=6):
    n, h, sigma = 400, 0.2, 0.3
    rng = np.random.default_rng(0)
    z0 = rng.standard_normal((n, 4))
    X = np.stack([z0, z0 + h * f(z0)], axis=1)
    Xe = 0.6 * rng.standard_normal((300, 4))
    Fe = f(Xe)
    D = np.stack([Fe.ravel(), Xe.ravel()], 1)
    out = {"naive": [], "simex_q": []}
    for d in range(n_draws):
        xi = np.random.default_rng(10 + d).standard_normal((n, 2, 2, 4))
        bank = Bank(z0=z0, times=np.array([0.0, h]), X=X, xi=xi, seed=(d,))
        ds = build_dataset("FWD", {"h": h}, 2 * n, bank, sigma)
        o, st = simex(ds, [est], SEKernel(), 1.0, 1e-3, {"e": Xe}, sigma, K=K, seed=d, L=np.full(4, 3.0))
        for k in out:
            out[k].append(np.linalg.lstsq(D, (o[est][k]["e"] - Fe).ravel(), rcond=None)[0])
    return {k: np.mean(v, 0) for k, v in out.items()}


def test_simex_removes_forward_anchoring_eiv_bias_in_unstructured_fit():
    b = _projected_bias("U-MP")
    assert b["naive"][1] < -0.25                 # large EIV component along x (theory ~ -0.41)
    assert abs(b["simex_q"][1]) < 0.15           # removed by reading-level RB-SIMEX


def test_hamiltonian_constraint_projects_out_forward_anchoring_eiv_bias():
    b = _projected_bias("MP", n_draws=3, K=2)
    assert abs(b["naive"][1]) < 0.05             # gradient-of-log-density bias is orthogonal to Hamiltonian fields
