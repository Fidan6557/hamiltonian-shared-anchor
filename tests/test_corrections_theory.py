"""Slope gates (discretization orders), SIMEX extrapolation, and the variance-factor table."""
import numpy as np
import pytest

from hamsched.corrections import extrapolation_weights
from hamsched.pipelines import richardson_weights
from hamsched.simulate import flow
from hamsched.systems import DoublePendulum
from hamsched.theory import pair_noise_covariances, variance_factor


def _mp_velocity_at_anchor(sys_, x, h, iters=30):
    """Exact-flow MP pair whose midpoint is exactly x: solve (z0 + phi_h(z0))/2 = x."""
    z0 = x - 0.5 * h * sys_.field(x)
    for _ in range(iters):
        z1 = flow(sys_, z0, h)
        z0 = z0 + (x - 0.5 * (z0 + z1))
    z1 = flow(sys_, z0, h)
    return (z1 - z0) / h


def _slope(hs, errs):
    return np.polyfit(np.log(hs), np.log(errs), 1)[0]


@pytest.fixture(scope="module")
def setup():
    dp = DoublePendulum()
    x = dp.sample(30, np.random.default_rng(0))
    return dp, x


def test_mp_target_is_second_order_and_richardson_fourth(setup):
    dp, x = setup
    f = dp.field(x)
    hs = np.array([0.02, 0.03, 0.045, 0.065])
    e_mp, e_r = [], []
    for h in hs:
        v1 = _mp_velocity_at_anchor(dp, x, h)
        v2 = _mp_velocity_at_anchor(dp, x, 2 * h)
        a1, a2 = richardson_weights(h, 2 * h)
        e_mp.append(np.linalg.norm(v1 - f))
        e_r.append(np.linalg.norm(a1 * v1 + a2 * v2 - f))
    assert 1.8 < _slope(hs, e_mp) < 2.2
    assert 3.6 < _slope(hs, e_r) < 4.4


def test_richardson_general_ratio_is_fourth_order(setup):
    dp, x = setup
    f = dp.field(x)
    hs = np.array([0.02, 0.03, 0.045])
    errs = []
    for h in hs:
        v1, v2 = _mp_velocity_at_anchor(dp, x, h), _mp_velocity_at_anchor(dp, x, 1.5 * h)
        a1, a2 = richardson_weights(h, 1.5 * h)
        errs.append(np.linalg.norm(a1 * v1 + a2 * v2 - f))
    assert 3.6 < _slope(hs, errs) < 4.4


def test_defect_correction_with_true_flow_is_fourth_order(setup):
    """Gate for SI-DC: chord-midpoint defect computed with the true flow removes the O(h^2) term."""
    dp, x = setup
    f = dp.field(x)
    hs = np.array([0.02, 0.03, 0.045, 0.065])
    e_dc, e_naive_shift = [], []
    for h in hs:
        v = _mp_velocity_at_anchor(dp, x, h)
        z0 = flow(dp, x, -0.5 * h)
        z1 = flow(dp, x, 0.5 * h)
        m = 0.5 * (z0 + z1)
        d = (z1 - z0) / h - dp.field(m)
        e_dc.append(np.linalg.norm(v - d - f))
        d_wrong = (z1 - z0) / h - dp.field(x)       # the anchor-shift trap
        e_naive_shift.append(np.linalg.norm(v - d_wrong - f))
    assert 3.5 < _slope(hs, e_dc) < 4.5
    assert _slope(hs, e_naive_shift) < 2.5           # trap leaves an O(h^2) error


def test_extrapolation_weights_exact_for_polynomials():
    z = np.array([0, 0.5, 1, 1.5, 2])
    vals = 1.0 + 2.0 * z - 0.7 * z**2
    assert abs(extrapolation_weights(z, 2) @ vals - (1 - 2 - 0.7)) < 1e-12
    vals1 = 3.0 - 0.5 * z
    assert abs(extrapolation_weights(z, 1) @ vals1 - 3.5) < 1e-12


def test_variance_factor_table():
    expect = {("SI", "MP"): 4.0, ("SI2", "MP"): 1.0, ("IP", "POOL"): 1.6, ("SA", "POOL"): 1.68,
              ("SA", "GLS"): 1.5, ("IP", "RICH"): 14.444444, ("RA", "RICH"): 14.444444,
              ("SA", "RICH"): 9.5, ("MS3", "RICH"): 1.5}
    for (plan, est), val in expect.items():
        assert abs(variance_factor(plan, est) - val) < 1e-5, (plan, est, variance_factor(plan, est))
    # Hu Sec. 6.3 protocol (independent stride noise): per-unit 0.944 vs 0.5 sigma^2/h^2
    assert abs(variance_factor("MS3_INDEP", "RICH") / 5 - 0.94444) < 1e-4
    assert abs(variance_factor("MS3", "RICH") / 3 - 0.5) < 1e-12


def test_pair_covariances_sa():
    c = pair_noise_covariances("SA", h1=0.1, r=2)
    np.testing.assert_allclose(c["WW"][0, 1], 1 / (0.1 * 0.2))
    np.testing.assert_allclose(c["UW"][0, 1], -1 / (2 * 0.2))   # Cov(U1, W2)
    np.testing.assert_allclose(c["UW"][1, 0], -1 / (2 * 0.1))   # Cov(U2, W1)
    np.testing.assert_allclose(np.diag(c["UW"]), 0.0, atol=1e-12)
