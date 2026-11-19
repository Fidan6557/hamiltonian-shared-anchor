import numpy as np
from scipy.integrate import solve_ivp

from hamsched.simulate import flow, integrate, make_bank, make_eval_set
from hamsched.systems import DoublePendulum, HenonHeiles, field_to_grad, grad_to_field


def _fd_grad(H, z, eps=1e-6):
    g = np.zeros_like(z)
    for k in range(z.shape[-1]):
        e = np.zeros(z.shape[-1])
        e[k] = eps
        g[..., k] = (H(z + e) - H(z - e)) / (2 * eps)
    return g


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    for sys_ in (DoublePendulum(), HenonHeiles()):
        z = sys_.sample(200, rng)
        np.testing.assert_allclose(sys_.grad(z), _fd_grad(sys_.H, z), rtol=1e-6, atol=1e-7)


def test_J_roundtrip():
    g = np.random.default_rng(1).standard_normal((10, 4))
    np.testing.assert_allclose(field_to_grad(grad_to_field(g)), g)


def test_sampler_respects_sublevel_set():
    rng = np.random.default_rng(2)
    dp = DoublePendulum(e_max=-1.5)
    z = dp.sample(5000, rng)
    assert np.all(dp.H(z) <= dp.e_max + 1e-12)
    lo, hi = dp.box
    assert np.all(z > lo) and np.all(z < hi)
    hh = HenonHeiles()
    z = hh.sample(5000, rng)
    assert np.all(hh.H(z) < 0.14) and np.all(hh.in_domain(z))


def test_stacked_integration_matches_individual_solves_and_conserves_energy():
    dp = DoublePendulum()
    rng = np.random.default_rng(3)
    z0 = dp.sample(40, rng)
    times = np.array([0.0, 0.05, 0.4, 0.8, 5.0])
    Z = integrate(dp, z0, times)
    for i in range(0, 40, 7):
        sol = solve_ivp(lambda t, y: dp.field(y[None])[0], (0, times[-1]), z0[i], method="DOP853",
                        t_eval=times, rtol=1e-13, atol=1e-13)
        assert np.max(np.abs(sol.y.T - Z[i])) < 1e-9
    drift = np.abs(dp.H(Z) - dp.H(z0)[:, None]).max()
    assert drift < 1e-10


def test_backward_flow_inverts_forward():
    dp = DoublePendulum()
    z = dp.sample(10, np.random.default_rng(4))
    z1 = flow(dp, z, 0.37)
    z0 = flow(dp, z1, -0.37)
    assert np.max(np.abs(z0 - z)) < 1e-10


def test_bank_and_evalset_shapes_and_reproducibility():
    dp = DoublePendulum()
    b1 = make_bank(dp, 12, [0.1, 0.2], seed=(7, 1))
    b2 = make_bank(dp, 12, [0.1, 0.2], seed=(7, 1))
    assert b1.X.shape == (12, 3, 4) and b1.xi.shape == (12, 3, 2, 4)
    np.testing.assert_array_equal(b1.X, b2.X)
    np.testing.assert_array_equal(b1.xi, b2.xi)
    ev = make_eval_set(dp, 5, 4, 2.0, seed=(9,))
    assert ev.X.shape == (20, 4)
    np.testing.assert_allclose(ev.F, dp.field(ev.X))
