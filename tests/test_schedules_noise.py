import numpy as np
import pytest

from hamsched.schedules import build_dataset
from hamsched.simulate import make_bank
from hamsched.systems import DoublePendulum
from hamsched.krr import noise_structure


@pytest.fixture(scope="module")
def bank():
    return make_bank(DoublePendulum(), 120, [0.1, 0.2, 0.15, 0.3], seed=(11, 0))


@pytest.mark.parametrize("plan,params,starts,pairs", [
    ("SI", {"h": 0.2}, 60, 60),
    ("IP", {"h1": 0.1, "h2": 0.2}, 60, 60),
    ("RA", {"h1": 0.1, "h2": 0.2}, 30, 60),
    ("SA", {"h1": 0.1, "h2": 0.2}, 40, 80),
    ("MS3", {"h": 0.1}, 40, 120),
])
def test_unique_reading_accounting(bank, plan, params, starts, pairs):
    B = 120
    ds = build_dataset(plan, params, B, bank, sigma=0.01)
    assert ds.info["n_readings"] == B
    assert ds.info["n_reading_rows"] == B
    assert ds.info["n_starts"] == starts
    assert ds.info["n_pairs"] == pairs


def test_shared_anchor_is_the_same_reading(bank):
    ds = build_dataset("SA", {"h1": 0.1, "h2": 0.2}, 120, bank, sigma=0.05)
    p, r = ds.pairs, ds.readings
    first = p.ia[p.group == 0]
    second = p.ia[p.group == 1]
    np.testing.assert_array_equal(first, second)           # same row index
    np.testing.assert_array_equal(r.ids[first], r.ids[second])
    np.testing.assert_array_equal(r.values[first], r.values[second])


def test_ra_reads_anchor_twice_with_independent_noise(bank):
    ds = build_dataset("RA", {"h1": 0.1, "h2": 0.2}, 120, bank, sigma=0.05)
    p, r = ds.pairs, ds.readings
    a0, a1 = p.ia[p.group == 0], p.ia[p.group == 1]
    assert np.all(r.ids[a0] != r.ids[a1])
    np.testing.assert_allclose(r.clean[a0], r.clean[a1])
    assert np.all(np.abs(r.values[a0] - r.values[a1]).sum(1) > 0)


def test_durations(bank):
    ds = build_dataset("SA", {"h1": 0.1, "h2": 0.3}, 120, bank, sigma=0.0)
    assert abs(ds.info["max_duration"] - 0.3) < 1e-12
    assert abs(ds.info["total_duration"] - 40 * 0.3) < 1e-9


def test_cross_plan_common_random_numbers(bank):
    si = build_dataset("SI", {"h": 0.2}, 120, bank, sigma=0.05)
    sa = build_dataset("SA", {"h1": 0.1, "h2": 0.2}, 120, bank, sigma=0.05)
    # reading (start 0, t=0) has identical noisy value in both plans
    i_si = np.where((si.readings.start == 0) & (si.readings.t == 0))[0][0]
    i_sa = np.where((sa.readings.start == 0) & (sa.readings.t == 0))[0][0]
    np.testing.assert_array_equal(si.readings.values[i_si], sa.readings.values[i_sa])


def _mc_noise(plan, params, n_mc=4000):
    """Monte Carlo of anchor/velocity noise for one unit structure using a zero-state bank."""
    rng = np.random.default_rng(5)
    from hamsched.simulate import Bank
    times = np.unique([0.0] + [v for v in params.values()] + ([2 * params["h"]] if plan == "MS3" else []))
    n_st = 2 * n_mc
    bank = Bank(z0=np.zeros((n_st, 4)), times=times, X=np.zeros((n_st, len(times), 4)),
                xi=rng.standard_normal((n_st, len(times), 2, 4)), seed=(0,))
    k = {"SI": 2, "SA": 3, "MS3": 3, "IP": 4, "RA": 4}[plan]
    ds = build_dataset(plan, params, k * n_mc, bank, sigma=1.0)
    U, W = ds.derive()
    return ds, U, W


def test_mp_anchor_and_velocity_noise_uncorrelated():
    ds, U, W = _mc_noise("SI", {"h": 0.2})
    c = np.mean(U[:, 0] * W[:, 0])
    assert abs(c) < 4 * np.sqrt(0.5 * 2 / 0.04 / len(U))


def test_shared_anchor_covariances_match_theory():
    h1, h2 = 0.1, 0.2
    ds, U, W = _mc_noise("SA", {"h1": h1, "h2": h2}, n_mc=20000)
    g = ds.pairs.group
    U1, U2, W1, W2 = U[g == 0, 0], U[g == 1, 0], W[g == 0, 0], W[g == 1, 0]
    def close(x, y, tol):
        assert abs(x - y) < tol, (x, y)
    close(np.mean(W1 * W2), 1 / (h1 * h2), 0.05 / (h1 * h2))
    close(np.mean(U1 * W2), -1 / (2 * h2), 0.15 / h2)
    close(np.mean(U2 * W1), -1 / (2 * h1), 0.15 / h1)
    close(np.mean(U1 * U2), 0.25, 0.02)
    close(np.mean(U1 * W1), 0.0, 0.1 / h1)


def test_rb_conditional_mean_matrix_matches_regression():
    ds, U, W = _mc_noise("MS3", {"h": 0.1}, n_mc=20000)
    idx = np.where(ds.pairs.unit == 0)[0]
    M = ds.unit_maps(idx)
    # empirical regression of W on U (stacked over units, coordinate 0)
    P = len(idx)
    Uu = U[:, 0].reshape(-1, P)
    Wu = W[:, 0].reshape(-1, P)
    M_hat = np.linalg.lstsq(Uu, Wu, rcond=None)[0].T
    np.testing.assert_allclose(M_hat, M, atol=0.6)


def test_gls_structure_reduces_to_ols_for_independent_pairs(bank):
    ds = build_dataset("IP", {"h1": 0.1, "h2": 0.2}, 120, bank, sigma=0.01)
    idx = np.arange(len(ds.pairs.h))
    kind, w = noise_structure(ds, idx, "ols")
    kind2, (r, c, v) = noise_structure(ds, idx, "gls")
    assert np.all(r == c)          # no off-diagonal entries for independent pairs
    np.testing.assert_allclose(np.sort(v), np.sort(w))
    ds2 = build_dataset("SA", {"h1": 0.1, "h2": 0.2}, 120, bank, sigma=0.01)
    _, (r2, c2, v2) = noise_structure(ds2, np.arange(len(ds2.pairs.h)), "gls")
    assert np.any(r2 != c2)        # shared reading -> within-unit correlation
