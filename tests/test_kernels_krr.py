import numpy as np
import pytest

from hamsched import linalg
from hamsched.kernels import Matern52Kernel, SEKernel, gram_grad, predict_grad
from hamsched.krr import GradFit
from hamsched.systems import DoublePendulum, field_to_grad


def _k(kernel, x, y, L):
    rho2 = np.sum(((x - y) / L) ** 2)
    return kernel.phis(np.array(rho2))[0]


@pytest.mark.parametrize("kernel", [SEKernel(), Matern52Kernel()])
def test_gradient_gram_matches_finite_differences(kernel):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 4))
    L = np.array([0.9, 1.3, 0.7, 1.1])
    G = gram_grad(X, L, kernel).reshape(3, 4, 3, 4)
    eps = 1e-4
    for i in range(3):
        for j in range(3):
            if i == j and kernel.name == "matern52":
                continue  # FD across the r=0 kink is inaccurate; diagonal checked analytically below
            for a in range(4):
                for b in range(4):
                    ea, eb = np.eye(4)[a] * eps, np.eye(4)[b] * eps
                    fd = (_k(kernel, X[i] + ea, X[j] + eb, L) - _k(kernel, X[i] + ea, X[j] - eb, L)
                          - _k(kernel, X[i] - ea, X[j] + eb, L) + _k(kernel, X[i] - ea, X[j] - eb, L)) / (4 * eps**2)
                    assert abs(G[i, a, j, b] - fd) < 1e-5
    # diagonal block at r = 0: -phi1(0)/L^2 on the diagonal
    _, p1, _ = kernel.phis(np.array(0.0))
    np.testing.assert_allclose(np.diag(G[0, :, 0, :]), -p1 / L**2)


def test_gram_is_psd_and_prediction_consistent():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 4))
    L = np.ones(4) * 1.2
    G = gram_grad(X, L, SEKernel())
    assert np.linalg.eigvalsh(G).min() > -1e-10
    c = rng.standard_normal((30, 4))
    pred = predict_grad(X, X, L, SEKernel(), c)
    np.testing.assert_allclose(pred.reshape(-1), G @ c.reshape(-1), rtol=1e-10, atol=1e-10)


def test_krr_recovers_double_pendulum_field_from_exact_gradients():
    dp = DoublePendulum()
    rng = np.random.default_rng(2)
    X = dp.sample(400, rng)
    g = dp.grad(X)
    L = 1.2 * X.std(0)
    fit = GradFit(X, SEKernel(), L, ("diag", np.ones(len(X))))
    fac, st = fit.factor(1e-8)
    assert fac is not None
    coef = fit.solve(fac, g)
    Xt = dp.sample(500, rng)
    f_hat = fit.field(coef, Xt)
    err = np.sqrt(np.sum((f_hat - dp.field(Xt)) ** 2) / np.sum(dp.field(Xt) ** 2))
    assert err < 0.06   # measured approximation floor for n=400 exact gradients is ~3-4%


def test_torch_and_scipy_backends_agree():
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    A = rng.standard_normal((50, 50))
    G = A @ A.T + 1e-3 * np.eye(50)
    b = rng.standard_normal(50)
    out = {}
    for be in ("scipy", "torch"):
        linalg.set_backend(be)
        f, st = linalg.factor_regularized(G, lambda M: None, 1.0)
        out[be] = f.solve(b)
    linalg.set_backend(None)
    np.testing.assert_allclose(out["scipy"], out["torch"], rtol=1e-8, atol=1e-10)


def test_targets_are_grad():
    dp = DoublePendulum()
    z = dp.sample(5, np.random.default_rng(4))
    np.testing.assert_allclose(field_to_grad(dp.field(z)), dp.grad(z))


@pytest.mark.parametrize("kernel", [SEKernel(), Matern52Kernel()])
def test_torch_kernel_paths_match_numpy(kernel):
    pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    X = rng.standard_normal((40, 4))
    Xe = rng.standard_normal((25, 4))
    L = np.array([0.8, 1.1, 0.9, 1.4])
    C = rng.standard_normal((40, 4, 3))
    res = {}
    for be in ("scipy", "torch"):
        linalg.set_backend(be)
        res[be] = (gram_grad(X, L, kernel), predict_grad(Xe, X, L, kernel, C))
    linalg.set_backend(None)
    np.testing.assert_allclose(res["scipy"][0], res["torch"][0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res["scipy"][1], res["torch"][1], rtol=1e-11, atol=1e-11)
