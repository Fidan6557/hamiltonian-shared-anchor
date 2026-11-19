"""Grid extensions must not perturb readings already used, and tuning fill-in must only append."""
import copy
import os

import numpy as np
import pandas as pd

from hamsched.experiment import Context, jobs_for, load_config, run_phase, tune_job

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_bank_extension_preserves_existing_draws(tmp_path):
    cfg = load_config(os.path.join(ROOT, "configs", "smoke.yaml"))
    ctx = Context(cfg, workdir=str(tmp_path), threads=2)
    b1 = ctx.bank("tune", 0, 30, [0.1, 0.2])
    b2 = ctx.bank("tune", 0, 30, [0.1, 0.2, 0.4])
    assert len(b2.times) == 4
    for t in (0.0, 0.1, 0.2):
        i, j = b1.tidx(t), b2.tidx(t)
        np.testing.assert_array_equal(b1.X[:, i], b2.X[:, j])
        np.testing.assert_array_equal(b1.xi[:, i], b2.xi[:, j])
    np.testing.assert_allclose(ctx.system.H(b2.X[:, b2.tidx(0.4)]), ctx.system.H(b2.z0), atol=1e-9)


def test_tuning_fill_in_only_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("HAMSCHED_RESULTS", str(tmp_path / "results"))
    cfg = load_config(os.path.join(ROOT, "configs", "smoke.yaml"))
    cfg["plans"] = [{"plan": "SI", "estimators": ["MP"]}]
    cfg["h_grid"] = [0.1, 0.2]
    ctx = Context(cfg, workdir=str(tmp_path), threads=2)
    d1 = run_phase(ctx, "tune", tune_job, jobs_for(cfg, 1), fill_in=True, purpose="tune")
    cfg2 = copy.deepcopy(cfg)
    cfg2["h_grid"] = [0.1, 0.2, 0.4]
    ctx2 = Context(cfg2, workdir=str(tmp_path), threads=2)
    d2 = run_phase(ctx2, "tune", tune_job, jobs_for(cfg2, 1), fill_in=True, purpose="tune")
    assert len(d2) == len(d1) * 3 // 2
    merged = d2.merge(d1, on=["plan", "h1", "est", "ell", "eta"], suffixes=("", "_old"))
    np.testing.assert_allclose(merged.val_err, merged.val_err_old)


def test_tuning_fill_in_adds_missing_etas_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HAMSCHED_RESULTS", str(tmp_path / "results"))
    cfg = load_config(os.path.join(ROOT, "configs", "smoke.yaml"))
    cfg["plans"] = [{"plan": "SI", "estimators": ["MP"]}]
    cfg["h_grid"] = [0.1, 0.2]
    cfg["etas"] = [1e-6]
    ctx = Context(cfg, workdir=str(tmp_path), threads=2)
    d1 = run_phase(ctx, "tune", tune_job, jobs_for(cfg, 1), fill_in=True, purpose="tune")
    cfg2 = copy.deepcopy(cfg)
    cfg2["etas"] = [1e-6, 1e-4]
    d2 = run_phase(Context(cfg2, workdir=str(tmp_path), threads=2), "tune", tune_job, jobs_for(cfg2, 1),
                   fill_in=True, purpose="tune")
    assert len(d2) == 2 * len(d1)
    assert d2.groupby(["h1", "eta"]).size().max() == 1
