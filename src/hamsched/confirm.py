"""Pre-registered confirmatory test of the minimum-interval contrast
(reports/confirmatory_min_interval_protocol_2026-11-19.md).

SA-RICH vs SI-MP and SA-RICH vs SI-DC at h >= h_min, B fixed, locked hyperparameters from the existing
constrained validation locks (no retuning), fresh paired replicates and a fresh held-out test set.
Outputs go to results/confirmatory/<name>/ and never overwrite original results.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .analysis import holm
from .experiment import (Context, PROJECT_ROOT, atomic_write_csv, env_metadata, load_config, load_lock, log,
                         n_bank, required_times, results_dir)
from .metrics import rel_err_path
from .pipelines import evaluate
from .schedules import build_dataset


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def locked_constrained(lock: dict, B: int, a: float, plan: str, est: str, hmin: float) -> dict:
    """(h1, h2, ell, eta) of the constrained validation selection h >= hmin, taken from the lock."""
    b = [x for x in lock["best"] if x["B"] == B and abs(x["a"] - a) < 1e-12 and x["plan"] == plan and x["est"] == est]
    if len(b) != 1:
        raise KeyError((B, a, plan, est))
    h1 = b[0]["constrained"][str(hmin)]["h1"]
    e = [x for x in lock["entries"] if x["B"] == B and abs(x["a"] - a) < 1e-12 and x["plan"] == plan
         and x["est"] == est and abs(x["h1"] - h1) < 1e-9]
    if len(e) != 1:
        raise KeyError((B, a, plan, est, h1))
    return {"h1": e[0]["h1"], "h2": e[0]["h2"], "ell": e[0]["ell"], "eta": e[0]["eta"]}


def min_reading_spacing(ds) -> float:
    """Smallest time gap between consecutive (distinct-time) readings on the same trajectory start."""
    r = ds.readings
    gaps = []
    for s in np.unique(r.start):
        t = np.unique(np.round(r.t[r.start == s], 12))
        if len(t) > 1:
            gaps.append(np.min(np.diff(t)))
    return float(np.min(gaps))


def _params(plan: str, st: dict) -> dict:
    if plan == "SI":
        return {"h": st["h1"]}
    return {"h1": st["h1"], "h2": st["h2"], "r": st["h2"] / st["h1"]}


def run_confirmation(ccfg: dict, workdir: str | None = None) -> pd.DataFrame:
    mcfg = load_config(ccfg["main_config"])
    ctx = Context(mcfg, workdir=workdir)
    lock = load_lock(mcfg["name"])
    if lock["sha256"] != ccfg["lock_sha256"]:
        raise RuntimeError(f"lock sha mismatch: {lock['sha256']} != {ccfg['lock_sha256']}")
    proto = os.path.join(PROJECT_ROOT, ccfg["protocol_file"])
    if not os.path.exists(proto):
        raise RuntimeError("protocol file missing: confirmation refuses to run without a recorded protocol")
    test = ctx._evalset(ccfg["testset_purpose"], ccfg["testset"])
    B, hmin = int(ccfg["B"]), float(ccfg["h_min"])
    out = os.path.join(ctx.root, ccfg["name"])
    os.makedirs(out, exist_ok=True)
    times = required_times(mcfg)
    settings = {(a, p, e): locked_constrained(lock, B, a, p, e, hmin)
                for a in ccfg["noise_levels"] for p, e in ccfg["methods"]}
    t_start = time.time()
    for rep in range(int(ccfg["reps"])):
        shard = os.path.join(out, f"r{rep}.csv")
        if os.path.exists(shard):
            continue
        bank = ctx.bank(ccfg["purpose"], rep, n_bank(mcfg), times)
        rows = []
        for a in ccfg["noise_levels"]:
            for plan, est in ccfg["methods"]:
                st = settings[(a, plan, est)]
                ds = build_dataset(plan, _params(plan, st), B, bank, a * ctx.zrms)
                spacing = min_reading_spacing(ds)
                if spacing < hmin - 1e-9:
                    raise RuntimeError(f"minimum-interval violation: {plan} spacing {spacing} < {hmin}")
                if ds.info["n_readings"] != B:
                    raise RuntimeError(f"budget violation: {ds.info['n_readings']} != {B}")
                t0 = time.time()
                pr, stt = evaluate(ds, [est], ctx.kernel, st["ell"], [st["eta"]], {"test": test.X})
                err = float(rel_err_path(pr[est]["test"], test.F)[0])
                rows.append(dict(rep=rep, a=a, plan=plan, est=est, pipeline=f"{plan}-{est}", h1=st["h1"],
                                 h2=st["h2"], ell=st["ell"], eta=st["eta"], test_err=err, status=int(stt[est][0]),
                                 min_spacing=spacing, fit_time=time.time() - t0, **ds.info))
        atomic_write_csv(pd.DataFrame(rows), shard)
        log(f"{ccfg['name']}: replicate {rep + 1}/{ccfg['reps']} ({(time.time() - t_start) / 60:.1f} min)")
    files = sorted(f for f in os.listdir(out) if f.endswith(".csv"))
    return pd.concat([pd.read_csv(os.path.join(out, f)) for f in files], ignore_index=True)


def _compare(a_err: np.ndarray, b_err: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    fa, fb = ~np.isfinite(a_err), ~np.isfinite(b_err)
    d = np.log(np.where(fa, np.inf, a_err)) - np.log(np.where(fb, np.inf, b_err))
    fin = np.isfinite(d)
    dd = d[fin]
    rng = np.random.default_rng(seed)
    bs = dd[rng.integers(0, len(dd), (n_boot, len(dd)))].mean(axis=1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    dw = np.where(np.isfinite(d), d, np.sign(d) * 1e6)
    p = float(wilcoxon(dw, alternative="two-sided").pvalue)
    return dict(n=int(len(d)), n_fail_first=int(fa.sum()), n_fail_second=int(fb.sum()),
                gm_ratio=float(np.exp(dd.mean())), ci_lo=float(np.exp(lo)), ci_hi=float(np.exp(hi)),
                wins_first=int(np.sum(d < 0)), p_wilcoxon=p)


def analyze_confirmation(ccfg: dict, raw: pd.DataFrame, t_run_start: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    for a in ccfg["noise_levels"]:
        sub = raw[np.isclose(raw.a, a)]
        piv = sub.pivot(index="rep", columns="pipeline", values="test_err")
        fail = sub.pivot(index="rep", columns="pipeline", values="status") == 2
        for first, second, direction in ccfg["comparisons"]:
            ea = np.where(fail[first], np.nan, piv[first].values)
            eb = np.where(fail[second], np.nan, piv[second].values)
            r = _compare(ea, eb, int(ccfg["bootstrap"]))
            rows.append(dict(a=a, comparison=f"{first} / {second}", predicted=("< 1" if direction == "less" else "> 1"),
                             **r))
    comp = pd.DataFrame(rows)
    comp["p_holm"] = holm(comp.p_wilcoxon.values)
    alpha = float(ccfg["alpha"])

    def decide(r):
        pred_dir = r.predicted == "< 1"
        in_dir = (r.gm_ratio < 1) if pred_dir else (r.gm_ratio > 1)
        if r.p_holm < alpha:
            return "supported" if in_dir else "contradicted"
        return "not supported (inconclusive)"
    comp["decision"] = comp.apply(decide, axis=1)
    proto = os.path.join(PROJECT_ROOT, ccfg["protocol_file"])
    summary = {
        "protocol_file": ccfg["protocol_file"], "protocol_sha256": _sha256(proto),
        "protocol_mtime_utc": _dt.datetime.fromtimestamp(os.path.getmtime(proto), _dt.timezone.utc).isoformat(),
        "lock_sha256": ccfg["lock_sha256"], "run_started_utc": t_run_start,
        "analysis_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "min_spacing_all_rows": float(raw.min_spacing.min()), "h_min": float(ccfg["h_min"]),
        "n_readings_all_rows": sorted(raw.n_readings.unique().tolist()),
        "n_rows": int(len(raw)), "n_failures": int((raw.status == 2).sum() + raw.test_err.isna().sum()),
        "n_jitter": int((raw.status == 1).sum()), "deviations": ccfg.get("deviations", []), "env": env_metadata(),
    }
    return comp, summary


def confirm(cfg_path: str, workdir: str | None = None):
    ccfg = load_config(cfg_path)
    t_run_start = _dt.datetime.now(_dt.timezone.utc).isoformat()
    raw = run_confirmation(ccfg, workdir)
    comp, summary = analyze_confirmation(ccfg, raw, t_run_start)
    d = os.path.join(results_dir(), "confirmatory", ccfg["name"])
    os.makedirs(d, exist_ok=True)
    atomic_write_csv(raw, os.path.join(d, "raw.csv"))
    atomic_write_csv(comp, os.path.join(d, "comparisons.csv"))
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return comp, summary


def posthoc_from_original(name: str = "main_dp", B: int = 1800, hmins=(0.2, 0.4)) -> pd.DataFrame:
    """The ORIGINAL post-hoc reading (not confirmatory): SA-RICH vs SI-MP / SI-DC under h >= h_min, computed
    from the original Phase B test rows and the pre-registered constrained locks. Saved separately."""
    from .analysis import paired
    t = pd.read_csv(os.path.join(results_dir(), f"{name}_test.csv"))
    lock = load_lock(name)
    rows = []
    for Bv in sorted(t.B.unique()):
        for a in sorted(t[t.B == Bv].a.unique()):
            for hm in (None,) + tuple(hmins):
                sel = {}
                for plan, est in (("SA", "RICH"), ("SI", "MP"), ("SI", "DC")):
                    b = [x for x in lock["best"] if x["B"] == Bv and abs(x["a"] - a) < 1e-12
                         and x["plan"] == plan and x["est"] == est][0]
                    sel[(plan, est)] = b["h1"] if hm is None else b["constrained"].get(str(hm), {}).get("h1")
                get = lambda p, e: t[(t.B == Bv) & np.isclose(t.a, a) & (t.plan == p) & (t.est == e)
                                     & np.isclose(t.h1, sel[(p, e)])]
                ra, rm, rd = get("SA", "RICH"), get("SI", "MP"), get("SI", "DC")
                rec = dict(B=int(Bv), a=float(a), h_min=0.0 if hm is None else hm, h_SA=sel[("SA", "RICH")],
                           h_SI_MP=sel[("SI", "MP")], h_SI_DC=sel[("SI", "DC")])
                for tag, other in (("MP", rm), ("DC", rd)):
                    p = paired(ra, other)
                    rec.update({f"SA_over_SI_{tag}": p["ratio"], f"lo_{tag}": p["ratio_lo"], f"hi_{tag}": p["ratio_hi"]})
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.insert(0, "analysis", "post hoc (not pre-registered; no multiplicity adjustment)")
    p = os.path.join(results_dir(), "analysis", f"{name}_posthoc_min_interval.csv")
    df.to_csv(p, index=False)
    return df
