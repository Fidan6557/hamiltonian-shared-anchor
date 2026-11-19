"""Tables and pre-registered contrasts from results CSVs (no model fitting here)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .experiment import PROJECT_ROOT, load_lock, results_dir

THREE_READING = [("SA", "RICH"), ("MS3", "RICH")]
SINGLE = [("SI", "MP"), ("SI", "DC")]


def _rd(name):
    p = os.path.join(results_dir(), name)
    return pd.read_csv(p) if os.path.exists(p) else None


def boot_ci(x, n=4000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = stat(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return tuple(np.quantile(bs, [0.025, 0.975]))


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame, col="test_err"):
    """Paired log-ratio log(err_a/err_b) over replicates; failures count as +inf (rank stats)."""
    m = a[["rep", col]].merge(b[["rep", col]], on="rep", suffixes=("_a", "_b"))
    ea, eb = m[f"{col}_a"].astype(float).values, m[f"{col}_b"].astype(float).values
    fa, fb = ~np.isfinite(ea), ~np.isfinite(eb)
    d = np.log(np.where(fa, np.inf, ea)) - np.log(np.where(fb, np.inf, eb))
    fin = np.isfinite(d)
    res = dict(n=len(d), n_fail_a=int(fa.sum()), n_fail_b=int(fb.sum()),
               mean_logratio=float(np.mean(d[fin])) if fin.any() else np.nan,
               median_logratio=float(np.median(d[fin])) if fin.any() else np.nan,
               win_frac_a=float(np.mean(np.where(fa & fb, 0.5, (ea < eb) | (fb & ~fa)))))
    lo, hi = boot_ci(d[fin])
    res["ci_lo"], res["ci_hi"] = lo, hi
    res["ratio"] = float(np.exp(res["mean_logratio"])) if fin.any() else np.nan
    res["ratio_lo"], res["ratio_hi"] = float(np.exp(lo)), float(np.exp(hi))
    dd = np.where(np.isfinite(d), d, np.sign(d) * 1e6)
    try:
        res["p_wilcoxon"] = float(wilcoxon(dd).pvalue) if np.any(dd != 0) and len(dd) >= 5 else np.nan
    except ValueError:
        res["p_wilcoxon"] = np.nan
    return res


def _rows(test, B, a, plan, est, h1):
    return test[(test.B == B) & (np.abs(test.a - a) < 1e-12) & (test.plan == plan) & (test.est == est)
                & (np.abs(test.h1 - h1) < 1e-9)]


def _best(lock, B, a, plan, est):
    for b in lock["best"]:
        if b["B"] == B and abs(b["a"] - a) < 1e-12 and b["plan"] == plan and b["est"] == est:
            return b
    return None


def _pick(lock, B, a, family, hmin=None):
    """Validation-only choice within a family (lowest val log-error, optionally under h >= hmin)."""
    best = None
    for plan, est in family:
        b = _best(lock, B, a, plan, est)
        if b is None:
            continue
        if hmin is None:
            h1, s = b["h1"], b["val_logerr"]
        else:
            c = b["constrained"].get(str(hmin))
            if c is None:
                continue
            h1, s = c["h1"], c["val_logerr"]
        if best is None or s < best[3]:
            best = (plan, est, h1, s)
    return best


def summarize(name: str, out_dir: str | None = None):
    out_dir = out_dir or os.path.join(results_dir(), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    test = _rd(f"{name}_test.csv")
    lock = load_lock(name)
    cells = sorted({(b["B"], b["a"]) for b in lock["best"]})

    # ---- 1. locked-best summary per pipeline
    rows = []
    for b in lock["best"]:
        sub = _rows(test, b["B"], b["a"], b["plan"], b["est"], b["h1"])
        e = sub.test_err.astype(float).values
        fin = e[np.isfinite(e)]
        lo, hi = boot_ci(fin, stat=np.median)
        rows.append(dict(B=b["B"], a=b["a"], plan=b["plan"], est=b["est"], h1=b["h1"], h2=b["h2"], ell=b["ell"],
                         eta=b["eta"], edge=b["edge"], val_err=np.exp(b["val_logerr"]), n=len(e),
                         n_fail=int((~np.isfinite(e)).sum() + (sub.status == 2).sum()),
                         median=float(np.median(fin)) if len(fin) else np.nan, med_lo=lo, med_hi=hi,
                         mean=float(np.mean(fin)) if len(fin) else np.nan,
                         n_starts=int(sub.n_starts.iloc[0]) if len(sub) else np.nan,
                         n_readings=int(sub.n_readings.iloc[0]) if len(sub) else np.nan,
                         n_pairs=int(sub.n_pairs.iloc[0]) if len(sub) else np.nan,
                         max_duration=float(sub.max_duration.iloc[0]) if len(sub) else np.nan,
                         total_duration=float(sub.total_duration.iloc[0]) if len(sub) else np.nan,
                         err_E0=float(np.nanmedian(sub.err_E0)) if len(sub) else np.nan,
                         err_E1=float(np.nanmedian(sub.err_E1)) if len(sub) else np.nan,
                         err_E2=float(np.nanmedian(sub.err_E2)) if len(sub) else np.nan))
    summ = pd.DataFrame(rows).sort_values(["B", "a", "median"])
    summ.to_csv(os.path.join(out_dir, f"{name}_best_summary.csv"), index=False)

    # ---- 2. primary contrasts
    con = []
    for (B, a) in cells:
        # P1: SA-RICH vs IP-RICH at identical (h1, h2) = SA-RICH locked h1
        b = _best(lock, B, a, "SA", "RICH")
        if b is not None:
            ra, rb = _rows(test, B, a, "SA", "RICH", b["h1"]), _rows(test, B, a, "IP", "RICH", b["h1"])
            if len(ra) and len(rb):
                con.append(dict(contrast="P1_SA-RICH_vs_IP-RICH_same_h", B=B, a=a, a_pipe=f"SA-RICH@{b['h1']:g}",
                                b_pipe=f"IP-RICH@{b['h1']:g}", **paired(ra, rb)))
        # P2: SA-RICH (locked) vs tuned SI (SI-MP locked)
        bs = _best(lock, B, a, "SI", "MP")
        if b is not None and bs is not None:
            ra, rb = _rows(test, B, a, "SA", "RICH", b["h1"]), _rows(test, B, a, "SI", "MP", bs["h1"])
            if len(ra) and len(rb):
                con.append(dict(contrast="P2_SA-RICH_vs_tuned_SI", B=B, a=a, a_pipe=f"SA-RICH@{b['h1']:g}",
                                b_pipe=f"SI-MP@{bs['h1']:g}", **paired(ra, rb)))
        # P3 / P4: best 3-reading vs best single-interval (validation choice within family)
        for hm in [None] + list(_hmins(lock)):
            p3 = _pick(lock, B, a, THREE_READING, hm)
            p1 = _pick(lock, B, a, SINGLE, hm)
            if p3 is None or p1 is None:
                continue
            ra, rb = _rows(test, B, a, p3[0], p3[1], p3[2]), _rows(test, B, a, p1[0], p1[1], p1[2])
            if len(ra) and len(rb):
                con.append(dict(contrast="P3_best3_vs_bestSI" if hm is None else f"P4_best3_vs_bestSI_hmin{hm:g}",
                                B=B, a=a, a_pipe=f"{p3[0]}-{p3[1]}@{p3[2]:g}", b_pipe=f"{p1[0]}-{p1[1]}@{p1[2]:g}",
                                **paired(ra, rb)))
    con = pd.DataFrame(con)
    if len(con):
        con["p_holm"] = np.nan
        for c, sub in con.groupby("contrast"):
            con.loc[sub.index, "p_holm"] = holm(sub.p_wilcoxon.values)
    con.to_csv(os.path.join(out_dir, f"{name}_contrasts.csv"), index=False)

    # ---- 3. descriptive: SA vs IP (RICH) at every common h1; IP -> RA -> SA decomposition
    desc = []
    for (B, a) in cells:
        for h1 in sorted(test[(test.B == B) & (np.abs(test.a - a) < 1e-12) & (test.plan == "SA") & (test.est == "RICH")].h1.unique()):
            rs = _rows(test, B, a, "SA", "RICH", h1)
            for other in (("IP", "RICH"), ("RA", "RICH")):
                ro = _rows(test, B, a, *other, h1)
                if len(rs) and len(ro):
                    desc.append(dict(B=B, a=a, h1=h1, a_pipe="SA-RICH", b_pipe="-".join(other), **paired(rs, ro)))
            ri, rr = _rows(test, B, a, "IP", "RICH", h1), _rows(test, B, a, "RA", "RICH", h1)
            if len(ri) and len(rr):
                desc.append(dict(B=B, a=a, h1=h1, a_pipe="RA-RICH", b_pipe="IP-RICH", **paired(rr, ri)))
            for other in (("SA", "POOL"), ("SA", "LONG")):
                ro = _rows(test, B, a, *other, h1)
                if len(rs) and len(ro):
                    desc.append(dict(B=B, a=a, h1=h1, a_pipe="SA-RICH", b_pipe="-".join(other), **paired(rs, ro)))
            rm = _rows(test, B, a, "MS3", "RICH", h1)
            if len(rs) and len(rm):
                desc.append(dict(B=B, a=a, h1=h1, a_pipe="MS3-RICH", b_pipe="SA-RICH", **paired(rm, rs)))
    pd.DataFrame(desc).to_csv(os.path.join(out_dir, f"{name}_sa_ip_ra_by_h.csv"), index=False)

    # ---- 4. SI-DC vs SI-MP (2x2 for the discretization correction: h tuned naive vs re-tuned)
    dc = []
    for (B, a) in cells:
        bm, bd = _best(lock, B, a, "SI", "MP"), _best(lock, B, a, "SI", "DC")
        if bm is None or bd is None:
            continue
        for hname, h in (("h_naive", bm["h1"]), ("h_dc", bd["h1"])):
            r_off, r_on = _rows(test, B, a, "SI", "MP", h), _rows(test, B, a, "SI", "DC", h)
            if len(r_off) and len(r_on):
                dc.append(dict(B=B, a=a, h_choice=hname, h=h, med_off=float(np.nanmedian(r_off.test_err)),
                               med_on=float(np.nanmedian(r_on.test_err)), **paired(r_on, r_off)))
    pd.DataFrame(dc).to_csv(os.path.join(out_dir, f"{name}_dc_2x2.csv"), index=False)

    # ---- 5. neighbor regret (knife-edge check) and failures
    nb = []
    for b in lock["best"]:
        sub = test[(test.B == b["B"]) & (np.abs(test.a - b["a"]) < 1e-12) & (test.plan == b["plan"]) & (test.est == b["est"])]
        hs = sorted(sub.h1.unique())
        if b["h1"] not in hs:
            continue
        i = hs.index(b["h1"])
        rec = dict(B=b["B"], a=b["a"], plan=b["plan"], est=b["est"], h1=b["h1"])
        for k, lab in ((-1, "minus"), (0, "at"), (1, "plus")):
            if 0 <= i + k < len(hs):
                rec[f"med_{lab}"] = float(np.nanmedian(sub[np.abs(sub.h1 - hs[i + k]) < 1e-9].test_err))
        nb.append(rec)
    pd.DataFrame(nb).to_csv(os.path.join(out_dir, f"{name}_neighbor_regret.csv"), index=False)
    fails = test.assign(fail=(test.status == 2) | ~np.isfinite(test.test_err.astype(float)),
                        jit=test.status == 1).groupby(["plan", "est"]).agg(
        n=("fail", "size"), n_fail=("fail", "sum"), n_jitter=("jit", "sum")).reset_index()
    fails.to_csv(os.path.join(out_dir, f"{name}_failures.csv"), index=False)
    tune = _rd(f"{name}_tune.csv")
    if tune is not None:
        tf = tune.assign(fail=tune.status == 2, jit=tune.status == 1).groupby(["plan", "est"]).agg(
            n=("fail", "size"), n_fail=("fail", "sum"), n_jitter=("jit", "sum")).reset_index()
        tf.to_csv(os.path.join(out_dir, f"{name}_failures_tune.csv"), index=False)
    # ---- lock ordering check
    meta_p = os.path.join(results_dir(), f"{name}_test_meta.json")
    chk = {"lock_sha256": lock["sha256"], "lock_created_utc": lock["created_utc"],
           "test_rows_with_lock_sha": int((test.lock_sha == lock["sha256"]).sum()), "test_rows": int(len(test))}
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        chk["test_started_utc"] = meta.get("time_utc")
        chk["lock_predates_test"] = bool(meta.get("time_utc", "") > lock["created_utc"])
    json.dump(chk, open(os.path.join(out_dir, f"{name}_lock_check.json"), "w"), indent=1)
    return summ, con


def _hmins(lock):
    hs = set()
    for b in lock["best"]:
        hs |= {float(k) for k in b["constrained"]}
    return sorted(hs)


def md_table(df: pd.DataFrame, cols, floatfmt="{:.3g}") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_md_tables(name: str):
    adir = os.path.join(results_dir(), "analysis")
    tdir = (os.path.join(results_dir(), "tables") if os.environ.get("HAMSCHED_RESULTS")
            else os.path.join(PROJECT_ROOT, "reports", "tables"))
    os.makedirs(tdir, exist_ok=True)
    s = pd.read_csv(os.path.join(adir, f"{name}_best_summary.csv"))
    s["pipeline"] = s.plan + "-" + s.est
    with open(os.path.join(tdir, f"{name}_best_summary.md"), "w") as f:
        f.write(md_table(s, ["B", "a", "pipeline", "h1", "n_starts", "n_pairs", "max_duration", "median", "med_lo",
                             "med_hi", "n_fail", "edge"]))
    cp = os.path.join(adir, f"{name}_contrasts.csv")
    if os.path.getsize(cp) < 5:          # no pre-registered contrast applies (e.g. robustness sub-experiments)
        return
    c = pd.read_csv(cp)
    with open(os.path.join(tdir, f"{name}_contrasts.md"), "w") as f:
        f.write(md_table(c, ["contrast", "B", "a", "a_pipe", "b_pipe", "ratio", "ratio_lo", "ratio_hi", "win_frac_a",
                             "p_holm", "n_fail_a", "n_fail_b"]))


def calibration_trigger(name: str, defaults: dict, budget: int = 1800, threshold: float = 1.10) -> dict:
    """Pre-registered trigger: some fixed default (plan, est, h) has >= 10% median regret vs the
    oracle-best locked pipeline (lowest median test error among locked bests) in a B = budget cell."""
    test = _rd(f"{name}_test.csv")
    lock = load_lock(name)
    out = {"threshold": threshold, "cells": []}
    for a in sorted({b["a"] for b in lock["best"] if b["B"] == budget}):
        meds = {}
        for b in lock["best"]:
            if b["B"] == budget and abs(b["a"] - a) < 1e-12:
                s = _rows(test, budget, a, b["plan"], b["est"], b["h1"])
                if len(s):
                    meds[f"{b['plan']}-{b['est']}"] = float(np.nanmedian(s.test_err))
        best = min(meds.values())
        cell = {"a": a, "oracle_best": best, "oracle_pipeline": min(meds, key=meds.get), "defaults": {}}
        for dname, d in defaults.items():
            s = _rows(test, budget, a, d["plan"], d["est"], d["h1"])
            if len(s):
                cell["defaults"][dname] = float(np.nanmedian(s.test_err)) / best
        out["cells"].append(cell)
    out["triggered"] = any(v >= threshold for c in out["cells"] for v in c["defaults"].values())
    with open(os.path.join(results_dir(), "analysis", f"{name}_calibration_trigger.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
