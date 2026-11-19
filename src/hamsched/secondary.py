"""Summaries (tables + figures) of the secondary studies, computed only from results CSVs."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .analysis import boot_ci, md_table, paired
from .experiment import PROJECT_ROOT, results_dir
from .theory import variance_factor


def _rd(name):
    p = os.path.join(results_dir(), name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _out(name, df, cols=None):
    adir = os.path.join(results_dir(), "analysis")
    tdir = (os.path.join(results_dir(), "tables") if os.environ.get("HAMSCHED_RESULTS")
            else os.path.join(PROJECT_ROOT, "reports", "tables"))
    os.makedirs(adir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    df.to_csv(os.path.join(adir, f"{name}.csv"), index=False)
    with open(os.path.join(tdir, f"{name}.md"), "w") as f:
        f.write(md_table(df, cols or list(df.columns)))


def hh_summary(name="hh_anchor"):
    d = _rd(f"{name}_hh_anchor.csv")
    if d is None:
        return None
    rows = []
    for (setting, plan, est), s in d.groupby(["setting", "plan", "est"]):
        s = s.sort_values("delta")
        ok = np.isfinite(s.test_err) & (s.test_err > 0)
        slope = np.polyfit(np.log(s.delta[ok]), np.log(s.test_err[ok]), 1)[0] if ok.sum() > 2 else np.nan
        rows.append(dict(setting=setting, pipeline=f"{plan}-{est}", slope=slope,
                         err_min=float(s.test_err.min()), err_at_0p1=float(s[np.abs(s.delta - 0.1) < 1e-9].test_err.iloc[0]),
                         err_at_0p3=float(s[np.abs(s.delta - 0.3) < 1e-9].test_err.iloc[0])))
    df = pd.DataFrame(rows)
    _out("hh_anchor_summary", df)
    return df


def decomposition_summary(name="main_dp"):
    d = _rd(f"{name}_decomp.csv")
    if d is None:
        return None
    agg = {"floor": "mean", "disc": "mean", "me_raw": "mean", "me2_corr": "mean", "var": "mean", "bias": "mean",
           "total_rmse": "mean", "cross": "mean", "fails": "sum", "n_draws": "sum", "h1": "first", "n_starts": "first"}
    for c in ["var_short", "var_long", "cov_short_long", "corr_short_long", "naive_sub_me_raw", "simex_q_me_raw",
              "simex_l_me_raw", "naive_sub_me2_corr", "simex_q_me2_corr", "simex_l_me2_corr", "naive_sub_rmse",
              "simex_q_rmse", "simex_l_rmse", "naive_sub_var", "simex_q_var"]:
        if c in d:
            agg[c] = "mean"
    g = d.groupby(["B", "a", "group", "name"]).agg(agg).reset_index()
    g["me_corr"] = np.sqrt(np.maximum(g.me2_corr, 0))
    g["sd"] = np.sqrt(g["var"])
    sv = _rd(f"{name}_decomp_startvar.csv")
    if sv is not None:
        g = g.merge(sv, on=["B", "a", "name"], how="left")
        g["start_sd"] = np.sqrt(g.start_var)
    head = g[g.group == "headline"]
    _out("decomp_headline", head, [c for c in ["B", "a", "name", "h1", "n_starts", "floor", "disc", "me_raw", "me_corr",
                                               "sd", "start_sd", "total_rmse", "fails"] if c in head])
    if "simex_q_me_raw" in g:
        sx = g[g.simex_q_me_raw.notna()]
        _out("decomp_simex", sx, ["a", "name", "h1", "naive_sub_me_raw", "simex_q_me_raw", "simex_l_me_raw",
                                  "naive_sub_rmse", "simex_q_rmse", "simex_l_rmse"])
    mech = g[g.group == "mechanism"].copy()
    if len(mech):
        theory = {"m-SI(h1)": variance_factor("SI", "MP"), "m-SI(2h1)": variance_factor("SI2", "MP"),
                  "m-SA-RICH": variance_factor("SA", "RICH"), "m-IP-RICH": variance_factor("IP", "RICH"),
                  "m-RA-RICH": variance_factor("RA", "RICH"), "m-MS3-RICH": variance_factor("MS3", "RICH")}
        rows = []
        for (B, a), s in mech.groupby(["B", "a"]):
            ref = float(s[s.name == "m-SI(2h1)"]["var"].iloc[0])
            for _, r in s.iterrows():
                rows.append(dict(B=B, a=a, name=r["name"], var_rel_to_SI2h1=r["var"] / ref,
                                 theory_rel=theory.get(r["name"], np.nan) / theory["m-SI(2h1)"],
                                 corr_short_long=r.get("corr_short_long", np.nan), floor=r["floor"], disc=r["disc"],
                                 sd=r["sd"], total_rmse=r["total_rmse"]))
            v_ms3 = float(s[s.name == "m-MS3-RICH"]["var"].iloc[0]) if (s.name == "m-MS3-RICH").any() else np.nan
            v_ms3i = float(s[s.name == "m-MS3I-RICH"]["var"].iloc[0]) if (s.name == "m-MS3I-RICH").any() else np.nan
            rows.append(dict(B=B, a=a, name="MS3I/MS3 per-unit variance (Hu Sec 6.3 protocol vs shared readings)",
                             var_rel_to_SI2h1=v_ms3i / v_ms3, theory_rel=0.944444 / 0.5))
        _out("decomp_mechanism", pd.DataFrame(rows))
    return g


def ablation_summary(name="main_dp"):
    d = _rd(f"{name}_ablate_test.csv")
    if d is None:
        return None
    rows = []
    for (B, a, plan, est), s in d.groupby(["B", "a", "plan", "est"]):
        rec = dict(B=B, a=a, pipeline=f"{plan}-{est}")
        for hc in ("h_naive", "h_simex"):
            for var in ("naive", "simex_q", "simex_l"):
                ss = s[(s.h_choice == hc) & (s.variant == var)]
                rec[f"{hc}:{var}"] = float(np.nanmedian(ss.test_err)) if len(ss) else np.nan
            rec[f"{hc}:h1"] = float(s[s.h_choice == hc].h1.iloc[0])
        a_ = s[(s.h_choice == "h_naive") & (s.variant == "simex_q")]
        b_ = s[(s.h_choice == "h_naive") & (s.variant == "naive")]
        p = paired(a_, b_)
        rec.update(simexq_vs_naive_ratio=p["ratio"], ratio_lo=p["ratio_lo"], ratio_hi=p["ratio_hi"])
        a2 = s[(s.h_choice == "h_simex") & (s.variant == "simex_q")]
        p2 = paired(a2, b_)
        rec.update(retuned_simexq_vs_naive_ratio=p2["ratio"], r2_lo=p2["ratio_lo"], r2_hi=p2["ratio_hi"])
        rows.append(rec)
    df = pd.DataFrame(rows)
    _out("ablation_simex_2x2", df)
    return df


def misspec_summary(name="main_dp"):
    d = _rd(f"{name}_misspec.csv")
    if d is None:
        return None
    rows = []
    base = d[(d.scenario == "correct") & (d.variant == "naive")]
    for (sc, kap, plan, est, var), s in d.groupby(["scenario", "kappa", "plan", "est", "variant"]):
        b = base[(base.plan == plan) & (base.est == est)]
        p = paired(s, b) if len(b) else {"ratio": np.nan, "ratio_lo": np.nan, "ratio_hi": np.nan}
        rows.append(dict(scenario=sc, kappa=kap, pipeline=f"{plan}-{est}", variant=var,
                         median=float(np.nanmedian(s.test_err)), ratio_vs_correct_naive=p["ratio"],
                         lo=p["ratio_lo"], hi=p["ratio_hi"], n=len(s)))
    df = pd.DataFrame(rows).sort_values(["scenario", "pipeline", "variant", "kappa"])
    _out("misspec_summary", df)
    return df


def calibration_summary(name="main_dp"):
    d = _rd(f"{name}_calibrate.csv")
    if d is None:
        return None
    rows = []
    for a, s in d.groupby("a"):
        orac = s[s.method.str.startswith("oracle_")]
        best_or = orac.groupby("rep").test_err.min().rename("oracle_best")
        for m, sm in s[s.method != "candidate"].groupby("method"):
            e = sm.set_index("rep").test_err
            reg = (e / best_or.reindex(e.index))
            lo, hi = boot_ci(np.log(reg.values))
            sel = sm.apply(lambda r: f"{r.plan}-{r.est}@{r.h1:g}", axis=1).value_counts().to_dict()
            rows.append(dict(a=a, method=m, median_err=float(np.nanmedian(e)), mean_regret=float(np.exp(np.nanmean(np.log(reg)))),
                             regret_lo=float(np.exp(lo)), regret_hi=float(np.exp(hi)), n=len(sm), selections=str(sel)))
        cands = s[s.method == "candidate"]
        if len(cands):
            from scipy.stats import spearmanr
            rhos = [spearmanr(c.pred_err, c.test_err).correlation for _, c in cands.groupby("rep") if c.test_err.notna().sum() > 2]
            rows.append(dict(a=a, method="pred_vs_realized_spearman", median_err=float(np.nanmedian(rhos)), n=len(rhos)))
            for ci, c in cands.groupby("cand"):
                rows.append(dict(a=a, method=f"candidate {c.plan.iloc[0]}-{c.est.iloc[0]}@{c.h1.iloc[0]:g}",
                                 median_err=float(np.nanmedian(c.test_err)), mean_regret=float(np.nanmedian(c.pred_err)),
                                 n=len(c), selections="(mean_regret column = median predicted error)"))
    df = pd.DataFrame(rows)
    _out("calibration_summary", df)
    return df


def robust_summary(name="main_dp"):
    rows = []
    for nm in ["robust_ratio_r15", "robust_ratio_r3", "robust_matern", "robust_unstr"]:
        t = _rd(f"{nm}_test.csv")
        if t is None:
            continue
        import json
        lock = json.load(open(os.path.join(results_dir(), "locks", f"{nm}.json")))
        for b in lock["best"]:
            s = t[(t.plan == b["plan"]) & (t.est == b["est"]) & (np.abs(t.h1 - b["h1"]) < 1e-9)]
            rows.append(dict(experiment=nm, pipeline=f"{b['plan']}-{b['est']}", h1=b["h1"], h2=b["h2"],
                             median=float(np.nanmedian(s.test_err)), n=len(s)))
        if nm.startswith("robust_ratio"):
            for b in lock["best"]:
                if b["plan"] == "SA":
                    sa = t[(t.plan == "SA") & (np.abs(t.h1 - b["h1"]) < 1e-9)]
                    ip = t[(t.plan == "IP") & (np.abs(t.h1 - b["h1"]) < 1e-9)]
                    p = paired(sa, ip)
                    rows.append(dict(experiment=nm, pipeline="SA-RICH / IP-RICH (same h)", h1=b["h1"], h2=b["h2"],
                                     median=p["ratio"], n=p["n"], lo=p["ratio_lo"], hi=p["ratio_hi"]))
    tu = _rd("robust_unstr_test.csv")
    if tu is not None:
        import json as _json
        lk = _json.load(open(os.path.join(results_dir(), "locks", "robust_unstr.json")))
        sel = {b["plan"]: (b["est"], b["h1"]) for b in lk["best"]}
        rows_u = {p: tu[(tu.plan == p) & (tu.est == e) & (np.abs(tu.h1 - h) < 1e-9)] for p, (e, h) in sel.items()}
        for pa, pb in (("SA", "SI"), ("MS3", "SI"), ("MS3", "SA")):
            if pa in rows_u and pb in rows_u:
                pr = paired(rows_u[pa], rows_u[pb])
                rows.append(dict(experiment="robust_unstr", pipeline=f"{pa}-{sel[pa][0]} / {pb}-{sel[pb][0]} (locked)",
                                 h1=sel[pa][1], median=pr["ratio"], n=pr["n"], lo=pr["ratio_lo"], hi=pr["ratio_hi"]))
    hf = _rd(f"{name}_hufixed.csv")
    if hf is not None:
        for h, s in hf.groupby("h1"):
            rows.append(dict(experiment="hu_fixed (Matern-5/2, ell=3 raw, eta=1e-9)", pipeline="SI-MP", h1=h,
                             median=float(np.nanmedian(s.test_err)), n=len(s)))
    df = pd.DataFrame(rows)
    if len(df):
        _out("robustness_summary", df)
    return df


def all_secondary(name="main_dp", hh_name="hh_anchor"):
    out = {}
    calls = [(hh_summary, (hh_name,)), (decomposition_summary, (name,)), (ablation_summary, (name,)),
             (misspec_summary, (name,)), (calibration_summary, (name,)), (robust_summary, (name,))]
    for f, args in calls:
        try:
            out[f.__name__] = f(*args)
        except Exception as e:  # report but continue
            print(f"{f.__name__} failed: {e!r}")
    return out
