"""Figures generated only from results CSVs/JSON (static PNG for the report).

Color follows the entity (fixed slot per pipeline, reference categorical palette, light mode), with
marker + line style as secondary encoding so identity never relies on color alone.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .experiment import PROJECT_ROOT, load_lock, results_dir  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8985"
STYLE = {  # pipeline -> (color, marker, linestyle)
    "SI-MP": ("#2a78d6", "o", "-"),
    "SA-RICH": ("#eb6834", "s", "-"),
    "MS3-RICH": ("#1baf7a", "D", "-"),
    "IP-RICH": ("#eda100", "^", "--"),
    "RA-RICH": ("#e87ba4", "v", "--"),
    "SI-DC": ("#008300", "o", "--"),
    "SA-POOL": ("#6250d6", "P", ":"),
    "IP-POOL": ("#e34948", "X", ":"),
    "SA-LONG": (NEUTRAL, "s", "-."),
    "SA-GLS": ("#6250d6", "*", "-."),
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5,
        "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5,
    })


def _pct_logy(ax):
    """Log y-axis with percent labels at 1-2-5 positions; minor labels hidden (no collisions)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _ratio_logy(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _fig_dir():
    d = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, name):
    p = os.path.join(_fig_dir(), name)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def _curve(test, B, a, plan, est):
    sub = test[(test.B == B) & (np.abs(test.a - a) < 1e-12) & (test.plan == plan) & (test.est == est)]
    if not len(sub):
        return None
    g = sub.groupby("h1").test_err
    return pd.DataFrame({"h": g.median().index, "med": g.median().values, "q1": g.quantile(0.25).values,
                         "q3": g.quantile(0.75).values, "n": g.size().values})


def fig_error_vs_h(name, groups, fname, title):
    _style()
    test = pd.read_csv(os.path.join(results_dir(), f"{name}_test.csv"))
    lock = load_lock(name)
    budgets = sorted(test.B.unique())
    noise = sorted(test.a.unique())
    fig, axes = plt.subplots(len(budgets), len(noise), figsize=(2.3 * len(noise), 2.3 * len(budgets) + 0.4),
                             sharex=True, squeeze=False)
    for i, B in enumerate(budgets):
        for j, a in enumerate(noise):
            ax = axes[i, j]
            for pipe in groups:
                plan, est = pipe.split("-")
                c = _curve(test, B, a, plan, est)
                if c is None:
                    continue
                col, mk, ls = STYLE[pipe]
                ax.plot(c.h, c.med, color=col, marker=mk, ls=ls, label=pipe, ms=3.5)
                ax.fill_between(c.h, c.q1, c.q3, color=col, alpha=0.12, lw=0)
                for b in lock["best"]:
                    if b["B"] == B and abs(b["a"] - a) < 1e-12 and b["plan"] == plan and b["est"] == est:
                        y = c.med[np.abs(c.h - b["h1"]) < 1e-9]
                        if len(y):
                            ax.plot([b["h1"]], [y.iloc[0]], marker="o", ms=9, mfc="none", mec=col, mew=1.4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"noise a = {a:g}")
            if j == 0:
                ax.set_ylabel(f"B = {B}\nrel. VF test error")
            if i == len(budgets) - 1:
                ax.set_xlabel("shortest interval h (h1)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(groups), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.07, fontsize=10)
    return _save(fig, fname)


def fig_contrasts(name, fname="fig_contrasts.png"):
    _style()
    c = pd.read_csv(os.path.join(results_dir(), "analysis", f"{name}_contrasts.csv"))
    kinds = [k for k in ["P1_SA-RICH_vs_IP-RICH_same_h", "P2_SA-RICH_vs_tuned_SI", "P3_best3_vs_bestSI"]
             + sorted(k for k in c.contrast.unique() if k.startswith("P4")) if k in set(c.contrast)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(2.9 * len(kinds), 2.8), sharey=True, squeeze=False)
    cols = {600: "#2a78d6", 1800: "#eb6834", 3600: "#1baf7a"}
    mks = {600: "o", 1800: "s", 3600: "D"}
    for ax, k in zip(axes[0], kinds):
        sub = c[c.contrast == k]
        for B, sb in sub.groupby("B"):
            x = sb.a.replace(0, 3e-4).values
            y = sb.ratio.values
            ax.errorbar(x * (1.08 if B == 1800 else 0.93), y,
                        yerr=[y - sb.ratio_lo.values, sb.ratio_hi.values - y], color=cols.get(B, INK2),
                        marker=mks.get(B, "o"), ls="none", capsize=2, label=f"B = {B}")
        ax.axhline(1.0, color=INK2, lw=0.9)
        ax.set_xscale("log")
        _ratio_logy(ax)
        ax.set_title(k.replace("_", " "), fontsize=8)
        ax.set_xlabel("noise a (a = 0 plotted at 3e-4)")
    sub = c[c.contrast.isin(kinds)]
    axes[0, 0].set_ylim(min(0.95, sub.ratio_lo.min() * 0.95), max(1.05, sub.ratio_hi.max() * 1.05))
    axes[0, 0].set_ylabel("error ratio (first / second)\ngeometric mean over paired reps, 95% CI")
    axes[0, 0].legend(loc="best")
    return _save(fig, fname)


def fig_sa_ip(name, fname="fig_sa_vs_ip_ra.png"):
    _style()
    d = pd.read_csv(os.path.join(results_dir(), "analysis", f"{name}_sa_ip_ra_by_h.csv"))
    budgets = sorted(d.B.unique())
    noise = sorted(d.a.unique())
    fig, axes = plt.subplots(len(budgets), len(noise), figsize=(2.3 * len(noise), 2.2 * len(budgets) + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    pairs = [("SA-RICH", "IP-RICH", "#eda100", "^"), ("SA-RICH", "RA-RICH", "#e87ba4", "v"),
             ("RA-RICH", "IP-RICH", NEUTRAL, "o"), ("MS3-RICH", "SA-RICH", "#1baf7a", "D")]
    for i, B in enumerate(budgets):
        for j, a in enumerate(noise):
            ax = axes[i, j]
            for pa, pb, col, mk in pairs:
                s = d[(d.B == B) & (np.abs(d.a - a) < 1e-12) & (d.a_pipe == pa) & (d.b_pipe == pb)].sort_values("h1")
                if len(s):
                    ax.errorbar(s.h1, s.ratio, yerr=[s.ratio - s.ratio_lo, s.ratio_hi - s.ratio], color=col,
                                marker=mk, ms=3.5, capsize=1.5, lw=1.0, label=f"{pa} / {pb}")
            ax.axhline(1.0, color=INK2, lw=0.9)
            ax.set_xscale("log")
            _ratio_logy(ax)
            if i == 0:
                ax.set_title(f"noise a = {a:g}")
            if j == 0:
                ax.set_ylabel(f"B = {B}\nerror ratio")
            if i == len(budgets) - 1:
                ax.set_xlabel("h1 (h2 = 2 h1)")
    sel = pd.concat([d[(d.a_pipe == pa) & (d.b_pipe == pb)] for pa, pb, _, _ in pairs])
    axes[0, 0].set_ylim(min(0.95, sel.ratio_lo.min() * 0.95), max(1.05, sel.ratio_hi.max() * 1.05))
    h, l = axes[0, -1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    return _save(fig, fname)


def fig_decomposition(name, fname="fig_decomposition.png"):
    """Stacked squared components (floor, discretization, ME bias, variance) with the total MSE marked.
    SI-DC (floor/disc split not defined for exact-derivative data) and the forward-difference positive control
    are omitted here; both are in the report tables."""
    _style()
    p = os.path.join(results_dir(), f"{name}_decomp.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d.group == "headline"]
    order = ["SI-MP", "SA-RICH", "MS3-RICH", "IP-RICH", "RA-RICH", "SA-POOL", "SA-LONG"]
    g = d.groupby(["a", "name"]).agg(floor=("floor", lambda x: np.mean(x**2)), disc=("disc", lambda x: np.mean(x**2)),
                                     me=("me2_corr", "mean"), var=("var", "mean"),
                                     total=("total_rmse", lambda x: np.mean(x**2))).reset_index()
    noise = sorted(g.a.unique())
    fig, axes = plt.subplots(1, len(noise), figsize=(3.3 * len(noise), 3.1), squeeze=False)
    comps = [("floor", "approximation floor", "#2a78d6"), ("disc", "discretization bias", "#eb6834"),
             ("me", "measurement-error bias (MC-corrected)", "#e34948"), ("var", "noise variance", "#1baf7a")]
    for k, (ax, a) in enumerate(zip(axes[0], noise)):
        s_ = g[g.a == a].set_index("name").reindex(order).reset_index()
        bottom = np.zeros(len(s_))
        for key, lab, col in comps:
            v = np.maximum(s_[key].values, 0)
            ax.bar(np.arange(len(s_)), v, bottom=bottom, color=col, label=lab, width=0.7, edgecolor=SURFACE, lw=1)
            bottom += v
        ax.scatter(np.arange(len(s_)), s_["total"], marker="_", s=260, color=INK, lw=2, zorder=5,
                   label="total MSE (incl. cross terms)")
        ax.set_xticks(np.arange(len(s_)))
        ax.set_xticklabels(s_["name"], rotation=50, ha="right", fontsize=7.5)
        ax.set_title(f"B = 1800, noise a = {a:g}")
        if k == 0:
            ax.set_ylabel("squared relative error")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 1.08))
    return _save(fig, fname)


def fig_phase0(fname="fig_phase0.png"):
    _style()
    fl = os.path.join(results_dir(), "phase0_floor.csv")
    tu = os.path.join(results_dir(), "phase0_tune.csv")
    if not (os.path.exists(fl) and os.path.exists(tu)):
        return None
    f = pd.read_csv(fl).groupby("n").val_err.min()
    t = pd.read_csv(tu)
    t["pipe"] = t.plan + "-" + t.est
    best = t.groupby(["a", "pipe", "h1"]).val_err.min().reset_index()
    noise = sorted(best.a.unique())
    fig, axes = plt.subplots(1, len(noise) + 1, figsize=(2.5 * (len(noise) + 1), 2.6), squeeze=False)
    ax = axes[0, 0]
    ax.plot(f.index, f.values, color="#2a78d6", marker="o")
    ax.set_xscale("log")
    _pct_logy(ax)
    ax.set_xlabel("n exact-gradient samples")
    ax.set_ylabel("best rel. VF error (validation)")
    ax.set_title("approximation floor F(n)")
    for ax, a in zip(axes[0, 1:], noise):
        for pipe in ["SI-MP", "SI-DC", "SA-RICH", "MS3-RICH", "IP-RICH"]:
            s = best[(best.a == a) & (best.pipe == pipe)].sort_values("h1")
            if len(s):
                col, mk, ls = STYLE[pipe]
                ax.plot(s.h1, s.val_err, color=col, marker=mk, ls=ls, label=pipe, ms=3.5)
        ax.set_xscale("log")
        _pct_logy(ax)
        ax.set_title(f"Phase 0, B=1800, a={a:g}")
        ax.set_xlabel("h (h1)")
    axes[0, -1].legend(fontsize=6.5)
    return _save(fig, fname)


def all_figures(name):
    out = []
    out.append(fig_phase0())
    out.append(fig_error_vs_h(name, ["SI-MP", "SI-DC", "SA-RICH", "MS3-RICH"], "fig_error_vs_h_main.png",
                              "Test error vs interval: single interval (plain / defect-corrected) vs 3-reading Richardson"))
    out.append(fig_error_vs_h(name, ["SA-RICH", "IP-RICH", "RA-RICH"], "fig_error_vs_h_sa_ip_ra.png",
                              "Shared anchor vs independent pairs vs re-measured anchor (Richardson)"))
    out.append(fig_error_vs_h(name, ["SI-MP", "SA-POOL", "IP-POOL", "SA-LONG"], "fig_error_vs_h_pooled.png",
                              "Pooled two-interval fits and the long-pair-only ablation vs single interval"))
    out.append(fig_contrasts(name))
    out.append(fig_sa_ip(name))
    out.append(fig_decomposition(name))
    return [o for o in out if o]


def fig_hh(fname="fig_hh_anchor.png"):
    _style()
    p = os.path.join(results_dir(), "hh_anchor_hh_anchor.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    spec = {("hu_fixed_matern52", "SI"): ("#2a78d6", "o", "--", "SI-MP, Hu fixed (Matern, l=3, eta=1e-9)"),
            ("hu_fixed_matern52", "SA"): ("#eb6834", "s", "--", "Richardson (d, 2d), Hu fixed"),
            ("tuned_se", "SI"): ("#2a78d6", "o", "-", "SI-MP, tuned SE"),
            ("tuned_se", "SA"): ("#eb6834", "s", "-", "Richardson (d, 2d), tuned SE")}
    for (st, plan), (col, mk, ls, lab) in spec.items():
        s = d[(d.setting == st) & (d.plan == plan)].sort_values("delta")
        ax.plot(s.delta, s.test_err, color=col, marker=mk, ls=ls, label=lab, ms=4)
    x = np.array([0.08, 0.3])
    ax.plot(x, 0.00108 * (x / 0.1) ** 2, color=NEUTRAL, lw=0.9, ls=":", label="slope 2 reference")
    ax.set_xscale("log")
    _pct_logy(ax)
    ax.set_xlabel("step delta")
    ax.set_ylabel("rel. VF test error (noise-free)")
    ax.set_title("Henon-Heiles implementation anchor (N = 1000)")
    ax.legend(fontsize=6.5)
    return _save(fig, fname)


def fig_secondary(name="main_dp"):
    """Ablation (SIMEX 2x2), misspecification and calibration panels (whatever exists)."""
    _style()
    out = []
    adir = os.path.join(results_dir(), "analysis")
    p = os.path.join(adir, "ablation_simex_2x2.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        fig, axes = plt.subplots(1, len(d.a.unique()), figsize=(3.6 * len(d.a.unique()), 3.0), squeeze=False)
        for ax, (a, s) in zip(axes[0], d.groupby("a")):
            x = np.arange(len(s))
            w = 0.2
            bars = [("h_naive:naive", "h naive, no corr.", "#2a78d6"), ("h_naive:simex_q", "h naive, SIMEX", "#eb6834"),
                    ("h_simex:naive", "h re-tuned, no corr.", "#1baf7a"), ("h_simex:simex_q", "h re-tuned, SIMEX", "#eda100")]
            for k, (col_, lab, c) in enumerate(bars):
                ax.bar(x + (k - 1.5) * w, s[col_], width=w * 0.9, color=c, label=lab, edgecolor=SURFACE)
            ax.set_xticks(x)
            ax.set_xticklabels(s.pipeline, fontsize=7.5)
            ax.set_title(f"SIMEX 2x2, B=1800, a={a:g}")
            ax.set_ylabel("median rel. VF test error")
        axes[0, 0].legend(fontsize=6.5)
        out.append(_save(fig, "fig_ablation_simex.png"))
    p = os.path.join(adir, "misspec_summary.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d[(d.variant != "simex_l")]
        d["lab"] = d.pipeline + np.where(d.variant == "simex_q", "+SIMEX", "") + \
            np.where(d.kappa != 1.0, " k=" + d.kappa.astype(str), "")
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        scen = ["correct", "hetero", "temporal"]
        cols = {"correct": "#2a78d6", "hetero": "#eb6834", "temporal": "#1baf7a"}
        labs = sorted(d.lab.unique())
        for k, sc in enumerate(scen):
            s = d[d.scenario == sc].set_index("lab").reindex(labs)
            ax.bar(np.arange(len(labs)) + (k - 1) * 0.27, s["median"], width=0.25, color=cols[sc], label=sc,
                   edgecolor=SURFACE)
        ax.set_xticks(np.arange(len(labs)))
        ax.set_xticklabels(labs, rotation=50, ha="right", fontsize=7)
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
        ax.set_ylabel("median rel. VF test error")
        ax.set_title("Noise-model misspecification (B=1800, a=0.03; methods assume iid isotropic)")
        ax.legend(fontsize=7, title="true noise", title_fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        out.append(_save(fig, "fig_misspec.png"))
    p = os.path.join(results_dir(), f"{name}_calibrate.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        fig, axes = plt.subplots(1, len(d.a.unique()), figsize=(4.2 * len(d.a.unique()), 3.2), squeeze=False)
        slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
        marks = ["o", "s", "D", "^", "v", "P"]
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        for ax, (a, s) in zip(axes[0], d.groupby("a")):
            c = s[s.method == "candidate"]
            for k, (ci, cc) in enumerate(c.groupby("cand")):
                lab = f"{cc.plan.iloc[0]}-{cc.est.iloc[0]}@{cc.h1.iloc[0]:g}"
                ax.scatter(cc.pred_err, cc.test_err, s=22, label=lab, color=slots[k % 6], marker=marks[k % 6],
                           edgecolor=SURFACE, linewidth=0.6)
            lim = [min(c.pred_err.min(), c.test_err.min()) * 0.9, max(c.pred_err.max(), c.test_err.max()) * 1.1]
            ax.plot(lim, lim, color=NEUTRAL, lw=0.8, ls=":")
            ax.set_xscale("log")
            _pct_logy(ax)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("surrogate-predicted error")
            ax.set_ylabel("realized test error (true system)")
            ax.set_title(f"calibration candidates, a={a:g}")
        axes[0, 0].legend(fontsize=6)
        out.append(_save(fig, "fig_calibration.png"))
    return out


_all_main = all_figures


def all_figures(name):  # noqa: F811
    out = _all_main(name)
    out.append(fig_hh())
    out.extend(fig_secondary(name))
    return [o for o in out if o]
