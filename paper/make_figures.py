"""Figures for the UNEC manuscript, generated directly from the saved result files (no re-running).

Inputs (read-only):
  results/analysis/main_dp_contrasts.csv                      -> Figure 2 (P1, P2 paired ratios, 95% CI)
  results/confirmatory/confirm_min_interval/raw.csv           -> Figure 3 (per-replicate paired ratios)
  results/confirmatory/confirm_min_interval/comparisons.csv   -> Figure 3 (geometric means, 95% CI)
Outputs: paper/figures/Figure1_sampling_schematic.{png,pdf} (design schematic, no data),
         paper/figures/Figure2_main_ratios.{png,pdf}, paper/figures/Figure3_confirmation.{png,pdf}

Run from the project root:  python paper/make_figures.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")

# Fixed categorical slots (never cycled); identity is also carried by marker shape, fill and line style,
# so the figures stay readable in grayscale.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.7, "savefig.dpi": 600,
})

NOISE = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
NOISE_LABELS = ["0", "0.001", "0.003", "0.01", "0.03", "0.1"]


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure_schematic():
    """Figure 1: independent pairs vs shared anchor (design schematic; no data). Aspect 16.8 x 5.98 cm."""
    fig, ax = plt.subplots(figsize=(16.8 / 2.54, 5.98 / 2.54))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.75)
    ax.axis("off")
    t0, t1, t2 = 4.0, 5.9, 7.8
    y1, y2, ysa = 2.62, 2.02, 0.66
    # headers and time ticks
    ax.text((t0 + t2) / 2, 3.55, "Reading time on each trajectory", ha="center", va="center", fontsize=10,
            fontweight="bold", color=INK)
    ax.text(9.15, 3.55, "Unique readings", ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    for t, lab in ((t0, "$0$"), (t1, "$h_1$"), (t2, "$h_2$")):
        ax.text(t, 3.12, lab, ha="center", va="center", fontsize=11, color=INK)
        ax.plot([t, t], [0.35, 2.92], ls=(0, (1, 2.5)), color=GRID, lw=0.9, zorder=0)
    ax.plot([0.25, 9.85], [1.5, 1.5], color=GRID, lw=0.8)
    # IP: two starts, one pair each
    ax.text(0.25, (y1 + y2) / 2, "IP  Independent pairs", ha="left", va="center", fontsize=10, fontweight="bold",
            color=INK)
    for y, tend, lab in ((y1, t1, "Start 1"), (y2, t2, "Start 2")):
        ax.text(3.75, y, lab, ha="right", va="center", fontsize=9, color=INK2)
        ax.plot([t0 - 0.12, t2 + 0.15], [y, y], color="#b9b8b4", lw=1.0, zorder=1)
        ax.plot([t0, tend], [y, y], color=INK, lw=1.8, zorder=2)
        for t in (t0, tend):
            ax.plot(t, y, "o", ms=6.5, mfc="white", mec=INK, mew=1.5, zorder=3)
    ax.text(9.15, (y1 + y2) / 2, "4 readings", ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    # SA: one start, the reading at 0 enters both pairs
    ax.text(0.25, ysa, "SA  Shared anchor", ha="left", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.plot([t0 - 0.12, t2 + 0.15], [ysa, ysa], color="#b9b8b4", lw=1.0, zorder=1)
    for tend, h, ls, lab, lx in ((t1, 0.38, "-", "pair $(0, h_1)$", None),
                                 (t2, 0.65, (0, (4, 2.2)), "pair $(0, h_2)$", t1 + 1.15)):
        xs = np.linspace(t0, tend, 100)
        ys = ysa + h * np.sin(np.pi * (xs - t0) / (tend - t0))
        ax.plot(xs, ys, color=BLUE, lw=1.6, ls=ls, zorder=2)
        if lx is None:  # label inside the short arc
            ax.text((t0 + tend) / 2, ysa + 0.07, lab, ha="center", va="bottom", fontsize=8.5, color=INK2)
        else:  # label just outside the long arc, on its right flank
            yl = ysa + h * np.sin(np.pi * (lx - t0) / (tend - t0))
            ax.text(lx + 0.12, yl + 0.05, lab, ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax.plot(t0, ysa, "o", ms=8, mfc=BLUE, mec=BLUE, mew=1.2, zorder=4)
    for t in (t1, t2):
        ax.plot(t, ysa, "o", ms=6.5, mfc="white", mec=INK, mew=1.5, zorder=3)
    ax.text(t0, ysa - 0.36, "same reading used in both pairs", ha="left", va="center", fontsize=8.5, color=INK2)
    ax.text(9.15, ysa, "3 readings", ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"Figure1_sampling_schematic.{ext}"))
    plt.close(fig)


def figure1():
    c = pd.read_csv(os.path.join(ROOT, "results", "analysis", "main_dp_contrasts.csv"))
    panels = [("P1_SA-RICH_vs_IP-RICH_same_h", "(a) SA-RICH / IP-RICH at the same intervals"),
              ("P2_SA-RICH_vs_tuned_SI", "(b) SA-RICH / tuned SI-MP")]
    series = [(600, BLUE, "o", "-", "B = 600 readings", -0.09, "full"),
              (1800, ORANGE, "s", "--", "B = 1800 readings", 0.09, "none")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.05), sharey=True)
    x = np.arange(len(NOISE))
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        ax.axhline(1.0, color=INK2, lw=1.0, zorder=1)
        for B, col, mk, ls, lab, dx, fill in series:
            d = c[(c.contrast == key) & (c.B == B)].set_index("a").loc[NOISE]
            y, lo, hi = d.ratio.values, d.ratio_lo.values, d.ratio_hi.values
            ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], color=col, ls=ls, lw=1.5, marker=mk, ms=6.2,
                        mfc=(col if fill == "full" else "white"), mec=col, mew=1.4, capsize=2.5, elinewidth=1.1,
                        label=lab, zorder=3)
        ax.set_xticks(x, NOISE_LABELS)
        ax.set_xlabel("Relative noise level $a$ (dimensionless)")
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(-0.45, len(NOISE) - 0.55)
    axes[0].set_ylabel("Error ratio (geometric mean of\n20 paired test replicates)")
    axes[0].set_ylim(0.6, 1.5)
    axes[0].text(len(NOISE) - 0.6, 1.015, "ratio = 1", ha="right", va="bottom", fontsize=8.5, color=INK2)
    axes[0].text(-0.35, 0.615, "below 1: SA-RICH has lower error", ha="left", va="bottom", fontsize=8.5,
                 color=INK2)
    axes[1].text(-0.35, 1.485, "above 1: SA-RICH has higher error", ha="left", va="top", fontsize=8.5,
                 color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"Figure2_main_ratios.{ext}"), bbox_inches="tight")
    plt.close(fig)


def figure2():
    d = os.path.join(ROOT, "results", "confirmatory", "confirm_min_interval")
    raw = pd.read_csv(os.path.join(d, "raw.csv"))
    comp = pd.read_csv(os.path.join(d, "comparisons.csv"))
    levels = [0.001, 0.003, 0.01]
    comps = [("SA-RICH / SI-MP", "SI-MP", BLUE, "o", "full", -0.17, "SA-RICH / SI-MP (predicted below 1)"),
             ("SA-RICH / SI-DC", "SI-DC", ORANGE, "D", "none", 0.17, "SA-RICH / SI-DC (predicted above 1)")]
    fig, ax = plt.subplots(figsize=(6.6, 3.15))
    _style(ax)
    ax.axhline(1.0, color=INK2, lw=1.0, zorder=1)
    rng = np.random.default_rng(0)  # horizontal jitter of replicate points only (display)
    x = np.arange(len(levels))
    for name, other, col, mk, fill, dx, lab in comps:
        gm, lo, hi = [], [], []
        for i, a in enumerate(levels):
            sub = raw[np.isclose(raw.a, a)].pivot(index="rep", columns="pipeline", values="test_err")
            r = (sub["SA-RICH"] / sub[other]).values
            jit = rng.uniform(-0.055, 0.055, len(r))
            ax.scatter(np.full(len(r), i + dx) + jit, r, s=11, facecolor="white", edgecolor=col, lw=0.7,
                       alpha=0.85, zorder=2, marker=mk)
            row = comp[np.isclose(comp.a, a) & (comp.comparison == name)].iloc[0]
            gm.append(row.gm_ratio)
            lo.append(row.ci_lo)
            hi.append(row.ci_hi)
        gm, lo, hi = map(np.asarray, (gm, lo, hi))
        ax.errorbar(x + dx + 0.13, gm, yerr=[gm - lo, hi - gm], color=col, ls="none", marker=mk, ms=6.5,
                    mfc=(col if fill == "full" else "white"), mec=col, mew=1.4, capsize=3, elinewidth=1.2,
                    zorder=4, label=lab)
    ax.set_xticks(x, ["0.001", "0.003", "0.01"])
    ax.set_xlim(-0.55, len(levels) - 0.45)
    ax.set_ylim(0.55, 1.4)
    ax.set_xlabel("Relative noise level $a$ (dimensionless); $B$ = 1800, $h_{\\min}$ = 0.4 for all methods")
    ax.set_ylabel("Error ratio (SA-RICH / comparator)")
    ax.text(len(levels) - 0.5, 1.012, "ratio = 1", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.text(-0.5, 1.385, "small open markers: the 20 individual paired replicates; large markers: geometric "
            "mean with 95% bootstrap CI", ha="left", va="top", fontsize=8.3, color=INK2)
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.2), fontsize=8.8,
              handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"Figure3_confirmation.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure_schematic()
    figure1()
    figure2()
    print("wrote", sorted(os.listdir(OUT)))
