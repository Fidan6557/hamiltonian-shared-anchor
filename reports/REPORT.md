# Does reusing one noisy initial reading across two intervals help recover a Hamiltonian vector field?

*Study date: 2026-11-19. The original experiments ran on a 15 W laptop CPU (i7-1355U, no GPU) with the code,
configurations and seeds in this folder. Experiment tables are generated from `results/` by `hamsched.analysis` /
`hamsched.secondary`; tables are in `reports/tables/`, figures in `figures/`. Synthetic checks and document
verification are identified separately. Nothing here is a reproduction of a published result unless explicitly
stated otherwise.*

## 1. Summary

**Question.** At a fixed number of *unique* noisy readings of (q, p), when does the shared-anchor plan (SA:
readings at 0, h1, h2 → pairs (0,h1), (0,h2), sharing reading 0) recover f = J∇H₀ better than independent pairs
(IP: 4 readings on two starts) or a well-tuned single interval (SI)?

**Answer (double pendulum, B ∈ {600, 1800} readings, noise a ∈ {0, …, 0.1}, 12 cells × 20 paired test replicates):**

1. **SA-RICH beats IP-RICH in all 12 tested conditions.** At identical (h1, h2), SA-RICH's error
   is 0.67–0.92× IP-RICH's (geometric mean of paired error ratios; 100% paired wins; Holm p < 0.001).
   The advantage is present in noise-free runs, so noise covariance cannot explain the full gain. Sharing
   saves a reading per unit, giving SA 4/3 more units than IP; finite-sample coverage and approximation also
   matter. A re-measured-anchor plan
   (same start, independent readings — Hu et al.'s Sec. 6.3 protocol in reading accounting) performs like IP
   (ratio ≈ 0.95–0.98), and SA beats it by ≈ 15%. The measured fit variances match the derived covariance
   structure (SA's two stride fits are positively correlated, +0.46, which helps the opposite-sign Richardson
   combination). This supports a variance mechanism under noise, rather than a complete explanation of the
   total-error advantage.
2. **With h chosen freely from the grid, tuned SI-MP beats SA-RICH in all 12 tested conditions.** SA-RICH is
   6–39% *worse* than the tuned plain single interval (paired geometric-mean ratios 1.06–1.39). The tested
   pooled, GLS and long-pair-only SA variants also do not improve on tuned SI-MP in their evaluated cells. For
   the two specified shared-anchor pairs, sharing helps relative to independent-pair Richardson in this grid,
   but does not beat tuned SI-MP when the interval is unrestricted. The minimum-interval case is reported
   separately (point 5, §4.9).
3. **MS3-RICH is a separate estimator using all three pairs.** Hu-style multistride on readings
   (0, h, 2h) — pairs (0,h), (h,2h), (0,2h) with Richardson (MS3-RICH) — is ≈ 23% better than SA-RICH on the same
   readings and is the best tested 3-reading estimator in all 12 tested conditions. This is distinct from
   SA-RICH, which uses only (0,h) and (0,2h). Against the best single-interval estimator, which
   is a single interval with a model-based defect correction (SI-DC, itself 10–17% better than plain SI at
   B = 1800 and 4–10% at B = 600), MS3-RICH
   wins by > 10% only without noise at B = 600 (0.79×), wins by 2–9% in 6 further cells, ties in 4, and loses by
   15% at (B = 1800, a = 0.001). Under an instrument constraint h ≥ 0.4 it loses or ties at low noise.
4. **No remaining measurement-error bias was detectable above Monte Carlo uncertainty in the tested settings.**
   The fixed-start decomposition estimates ≤ ~0.5% relative measurement-error bias, within Monte Carlo error;
   approximation floor, discretization bias and noise variance dominate the measured errors. For a single
   forward-anchored pair with independent isotropic Gaussian reading noise and vanishing boundary terms,
   the correlation term +(σ²/h)∇log ρ̃ is L²(ρ̃)-orthogonal to Hamiltonian fields. This population argument
   covers that term; it does not remove the remaining attenuation or establish cancellation of cross-pair
   effects in pooled fits. The synthetic setting shows a −45% radial component for an unstructured fit and
   approximately zero for a Hamiltonian fit, which tests this component rather than total EIV bias. The known-Σ
   correction (reading-level Rao–Blackwellized SIMEX) raises test error by +4% to +32% in all six
   ablation cells, with or without re-tuning h. It removes the −45% radial component in the unstructured
   synthetic case; its lack of benefit on this benchmark is empirical.
5. **Minimum sampling interval h ≥ 0.4 (separate, pre-registered confirmation; §4.9).** In the tested
   conditions only — double pendulum, B = 1800, h_min = 0.4 imposed on all methods, a ∈ {0.001, 0.003, 0.01},
   locked constrained settings, 20 fresh paired replicates and a fresh test set — SA-RICH had lower held-out
   vector-field error than the constrained SI-MP (ratios 0.63, 0.69, 0.84) and higher error than the constrained
   SI-DC (ratios 1.19, 1.19, 1.23). All six pre-registered predictions were supported (Holm-adjusted p ≈ 1e-5).
   This pattern was first seen post hoc in the original data and was then confirmed on new seeds; it is not
   claimed for other minimum intervals, budgets, noise levels or systems.

**What this adds to Hu et al. (arXiv:2610.17835):** unique-reading accounting with genuinely shared readings,
the covariance structure that sharing induces and its sign-dependent effect (measured), a tuned head-to-head
under noise on a nonseparable system, the split of discretization from measurement-error bias, and the
single-pair orthogonality argument for the forward-anchor correlation term. The empirical scope is
one benchmark (double pendulum, one energy band), one estimator family (gradient kernel ridge with
implicit-midpoint anchors), oracle-validation tuning, and i.i.d.-noise truth in the main grid.

**Secondary findings.** The ranking is unchanged under heteroscedastic or temporally correlated true noise
(§4.6), under a different kernel or without Hamiltonian structure, and for interval ratios 1.5–3 (§4.8). A
14.7% calibration set (counted in the budget) picked a good schedule and interval with a surrogate model —
MS3-RICH in 15 of 16 replicates — beating fixed no-calibration defaults by 5–25% and landing within 10–23% of
the oracle-tuned best (§4.7). Hénon–Heiles reproduces the qualitative O(δ²) behaviour reported by Hu et al.
with our implementation (slope 2.00).

---

## 2. Relation to prior work (verified by reading the paper)

Hu, Martín de Diego, Ortega & Vaquero, *Learning Hamiltonians from Trajectory Data with Kernel Methods*,
arXiv:2610.17835 (v1, 13 Oct 2026), main text read in full. By section:

* **Sec. 4.3** — a *pathwise* noise-stability bound (velocity noise W = (e₁ − e₀)/δ enters with η^{-1/2},
  anchor noise with η^{-1}). No errors-in-variables analysis, no bias derivation, no correction; an EIV
  treatment and data-driven choice of (δ, η, Q) under noise are listed as open (Sec. 7).
* **Sec. 5** — multistride Richardson: separate stride fits combined with Vandermonde weights (MP, Q = 2:
  4/3, −1/3), raising the integrator-error order without new trajectories.
* **Sec. 6.3** (Hénon–Heiles, separable; N = 1000, L = 1, η = 1e-9, ℓ = 3 fixed): Richardson vs δ without
  noise, and a noise sweep at δ = 0.1 in which *independent noise realizations are used for different
  strides* — shared states are never shared measurements.
* **Sec. 6.4** — noise-free; budget NL = 400 split into (N, L); it counts pairs, not unique readings.
* No baselines, no hyperparameter tuning; the double pendulum appears only as a reconstruction plot (Sec. 6.5).

Our implementation follows their estimator family (gradient kernel ridge regression with MP anchors,
endpoint-defect loss) but is a **re-implementation**, not their code, and uses a tuned squared-exponential
kernel. Implementation anchor (§4.8): on Hénon–Heiles, noise-free, with their N, δ range and E < 0.14, our MP
error scales with slope 2.00 in δ (they report ≈ 2.03 for a Hamiltonian-error metric on a slice); with their
exact fixed settings (Matérn-5/2, ℓ = 3, η = 1e-9) the slope is 1.49 over δ ∈ [0.08, 0.3] because of a 0.15%
error floor at small δ (≈ 2.0 for δ ≥ 0.2), and Richardson (δ, 2δ) lowers the error 7–11× until a common
floor, as in their Sec. 6.3. This is a qualitative consistency check, not a reproduction of their numbers.

Related work used or cited: inverse modified differential equations (Zhu, Jin, Zhu & Tang, arXiv:2009.01058);
learning inverse modified Hamiltonians (Offen & Ober-Blöbaum, *Chaos* 32:013122, 2022); post-training
correction to the true Hamiltonian (David & Méhats, *J. Comput. Phys.* 494:112495, 2023) — our SI-DC is a
member of this backward-error post-processing family, not a new idea; SIMEX (Cook & Stefanski, *JASA*
89:1314–1328, 1994); GP regression with input noise (McHutchon & Rasmussen, NeurIPS 2011); difference-based
variance estimation (Rice, *Ann. Stat.* 12:1215–1230, 1984; Hall, Kay & Titterington, *Biometrika* 77:521–528,
1990). Found by search but not studied in detail: finite-difference variance under correlated noise for
Hamiltonian learning (arXiv:2604.17470) and variable-step multistep port-Hamiltonian GPs (arXiv:2510.00384).

## 3. Setup

**System.** Planar double pendulum (unit masses, lengths, g = 1),
H = [p₁² + 2p₂² − 2p₁p₂cos(q₁−q₂)] / [2(1 + sin²(q₁−q₂))] − 2cos q₁ − cos q₂ — nonlinear and nonseparable
(the mass matrix depends on q). States follow ρ ∝ g(H) on {H ≤ −1.5} (g = 1 below 60% of the energy range,
raised-cosine taper to 0; |q₁| ≤ 75°, |q₂| ≤ 120°, no flips). ρ depends on z only through H, so it is invariant
under the flow: clean states at any fixed time, validation states and test states share one distribution.
Arithmetic pair midpoints need not have that distribution or identical distributions across lags; Gaussian
measurement noise further changes the anchor distribution. Simulation: DOP853 with
rtol = atol = 1e-12 on stacked batches (verified per trajectory to < 1e-9; energy drift < 1e-10, far below
the smallest noise σ = 7e-4). Small-oscillation frequencies 0.77 and 1.85.

**Noise and budgets.** Isotropic Gaussian, σ = a·z_rms with z_rms = 0.6926 (Hu et al.'s relative
convention), a ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}; B ∈ {600, 1800} unique readings. Noise is drawn per
(start, time) and reused across plans and noise levels (common random numbers), so a shared reading has the
same value and ID in both pairs (unit-tested).

**Plans** (per budget B; observation durations reported per start):

| plan | readings per unit | starts | pairs | max duration per start |
|---|---|---|---|---|
| SI(h) | (0, h) | B/2 | B/2 | h |
| IP(h1, 2h1) | (0, h1) on one start + (0, h2) on another | B/2 | B/2 | h1 or h2 |
| RA(h1, 2h1) | 0 read twice (independent), h1, h2 on one start | B/4 | B/2 | h2 |
| SA(h1, 2h1) | (0, h1, h2); pairs (0,h1), (0,h2) share reading 0 | B/3 | 2B/3 | h2 |
| MS3(h) | (0, h, 2h); pairs (0,h), (h,2h), (0,2h) | B/3 | B | 2h |

Example at (B = 1800, a = 0.01) with locked h: SI-MP 900 starts, 900 pairs, total observed time 255;
SA-RICH 600 starts, 1200 pairs, 480; MS3-RICH 600 starts, 1800 pairs, 339; IP-RICH 900 starts, 900 pairs,
540; RA-RICH 450 starts, 900 pairs, 360 (`results/analysis/main_dp_best_summary.csv` lists all cells).

**Estimators** (one gradient-KRR core; squared-exponential kernel with lengthscale ℓ × per-coordinate spread of
the training readings; ridge λ = η·n·mean Gram diagonal; Cholesky with one jitter ladder for all plans):
MP (single interval); **DC** (MP + cross-fitted model-based MP defect evaluated at the chord midpoint —
`test_defect_correction_with_true_flow_is_fourth_order` checks the O(h⁴) gate and the O(h²) anchor-shift trap);
POOL (all pairs, endpoint-defect weighting); GLS (full within-unit noise covariance, known Σ up to scale);
**RICH** (separate fits per interval with *shared* (ℓ, η), weights α₁ = r²/(r²−1), α₂ = −1/(r²−1), r = h2/h1;
O(h⁴) clean-target gate verified for r = 2 and 1.5); LONG (long pair only); reading-level Rao–Blackwellized SIMEX
(known Σ) as the measurement-error correction.

Richardson cancels the h² term of the clean theoretical targets at a common anchor in the small-step
regime. Shared kernel hyperparameters do not guarantee identical fitted smoothing operators: different
anchor distributions, realized samples and Gram matrices affect regularized finite-sample KRR. The clean
target slope gate therefore does not establish O(h⁴) error for these fitted estimators; their observed
discretization reduction is reported separately in §4.4 (see `docs/theory.md` §3).

**Metric.** Relative vector-field error √(Σ‖f̂ − f‖²/Σ‖f‖²) on 300 clean held-out test trajectories × 20
states (t ∈ [0, 9.5]); invariant to H's additive constant; also by energy tercile.

**Protocol** (reports/preregistration.md). Phase 0 (separate seeds) fixed grids and recorded predictions.
Phase A tuned (h, ℓ, η) per (B, a, plan, estimator) on 150 clean validation trajectories (an **oracle**
validation set, disjoint from test) with 2 tuning datasets; locks were written with sha256 and UTC time
(`results/locks/main_dp.json`, sha 888a9b3d4758, created 13:18:37 UTC); Phase B evaluated 20 fresh datasets per
cell at locked settings (started 13:18:40 UTC; every test row carries the lock hash; verified in
`results/analysis/main_dp_lock_check.json`). Three grid amendments were made **before** the lock under the
pre-registered edge rule (B = 600 lengthscale windows, h windows extended by one doubling to h2 ≤ 1.6, lower ℓ
and η values at B = 1800); remaining edge selections (4 at B = 1800, 37 at B = 600, of which 24 ℓ-edges are
bracketed by the discarded first pass) are listed in the lock and in
`results/analysis/main_dp_B600_ell_edge_bracketing.csv`.

## 4. Results

### 4.1 Pre-registered primary contrasts

Ratio = geometric mean over the 20 paired test replicates of err(first)/err(second), bootstrap 95% CI;
"(ns)" = Holm-adjusted Wilcoxon p ≥ 0.05. Ratio < 1 favours the first pipeline. Validation chose MS3-RICH as the
best 3-reading estimator and SI-DC as the best single-interval estimator in every cell (P3/P4).

| B | a | P1: SA-RICH / IP-RICH, same (h1,h2) | P2: SA-RICH / tuned SI-MP | P3: MS3-RICH / SI-DC | P4: P3 with h ≥ 0.4 |
|---|---|---|---|---|---|
| 600 | 0 | 0.75 [0.72, 0.78] | 1.27 [1.22, 1.32] | 0.79 [0.76, 0.83] | 1.07 [1.05, 1.10] |
| 600 | 0.001 | 0.75 [0.72, 0.78] | 1.33 [1.28, 1.39] | 0.94 [0.91, 0.98] | 0.97 [0.95, 1.01] (ns) |
| 600 | 0.003 | 0.78 [0.75, 0.81] | 1.22 [1.18, 1.26] | 0.99 [0.96, 1.01] (ns) | 0.99 [0.96, 1.01] (ns) |
| 600 | 0.01 | 0.85 [0.83, 0.86] | 1.16 [1.13, 1.19] | 1.01 [0.98, 1.03] (ns) | 1.01 [0.98, 1.03] (ns) |
| 600 | 0.03 | 0.88 [0.86, 0.90] | 1.15 [1.13, 1.18] | 0.99 [0.96, 1.01] (ns) | 0.99 [0.96, 1.01] (ns) |
| 600 | 0.1 | 0.92 [0.90, 0.94] | 1.13 [1.11, 1.15] | 0.97 [0.95, 0.99] | 0.97 [0.95, 0.99] (ns) |
| 1800 | 0 | 0.67 [0.65, 0.70] | 1.39 [1.35, 1.43] | 0.91 [0.88, 0.95] | 1.10 [1.10, 1.11] |
| 1800 | 0.001 | 0.78 [0.76, 0.81] | 1.14 [1.12, 1.16] | 1.15 [1.11, 1.19] | 1.10 [1.09, 1.11] |
| 1800 | 0.003 | 0.78 [0.77, 0.80] | 1.11 [1.09, 1.13] | 0.99 [0.98, 1.00] (ns) | 1.03 [1.02, 1.05] |
| 1800 | 0.01 | 0.85 [0.83, 0.86] | 1.06 [1.05, 1.07] | 0.95 [0.94, 0.97] | 1.00 [0.99, 1.02] (ns) |
| 1800 | 0.03 | 0.89 [0.87, 0.90] | 1.10 [1.09, 1.12] | 0.98 [0.96, 0.99] | 0.98 [0.96, 0.99] (ns) |
| 1800 | 0.1 | 0.89 [0.87, 0.90] | 1.13 [1.12, 1.15] | 0.98 [0.97, 0.99] | 0.98 [0.97, 0.99] |

P4 with h ≥ 0.1 and h ≥ 0.2 is in `results/analysis/main_dp_contrasts.csv` (≈ P3 except B = 1800, a = 0:
0.95 at h ≥ 0.2). Figure: `figures/fig_contrasts.png`.

### 4.2 Locked-best median test error (%) and selected shortest interval h

B = 1800 (900 starts for SI/IP, 600 for SA/MS3, 450 for RA):

| pipeline | a = 0 | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 |
|---|---|---|---|---|---|---|
| SI-MP (plain single interval) | 1.25 (h .071) | 1.95 (.141) | 2.78 (.2) | 4.54 (.283) | 7.19 (.4) | 12.20 (.566) |
| SI-DC (single interval + defect corr.) | 1.05 (.2) | **1.67** (.283) | 2.32 (.283) | 3.89 (.4) | 6.11 (.4) | 11.00 (.566) |
| SA-RICH (shared anchor, Richardson) | 1.71 (.2) | 2.24 (.283) | 3.12 (.283) | 4.83 (.4) | 7.97 (.4) | 13.97 (.566) |
| MS3-RICH (all 3 pairs, Richardson) | **0.93** (.141) | 1.92 (.2) | **2.29** (.283) | **3.74** (.283) | **6.03** (.4) | **10.75** (.566) |
| IP-RICH (independent pairs) | 2.53 (.283) | 2.87 (.283) | 3.98 (.283) | 5.73 (.4) | 8.81 (.566) | 15.53 (.566) |
| RA-RICH (re-measured anchor) | 2.42 (.2) | 2.82 (.283) | 4.07 (.4) | 5.61 (.4) | 8.81 (.566) | 15.57 (.566) |
| SA-POOL (shared anchor, pooled) | 1.61 (.05) | 2.43 (.1) | 3.38 (.1) | 5.07 (.141) | 7.92 (.2) | 13.16 (.283) |
| IP-POOL | 1.28 (.05) | 2.15 (.1) | 3.06 (.1) | 4.83 (.141) | 7.76 (.2) | 13.07 (.283) |
| SA-LONG (SA without middle reading) | 1.86 (.05) | 2.53 (.1) | 3.40 (.1) | 5.15 (.141) | 8.22 (.2) | 13.42 (.283) |
| SA-GLS (known-Σ correlated weights) | – | – | 3.33 (.1) | – | 8.01 (.2) | – |

B = 600 (300 / 200 / 150 starts):

| pipeline | a = 0 | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 |
|---|---|---|---|---|---|---|
| SI-MP | 4.06 (.2) | 4.14 (.2) | 4.84 (.283) | 6.85 (.283) | 9.77 (.4) | 15.94 (.566) |
| SI-DC | 3.77 (.283) | 3.82 (.283) | 4.43 (.4) | **6.18** (.4) | 8.90 (.566) | 15.36 (.566) |
| SA-RICH | 5.06 (.4) | 5.48 (.4) | 5.80 (.4) | 8.01 (.4) | 11.33 (.566) | 18.00 (.8) |
| MS3-RICH | **2.99** (.283) | **3.52** (.283) | **4.36** (.4) | 6.22 (.4) | **8.83** (.566) | **15.03** (.566) |
| IP-RICH | 6.83 (.4) | 7.35 (.4) | 7.59 (.4) | 9.37 (.4) | 12.94 (.566) | 19.70 (.8) |
| RA-RICH | 6.35 (.4) | 6.89 (.4) | 7.47 (.4) | 9.36 (.566) | 12.60 (.566) | 19.25 (.8) |
| SA-POOL | 4.82 (.141) | 5.22 (.141) | 5.67 (.141) | 7.58 (.2) | 11.19 (.283) | 17.57 (.283) |
| IP-POOL | 4.21 (.035) | 4.48 (.1) | 5.23 (.141) | 7.22 (.2) | 10.87 (.283) | 17.75 (.4) |
| SA-LONG | 5.18 (.141) | 5.69 (.141) | 5.98 (.141) | 8.02 (.2) | 11.61 (.283) | 17.95 (.283) |

Bootstrap CIs of the medians are in `reports/tables/main_dp_best_summary.md` (typical half-width 1–3% of the
median). Errors are largest in the high-energy tercile for every pipeline; rankings are the same in all three
energy terciles (`err_E0..2` columns). Figures: `figures/fig_error_vs_h_main.png`,
`fig_error_vs_h_sa_ip_ra.png`, `fig_error_vs_h_pooled.png`.

**Fit failures: none.** 0 Cholesky failures and 0 jitter escalations in ≈ 33,000 tuning fits and 13,980 test
fits (`results/analysis/main_dp_failures*.csv`).

### 4.3 Why: what sharing does, and why the single interval still wins

**Exact covariance structure (docs/theory.md §2).** With i.i.d. reading noise, SA's derived pairs satisfy
Cov(W1, W2) = Σ/(h1h2) > 0, Cov(U1, W2) = −Σ/(2h2), Cov(U2, W1) = −Σ/(2h1), Cov(U1, U2) = Σ/4, and zero
anchor–velocity covariance within each pair (MP anchoring).

**Mechanism check** (fixed starts, 12 noise draws, common h1 = 0.2 and common (ℓ, η); B = 1800, a = 0.01).
Variance relative to SI(2h1), measured vs the local-averaging prediction (docs/theory.md §4):

| plan / estimator | measured | predicted | corr(stride-1 fit, stride-2 fit) |
|---|---|---|---|
| SI(h1) | 3.50 | 4.0 | – |
| SA-RICH | 9.12 | 9.5 | +0.46 |
| IP-RICH | 14.18 | 14.4 | +0.01 |
| RA-RICH | 14.97 | 14.4 | +0.00 |
| MS3-RICH | 2.37 | 1.5 (idealized lower bound) | +0.76 |
| MS3 with Hu Sec. 6.3 noise protocol / MS3 (per unit) | 1.50 | 1.89 | – |

So (i) the variance accounting is quantitatively right for SI, SA, IP and RA; (ii) the positive correlation
created by the shared reading is real and reduces the Richardson variance, alongside the benefit of 4/3 more
units per budget; these explain the variance comparison, not the whole total-error gap, which also appears
without noise; (iii) MS3's stride-1 average telescopes to the stride-2 noise, so its Richardson combination
amplifies noise far less than SA's one-sided stencil — the reason MS3 beats SA on identical readings; (iv) the
independent-noise-per-stride protocol used in Hu et al.'s Sec. 6.3 overstates the noise cost of multistride
Richardson by ≈ 1.5× relative to truly shared readings.

**IP → RA → SA** (paired, same h1; `results/analysis/main_dp_sa_ip_ra_by_h.csv`): RA/IP median ratio 0.95
(B = 600) and 0.98 (B = 1800) — sharing a *start* (half as many starts, independent noise) is nearly free;
SA/RA ≈ 0.84–0.85 — sharing the *reading* changes both reading efficiency (one saved reading per unit) and
noise covariance. This comparison does not isolate covariance as the cause of the full error reduction.

**Why SA still loses to the single interval.** SA-RICH's variance factor (9.5) is ≈ 2.4× SI(h1)'s and ≈ 9.5× SI(2h1)'s;
Richardson reduces discretization bias that, at the selected h in these noisy settings, is not large enough
to pay for that variance cost and the approximation floor. Without noise, covariance provides no explanation
of the error ranking. SA without Richardson also loses in the evaluated comparisons: for a straight-line
(locally constant velocity) model the middle reading
carries no slope information (GLS puts all weight on the long pair), and the pooled, GLS and long-pair-only
variants are 8–49% worse than tuned SI-MP (Richardson changes SA little: paired SA-RICH/SA-POOL median ratio
0.94–1.06 across h). The single interval spends its budget on the two most
informative readings (the endpoints) and on 1.5× more starts.

**Why MS3 helps and DC rivals it.** MS3-RICH has an O(h⁴) clean target in the small-step regime and measured
variance close to SI(2h) on 2/3 of the starts. The target order does not guarantee O(h⁴) error for the
regularized fits. Its anchors at three distinct points per start raise coverage (noise-free it beats the SI floor).
SI-DC removes most O(h²) discretization bias from a single interval using the fitted model's own defect, so the
remaining advantage of a third reading is small (P3 ratios 0.95–1.01 for 0.003 ≤ a ≤ 0.03).

### 4.4 Discretization bias vs measurement-error bias

Fixed-start decomposition at the locked settings (B = 1800; 4 start sets × 12 noise draws; sequential
components in % of ‖f‖, test set; `results/analysis/decomp_headline.csv`). *floor* = same estimator fitted to
exact f at the plan's clean anchors; *disc* = clean-data fit − floor fit (discretization); *ME bias* = mean
noisy fit − clean fit with the Monte Carlo correction −tr(Var)/R (shown as √max(·,0)); *noise sd* = spread over
noise draws; *start sd* = spread of the noise-averaged estimate over start sets.

| a | pipeline (locked h) | floor | disc | ME bias | noise sd | start sd | total RMSE |
|---|---|---|---|---|---|---|---|
| 0.003 | SI-MP (0.2) | 1.98 | 1.58 | 0.00 | 1.56 | 1.06 | 2.79 |
| 0.003 | SA-RICH (0.283) | 2.83 | 1.97 | 0.02 | 1.99 | 1.44 | 3.16 |
| 0.003 | MS3-RICH (0.283) | 2.04 | 1.62 | 0.06 | 1.16 | 0.96 | 2.33 |
| 0.003 | IP-RICH (0.283) | 3.77 | 2.47 | 0.00 | 2.63 | 2.03 | 3.96 |
| 0.01 | SI-MP (0.283) | 3.05 | 2.77 | 0.00 | 2.42 | 1.35 | 4.54 |
| 0.01 | SA-RICH (0.4) | 3.92 | 3.15 | 0.00 | 2.93 | 1.83 | 4.91 |
| 0.01 | MS3-RICH (0.283) | 2.95 | 1.44 | 0.00 | 2.33 | 1.34 | 3.76 |
| 0.01 | SA-POOL (0.141) | 3.18 | 2.42 | 0.00 | 3.48 | 1.73 | 5.08 |
| 0.03 | SI-MP (0.4) | 3.59 | 4.69 | 0.00 | 4.58 | 1.63 | 7.22 |
| 0.03 | SA-RICH (0.4) | 6.19 | 2.44 | 0.39 | 5.09 | 2.73 | 8.05 |
| 0.03 | MS3-RICH (0.4) | 3.61 | 2.69 | 0.20 | 4.62 | 1.70 | 6.10 |
| 0.03 | SA-POOL (0.2) | 5.43 | 3.68 | 0.24 | 4.47 | 2.27 | 7.96 |
| 0.03 | FWD (h = 0.1, positive control) | 7.66 | 5.16 | 0.00 | 9.51 | 3.55 | 13.36 |

(The floor/disc split is not meaningful for SI-DC — the defect correction is defined for MP-pair data and
distorts exact-derivative data — so SI-DC is omitted here; its totals are in §4.2.)

Findings. (i) The estimated measurement-error bias is at most 0.4% in the displayed decomposition cells:
no remaining measurement-error bias was detectable above Monte Carlo uncertainty in the tested settings,
including SA-POOL whose pooled pairs carry cross-pair anchor–velocity covariance. Clipping the corrected
squared estimate at zero is not evidence that the population bias is exactly zero. (ii) Richardson reduces
the measured discretization term: at a = 0.01 MS3-RICH's disc is 1.4% vs SI-MP's 2.8%; SA-RICH's
discretization reduction is smaller at its larger selected h. Its noise sd is also larger, adding a
separate cost to total error.
(iii) The forward-difference positive control (anchor = first reading, Cov(U, W) = −Σ/h) also shows no
detectable EIV bias — and an exploratory stronger control at (B = 600, a = 0.1, h = 0.05) shows none either
(Monte Carlo-corrected squared bias −0.0013 ± 0.0047, while the noise RMSE is ≈ 60%).

**Single-pair explanation and its limits (docs/theory.md §7).** With x̃ = x + U, C = Cov(W,U), and jointly
Gaussian reading noise independent of the clean state, E[U | x̃] = −Σ_U∇log ρ̃ and
E[W | x̃] = −C∇log ρ̃. For forward anchoring C = −Σ/h, so the correlation term has the **positive** sign
+(Σ/h)∇log ρ̃. With isotropic Σ = σ²I and vanishing boundary terms,
∫ρ̃∇log ρ̃·J∇H = −∫ρ̃ div(J∇H) = 0. Thus this correlation term is L²(ρ̃)-orthogonal to Hamiltonian
fields and contributes nothing to the population least-squares projection. The remaining O(σ²) attenuation
need not be orthogonal or zero; its score part vanishes only where the noisy density ρ̃ is flat, which a flat
clean density does not ensure exactly. This argument does not establish removal of general EIV bias,
cross-pair effects in SA-POOL or MS3 fits, or finite-sample bias. Synthetic check
(`tests/test_simex_bias_removal.py`, linear Hamiltonian field, Gaussian anchors, forward anchoring, h = 0.2,
σ = 0.3): the unstructured vector-KRR fit has a bias component of −0.45 along x (linear theory −0.41), the
Hamiltonian fit ≈ 0.002 ± 0.004. These are the originally reported exploratory values; the targeted
verification in §7 checks the sign and test assertions, rather than reconstructing those uncertainty ranges.
For Gaussian anchors ∇log ρ̃ = −x̃/(1+σ²), so the positive score prefactor gives the negative coefficient
−σ²/[h(1+σ²)] ≈ −0.41. A near-zero radial component does not show that attenuation along the true field is zero.

### 4.5 Correction ablations (schedule change vs discretization correction vs noise correction)

**Discretization correction, single interval (2×2 from the main grid, B = 1800).** At the h tuned for plain
SI, adding the defect correction changes the error by 0.85–1.01×; re-tuning h for the corrected estimator
moves h up (e.g. 0.071 → 0.2 at a = 0; 0.141 → 0.283 at a = 0.001) and gives 0.56–0.90× of plain SI *at that
h*, and 0.83–0.90× of plain SI at its own best h (`results/analysis/main_dp_dc_2x2.csv`). The correction is
worth as much as, or more than, changing the schedule to SA, and it requires no extra reading.

**Schedule change vs Richardson.** Adding a second interval without Richardson (SA-POOL, IP-POOL, SA-LONG,
SA-GLS) does not beat tuned SI-MP in the evaluated cells (§4.2). Adding Richardson to SA changes little
(paired SA-RICH/SA-POOL 0.94–1.06);
adding it to all three pairs of the same readings (MS3) is what pays.

**Measurement-error correction, SIMEX 2×2** (B = 1800, a ∈ {0.03, 0.1}, 10 paired test replicates; h either
tuned for the naive estimator or re-tuned for the corrected one on separate validation seeds):

| a | pipeline | naive @ h_naive | SIMEX-quad @ h_naive | ratio [95% CI] | SIMEX-quad @ h re-tuned | ratio vs naive |
|---|---|---|---|---|---|---|
| 0.03 | SI-MP | 7.29% | 7.63% | 1.04 [1.03, 1.05] | 7.63% | 1.04 |
| 0.03 | SA-RICH | 8.20% | 8.59% | 1.04 [1.03, 1.05] | 8.59% | 1.04 |
| 0.03 | MS3-RICH | 6.18% | 8.12% | 1.32 [1.30, 1.34] | 8.06% | 1.32 |
| 0.1 | SI-MP | 12.20% | 12.92% | 1.05 [1.04, 1.07] | 12.92% | 1.05 |
| 0.1 | SA-RICH | 14.08% | 15.85% | 1.14 [1.13, 1.15] | 16.43% | 1.16 |
| 0.1 | MS3-RICH | 11.01% | 13.76% | 1.26 [1.24, 1.28] | 13.76% | 1.26 |

The linear extrapolant is nearly neutral (≤ +3%). With no remaining measurement-error bias detectable above
Monte Carlo uncertainty in the tested settings, these results are consistent with added pseudo-noise variance
and over-correction of the fixed-η estimator's noise response; re-tuning h does not improve on naive fits here.
The correction itself is not broken: in the synthetic unstructured case it removes the −0.45 bias component
(to +0.03 ± 0.03 in the originally reported exploratory calculation). This demonstrates removal of one
synthetic radial component, not all EIV bias. The known-Σ correction provides no observed advantage for the
tested Hamiltonian pipelines; its misspecification is tested next.

### 4.6 Noise-model misspecification

(B = 1800, a = 0.03, 8 paired replicates; all methods keep the correctly-specified locks and assume i.i.d.
isotropic noise with known σ; ratio vs the same pipeline under correctly specified noise;
`results/analysis/misspec_summary.csv`, `figures/fig_misspec.png`.)

| pipeline | correct (median) | heteroscedastic truth, same mean σ² | temporally correlated truth (ρ = 0.5 at Δt = 0.2) |
|---|---|---|---|
| SI-MP | 7.20% | 1.04 [1.03, 1.04] | 0.96 [0.95, 0.97] |
| SI-DC | 6.07% | 1.05 [1.04, 1.06] | 0.95 [0.93, 0.96] |
| SA-RICH | 7.80% | 1.03 [1.03, 1.04] | 0.96 [0.95, 0.96] |
| MS3-RICH | 5.95% | 1.07 [1.06, 1.08] | 0.93 [0.92, 0.95] |
| IP-RICH | 8.84% | 1.02 [1.01, 1.03] | 0.98 [0.98, 0.99] |
| SA-POOL | 7.76% | 1.03 [1.02, 1.03] | 0.96 [0.96, 0.97] |
| SA-GLS | 7.88% | 1.02 [1.02, 1.03] | 0.97 [0.97, 0.98] |

Heteroscedastic noise (σ larger at high momentum) is the case that breaks the MP anchor–velocity
decorrelation (Cov(U, W) ≈ ½∇σ²·f ≠ 0); it costs 2–7%, most for the 3-reading pipelines. Temporally correlated
noise partly cancels in differences and helps every pipeline. **No ranking changes**: SA-RICH < IP-RICH and
SI-MP < SA-RICH in all scenarios; MS3-RICH vs SI-DC remains within ±5%. Misspecifying the assumed noise scale
in SIMEX (Σ̂ = κΣ) makes the quadratic SIMEX monotonically worse (κ = 0.5: +2–3%, κ = 1: +4–5%, κ = 2: +8–9%);
the linear extrapolant stays within ±1% of naive. GLS's assumed correlation structure was wrong under
temporal correlation; it still behaves like SA-POOL.

### 4.7 Can a small calibration set choose the schedule? (non-oracle selection, budget counted)

Pre-registered trigger (≥ 10% regret of some fixed default vs the oracle-best in a headline cell) fired
(`results/analysis/main_dp_calibration_trigger.json`). Design (B = 1800; 8 replicates at a = 0.003 and 0.03;
`configs/calibration.yaml`): spend B_cal = 264 readings (14.7%) on 88 triplets (0, 0.2, 0.4) from the true
system; fit a surrogate (MS3-RICH, hyperparameters by 4-fold rollout CV on the calibration triplets); simulate
each of 6 pre-specified candidates (SI-MP@0.2, SI-MP@0.4, SI-DC@0.4, MS3-RICH@0.2, MS3-RICH@0.4, SA-RICH@0.4) on
the surrogate with the remaining 1536 readings and the known σ; pick the lowest predicted error; collect the
remaining readings from the true system with that schedule; choose (ℓ, η) by the schedule-independent noisy
criterion (multi-step prediction of the calibration readings). Regret = test error / per-replicate best of the
oracle-locked pipelines (full budget, oracle validation); geometric mean, 95% CI.

| method | a = 0.003: selections | regret | a = 0.03: selections | regret |
|---|---|---|---|---|
| calibrated (all candidates) | MS3-RICH@0.2 ×8 | 1.23 [1.16, 1.32] | MS3-RICH@0.4 ×7, SI-DC@0.4 ×1 | 1.10 [1.09, 1.12] |
| calibrated, SI candidates only | SI-DC@0.4 ×8 | 1.35 [1.32, 1.38] | SI-DC@0.4 ×8 | 1.15 [1.12, 1.18] |
| no calibration: SI-MP@0.2, full budget, rollout-CV (ℓ, η) | – | 1.37 [1.30, 1.45] | – | 1.47 [1.43, 1.51] |
| no calibration: MS3-RICH@0.2, full budget | – | 1.30 [1.18, 1.41] | – | 1.35 [1.30, 1.41] |
| oracle-locked SI-DC / MS3-RICH (reference) | – | 1.03 / 1.00 | – | 1.02 / 1.00 |

Surrogate-predicted vs realized candidate errors: Spearman 0.94 (a = 0.003) and 1.00 (a = 0.03) per replicate
(median); predicted errors are optimistic in absolute terms (e.g. 1.6% vs 2.8% realized for MS3-RICH@0.2 at
a = 0.003), as expected for an oversmoothed surrogate, but the ranking is right (`figures/fig_calibration.png`).
So **yes — a 15% calibration set chose a good schedule and h**, beating the fixed no-calibration defaults
by 5–25% despite its cost, and choosing *among plans* (not just h) added 4–10% over SI-only selection. The
remaining 10–23% regret to the oracle reflects the spent readings and non-oracle hyperparameter selection.
Caveats: two cells, 8 replicates, a fixed candidate list, and SA-RICH was never selected (it was predicted and
realized worse than the alternatives).

### 4.8 Robustness checks (one cell each, B = 1800, a = 0.01, 10 paired test replicates) and implementation anchor

| check | result |
|---|---|
| Interval ratio r = h2/h1 (SA-RICH / IP-RICH, same intervals) | r = 1.5: 0.75 [0.74, 0.77]; r = 2 (main): 0.85 [0.83, 0.86]; r = 3: 0.85 [0.84, 0.86]. Local-averaging prediction √(variance ratio): 0.74, 0.81, 0.85 |
| Matérn-5/2 kernel instead of SE (tuned) | SA-RICH / SI-MP 1.07 [1.05, 1.08]; MS3-RICH / SI-MP 0.80 [0.79, 0.80]; all errors slightly higher than with SE (SI-MP 4.94% vs 4.54%) |
| Unstructured vector KRR (no Hamiltonian structure) | SA / SI 1.12 [1.10, 1.14]; MS3 / SI 0.85 [0.83, 0.87]; MS3 / SA 0.76 [0.75, 0.77]; the Hamiltonian versions are 26–30% better in absolute terms (SI 6.16% vs 4.54%) |
| Hu et al.'s fixed estimator settings (Matérn-5/2, ℓ = 3 raw, η = 1e-9, untuned) on the double pendulum | best 6.43% (h = 0.4) vs 4.54% for the tuned SI-MP |
| Hénon–Heiles noise-free anchor (N = 1000, Hu et al. Sec. 6.1–6.2 settings) | MP slope 2.00 (tuned SE) / 1.49 (Hu-fixed; ≈ 2.0 for δ ≥ 0.2); Richardson 7–11× lower until a common floor (`figures/fig_hh_anchor.png`) |

The qualitative conclusions with h chosen freely (SA beats IP; SA loses to SI; MS3 is the best 3-reading
estimator) hold under all three estimator changes and both other interval ratios. The robust_unstr lock selected
the lower ℓ edge for all three pipelines (ℓ grid not widened for this check).

### 4.9 Minimum sampling interval h ≥ 0.4 (separate from the unrestricted results)

This section is kept separate from §4.1–4.8, where the interval was chosen freely from the grid.

**Origin (post hoc).** After the main analysis, a read-only reanalysis of the original Phase B rows with the
pre-registered constrained validation locks (h ≥ h_min) suggested that, at B = 1800 and low noise, SA-RICH has
lower error than SI-MP when both must use h ≥ 0.4, while SI-DC remains better than SA-RICH. That contrast was not
pre-registered and was not multiplicity-adjusted. It is saved separately in
`results/analysis/main_dp_posthoc_min_interval.csv` and is not used as evidence below.

**Confirmation (pre-registered).** Protocol: `reports/confirmatory_min_interval_protocol_2026-11-19.md`,
recorded at 19:43:21Z (sha256 `ce62121b…`), before any confirmatory data existed; the run started at 19:46:13Z.
* Design: B = 1800 unique readings; h_min = 0.4 for all three methods; a ∈ {0.001, 0.003, 0.01}. Settings taken
  from the existing constrained locks without retuning: every method uses h = 0.4 (SA-RICH: h1 = 0.4, h2 = 0.8);
  ℓ and η as listed in the protocol.
* Schedules: SI reads (0, 0.4) on 900 starts; SA reads (0, 0.4, 0.8) on 600 starts. The minimum spacing between
  consecutive readings on the same trajectory was checked for every dataset: 0.4 in all 180 rows, with exactly
  1800 unique readings in each.
* 20 fresh paired replicates per noise level (seed purpose `confirm`, never used for tuning or the original test)
  and a fresh held-out test set of 300 trajectories × 20 states. No fit failures or jitter escalations.
* Statistics: paired geometric-mean error ratio, 95% percentile bootstrap CI (10,000 resamples), two-sided exact
  Wilcoxon signed-rank test, Holm adjustment across the six tests.

| a | SA-RICH / SI-MP (predicted < 1) | SA-RICH / SI-DC (predicted > 1) | median error: SA-RICH, SI-MP, SI-DC |
|---|---|---|---|
| 0.001 | 0.634 [0.629, 0.639]; 20/20 SA lower; p_Holm = 1.1e-5 | 1.195 [1.186, 1.204]; 0/20 SA lower; p_Holm = 1.1e-5 | 3.38%, 5.31%, 2.82% |
| 0.003 | 0.689 [0.682, 0.695]; 20/20; p_Holm = 1.1e-5 | 1.191 [1.180, 1.203]; 0/20; p_Holm = 1.1e-5 | 3.76%, 5.45%, 3.14% |
| 0.01 | 0.841 [0.831, 0.853]; 20/20; p_Holm = 1.1e-5 | 1.226 [1.210, 1.245]; 0/20; p_Holm = 1.1e-5 | 5.02%, 5.98%, 4.13% |

Both predictions were supported at all three noise levels (`results/confirmatory/confirm_min_interval/`:
`raw.csv`, `comparisons.csv`, `summary.json`). Interpretation, limited to these conditions: when the sampling
interval cannot be shorter than 0.4 (ω_max·h ≈ 0.74), the plain single interval carries discretization bias it
cannot tune away, and the shared-anchor Richardson schedule reduces the error by 16–37% relative to it; against a
single interval with model-based defect correction at the same minimum interval, SA-RICH's error is still 19–23%
higher.
The advantage over SI-MP shrinks as noise grows within the tested range. Nothing is claimed here about other
minimum intervals, budgets, noise levels, systems or estimator families.

## 5. What the evidence supports

* At equal unique-reading budgets and identical intervals, **sharing the initial reading beats taking
  independent pairs** for Richardson-type two-interval estimation (all 12 tested conditions; 8–33% lower error).
  Positive correlation between the two stride fits and the saved reading per unit support the variance
  mechanism under noise; covariance cannot explain the full gain because it also occurs without noise.
* **With h chosen freely, tuned SI-MP beats SA-RICH** (two pairs (0,h1), (0,h2)) in all 12 tested conditions;
  SA-RICH's error is 6–39% higher. The tested SA variants and noise corrections do not improve this conclusion
  in their evaluated cells.
* **Under a minimum sampling interval h ≥ 0.4 (separate, pre-registered confirmation, §4.9)**, at B = 1800 and
  a ∈ {0.001, 0.003, 0.01}, SA-RICH beat the constrained SI-MP (ratios 0.63–0.84) and lost to the constrained
  SI-DC (ratios 1.19–1.23); all six predictions were supported after Holm adjustment. This is the tested
  condition in which reusing the initial reading improved recovery relative to a well-tuned plain single
  interval; it is not generalized beyond those conditions.
* **The value of a third reading depends on how its pairs are used.** Using all three pairs (multistride, Hu et
  al. Sec. 5 style, on truly shared readings) is the best tested 3-reading estimator in all 12 conditions and
  matches or modestly beats the best single-interval estimator (single interval + defect correction) in 11 of 12 cells; > 10%
  advantage only without noise at B = 600; a 15% disadvantage at (1800, 0.001).
* **Approximation, discretization and variance dominate the observed errors.** No remaining measurement-error
  bias was detectable above Monte Carlo uncertainty in the tested settings. The population orthogonality
  result applies to the single-pair forward-anchor correlation term under isotropic Gaussian reading noise
  and vanishing boundary terms (§4.4); it does not eliminate attenuation or pooled cross-pair effects.
  Known-Σ SIMEX increased test error in the six tested ablation cells.
* Hu et al.'s independent-noise-per-stride protocol overstates the noise cost of multistride Richardson
  (measured ≈ 1.5×) relative to readings that are genuinely shared.

## 6. Limitations and what remains unverified

* **One benchmark, one energy band.** Double pendulum on {H ≤ −1.5} with a flat-interior density. A density
  with strong gradients, anisotropic noise or a different system could change the size of the EIV terms and
  of discretization bias. Hénon–Heiles was used only as a noise-free implementation anchor; the schedule
  comparison was **not** repeated on it.
* **Oracle-validation tuning.** Hyperparameters and h were chosen on clean validation trajectories (disjoint
  from test). This gives every pipeline its best case and slightly favours the more tuning-sensitive ones
  (Richardson, DC). The non-oracle calibration study (§4.7) addresses realistic selection only at two cells.
* **Tuning noise.** Only 2 tuning datasets per cell; neighbour-grid checks show locked h is sometimes one step
  off the test optimum (e.g. MS3-RICH at (1800, 0.001) would be 5% better one step up; still worse than SI-DC).
* **Grid edges.** 41 locked selections lie on a grid edge (4 at B = 1800). At B = 600, the Richardson plans
  often prefer the largest allowed h1 (h2 up to 1.6, where ω_max·h ≈ 3 approaches the π radius of the MP
  inverse modified Hamiltonian); their B = 600 errors are therefore upper bounds. The error curves are flat
  near the edge (one grid step inward costs 2–11%), so this is unlikely to overturn P2 (SA-RICH vs SI, 13–33%
  margins at B = 600), but it could favour MS3 slightly further.
* **Estimator family.** Only pairwise one-step regression (gradient kernel ridge with MP anchors) was studied.
  Trajectory-shooting estimators that integrate the learned flow (no discretization bias), Gaussian-process
  marginal-likelihood tuning, and neural baselines (HNN, SympNets) were **not run**; the unstructured vector
  KRR and the Matérn kernel are the only estimator-class robustness checks (§4.8).
* **SIMEX positive control.** No empirical double-pendulum regime with detectable EIV bias was found, so the
  correction's efficacy is shown only on the synthetic unstructured case.
* **Midpoint distributions and fitted smoothing.** Flow invariance does not equate arithmetic-midpoint
  distributions across lags. Shared hyperparameters do not equate the KRR smoothing operators. Clean-target
  Richardson cancellation therefore does not by itself prove the discretization order of the fitted models.
* **Budgets.** B ∈ {600, 1800} only; B = 3600 was not run (memory/compute on this laptop).
* Theory of §4.3 is a local-averaging idealization; it matched measurements for SI/SA/IP/RA but not exactly
  for MS3 (2.37 vs the idealized 1.5).

## 7. Reproducibility, compute and deviations

* Commands: `README.md` (or `scripts/run_all.sh` / `scripts/run_all.ps1`). Pinned dependencies:
  `requirements.txt` (core) and `requirements-accel.txt` (optional torch CPU, used here for float64 Cholesky
  and prediction; scipy and torch agree to 1e-8 in tests). Python 3.12.6, Windows 11, i7-1355U, CPU only.
* Original study tests: `python -m pytest` — 39 passed (12.5 s) in the pinned venv (unit tests, three discretization slope gates,
  Monte Carlo covariance checks, SIMEX/orthogonality synthetic checks, bank-extension/fill-in checks, smoke pipeline).
* Document verification (2026-11-19): recomputed P1/P2 directly from `results/main_dp_test.csv` using
  `results/locks/main_dp.json`, pairing the 20 replicates in each condition. All 24 geometric-mean ratios,
  bootstrap CIs, win fractions and Holm-adjusted p-values agree with `results/analysis/main_dp_contrasts.csv`.
  P1 ranges from 0.670023 to 0.919615 (240/240 paired SA wins); P2 from 1.059715 to 1.385294
  (SA has higher error on 238/240 individual replicates and in all 12 condition-level comparisons).
* Targeted sign check: executed both existing test functions in `tests/test_simex_bias_removal.py` with their
  original seeds, draw counts and assertions, using scipy and two BLAS threads; both passed. The unstructured
  radial coefficient was −0.45908 before SIMEX and −0.01031 afterward; the Hamiltonian coefficient was
  −0.00318. Also checked the executable forward-anchor pseudo-noise map M = −I/h. The wrong Tweedie sign
  is explanatory prose, including the test's docstring, rather than an executable sign error. The test
  source was left unchanged in this two-document revision. No synthetic or main numerical output was changed.
  The earlier synthetic ± uncertainty ranges quoted in §4.4–4.5 have no saved per-draw output here and were
  not independently reconstructed; the current tests return means and check the qualitative components.
* Minimum-interval confirmation (added after the main analysis; §4.9): protocol recorded before data
  generation (`reports/confirmatory_min_interval_protocol_2026-11-19.md`, sha256 recorded in
  `results/confirmatory/confirm_min_interval/summary.json`); run with
  `python -m hamsched.cli confirm --config configs/confirm_min_interval.yaml` (≈ 7 min). No deviations from the
  protocol. The original test results and lock were verified byte-identical (sha256) after the run. The
  post-hoc reading that motivated it is saved separately in `results/analysis/main_dp_posthoc_min_interval.csv`.
* Wall-clock on this laptop: Phase 0 ≈ 20 min; Phase A ≈ 1 h 45 min (+ 52 min fill-in after amendments);
  Phase B ≈ 2 h; decomposition ≈ 42 min; SIMEX ablation ≈ 52 min; misspecification ≈ 20 min;
  calibration ≈ 58 min; robustness ≈ 20 min; Hénon–Heiles ≈ 10 min; exploratory positive control ≈ 12 min.
  Total ≈ 10 h of wall-clock on one laptop (one experiment process at a time; ≈ 1–2 GB RAM free).
* Deviations from the plan: three pre-lock grid amendments (edge rule); a discarded first Phase A pass
  (B = 600 only); B = 3600 not run; Hénon–Heiles schedule comparison not run; post-hoc exploratory positive
  control and synthetic SIMEX tests (labelled as such). All are recorded in `reports/preregistration.md`.

