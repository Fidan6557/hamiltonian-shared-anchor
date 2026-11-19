# Confirmatory protocol: shared anchor vs single interval under a minimum sampling interval

**Recorded:** 2026-11-19 23:43 local (2026-11-19T19:43Z), **before** any confirmatory data were generated
(no `bank_confirm_*` file existed; no confirmation code had been written or run).

## Motivation (post hoc, not confirmatory)

A read-only reanalysis of the original Phase B test results, using the pre-registered constrained
validation locks (h ≥ h_min, lock sha256 `888a9b3d4758`), suggested that under h ≥ 0.4 at B = 1800 the
shared-anchor Richardson plan (SA-RICH) has lower test error than the tuned plain single interval (SI-MP),
but higher error than the defect-corrected single interval (SI-DC). That analysis was not a pre-registered
contrast and was not multiplicity-adjusted; it is saved separately as
`results/analysis/main_dp_posthoc_min_interval.csv`. Post-hoc point estimates (shown only so that the
reader can see what is being tested): SA-RICH/SI-MP ≈ 0.63, 0.69, 0.85 and SA-RICH/SI-DC ≈ 1.20, 1.20, 1.23
at a = 0.001, 0.003, 0.01.

## Design (fixed)

* System, sampling density, noise model, kernel, estimators and code: unchanged from `configs/main_dp.yaml`
  (double pendulum, E_max = −1.5, isotropic Gaussian noise σ = a·z_rms, z_rms = 0.6926).
* Budget B = 1800 unique readings. Noise a ∈ {0.001, 0.003, 0.01}. Minimum sampling interval h_min = 0.4 for
  all three methods.
* Methods and settings, taken **without retuning** from the existing constrained validation locks
  (`results/locks/main_dp.json`, `best[...].constrained["0.4"]` and the lock entry at that h):

| a | SA-RICH (h1, h2, ℓ, η) | SI-MP (h, ℓ, η) | SI-DC (h, ℓ, η) |
|---|---|---|---|
| 0.001 | 0.4, 0.8, 1.15, 1e-7 | 0.4, 1.15, 1e-7 | 0.4, 1.15, 1e-7 |
| 0.003 | 0.4, 0.8, 1.10, 1e-6 | 0.4, 1.10, 1e-6 | 0.4, 1.10, 1e-7 |
| 0.01 | 0.4, 0.8, 1.15, 1e-5 | 0.4, 1.15, 1e-6 | 0.4, 1.15, 1e-6 |

* Schedules: SI reads each start at t = 0 and 0.4 (900 starts); SA reads each start at t = 0, 0.4, 0.8
  (600 starts; pairs (0, 0.4) and (0, 0.8) share the reading at 0). **Check:** for every dataset, the minimum
  spacing between consecutive readings on the same trajectory is computed and must be ≥ 0.4 (−1e-9); the run
  aborts otherwise. The spacing is stored with every result row.
* Replicates: 20 fresh paired replicates per noise level, seeds `(20261119, 13, rep, 0)` for rep = 0…19
  (seed purpose `confirm` = 13; never used for tuning (1), the original test (2) or any other study). Within a
  replicate the three methods use the same trajectory bank and noise draws (paired); as in the main study, the
  replicate's bank is shared across the three noise levels with noise scaled by a.
* Held-out evaluation: a **fresh** clean test set of 300 trajectories × 20 states from the same density
  (seed purpose `confirm_testset` = 102), not the original test set. Metric: relative vector-field error
  √(Σ‖f̂ − f‖² / Σ‖f‖²).

## Predictions (directional)

1. **P-MP:** SA-RICH has lower held-out vector-field error than SI-MP (ratio SA-RICH/SI-MP < 1).
2. **P-DC:** SA-RICH has higher held-out vector-field error than SI-DC (ratio SA-RICH/SI-DC > 1).

Each is tested separately at each of the three noise levels: six confirmatory tests.

## Analysis (fixed)

* Per comparison: paired log error ratios over the 20 replicates; geometric-mean ratio = exp(mean log ratio);
  95% percentile bootstrap CI (10,000 resamples, seed 0); two-sided exact Wilcoxon signed-rank p-value on the
  log ratios; **Holm adjustment across the six tests**.
* Decision rule: a prediction is **supported** at a noise level iff the geometric-mean ratio is in the predicted
  direction and the Holm-adjusted p < 0.05; **contradicted** iff the ratio is in the opposite direction with
  Holm-adjusted p < 0.05; otherwise **not supported (inconclusive)**. No pooling across noise levels.
* Failures (Cholesky failure after the jitter ladder, or non-finite predictions) are counted and reported; in
  the rank test a failed replicate counts as +∞ error for that method; geometric means exclude failures.
* No retuning, no change of settings, seeds or analysis after viewing outcomes. Any deviation will be listed
  as a deviation in the results file and in the report.

## Outputs

* `results/confirmatory/confirm_min_interval/raw.csv` (one row per replicate × noise level × method),
  `comparisons.csv` (six rows), `summary.json` (including the sha256 of this protocol file and of the lock).
* The manuscript keeps unrestricted-h results and minimum-interval results in separate sections, and states
  the confirmatory result only for the tested conditions (double pendulum, E_max = −1.5, B = 1800, h_min = 0.4,
  a ∈ {0.001, 0.003, 0.01}, gradient kernel ridge estimators with the locked hyperparameters).
