# Pre-registration (written after Phase 0, before Phase A locks and before any Phase B test run)

Date: 2026-11-19. All choices below were fixed from Phase 0/0b (separate seeds, purpose `phase0`,
results in `results/phase0_floor.csv`, `results/phase0_tune.csv`, `results/phase0b_ell.csv`) and
from the theory in `docs/theory.md`. No test-set information was available when this was written.

## Phase 0 findings that set the design (exploratory, 1 replicate, B = 1800)

* Approximation floor with exact gradients (validation, best over ℓ, η): F(300)=4.9%, F(450)=3.1%,
  F(600)=2.1%, F(900)=1.27%, F(1200)=1.09%.
* Pointwise MP discretization error at exact-flow anchors: 0.45% (h=0.1), 1.7% (h=0.2), 5.3% (h=0.4);
  Richardson (h, 2h): 0.05%, 0.5%, 2.9%. The O(h⁴) regime holds only for h ≲ 0.05–0.1.
* Best validation error per pipeline (single replicate; ranking, not inference):

| a | SI-MP | SI-DC | MS3-RICH | SA-RICH | IP-RICH |
|---|---|---|---|---|---|
| 0 | 1.40% | 1.27% | 1.02% | 2.06% | 3.48% |
| 0.003 | 3.27% | 2.98% | 2.82% | 3.80% | 4.43% |
| 0.01 | 5.16% | 3.97% | 4.09% | 5.30% | 6.67% |
| 0.03 | 7.97% | 7.17% | 6.78% | 8.46% | 9.71% |
| 0.1 | 13.6% | 12.3% | 12.5% | 16.3% | 19.2% |

* **Falsified theory prediction (recorded now):** "without noise the single interval wins" is false
  at B = 1800 — MS3-RICH was best at a = 0, plausibly because it yields one pair per reading with
  anchors at three distinct points per trajectory (more distinct anchors than SI's B/2).
* Grid edges hit in Phase 0: ℓ = 1.3 (max) at every noisy level; h1 = 0.4 (max allowed, h2 = 0.8)
  for two-interval plans at a ≥ 0.01; SI-DC h = 0.8 (max) at a = 0.1. Per the edge rule the main
  grids were widened: noise-dependent ℓ windows (Phase 0b probe: optimum 0.8–1.0 clean, ≈1.3 for
  a = 0.003–0.01, ≈1.8 for a = 0.1), η windows at factor-10 spacing, and h up to 1.131 (h1 ≤ 0.566).

## Main design (configs/main_dp.yaml)

Double pendulum, ρ ∝ g(H) with E_max = −1.5 (taper from 60% of the energy range), budgets
B ∈ {600, 1800} unique readings, noise a ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1} (σ = a·z_rms,
z_rms = 0.6926), plans SI, IP, RA, SA, MS3 (r = 2), estimators as in the config; SA-GLS only at
(1800, 0.003) and (1800, 0.03). Phase A: R_tune = 2, oracle validation (150 clean trajectories ×
20 states), selection by mean log validation error. Phase B: R_test = 20 fresh seeds.
Constrained (instrument) selections: h ≥ h_min for h_min ∈ {0.1, 0.2, 0.4}.

## Primary contrasts (Holm across the 12 cells within each family)

* **P1** SA-RICH vs IP-RICH at identical (h1, h2) = SA-RICH's locked h1.
* **P2** SA-RICH (locked) vs the tuned single interval SI-MP (locked).
* **P3** best 3-reading estimator (SA-RICH or MS3-RICH, chosen by validation) vs best single interval
  (SI-MP or SI-DC, chosen by validation).
* **P4** P3 under each instrument constraint h ≥ h_min.

Statistic: mean paired log error ratio over the 20 test replicates (bootstrap 95% CI), Wilcoxon
signed-rank p-value, Holm-adjusted; win fractions reported. A ratio is called a difference only if
the CI excludes 1 and p_Holm < 0.05; "practically relevant" means |ratio − 1| ≥ 10%.

**Predictions** (from theory + Phase 0): P1 ratio < 1 (sharing helps vs independent pairs);
P2 ratio ≥ 1 at most cells (SA-RICH does not beat the tuned single interval); P3: MS3-RICH and SI-DC
close (within ~10%), with MS3-RICH favoured at low noise; pooled two-interval fits (SA-POOL,
IP-POOL) do not beat SI-MP; SA-LONG ≈ SA-POOL-GLS (middle reading adds little without Richardson).

## Failure policy

A fit fails if Cholesky fails after the jitter ladder (0, 1e-10, 1e-8, 1e-6 × mean diagonal) or
predictions are non-finite. Failures are counted per plan/estimator and reported; in paired rank
statistics a failure counts as +∞ error; means/medians exclude failures (counts shown).

## Secondary studies (cells fixed now)

* Decomposition: (1800, a) for a ∈ {0.003, 0.01, 0.03}; 4 start sets × 12 noise draws; mechanism set
  at (1800, 0.01), h1 = 0.2, common (ℓ, η) = SA-RICH's lock entry; SIMEX checks at (1800, 0.03)
  incl. forward-difference positive control at h = 0.1.
* SIMEX 2×2 ablation: (1800, 0.03) and (1800, 0.1); SI-MP, SA-RICH, MS3-RICH; K = 4, ζ ∈ {0.5,1,1.5,2}.
* Misspecification: (1800, 0.03): SIMEX with Σ̂ = κΣ (κ ∈ {0.5, 2}); heteroscedastic truth; temporally
  correlated truth (corr 0.5 at Δt = 0.2).
* Calibration trigger: run if, in Phase B, some fixed default (plan, h) has ≥ 10% median regret
  relative to the oracle-best pipeline in at least one headline cell (B = 1800). Calibration cells
  (1800, 0.003) and (1800, 0.03); B_cal = 264; candidates and defaults as in configs/calibration.yaml.
* Robustness at (1800, 0.01): r ∈ {1.5, 3}; Matérn-5/2; unstructured vector KRR; Hu-fixed estimator.
* Hénon–Heiles implementation anchor (noise-free, Hu et al. settings).

## Amendment 1 (2026-11-19T10:44Z, before any lock or Phase B run)

The first Phase A pass (9 of 24 jobs, all B = 600) selected the upper edge of every B = 600 lengthscale
window and, for a ≤ 0.001, the upper edge of the η window: with fewer training points smoother fits are
preferred, so the Phase 0 windows (fitted at B = 1800) do not transfer to B = 600. Following the edge rule,
that pass was discarded (shards archived in the work directory under tune_aborted_pass1/) and the grids were
made budget-specific: B = 600 uses 3 lengthscales per noise level shifted upward and 6 η values (one more
decade upward); B = 1800 adds a third lengthscale at a ∈ {0, 0.001}, where Phase 0 centering was least
certain. No other design element changed. The lock now also flags ℓ- and η-edge selections; remaining
edge selections are reported, not silently accepted.

## Amendment 2 (2026-11-19T11:20Z, before any lock or Phase B run)

With the budget-specific grids, B = 600 tuning (validation only) still selected the **upper end of the h
window** for the multi-interval Richardson plans at most noise levels (their validation error was still
decreasing at h2 = max h), and, at a = 0.03, MS3-RICH validation error at ℓ = 1.3 in the discarded pass was
lower than anything in the new window. Following the edge rule, every h window is extended by one
doubling step (a ∈ {0, 0.001}: + 0.8; a = 0.003: + 1.131; a ≥ 0.01: + 1.6, i.e. h2 up to 1.6 with
ω_max·h2 ≈ 3, close to the π radius of the MP inverse modified Hamiltonian, so the Richardson expansion is
no longer an asymptotic justification there — it is simply evaluated as an estimator), and the B = 600 ℓ
windows at a ∈ {0.03, 0.1} get one lower value. Implementation: cached trajectory banks are *extended*
(existing states and noise draws unchanged; new times from the same starts, per-time noise seeds; test
`test_bank_extension_preserves_existing_draws`) and Phase A is *filled in* (only missing (spec, ℓ) combinations
are computed into the existing replicate shards; test `test_tuning_fill_in_only_appends`). The same
extension is applied to B = 1800 once its first pass finishes. The discarded first pass is not used for
any selection.

## Amendment 3 (2026-11-19T12:14Z, before any lock or Phase B run)

B = 1800 Phase A (validation only, a ≤ 0.01 complete) selected the lower ℓ edge at a ∈ {0.003, 0.01} for
almost all pipelines (and for MS3-RICH at a = 0) and the lower η edge at a ∈ {0.001, 0.003}; B = 600 at a = 0
selected the upper η edge. Added by fill-in (η-granular; existing rows untouched): B = 1800 ℓ 0.65 (a = 0),
0.85 (a = 0.003), 0.9 (a = 0.01); η 1e-11 (a = 0.001), 1e-10 (a = 0.003); B = 600 η 1e-5, 1e-4 (a = 0).
Not extended (reported as edge selections): η = 1e-11 at a = 0 (ridge ≈ 1e-8 of the Gram diagonal; float64
conditioning floor), single-pipeline ℓ edges (e.g. IP-RICH at a = 0.001). a ∈ {0.03, 0.1} at B = 1800 are
checked after their jobs finish and extended by the same rule if needed. They were: lower-ℓ edge for all
pipelines → ℓ 1.0 (a = 0.03) and 1.15 (a = 0.1) added by the same fill-in.

## Stopping rule for grid extension (2026-11-19T12:57Z)

After the Amendment 2–3 fill-in the lock is written regardless of any remaining edge selections (to avoid
open-ended grid chasing); all remaining h/ℓ/η edge selections are listed in the lock output
(results/locks/main_dp.json fields edge, ell_edge, eta_edge) and in the report.

## Post-hoc (exploratory) additions, after Phase B

* Stronger EIV positive control (configs/positive_control.yaml: B = 600, a = 0.1, forward anchoring at h = 0.05
  and 0.141): added because the pre-registered positive control showed no detectable bias. Exploratory.
* Synthetic correctness tests of SIMEX and of the Hamiltonian-orthogonality explanation
  (tests/test_simex_bias_removal.py). These are code tests on a synthetic linear field, not benchmark results.

## Later pre-registered confirmation (separate study)

The minimum-interval contrast (SA-RICH vs SI-MP / SI-DC at h ≥ 0.4) was observed post hoc after Phase B and then
tested in a separately pre-registered confirmation: `reports/confirmatory_min_interval_protocol_2026-11-19.md`
(outputs in `results/confirmatory/confirm_min_interval/`). Nothing above was changed.
