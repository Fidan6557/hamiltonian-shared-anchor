# Shared-anchor measurement scheduling for Hamiltonian vector-field recovery

Reproducible study of **which measurement schedule recovers the physical Hamiltonian vector field
f = J∇H₀ best from noisy (q, p) readings under a fixed budget of unique readings**:

| plan | readings | starts per B | pairs |
|---|---|---|---|
| SI(h) single interval | (0, h) | B/2 | B/2 |
| IP(h1,h2) independent pairs | (0,h1) and (0,h2) on two different starts | B/2 | B/2 |
| RA(h1,h2) re-measured anchor (Hu et al. Sec. 6.3 noise protocol) | 0 twice, h1, h2 on one start | B/4 | B/2 |
| **SA(h1,h2) shared anchor** | (0, h1, h2), pairs (0,h1), (0,h2) share reading 0 | B/3 | 2B/3 |
| MS3(h) multistride (Hu et al. Sec. 5 style) | (0, h, 2h), pairs (0,h), (h,2h), (0,2h) | B/3 | B |

Main benchmark: the planar double pendulum (nonlinear, nonseparable). Secondary: Hénon–Heiles, used
only as an implementation anchor against Hu et al. (arXiv:2610.17835).
The report is in [reports/REPORT.md](reports/REPORT.md); derivations in [docs/theory.md](docs/theory.md).

## Requirements

* Windows/Linux/macOS, CPython 3.12 (tested: 3.12.6 on Windows 11), **CPU only** (no GPU used; the
  code is numpy/scipy, with torch as an optional float64 Cholesky/prediction accelerator on CPU).
* ~2 GB free RAM for the largest fits (4800 × 4800 float64 systems).

## Setup

```bash
# from the project folder
uv venv ../.venvs/hamsched --python 3.12          # or: python -m venv .venv
uv pip install --python ../.venvs/hamsched/Scripts/python.exe -r requirements.txt
uv pip install --python ../.venvs/hamsched/Scripts/python.exe -e .
# optional accelerator (CPU wheel):
uv pip install --python ../.venvs/hamsched/Scripts/python.exe -r requirements-accel.txt --index-url https://download.pytorch.org/whl/cpu
```

With plain pip: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && pip install -e .`
(On OneDrive-synced folders, keep the virtual environment outside the synced folder.)

Without installing the package you can set `PYTHONPATH=src` and call `python -m hamsched.cli ...`.
Force the scipy backend with `HAMSCHED_LINALG=scipy` (results agree to ~1e-10; see tests).

## Reproduce

All commands are run from the project folder. Intermediate shards/caches go to
`%LOCALAPPDATA%\hamsched_runs` (override with `--workdir` or `HAMSCHED_WORKDIR`); combined
machine-readable results are written to `results/`, figures to `figures/`.

```bash
python -m pytest -q                                              # unit + slope-gate + smoke tests
python -m hamsched.cli smoke                                     # tiny end-to-end run (~20 s)

python -m hamsched.cli phase0    --config configs/phase0.yaml     # exploratory (grids fixed from it)
python -m hamsched.cli tune      --config configs/main_dp.yaml    # Phase A: oracle validation tuning
python -m hamsched.cli lock      --config configs/main_dp.yaml    # results/locks/main_dp.json (sha256, UTC time)
python -m hamsched.cli test      --config configs/main_dp.yaml    # Phase B: fresh seeds, locked settings
python -m hamsched.cli analyze   --config configs/main_dp.yaml    # tables + pre-registered contrasts (Holm)
python -m hamsched.cli decompose --config configs/decomposition.yaml
python -m hamsched.cli ablate    --config configs/ablation.yaml
python -m hamsched.cli misspec   --config configs/misspec.yaml
python -m hamsched.cli calibrate --config configs/calibration.yaml
python -m hamsched.cli hh        --config configs/hh_anchor.yaml
python -m hamsched.cli hufixed   --config configs/hufixed.yaml
python -m hamsched.cli tune|lock|test|analyze --config configs/robust_*.yaml   # robustness sub-experiments
python -m hamsched.cli secondary --config configs/main_dp.yaml   # tables for all secondary studies
python -m hamsched.cli figures   --config configs/main_dp.yaml
# exploratory (post hoc, labelled as such in the report):
python -m hamsched.cli decompose --config configs/positive_control.yaml
python scripts/phase0b_ell_probe.py                              # Phase 0b lengthscale probe
# pre-registered minimum-interval confirmation (protocol in reports/, outputs in results/confirmatory/):
python -m hamsched.cli confirm --config configs/confirm_min_interval.yaml
# figures of the journal manuscript (read-only from results/; written to paper/figures/):
python paper/make_figures.py
```

`scripts/post_phaseB_chain.sh` runs everything after Phase B in the order used here; `scripts/run_all.sh` /
`scripts/run_all.ps1` run the whole study. Each step skips shards that already exist; Phase A also
*fills in* grid points added to a config (only missing (spec, ℓ, η) combinations are computed, existing
noise draws are reused), which is how the pre-registered grid amendments were applied.

Every step is resumable (per-cell/replicate shards are skipped if present). The full study took ≈ 10 h of
wall-clock on a 15 W laptop (i7-1355U, CPU only, one experiment process at a time); the smoke run takes ≈ 20 s
and the test suite ≈ 15 s.

## Results at a glance (details and caveats in reports/REPORT.md)

* Shared anchor vs independent pairs at identical intervals: SA-RICH has lower error in all 12 (budget, noise)
  cells (error ratio 0.67–0.92). The gain is also present without noise; the positive correlation between the
  two stride fits (+0.46) is one mechanism under noise, not the whole explanation.
* Shared anchor vs a tuned single interval with the interval chosen freely: SA-RICH has higher error in all 12
  cells (ratio 1.06–1.39).
* Minimum sampling interval h ≥ 0.4 (pre-registered confirmation; B = 1800, a = 0.001, 0.003, 0.01; 20 fresh
  paired replicates and a fresh test set): SA-RICH has lower error than plain SI-MP (ratios 0.634–0.841) but
  higher error than the defect-corrected SI-DC (ratios 1.191–1.226); all six predictions supported after Holm
  adjustment. Not claimed for other budgets, intervals, noise levels or systems.
* Using all three pairs of the same readings (multistride Richardson, MS3-RICH) is the best tested 3-reading
  estimator and ties or modestly beats a single interval with model-based defect correction in 11 of 12 cells.
* No remaining measurement-error bias was detectable above Monte Carlo error in the tested settings. For a single
  forward-anchored pair under isotropic Gaussian noise the correlation term is orthogonal to Hamiltonian fields
  (docs/theory.md §7); this does not remove all measurement-error bias. Known-Σ SIMEX increased test error by
  4–32% in the ablation cells.
* A 15% calibration set picked a good schedule and interval (regret 1.10–1.23 vs oracle; 5–25% better than
  fixed defaults).

## Layout

```
src/hamsched/   systems (H, ∇H, invariant samplers) · simulate (DOP853 banks) · schedules (plans, reading IDs,
                budget accounting) · kernels/krr/linalg (gradient KRR) · pipelines (MP, DC, POOL, GLS, RICH, LONG)
                · corrections (RB-SIMEX) · theory · experiment (Phase A/B, locks) · studies · analysis · plotting · cli
configs/        all experiment configurations (seeds included)
tests/          unit tests, slope gates, Monte Carlo covariance checks, smoke pipeline
results/        CSV/JSON results, locks, analysis tables      figures/  PNG figures generated from results
reports/        REPORT.md (+ tables/)                          docs/     theory.md
notebooks/      colab_wrapper.ipynb (installs the package and calls the same CLI)
paper/          make_figures.py + figures/ for the journal manuscript (the manuscript itself is not included)
```
