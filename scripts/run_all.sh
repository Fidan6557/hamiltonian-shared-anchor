#!/usr/bin/env bash
# Full reproduction, in order (resumable: re-running skips finished shards). Run from the project folder
# with the virtual environment's python on PATH (or set PY=/path/to/python).
set -euo pipefail
PY=${PY:-python}
$PY -m pytest -q
$PY -m hamsched.cli phase0    --config configs/phase0.yaml
$PY scripts/phase0b_ell_probe.py
$PY -m hamsched.cli tune      --config configs/main_dp.yaml
$PY -m hamsched.cli lock      --config configs/main_dp.yaml
$PY -m hamsched.cli test      --config configs/main_dp.yaml
$PY -m hamsched.cli analyze   --config configs/main_dp.yaml
$PY -m hamsched.cli decompose --config configs/decomposition.yaml
$PY -m hamsched.cli ablate    --config configs/ablation.yaml
$PY -m hamsched.cli misspec   --config configs/misspec.yaml
$PY -m hamsched.cli calibrate --config configs/calibration.yaml
$PY -m hamsched.cli hh        --config configs/hh_anchor.yaml
$PY -m hamsched.cli hufixed   --config configs/hufixed.yaml
for c in robust_ratio_r15 robust_ratio_r3 robust_matern robust_unstr; do
  for s in tune lock test; do $PY -m hamsched.cli $s --config configs/$c.yaml; done
done
$PY -m hamsched.cli secondary --config configs/main_dp.yaml
$PY -m hamsched.cli figures   --config configs/main_dp.yaml
