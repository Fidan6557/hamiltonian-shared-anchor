#!/usr/bin/env bash
# Runs all studies that follow the main Phase B, sequentially (one heavy process at a time).
# Usage: PY=/path/to/python bash scripts/post_phaseB_chain.sh   (from the project folder)
set -u
PY=${PY:-python}
L=${LOGDIR:-"$LOCALAPPDATA/hamsched_runs/logs"}
mkdir -p "$L"
run () { echo "[$(date +%H:%M:%S)] START $*"; "$PY" -m hamsched.cli "$@" > "$L/chain_$1_$(basename ${3:-x} .yaml).log" 2>&1; rc=$?; echo "[$(date +%H:%M:%S)] END rc=$rc $*"; return $rc; }
run analyze   --config configs/main_dp.yaml
run decompose --config configs/decomposition.yaml
run ablate    --config configs/ablation.yaml
run misspec   --config configs/misspec.yaml
if "$PY" -c "import json,sys; sys.exit(0 if json.load(open('results/analysis/main_dp_calibration_trigger.json'))['triggered'] else 1)"; then
  run calibrate --config configs/calibration.yaml
else
  echo "[$(date +%H:%M:%S)] calibration NOT triggered (pre-registered rule) - skipped"
fi
for c in robust_ratio_r15 robust_ratio_r3 robust_matern robust_unstr; do
  for s in tune lock test analyze; do run $s --config configs/$c.yaml; done
done
run secondary --config configs/main_dp.yaml
run figures   --config configs/main_dp.yaml
echo "[$(date +%H:%M:%S)] CHAIN DONE"
