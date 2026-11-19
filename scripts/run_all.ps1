# Full reproduction on Windows PowerShell (resumable). Run from the project folder.
$ErrorActionPreference = "Stop"
$PY = if ($env:PY) { $env:PY } else { "python" }
function Run { param([string[]]$a) & $PY @a; if ($LASTEXITCODE -ne 0) { throw "failed: $a" } }
Run @("-m","pytest","-q")
Run @("-m","hamsched.cli","phase0","--config","configs/phase0.yaml")
Run @("scripts/phase0b_ell_probe.py")
foreach ($s in "tune","lock","test","analyze") { Run @("-m","hamsched.cli",$s,"--config","configs/main_dp.yaml") }
Run @("-m","hamsched.cli","decompose","--config","configs/decomposition.yaml")
Run @("-m","hamsched.cli","ablate","--config","configs/ablation.yaml")
Run @("-m","hamsched.cli","misspec","--config","configs/misspec.yaml")
Run @("-m","hamsched.cli","calibrate","--config","configs/calibration.yaml")
Run @("-m","hamsched.cli","hh","--config","configs/hh_anchor.yaml")
Run @("-m","hamsched.cli","hufixed","--config","configs/hufixed.yaml")
foreach ($c in "robust_ratio_r15","robust_ratio_r3","robust_matern","robust_unstr") {
  foreach ($s in "tune","lock","test") { Run @("-m","hamsched.cli",$s,"--config","configs/$c.yaml") }
}
Run @("-m","hamsched.cli","secondary","--config","configs/main_dp.yaml")
Run @("-m","hamsched.cli","figures","--config","configs/main_dp.yaml")
