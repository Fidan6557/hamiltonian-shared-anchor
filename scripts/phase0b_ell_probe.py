"""Phase 0b (exploratory, phase0 seeds): lengthscale range probe after Phase 0 selected the grid edge
ell = 1.3 at every noisy level.  Output: results/phase0b_ell.csv.  Used only to set the main ell windows."""
import os
import sys

import numpy as np
import pandas as pd

from hamsched.experiment import Context, load_config, n_bank, required_times, results_dir
from hamsched.metrics import rel_err_path
from hamsched.pipelines import evaluate
from hamsched.schedules import build_dataset

ctx = Context(load_config("configs/phase0.yaml"))
bank = ctx.bank("phase0", 0, n_bank(ctx.cfg), required_times(ctx.cfg))
ETAS = {0.0: [1e-11, 1e-10, 1e-9, 1e-8, 1e-7], 0.003: [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        0.01: [1e-8, 1e-7, 1e-6, 1e-5, 1e-4], 0.1: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
rows = []
for a, etas in ETAS.items():
    for plan, params, est in (("SI", {"h": 0.2 if a < 0.01 else 0.4}, "MP"),
                              ("MS3", {"h": 0.2 if a < 0.01 else 0.4}, "RICH"),
                              ("SA", {"h1": 0.2 if a < 0.01 else 0.4, "h2": 0.4 if a < 0.01 else 0.8}, "RICH")):
        ds = build_dataset(plan, params, 1800, bank, a * ctx.zrms)
        for ell in ([0.6, 0.8, 1.0, 1.3] if a == 0 else [1.3, 1.8, 2.5, 3.5]):
            p, st = evaluate(ds, [est], ctx.kernel, ell, etas, {"val": ctx.val.X})
            e = rel_err_path(p[est]["val"], ctx.val.F)
            j = int(np.nanargmin(e))
            rows.append(dict(a=a, plan=plan, est=est, ell=ell, best_eta=etas[j], val_err=e[j]))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(results_dir(), "phase0b_ell.csv"), index=False)
