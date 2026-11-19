"""Command-line entry point:  python -m hamsched.cli <command> --config configs/X.yaml [--workdir DIR]

Commands
  phase0     exploratory floor/bias/variance probe on separate seeds (grids are fixed afterwards)
  tune       Phase A: oracle validation tuning over h, ell, eta (R_tune replicates per cell)
  lock       write results/locks/<name>.json (sha256 + timestamp) from the tuning results
  test       Phase B: fresh seeds, locked settings, clean held-out test trajectories
  decompose  error decomposition + mechanism checks at headline cells
  ablate     correction ablation (SIMEX 2x2)
  misspec    noise-model misspecification
  calibrate  non-oracle schedule selection from a calibration set (counted in the budget)
  hh         Henon-Heiles implementation anchor (Hu et al. settings, noise-free slope)
  robust     robustness at one cell (ratio r, Matern kernel, unstructured KRR, Hu-fixed estimator)
  analyze    tables + primary contrasts (Holm) from results CSVs
  figures    figures from results CSVs
  smoke      tiny end-to-end run (tune -> lock -> test -> analyze) for CI / quick checks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd


def _ctx(args):
    from .experiment import Context, load_config
    cfg = load_config(args.config)
    return Context(cfg, workdir=args.workdir, threads=args.threads)


def cmd_phase0(args):
    from .experiment import atomic_write_csv, jobs_for, log, results_dir, run_phase, tune_job
    from .krr import GradFit
    from .metrics import rel_err_path
    ctx = _ctx(args)
    cfg = ctx.cfg
    fl = cfg["floor"]
    path = os.path.join(results_dir(), "phase0_floor.csv")
    if not os.path.exists(path):
        rows = []
        rng = np.random.default_rng((ctx.seed, 0, 999))
        for n in fl["n"]:
            X = ctx.system.sample(n, rng)
            g = ctx.system.grad(X)
            for ell in fl["ells"]:
                t0 = time.time()
                fit = GradFit(X, ctx.kernel, ell * X.std(0), ("diag", np.ones(n)))
                for eta in fl["etas"]:
                    fac, st = fit.factor(eta)
                    err = np.nan
                    if fac is not None:
                        c = fit.solve(fac, g)
                        err = rel_err_path(fit.field(c, ctx.val.X)[None], ctx.val.F)[0]
                    rows.append(dict(n=n, ell=ell, eta=eta, val_err=err, status=st, time=time.time() - t0))
                log(f"floor n={n} ell={ell} best={min(r['val_err'] for r in rows[-len(fl['etas']):]):.4f}")
        atomic_write_csv(pd.DataFrame(rows), path)
    jobs = jobs_for(cfg, 1)
    run_phase(ctx, "tune", tune_job, jobs, purpose="phase0")


def cmd_tune(args):
    from .experiment import jobs_for, run_phase, tune_job
    ctx = _ctx(args)
    jobs = jobs_for(ctx.cfg, int(ctx.cfg["reps"]["tune"]))
    # fill-in: existing shards are completed with any (spec, ell) combinations added to the config
    run_phase(ctx, "tune", tune_job, jobs, fill_in=True, purpose="tune")


def cmd_lock(args):
    from .experiment import combine, load_config, make_locks, save_lock
    from .experiment import Context
    cfg = load_config(args.config)
    ctx = Context(cfg, workdir=args.workdir, threads=args.threads)
    df = combine(ctx, "tune")
    lock = make_locks(df, cfg)
    p = save_lock(lock, cfg["name"])
    edges = [b for b in lock["best"] if b["edge"] or b.get("eta_edge") or b.get("ell_edge")]
    print(f"lock written: {p} sha={lock['sha256']} entries={len(lock['entries'])} "
          f"evaluate={sum(e['evaluate'] for e in lock['entries'])} edge_selections={len(edges)}")
    for b in edges:
        kinds = [k for k, f in (("h", b["edge"]), ("eta", b.get("eta_edge")), ("ell", b.get("ell_edge"))) if f]
        print(f"  edge[{','.join(kinds)}]: B={b['B']} a={b['a']} {b['plan']}-{b['est']} h1={b['h1']} "
              f"ell={b['ell']} eta={b['eta']:.0e}")


def cmd_test(args):
    from .experiment import env_metadata, jobs_for, load_lock, results_dir, run_phase, test_job
    ctx = _ctx(args)
    lock = load_lock(ctx.cfg["name"])
    meta = {"lock_sha256": lock["sha256"], "lock_created_utc": lock["created_utc"], **env_metadata()}
    mp = os.path.join(results_dir(), f"{ctx.cfg['name']}_test_meta.json")
    if not os.path.exists(mp):
        json.dump(meta, open(mp, "w"), indent=1)
    jobs = jobs_for(ctx.cfg, int(ctx.cfg["reps"]["test"]))
    run_phase(ctx, "test", test_job, jobs, lock=lock)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hamsched")
    ap.add_argument("command")
    ap.add_argument("--config", required=False)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)
    cmds = {k[4:]: v for k, v in globals().items() if k.startswith("cmd_")}
    try:
        from . import extra_cli
        cmds.update(extra_cli.COMMANDS)
    except ImportError:
        pass
    if args.command not in cmds:
        print(f"unknown command {args.command}; available: {sorted(cmds)}")
        sys.exit(2)
    cmds[args.command](args)


if __name__ == "__main__":
    main()
