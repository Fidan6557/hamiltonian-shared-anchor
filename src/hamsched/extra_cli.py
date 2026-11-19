"""Additional CLI commands (secondary studies, analysis, figures, smoke)."""
from __future__ import annotations

import json
import os
import shutil

import yaml


def _main_ctx(args):
    """Context of the main experiment named in the study config (`main_config`), plus the study cfg."""
    from .experiment import Context, load_config
    scfg = load_config(args.config)
    mcfg = load_config(scfg["main_config"])
    ctx = Context(mcfg, workdir=args.workdir, threads=args.threads)
    return ctx, scfg


def cmd_decompose(args):
    from .experiment import load_lock
    from .studies import decompose
    ctx, scfg = _main_ctx(args)
    decompose(ctx, scfg, load_lock(ctx.cfg["name"]))


def cmd_ablate(args):
    from .experiment import load_lock
    from .studies import ablate
    ctx, scfg = _main_ctx(args)
    ablate(ctx, scfg, load_lock(ctx.cfg["name"]))


def cmd_misspec(args):
    from .experiment import load_lock
    from .studies import misspec
    ctx, scfg = _main_ctx(args)
    misspec(ctx, scfg, load_lock(ctx.cfg["name"]))


def cmd_calibrate(args):
    from .experiment import load_lock
    from .studies import calibrate
    ctx, scfg = _main_ctx(args)
    calibrate(ctx, scfg, load_lock(ctx.cfg["name"]))


def cmd_hufixed(args):
    from .studies import hu_fixed_dp
    ctx, scfg = _main_ctx(args)
    hu_fixed_dp(ctx, scfg)


def cmd_hh(args):
    from .experiment import Context, load_config
    from .studies import hh_anchor
    cfg = load_config(args.config)
    ctx = Context(cfg, workdir=args.workdir, threads=args.threads)
    hh_anchor(ctx, cfg["hh"])


def cmd_analyze(args):
    from .analysis import calibration_trigger, summarize, write_md_tables
    from .experiment import load_config
    cfg = load_config(args.config)
    summarize(cfg["name"])
    write_md_tables(cfg["name"])
    cal = os.path.join(os.path.dirname(os.path.abspath(args.config)), "calibration.yaml")
    if cfg["name"] == "main_dp" and os.path.exists(cal):
        trig = calibration_trigger(cfg["name"], load_config(cal)["defaults"])
        print("calibration trigger:", trig["triggered"])
    print(f"analysis written for {cfg['name']}")


def cmd_figures(args):
    from .experiment import load_config
    from .plotting import all_figures
    cfg = load_config(args.config)
    for p in all_figures(cfg["name"]):
        print("figure:", p)


def cmd_smoke(args):
    """Tiny end-to-end run in a throwaway work/results directory (does not touch results/)."""
    import tempfile

    from . import cli
    root = args.workdir or tempfile.mkdtemp(prefix="hamsched_smoke_")
    os.environ["HAMSCHED_RESULTS"] = os.path.join(root, "results")
    cfgp = args.config or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                       "configs", "smoke.yaml")
    for c in ("tune", "lock", "test", "analyze"):
        cli.main([c, "--config", cfgp, "--workdir", root])
    print("smoke OK:", root)
    if not args.workdir:
        shutil.rmtree(root, ignore_errors=True)


COMMANDS = {k[4:]: v for k, v in dict(globals()).items() if k.startswith("cmd_")}
_ = yaml


def cmd_secondary(args):
    """Summaries of all secondary studies present in results/ (tables + robustness)."""
    from .experiment import load_config
    from .secondary import all_secondary
    for k, v in all_secondary(load_config(args.config)["name"]).items():
        print(k, "rows:", None if v is None else len(v))


COMMANDS["secondary"] = cmd_secondary


def cmd_confirm(args):
    """Pre-registered confirmatory run + analysis (separate outputs under results/confirmatory/)."""
    from .confirm import confirm
    comp, summary = confirm(args.config, workdir=args.workdir)
    print(comp.to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "env"}, indent=1))


def cmd_posthoc_min_interval(args):
    """Save the original post-hoc minimum-interval reading (from the original test rows) separately."""
    from .confirm import posthoc_from_original
    print(posthoc_from_original().round(3).to_string(index=False))


COMMANDS["confirm"] = cmd_confirm
COMMANDS["posthoc_min_interval"] = cmd_posthoc_min_interval
