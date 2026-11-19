"""Experiment orchestration: cached contexts (eval sets, banks), Phase A (tune), locks, Phase B (test).

Per-(cell, replicate) shards are written atomically to a work directory OUTSIDE OneDrive (default
%LOCALAPPDATA%/hamsched_runs) and combined into results/<experiment>_<phase>.csv in the project.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import yaml

from .kernels import make_kernel
from .linalg import backend, set_threads
from .metrics import energy_strata, rel_err_path, stratified_rel_err
from .pipelines import evaluate
from .schedules import build_dataset, plan_times
from .simulate import make_bank, make_eval_set
from .systems import make_system

PURPOSE = {"phase0": 0, "tune": 1, "test": 2, "decomp": 3, "ablate_tune": 4, "ablate_test": 5,
           "misspec": 6, "calib": 7, "robust_tune": 8, "robust_test": 9, "calib_true": 10,
           "calib_sur": 11, "hh": 12, "confirm": 13, "confirm_dryrun": 14, "val": 100, "testset": 101,
           "confirm_testset": 102, "confirm_dryrun_testset": 103}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = path
    return cfg


def default_workdir() -> str:
    return os.environ.get("HAMSCHED_WORKDIR") or os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "hamsched_runs")


def results_dir() -> str:
    d = os.environ.get("HAMSCHED_RESULTS") or os.path.join(PROJECT_ROOT, "results")
    os.makedirs(d, exist_ok=True)
    return d


def atomic_write_csv(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _h(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:12]


def log(msg: str):
    print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class Context:
    def __init__(self, cfg: dict, workdir: str | None = None, threads: int | None = None):
        self.cfg = cfg
        self.root = workdir or default_workdir()
        self.system = make_system(cfg["system"])
        self.kernel = make_kernel(cfg.get("kernel", "se"))
        self.seed = int(cfg.get("seed", 20261119))
        self.syskey = _h(cfg["system"])
        self.cache = os.path.join(self.root, "cache", f"{cfg['system']['name']}_{self.syskey}")
        self.out = os.path.join(self.root, cfg["name"])
        os.makedirs(self.cache, exist_ok=True)
        os.makedirs(self.out, exist_ok=True)
        set_threads(threads or int(cfg.get("threads", 10)))
        self.zrms = self._zrms()
        ev = cfg.get("eval", {})
        self.val = self._evalset("val", ev.get("val", {"n_traj": 150, "n_states": 20, "t_max": 9.5}))
        self.test = self._evalset("testset", ev.get("test", {"n_traj": 300, "n_states": 20, "t_max": 9.5}))
        self.test_strata = energy_strata(self.test.E)

    def _zrms(self) -> float:
        p = os.path.join(self.cache, "zrms.json")
        if os.path.exists(p):
            return json.load(open(p))["zrms"]
        z = self.system.zrms(seed=self.seed)
        json.dump({"zrms": z}, open(p, "w"))
        return z

    def _evalset(self, which: str, spec: dict):
        p = os.path.join(self.cache, f"eval_{which}_{_h(spec)}.npz")
        from .simulate import EvalSet
        if os.path.exists(p):
            d = np.load(p)
            return EvalSet(X=d["X"], F=d["F"], E=d["E"], traj=d["traj"])
        es = make_eval_set(self.system, spec["n_traj"], spec["n_states"], spec["t_max"],
                           seed=(self.seed, PURPOSE[which]))
        np.savez(p, X=es.X, F=es.F, E=es.E, traj=es.traj)
        return es

    def bank(self, purpose: str, rep: int, n_starts: int, times, noise_model: str = "iid", tau=None, sub: int = 0):
        """Cached trajectory bank.  If a cached iid bank of the same (purpose, rep, n_starts, sub) holds a
        SUBSET of the requested times, it is extended: existing states and noise draws are kept unchanged,
        new times are integrated from the same starts and get noise from a per-time seed (so grid
        extensions do not perturb any reading already used)."""
        times = np.unique(np.round(np.concatenate([[0.0], np.asarray(times, float)]), 9))
        key = _h([purpose, rep, n_starts, times.tolist(), noise_model, tau, sub])
        p = os.path.join(self.cache, f"bank_{purpose}_{rep}_{key}.npz")
        from .simulate import Bank, integrate
        if os.path.exists(p):
            d = np.load(p)
            return Bank(z0=d["z0"], times=d["times"], X=d["X"], xi=d["xi"], seed=tuple(d["seed"]))
        seed = (self.seed, PURPOSE[purpose], rep, sub)
        base = None
        if noise_model == "iid":
            import glob
            for f in sorted(glob.glob(os.path.join(self.cache, f"bank_{purpose}_{rep}_*.npz"))):
                d = np.load(f)
                if (len(d["z0"]) == n_starts and tuple(d["seed"]) == seed and str(d.get("noise_model", "iid")) == "iid"
                        and np.all(np.isin(np.round(d["times"], 9), times))):
                    if base is None or len(d["times"]) > len(base["times"]):
                        base = {k: d[k] for k in ("z0", "times", "X", "xi")}
        if base is not None and len(base["times"]) < len(times):
            old_t = np.round(base["times"], 9)
            extra = np.array([t for t in times if t not in set(old_t.tolist())])
            Xe = integrate(self.system, base["z0"], extra)
            xie = np.stack([np.random.default_rng(np.random.SeedSequence(
                [self.seed, PURPOSE[purpose], rep, sub, 7777, int(round(t * 1e6))])).standard_normal((n_starts, 2, 4))
                for t in extra], axis=1)
            allt = np.concatenate([base["times"], extra])
            order = np.argsort(allt)
            b = Bank(z0=base["z0"], times=allt[order], X=np.concatenate([base["X"], Xe], axis=1)[:, order],
                     xi=np.concatenate([base["xi"], xie], axis=1)[:, order], seed=seed)
        else:
            b = make_bank(self.system, n_starts, times, seed=seed, noise_model=noise_model, tau=tau)
        np.savez(p, z0=b.z0, times=b.times, X=b.X, xi=b.xi, seed=np.array(seed), noise_model=noise_model)
        return b


# ----------------------------------------------------------------------------- plan specs

def _match(val, lst):
    return any(abs(float(val) - float(x)) < 1e-12 for x in lst)


def grid_for(cfg: dict, a: float):
    """h grid: a list, or a dict keyed by noise level (with 'default')."""
    g = cfg["h_grid"]
    if isinstance(g, dict):
        for k, v in g.items():
            if k != "default" and abs(float(k) - a) < 1e-12:
                return [float(x) for x in v]
        return [float(x) for x in g["default"]]
    return [float(x) for x in g]


def expand_specs(cfg: dict, B: int, a: float, key: str = "plans"):
    specs = []
    grid = grid_for(cfg, a)
    hmax = max(grid)
    for ps in cfg[key]:
        if "budgets" in ps and B not in ps["budgets"]:
            continue
        if "noise_levels" in ps and not _match(a, ps["noise_levels"]):
            continue
        if "cells" in ps and not any(B == c[0] and abs(a - c[1]) < 1e-12 for c in ps["cells"]):
            continue
        plan, ests = ps["plan"], ps["estimators"]
        if plan in ("SI", "FWD"):
            for h in ps.get("h", grid):
                specs.append((plan, {"h": float(h)}, ests))
        elif plan == "MS3":
            for h in ps.get("h", [x for x in grid if 2 * x <= hmax + 1e-12]):
                specs.append((plan, {"h": float(h)}, ests))
        else:
            r = float(ps.get("r", 2.0))
            for h1 in ps.get("h1", grid):
                h2 = r * float(h1)
                if h2 <= hmax * (1 + 1e-9) or ps.get("allow_beyond", False):
                    specs.append((plan, {"h1": float(h1), "h2": float(h2), "r": r}, ests))
    return specs


def required_times(cfg: dict, key: str = "plans"):
    ts = set()
    for B in cfg["budgets"]:
        for a in cfg["noise_levels"]:
            for plan, params, _ in expand_specs(cfg, B, a, key):
                for t in plan_times(plan, params):
                    ts.add(round(float(t), 9))
    return sorted(ts)


def spec_cols(plan, params):
    if plan in ("SI", "FWD"):
        return params["h"], np.nan
    if plan in ("MS3", "MS3I"):
        return params["h"], 2 * params["h"]
    return params["h1"], params["h2"]


def _by_budget(e, B):
    if isinstance(e, dict) and B is not None and f"B{B}" in e:
        return e[f"B{B}"]
    return e


def etas_for(cfg: dict, a: float, B: int | None = None):
    e = _by_budget(cfg["etas"], B)
    if isinstance(e, dict):
        for k, v in e.items():
            if abs(float(k) - a) < 1e-12:
                return [float(x) for x in v]
        return [float(x) for x in e["default"]]
    return [float(x) for x in e]


def ells_for(cfg: dict, plan: str, a: float | None = None, B: int | None = None):
    """Lengthscale grid: a list, or a dict keyed by noise level (with 'default'), optionally per budget ('B600')."""
    e = _by_budget(cfg["ells"], B)
    if isinstance(e, dict):
        if a is not None:
            for k, v in e.items():
                if k != "default" and abs(float(k) - a) < 1e-12:
                    return [float(x) for x in v]
        return [float(x) for x in e["default"]]
    return [float(x) for x in e]


def n_bank(cfg: dict) -> int:
    return max(cfg["budgets"]) // 2


# ----------------------------------------------------------------------------- Phase A

def _done_set(existing):
    """(plan, h1, h2, est, ell) -> set of rounded log10(eta) already computed."""
    done = {}
    if existing is None or not len(existing):
        return done
    h2 = existing["h2"].fillna(-1.0)
    for k, e in zip(zip(existing["plan"], existing["h1"].round(9), h2.round(9), existing["est"],
                        existing["ell"].round(9)), existing["eta"]):
        done.setdefault(k, set()).add(round(float(np.log10(e)), 6))
    return done


def tune_job(ctx: Context, B: int, a: float, rep: int, purpose: str = "tune", key: str = "plans",
             existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Phase A rows; with `existing`, only (spec, ell) combinations not yet present are computed (fill-in)."""
    cfg = ctx.cfg
    bank = ctx.bank(purpose, rep, n_bank(cfg), required_times(cfg, key))
    sigma = a * ctx.zrms
    etas = etas_for(cfg, a, B)
    done = _done_set(existing)
    rows = []
    for plan, params, ests in expand_specs(cfg, B, a, key):
        ds = build_dataset(plan, params, B, bank, sigma)
        h1, h2 = spec_cols(plan, params)
        for ell in ells_for(cfg, plan, a, B):
            key0 = (plan, round(h1, 9), round(-1.0 if np.isnan(h2) else h2, 9))
            miss = {e: [eta for eta in etas if round(float(np.log10(eta)), 6) not in done.get(key0 + (e, round(ell, 9)), set())]
                    for e in ests}
            todo = [e for e in ests if miss[e]]
            if not todo:
                continue
            etas_run = sorted({eta for e in todo for eta in miss[e]})
            t0 = time.time()
            preds, st = evaluate(ds, todo, ctx.kernel, ell, etas_run, {"val": ctx.val.X})
            dt = time.time() - t0
            for e in todo:
                errs = rel_err_path(preds[e]["val"], ctx.val.F)
                for j, eta in enumerate(etas_run):
                    if eta not in miss[e]:
                        continue
                    rows.append(dict(B=B, a=a, rep=rep, plan=plan, h1=h1, h2=h2, est=e, ell=ell, eta=eta,
                                     val_err=errs[j], status=int(st[e][j]), fit_time=dt, **ds.info))
    return pd.DataFrame(rows)


def run_phase(ctx: Context, phase: str, fn, jobs, fill_in: bool = False, **kw) -> pd.DataFrame:
    outdir = os.path.join(ctx.out, phase)
    os.makedirs(outdir, exist_ok=True)
    t_start = time.time()
    for k, (B, a, rep) in enumerate(jobs):
        path = os.path.join(outdir, f"B{B}_a{a:g}_r{rep}.csv")
        existing = None
        if os.path.exists(path):
            if not fill_in:
                continue
            existing = pd.read_csv(path)
        t0 = time.time()
        df = fn(ctx, B, a, rep, existing=existing, **kw) if fill_in else fn(ctx, B, a, rep, **kw)
        if existing is not None:
            if not len(df):
                continue
            df = pd.concat([existing, df], ignore_index=True)
        atomic_write_csv(df, path)
        log(f"{ctx.cfg['name']}/{phase}: job {k + 1}/{len(jobs)} B={B} a={a:g} rep={rep} "
            f"rows={len(df)} {time.time() - t0:.0f}s (elapsed {(time.time() - t_start) / 60:.1f} min)")
    return combine(ctx, phase)


def combine(ctx: Context, phase: str) -> pd.DataFrame:
    outdir = os.path.join(ctx.out, phase)
    files = sorted(f for f in os.listdir(outdir) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(outdir, f)) for f in files], ignore_index=True) if files else pd.DataFrame()
    if len(df):
        atomic_write_csv(df, os.path.join(results_dir(), f"{ctx.cfg['name']}_{phase}.csv"))
    return df


# ----------------------------------------------------------------------------- locks

CFG_KEYS = ["B", "a", "plan", "h1", "h2", "est"]


def _score(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["logerr"] = np.log(d["val_err"].astype(float))
    d.loc[~np.isfinite(d["logerr"]), "logerr"] = np.inf
    g = d.groupby(CFG_KEYS + ["ell", "eta"], dropna=False)["logerr"].mean().reset_index()
    return g


def make_locks(tune_df: pd.DataFrame, cfg: dict) -> dict:
    g = _score(tune_df)
    g = g.sort_values("logerr", kind="stable")
    best_hp = g.groupby(CFG_KEYS, dropna=False).head(1).reset_index(drop=True)
    entries = []
    best = []
    hmins = cfg.get("h_min", [])
    test_all = set(tuple(x) for x in cfg.get("test_all_h", []))
    # selected h (and constrained selections) of every estimator, per plan: siblings are evaluated there too
    sel_by_plan = {}
    for (B, a, plan, est), sub in best_hp.groupby(["B", "a", "plan", "est"], dropna=False):
        sub = sub.sort_values("h1").reset_index(drop=True)
        hs = {float(sub.h1[int(np.argmin(sub["logerr"].values))])}
        for hm in hmins:
            ok = np.where(sub.h1.values >= hm - 1e-12)[0]
            if len(ok):
                hs.add(float(sub.h1[ok[int(np.argmin(sub["logerr"].values[ok]))]]))
        sel_by_plan.setdefault((int(B), float(a), plan), set()).update(hs)
    for (B, a, plan, est), sub in best_hp.groupby(["B", "a", "plan", "est"], dropna=False):
        sub = sub.sort_values("h1").reset_index(drop=True)
        ib = int(np.argmin(sub["logerr"].values))
        hgrid = sub["h1"].values
        chosen = {ib}
        rec = {"B": int(B), "a": float(a), "plan": plan, "est": est,
               "h1": float(sub.h1[ib]), "h2": None if pd.isna(sub.h2[ib]) else float(sub.h2[ib]),
               "ell": float(sub.ell[ib]), "eta": float(sub.eta[ib]), "val_logerr": float(sub.logerr[ib]),
               "edge": bool(ib == 0 or ib == len(sub) - 1), "constrained": {}}
        try:
            eg = etas_for(cfg, float(a), int(B))
            lg = ells_for(cfg, plan, float(a), int(B))
            rec["eta_edge"] = bool(abs(np.log(rec["eta"] / min(eg))) < 1e-9 or abs(np.log(rec["eta"] / max(eg))) < 1e-9)
            rec["ell_edge"] = bool(len(lg) > 1 and (abs(rec["ell"] - min(lg)) < 1e-12 or abs(rec["ell"] - max(lg)) < 1e-12))
        except (KeyError, ValueError):
            pass
        for hm in hmins:
            ok = np.where(hgrid >= hm - 1e-12)[0]
            if len(ok):
                j = ok[int(np.argmin(sub["logerr"].values[ok]))]
                chosen.add(j)
                rec["constrained"][str(hm)] = {"h1": float(sub.h1[j]), "val_logerr": float(sub.logerr[j])}
        best.append(rec)
        for hsel in sel_by_plan.get((int(B), float(a), plan), set()):
            jj = np.where(np.abs(hgrid - hsel) < 1e-9)[0]
            if len(jj):
                chosen.add(int(jj[0]))
        neigh = set()
        for j in chosen:
            neigh |= {j - 1, j, j + 1}
        for j in range(len(sub)):
            evaluate_it = ((plan, est) in test_all) or (j in neigh)
            entries.append({"B": int(B), "a": float(a), "plan": plan, "est": est, "h1": float(sub.h1[j]),
                            "h2": None if pd.isna(sub.h2[j]) else float(sub.h2[j]), "ell": float(sub.ell[j]),
                            "eta": float(sub.eta[j]), "val_logerr": float(sub.logerr[j]),
                            "evaluate": bool(evaluate_it)})
    body = {"experiment": cfg["name"], "entries": entries, "best": best}
    lock = {"created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "sha256": _h(body), **body}
    return lock


def save_lock(lock: dict, name: str) -> str:
    d = os.path.join(results_dir(), "locks")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}.json")
    with open(p, "w") as f:
        json.dump(lock, f, indent=1)
    return p


def load_lock(name: str) -> dict:
    with open(os.path.join(results_dir(), "locks", f"{name}.json")) as f:
        return json.load(f)


def _params_from_entry(e):
    if e["plan"] in ("SI", "FWD"):
        return {"h": e["h1"]}
    if e["plan"] == "MS3":
        return {"h": e["h1"]}
    return {"h1": e["h1"], "h2": e["h2"], "r": e["h2"] / e["h1"]}


# ----------------------------------------------------------------------------- Phase B

def test_job(ctx: Context, B: int, a: float, rep: int, lock: dict, purpose: str = "test", key: str = "plans",
             scale_fn=None, noise_model: str = "iid", tau=None, tag: str = "") -> pd.DataFrame:
    cfg = ctx.cfg
    bank = ctx.bank(purpose, rep, n_bank(cfg), required_times(cfg, key), noise_model=noise_model, tau=tau)
    sigma = a * ctx.zrms
    ents = [e for e in lock["entries"] if e["B"] == B and abs(e["a"] - a) < 1e-12 and e["evaluate"]]
    rows = []
    cache = {}
    for e in ents:
        params = _params_from_entry(e)
        k = (e["plan"], json.dumps(params, sort_keys=True))
        if k not in cache:
            cache = {k: build_dataset(e["plan"], params, B, bank, sigma, scale_fn=scale_fn)}
        ds = cache[k]
        t0 = time.time()
        preds, st = evaluate(ds, [e["est"]], ctx.kernel, e["ell"], [e["eta"]], {"test": ctx.test.X})
        P = preds[e["est"]]["test"][0]
        err = rel_err_path(P[None], ctx.test.F)[0]
        strat = stratified_rel_err(P, ctx.test.F, ctx.test_strata) if np.isfinite(err) else [np.nan] * 3
        rows.append(dict(B=B, a=a, rep=rep, plan=e["plan"], h1=e["h1"], h2=e["h2"], est=e["est"], ell=e["ell"],
                         eta=e["eta"], test_err=err, err_E0=strat[0], err_E1=strat[1], err_E2=strat[2],
                         status=int(st[e["est"]][0]), fit_time=time.time() - t0, lock_sha=lock["sha256"],
                         tag=tag, **ds.info))
    return pd.DataFrame(rows)


def jobs_for(cfg: dict, n_rep: int, budgets=None, noise_levels=None):
    budgets = budgets or cfg["budgets"]
    noise_levels = noise_levels if noise_levels is not None else cfg["noise_levels"]
    jobs = []
    for B in sorted(budgets):
        for a in noise_levels:
            for r in range(n_rep):
                jobs.append((int(B), float(a), r))
    return jobs


def env_metadata() -> dict:
    import platform

    import numpy
    import scipy
    md = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
          "pandas": pd.__version__, "linalg_backend": backend(), "platform": platform.platform(),
          "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    try:
        import torch
        md["torch"] = torch.__version__
    except Exception:
        pass
    return md
