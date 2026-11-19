"""Secondary studies: error decomposition & mechanism, SIMEX ablation, misspecification, calibration,
Henon-Heiles implementation anchor, Hu-fixed reference."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .corrections import simex
from .experiment import (PURPOSE, _params_from_entry, atomic_write_csv, log, n_bank, required_times,
                         results_dir, spec_cols)
from .kernels import make_kernel
from .metrics import rel_err, rel_err_path, stratified_rel_err
from .pipelines import evaluate, fit_field_model, rk4_flow
from .schedules import build_dataset
from .simulate import Bank


def _best(lock, B, a, plan, est):
    for b in lock["best"]:
        if b["B"] == B and abs(b["a"] - a) < 1e-12 and b["plan"] == plan and b["est"] == est:
            return b
    raise KeyError((B, a, plan, est))


def _entry(lock, B, a, plan, est, h1):
    for e in lock["entries"]:
        if (e["B"] == B and abs(e["a"] - a) < 1e-12 and e["plan"] == plan and e["est"] == est
                and abs(e["h1"] - h1) < 1e-9):
            return e
    raise KeyError((B, a, plan, est, h1))


def _grid_neighbors(lock, B, a, plan, est, h1):
    hs = sorted({e["h1"] for e in lock["entries"] if e["B"] == B and abs(e["a"] - a) < 1e-12
                 and e["plan"] == plan and e["est"] == est})
    i = int(np.argmin(np.abs(np.array(hs) - h1)))
    return [hs[j] for j in (i - 1, i, i + 1) if 0 <= j < len(hs)]


def _redraw(bank: Bank, seed) -> Bank:
    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    return Bank(z0=bank.z0, times=bank.times, X=bank.X, xi=rng.standard_normal(bank.xi.shape), seed=tuple(seed))


def _params(plan, h1, h2=None):
    if plan in ("SI", "FWD", "MS3", "MS3I"):
        return {"h": h1}
    return {"h1": h1, "h2": h2, "r": h2 / h1}


def _budget(plan, B):
    return 5 * B // 3 if plan == "MS3I" else B


# ============================================================================ decomposition

def decompose(ctx, dcfg: dict, lock: dict):
    """Sequential decomposition at fixed starts: floor (exact f at the plan's clean anchors),
    discretization (clean - exact), measurement-error bias (mean noisy - clean, MC-corrected), variance."""
    out = os.path.join(ctx.out, dcfg.get("out_name", "decomp"))
    os.makedirs(out, exist_ok=True)
    F = ctx.test.F
    nF2 = float(np.sum(F**2))
    times = required_times(ctx.cfg)
    for (B, a) in dcfg["cells"]:
        sigma = a * ctx.zrms
        for s in range(dcfg["n_startsets"]):
            shard = os.path.join(out, f"B{B}_a{a:g}_s{s}.csv")
            if os.path.exists(shard):
                continue
            t0 = time.time()
            bank0 = ctx.bank("decomp", s, n_bank(ctx.cfg), times)
            draws = [_redraw(bank0, (ctx.seed, PURPOSE["decomp"], s, 1000 + d)) for d in range(dcfg["n_draws"])]
            rows = []
            # pipeline list: locked headline pipelines + mechanism set at a common h1 and (ell, eta)
            pipes = []
            for p in dcfg["headline"]:
                b = _best(lock, B, a, p["lock"][0], p["lock"][1])
                if "h_override" in p:
                    b = dict(_entry(lock, B, a, p["lock"][0], p["lock"][1], p.get("hp_from_h", p["h_override"])))
                    b["h1"] = p["h_override"]
                h1 = b["h1"]
                h2 = b["h2"] if b["h2"] is not None else None
                pipes.append(dict(name=p["name"], plan=p["plan"], est=p["est"], h1=h1, h2=h2, ell=b["ell"],
                                  eta=b["eta"], simex=p.get("simex", False), group="headline"))
            mech = dcfg.get("mechanism")
            if mech and any(B == c[0] and abs(a - c[1]) < 1e-12 for c in dcfg.get("mechanism_cells", dcfg["cells"])):
                hm = mech["h1"]
                ref = _entry(lock, B, a, mech["ref"][0], mech["ref"][1], hm)
                for p in mech["pipelines"]:
                    h1 = hm * p.get("h_mult", 1.0)
                    h2 = (p.get("r", 2.0) * h1) if p["plan"] not in ("SI", "FWD") else None
                    pipes.append(dict(name=p["name"], plan=p["plan"], est=p["est"], h1=h1, h2=h2, ell=ref["ell"],
                                      eta=ref["eta"], simex=False, group="mechanism"))
            for pp in pipes:
                plan, est = pp["plan"], pp["est"]
                params = _params(plan, pp["h1"], pp["h2"])
                Bp = _budget(plan, B)
                ds0 = build_dataset(plan, params, Bp, bank0, sigma)
                L = pp["ell"] * ds0.readings.clean.std(0)
                ests = [est] + (["SHORT", "LONG"] if est == "RICH" else [])
                ev = {"test": ctx.test.X}
                pe, ste = evaluate(ds0, [est], ctx.kernel, pp["ell"], [pp["eta"]], ev, L=L,
                                   values=ds0.readings.clean, exact_fn=ctx.system.field)
                pc, stc = evaluate(ds0, [est], ctx.kernel, pp["ell"], [pp["eta"]], ev, L=L,
                                   values=ds0.readings.clean)
                f_exact, f_clean = pe[est]["test"][0], pc[est]["test"][0]
                noisy, short, long_, fails = [], [], [], 0
                for bd in draws:
                    dsd = build_dataset(plan, params, Bp, bd, sigma)
                    pn, stn = evaluate(dsd, ests, ctx.kernel, pp["ell"], [pp["eta"]], ev, L=L)
                    if stn[est][0] == 2 or not np.all(np.isfinite(pn[est]["test"][0])):
                        fails += 1
                        continue
                    noisy.append(pn[est]["test"][0])
                    if est == "RICH":
                        short.append(pn["SHORT"]["test"][0])
                        long_.append(pn["LONG"]["test"][0])
                R = len(noisy)
                Nz = np.stack(noisy)
                mean = Nz.mean(0)
                var_tr = float(np.sum((Nz - mean) ** 2) / max(R - 1, 1))
                floor_v = f_exact - F
                disc_v = f_clean - f_exact
                me_v = mean - f_clean
                bias_v = mean - F
                me2_raw = float(np.sum(me_v**2))
                row = dict(B=B, a=a, startset=s, name=pp["name"], group=pp["group"], plan=plan, est=est,
                           h1=pp["h1"], h2=pp["h2"], ell=pp["ell"], eta=pp["eta"], n_draws=R, fails=fails,
                           n_starts=ds0.info["n_starts"], n_readings=ds0.info["n_readings"],
                           floor=np.sqrt(np.sum(floor_v**2) / nF2), disc=np.sqrt(np.sum(disc_v**2) / nF2),
                           me_raw=np.sqrt(me2_raw / nF2), me2_corr=(me2_raw - var_tr / R) / nF2,
                           var=var_tr / nF2, bias=np.sqrt(np.sum(bias_v**2) / nF2),
                           cross=(np.sum(bias_v**2) - np.sum(floor_v**2) - np.sum(disc_v**2) - me2_raw) / nF2,
                           total_rmse=np.sqrt(np.mean(np.sum((Nz - F) ** 2, axis=(1, 2))) / nF2))
                if est == "RICH" and R > 2:
                    S, Lg = np.stack(short), np.stack(long_)
                    dS, dL = S - S.mean(0), Lg - Lg.mean(0)
                    row["var_short"] = float(np.sum(dS**2) / (R - 1) / nF2)
                    row["var_long"] = float(np.sum(dL**2) / (R - 1) / nF2)
                    row["cov_short_long"] = float(np.sum(dS * dL) / (R - 1) / nF2)
                    row["corr_short_long"] = row["cov_short_long"] / np.sqrt(row["var_short"] * row["var_long"])
                if pp["simex"] and any(B == c[0] and abs(a - c[1]) < 1e-12 for c in dcfg.get("simex_cells", [])):
                    sx = []
                    for d, bd in enumerate(draws[:dcfg.get("n_simex_draws", 6)]):
                        dsd = build_dataset(plan, params, Bp, bd, sigma)
                        o, st = simex(dsd, [est], ctx.kernel, pp["ell"], pp["eta"], ev, sigma, K=dcfg.get("simex_K", 4),
                                      seed=7000 + d, L=L)
                        sx.append((o[est]["naive"]["test"], o[est]["simex_q"]["test"], o[est]["simex_l"]["test"]))
                    nv, sq, sl = (np.stack([x[i] for x in sx]) for i in range(3))
                    Rs = len(sx)
                    for tag, arr in (("naive_sub", nv), ("simex_q", sq), ("simex_l", sl)):
                        m = arr.mean(0)
                        v = float(np.sum((arr - m) ** 2) / max(Rs - 1, 1))
                        row[f"{tag}_me_raw"] = np.sqrt(np.sum((m - f_clean) ** 2) / nF2)
                        row[f"{tag}_me2_corr"] = (np.sum((m - f_clean) ** 2) - v / Rs) / nF2
                        row[f"{tag}_var"] = v / nF2
                        row[f"{tag}_rmse"] = np.sqrt(np.mean(np.sum((arr - F) ** 2, axis=(1, 2))) / nF2)
                rows.append(row)
                np.save(os.path.join(out, f"mean_B{B}_a{a:g}_s{s}_{pp['name']}.npy"), mean)
            atomic_write_csv(pd.DataFrame(rows), shard)
            log(f"decomp B={B} a={a:g} startset={s} {time.time() - t0:.0f}s")
    files = sorted(f for f in os.listdir(out) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(out, f)) for f in files], ignore_index=True)
    # start-sampling variance of the noise-averaged estimate, across start sets
    extra = []
    for (B, a, name), sub in df.groupby(["B", "a", "name"]):
        ms = [np.load(os.path.join(out, f"mean_B{B}_a{a:g}_s{s}_{name}.npy")) for s in sorted(sub.startset)
              if os.path.exists(os.path.join(out, f"mean_B{B}_a{a:g}_s{s}_{name}.npy"))]
        if len(ms) > 1:
            M = np.stack(ms)
            extra.append(dict(B=B, a=a, name=name, start_var=float(np.sum((M - M.mean(0)) ** 2) / (len(ms) - 1) / nF2)))
    tag = dcfg.get("out_name", "decomp")
    df.to_csv(os.path.join(results_dir(), f"{ctx.cfg['name']}_{tag}.csv"), index=False)
    if extra:
        pd.DataFrame(extra).to_csv(os.path.join(results_dir(), f"{ctx.cfg['name']}_{tag}_startvar.csv"), index=False)
    return df


# ============================================================================ SIMEX ablation (2x2)

def ablate(ctx, acfg: dict, lock: dict):
    out = os.path.join(ctx.out, "ablate")
    os.makedirs(out, exist_ok=True)
    times = required_times(ctx.cfg)
    K, zetas = acfg.get("K", 4), acfg.get("zetas", [0.5, 1.0, 1.5, 2.0])
    lock_path = os.path.join(results_dir(), "locks", f"{ctx.cfg['name']}_ablate.json")
    # ---- tuning of h for the corrected pipeline (validation oracle, separate seeds) ----
    if os.path.exists(lock_path):
        alock = json.load(open(lock_path))
    else:
        rows = []
        for (B, a) in acfg["cells"]:
            sigma = a * ctx.zrms
            for p in acfg["pipelines"]:
                plan, est = p["plan"], p["est"]
                b = _best(lock, B, a, plan, est)
                for h1 in _grid_neighbors(lock, B, a, plan, est, b["h1"]):
                    e = _entry(lock, B, a, plan, est, h1)
                    params = _params_from_entry(e)
                    for rep in range(acfg["reps_tune"]):
                        bank = ctx.bank("ablate_tune", rep, n_bank(ctx.cfg), times)
                        ds = build_dataset(plan, params, B, bank, sigma)
                        o, st = simex(ds, [est], ctx.kernel, e["ell"], e["eta"], {"val": ctx.val.X}, sigma,
                                      zetas=zetas, K=K, seed=100 + rep)
                        for var in ("naive", "simex_q", "simex_l"):
                            rows.append(dict(B=B, a=a, plan=plan, est=est, h1=h1, rep=rep, variant=var,
                                             val_err=rel_err(o[est][var]["val"], ctx.val.F), status=st[est]))
                log(f"ablate-tune B={B} a={a:g} {plan}-{est} done")
        tdf = pd.DataFrame(rows)
        tdf.to_csv(os.path.join(results_dir(), f"{ctx.cfg['name']}_ablate_tune.csv"), index=False)
        choice = []
        for (B, a, plan, est), sub in tdf[tdf.variant == "simex_q"].groupby(["B", "a", "plan", "est"]):
            sc = sub.assign(l=np.log(sub.val_err)).groupby("h1").l.mean()
            choice.append(dict(B=int(B), a=float(a), plan=plan, est=est, h1_simex=float(sc.idxmin()),
                               h1_naive=float(_best(lock, int(B), float(a), plan, est)["h1"])))
        import datetime as _dt
        alock = {"created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "choices": choice}
        json.dump(alock, open(lock_path, "w"), indent=1)
    # ---- test ----
    rows = []
    shard = os.path.join(out, "test.csv")
    if os.path.exists(shard):
        return pd.read_csv(shard)
    for c in alock["choices"]:
        B, a, plan, est = c["B"], c["a"], c["plan"], c["est"]
        sigma = a * ctx.zrms
        for which, h1 in (("h_naive", c["h1_naive"]), ("h_simex", c["h1_simex"])):
            e = _entry(lock, B, a, plan, est, h1)
            params = _params_from_entry(e)
            for rep in range(acfg["reps_test"]):
                bank = ctx.bank("ablate_test", rep, n_bank(ctx.cfg), times)
                ds = build_dataset(plan, params, B, bank, sigma)
                o, st = simex(ds, [est], ctx.kernel, e["ell"], e["eta"], {"test": ctx.test.X}, sigma,
                              zetas=zetas, K=K, seed=500 + rep)
                for var in ("naive", "simex_q", "simex_l"):
                    rows.append(dict(B=B, a=a, plan=plan, est=est, h_choice=which, h1=h1, rep=rep, variant=var,
                                     test_err=rel_err(o[est][var]["test"], ctx.test.F), status=st[est]))
        log(f"ablate-test B={B} a={a:g} {plan}-{est} done")
    df = pd.DataFrame(rows)
    atomic_write_csv(df, shard)
    df.to_csv(os.path.join(results_dir(), f"{ctx.cfg['name']}_ablate_test.csv"), index=False)
    return df


# ============================================================================ misspecification

def hetero_scale_fn(ctx):
    m_p = float(np.mean(np.sum(ctx.val.X[:, 2:4] ** 2, axis=1)))

    def fn(z):
        return np.sqrt((0.25 + np.sum(z[:, 2:4] ** 2, axis=1) / m_p) / 1.25)
    return fn


def misspec(ctx, mcfg: dict, lock: dict):
    out = os.path.join(ctx.out, "misspec")
    os.makedirs(out, exist_ok=True)
    times = required_times(ctx.cfg)
    B, a = mcfg["cell"]
    sigma = a * ctx.zrms
    scen = {"correct": dict(noise_model="iid", scale_fn=None),
            "hetero": dict(noise_model="iid", scale_fn=hetero_scale_fn(ctx)),
            "temporal": dict(noise_model="temporal", scale_fn=None, tau=mcfg["tau"])}
    for rep in range(mcfg["reps"]):
        shard = os.path.join(out, f"r{rep}.csv")
        if os.path.exists(shard):
            continue
        t0 = time.time()
        rows = []
        for sname, sc in scen.items():
            bank = ctx.bank("misspec", rep, n_bank(ctx.cfg), times, noise_model=sc["noise_model"], tau=sc.get("tau"))
            for p in mcfg["pipelines"]:
                plan, est = p["plan"], p["est"]
                try:
                    b = _best(lock, B, a, plan, est)
                except KeyError:
                    continue
                e = _entry(lock, B, a, plan, est, b["h1"])
                params = _params_from_entry(e)
                ds = build_dataset(plan, params, B, bank, sigma, scale_fn=sc["scale_fn"])
                if p.get("simex"):
                    kappas = mcfg.get("kappas", [0.5, 2.0]) if sname == "correct" else []
                    for kap in [1.0] + kappas:
                        o, st = simex(ds, [est], ctx.kernel, e["ell"], e["eta"], {"test": ctx.test.X},
                                      sigma * np.sqrt(kap), K=mcfg.get("K", 4), seed=900 + rep)
                        for var in ("naive", "simex_q", "simex_l"):
                            if var == "naive" and kap != 1.0:
                                continue
                            rows.append(dict(B=B, a=a, rep=rep, scenario=sname, kappa=kap, plan=plan, est=est,
                                             h1=b["h1"], variant=var, test_err=rel_err(o[est][var]["test"], ctx.test.F),
                                             status=st[est]))
                else:
                    pr, st = evaluate(ds, [est], ctx.kernel, e["ell"], [e["eta"]], {"test": ctx.test.X})
                    rows.append(dict(B=B, a=a, rep=rep, scenario=sname, kappa=1.0, plan=plan, est=est, h1=b["h1"],
                                     variant="naive", test_err=rel_err_path(pr[est]["test"], ctx.test.F)[0],
                                     status=int(st[est][0])))
        atomic_write_csv(pd.DataFrame(rows), shard)
        log(f"misspec rep={rep} {time.time() - t0:.0f}s")
    files = sorted(f for f in os.listdir(out) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(out, f)) for f in files], ignore_index=True)
    df.to_csv(os.path.join(results_dir(), f"{ctx.cfg['name']}_misspec.csv"), index=False)
    return df


# ============================================================================ calibration

def bank_from_field(field, sampler_rng, system, n, times, xi_seed, dt=0.005) -> Bank:
    z0 = system.sample(n, sampler_rng)
    times = np.unique(np.concatenate([[0.0], np.asarray(times, float)]))
    X = np.empty((n, len(times), 4))
    z = z0.copy()
    tprev = 0.0
    X[:, 0] = z0
    for j, t in enumerate(times[1:], start=1):
        z = rk4_flow(field, z, t - tprev, dt=dt)
        X[:, j] = z
        tprev = t
    xi = np.random.default_rng(np.random.SeedSequence(list(xi_seed))).standard_normal((n, len(times), 2, 4))
    return Bank(z0=z0, times=times, X=X, xi=xi, seed=tuple(xi_seed))


def rollout_loss(model, ds_held, dt=0.01):
    """Schedule-independent noisy criterion: predict every later reading of a start from its t=0 reading."""
    r = ds_held.readings
    is0 = np.where((r.t == 0) & (r.copy == 0))[0]
    first = {int(r.start[i]): int(i) for i in is0}
    later = np.where(r.t > 0)[0]
    loss, cnt = 0.0, 0
    for t in np.unique(r.t[later]):
        idx = [int(i) for i in later[np.abs(r.t[later] - t) < 1e-12] if int(r.start[i]) in first]
        if not idx:
            continue
        y0 = r.values[[first[int(r.start[i])] for i in idx]]
        yt = r.values[idx]
        pred = rk4_flow(model, y0, float(t), dt=dt)
        loss += float(np.sum((pred - yt) ** 2))
        cnt += yt.size
    return loss / max(cnt, 1)


def select_by_rollout(ds_train, est, kernel, ells, etas, ds_held):
    best = (np.inf, None, None, None)
    for ell in ells:
        for eta in etas:
            m, st = fit_field_model(ds_train, est, kernel, ell, eta)
            if m is None:
                continue
            lv = rollout_loss(m, ds_held)
            if lv < best[0]:
                best = (lv, ell, eta, m)
    return best


def calibrate(ctx, ccfg: dict, lock: dict):
    out = os.path.join(ctx.out, "calibrate")
    os.makedirs(out, exist_ok=True)
    sysm = ctx.system
    kernel = ctx.kernel
    hc = ccfg["h_c"]
    Bcal, Bfull = ccfg["B_cal"], ccfg["B"]
    Brest = Bfull - Bcal
    cand = ccfg["candidates"]
    surX = sysm.sample(ccfg.get("n_surrogate_test", 2000), np.random.default_rng((ctx.seed, 77)))
    cand_times = sorted({round(t, 9) for c in cand for t in (
        [c["h1"]] if c["plan"] == "SI" else [c["h1"], 2 * c["h1"]])} | {hc, 2 * hc})
    for a in ccfg["noise_levels"]:
        sigma = a * ctx.zrms
        ells = [float(x) for x in ccfg["ells"][str(a)]]
        etas = [float(x) for x in ccfg["etas"][str(a)]]
        for rep in range(ccfg["reps"]):
            shard = os.path.join(out, f"a{a:g}_r{rep}.csv")
            if os.path.exists(shard):
                continue
            t0 = time.time()
            rows = []
            # 1) calibration triplets from the TRUE system (counted in the budget)
            cbank = ctx.bank("calib", rep, Bcal // 3, [hc, 2 * hc])
            dcal = build_dataset("MS3", {"h": hc}, Bcal, cbank, sigma)
            # 2) surrogate: MS3-RICH, (ell, eta) by 4-fold rollout CV over calibration starts
            starts = np.unique(dcal.readings.start)
            folds = np.arange(len(starts)) % 4
            cv = {}
            for ell in ells:
                for eta in etas:
                    tot = 0.0
                    for k in range(4):
                        tr_st, te_st = starts[folds != k], starts[folds == k]
                        dtr = _subset_ds(dcal, tr_st)
                        dte = _subset_ds(dcal, te_st)
                        m, st = fit_field_model(dtr, "RICH", kernel, ell, eta)
                        tot += np.inf if m is None else rollout_loss(m, dte)
                    cv[(ell, eta)] = tot
            ell_s, eta_s = min(cv, key=cv.get)
            sur, _ = fit_field_model(dcal, "RICH", kernel, ell_s, eta_s)
            Fsur = sur(surX)
            # 3) predicted error of each candidate on the surrogate (oracle tuning inside the surrogate world)
            pred_err = []
            for ci, c in enumerate(cand):
                errs = []
                for rs in range(ccfg["surrogate_reps"]):
                    sb = bank_from_field(sur, np.random.default_rng((ctx.seed, 11, rep, ci, rs)), sysm, Brest // 2,
                                         cand_times, xi_seed=(ctx.seed, PURPOSE["calib_sur"], rep, ci, rs))
                    ds = build_dataset(c["plan"], _params(c["plan"], c["h1"], 2 * c["h1"] if c["plan"] not in ("SI", "MS3") else None),
                                       Brest, sb, sigma)
                    best = np.inf
                    for ell in ells:
                        pr, st = evaluate(ds, [c["est"]], kernel, ell, etas, {"s": surX})
                        e = rel_err_path(pr[c["est"]]["s"], Fsur)
                        best = min(best, np.nanmin(e) if np.any(np.isfinite(e)) else np.inf)
                    errs.append(best)
                pred_err.append(float(np.mean(errs)))
            sel = int(np.argmin(pred_err))
            si_ids = [i for i, c in enumerate(cand) if c["plan"] == "SI"]
            sel_si = si_ids[int(np.argmin([pred_err[i] for i in si_ids]))]
            # 4) collect the remaining budget from the TRUE system for every candidate (realized curve) and
            #    choose (ell, eta) by the schedule-independent rollout criterion on the calibration readings
            tbank = ctx.bank("calib_true", rep, Bfull // 2, cand_times)
            realized = []
            for ci, c in enumerate(cand):
                ds = build_dataset(c["plan"], _params(c["plan"], c["h1"], 2 * c["h1"] if c["plan"] not in ("SI", "MS3") else None),
                                   Brest, tbank, sigma)
                lv, ell, eta, m = select_by_rollout(ds, c["est"], kernel, ells, etas, dcal)
                err = rel_err(m(ctx.test.X), ctx.test.F) if m is not None else np.nan
                realized.append(err)
                rows.append(dict(a=a, rep=rep, method="candidate", cand=ci, plan=c["plan"], est=c["est"], h1=c["h1"],
                                 budget_used=Bfull, B_cal=Bcal, pred_err=pred_err[ci], test_err=err, ell=ell, eta=eta,
                                 ell_sur=ell_s, eta_sur=eta_s))
            rows.append(dict(a=a, rep=rep, method="calibrated", cand=sel, plan=cand[sel]["plan"], est=cand[sel]["est"],
                             h1=cand[sel]["h1"], budget_used=Bfull, B_cal=Bcal, pred_err=pred_err[sel],
                             test_err=realized[sel]))
            rows.append(dict(a=a, rep=rep, method="calibrated_SI_only", cand=sel_si, plan=cand[sel_si]["plan"],
                             est=cand[sel_si]["est"], h1=cand[sel_si]["h1"], budget_used=Bfull, B_cal=Bcal,
                             pred_err=pred_err[sel_si], test_err=realized[sel_si]))
            # 5) no-calibration baselines with the full budget: fixed defaults, (ell, eta) by 2-fold rollout CV
            for dname, d in ccfg["defaults"].items():
                ds = build_dataset(d["plan"], _params(d["plan"], d["h1"], 2 * d["h1"] if d["plan"] not in ("SI", "MS3") else None),
                                   Bfull, tbank, sigma)
                st_all = np.unique(ds.readings.start)
                half = st_all[np.arange(len(st_all)) % 2 == 0]
                other = st_all[np.arange(len(st_all)) % 2 == 1]
                lv, ell, eta, _ = select_by_rollout(_subset_ds(ds, half), d["est"], kernel, ells, etas, _subset_ds(ds, other))
                m, _ = fit_field_model(ds, d["est"], kernel, ell, eta)
                rows.append(dict(a=a, rep=rep, method=f"default_{dname}", plan=d["plan"], est=d["est"], h1=d["h1"],
                                 budget_used=Bfull, B_cal=0, test_err=rel_err(m(ctx.test.X), ctx.test.F), ell=ell, eta=eta))
            # 6) oracle reference: main-lock best pipeline (oracle validation tuning), full budget
            for oname, o in ccfg["oracle"].items():
                b = _best(lock, Bfull, a, o["plan"], o["est"])
                e = _entry(lock, Bfull, a, o["plan"], o["est"], b["h1"])
                obank = ctx.bank("calib_true", rep, Bfull // 2, cand_times + [b["h1"], 2 * b["h1"]])
                ds = build_dataset(o["plan"], _params_from_entry(e), Bfull, obank, sigma)
                pr, st = evaluate(ds, [o["est"]], kernel, e["ell"], [e["eta"]], {"test": ctx.test.X})
                rows.append(dict(a=a, rep=rep, method=f"oracle_{oname}", plan=o["plan"], est=o["est"], h1=b["h1"],
                                 budget_used=Bfull, B_cal=0, test_err=rel_err_path(pr[o["est"]]["test"], ctx.test.F)[0]))
            atomic_write_csv(pd.DataFrame(rows), shard)
            log(f"calibrate a={a:g} rep={rep}: selected {cand[sel]} {time.time() - t0:.0f}s")
    files = sorted(f for f in os.listdir(out) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(out, f)) for f in files], ignore_index=True)
    df.to_csv(os.path.join(results_dir(), f"{ctx.cfg['name']}_calibrate.csv"), index=False)
    return df


def _subset_ds(ds, starts):
    """Sub-dataset restricted to the given trajectory starts (readings and pairs re-indexed)."""
    from copy import deepcopy

    from .schedules import Dataset, Pairs, Readings, accounting
    r, p = ds.readings, ds.pairs
    keep_r = np.isin(r.start, starts)
    new_idx = -np.ones(len(r.ids), dtype=int)
    new_idx[keep_r] = np.arange(keep_r.sum())
    R = Readings(values=r.values[keep_r], clean=r.clean[keep_r], start=r.start[keep_r], tidx=r.tidx[keep_r],
                 copy=r.copy[keep_r], t=r.t[keep_r], ids=r.ids[keep_r], unit=r.unit[keep_r])
    keep_p = keep_r[p.ia] & keep_r[p.ib]
    P = Pairs(ia=new_idx[p.ia[keep_p]], ib=new_idx[p.ib[keep_p]], h=p.h[keep_p], group=p.group[keep_p],
              unit=p.unit[keep_p], anchor=p.anchor)
    out = Dataset(plan=ds.plan, params=deepcopy(ds.params), readings=R, pairs=P, group_h=dict(ds.group_h))
    out.info = accounting(out)
    return out


# ============================================================================ Henon-Heiles anchor / Hu-fixed

def hh_anchor(ctx, hcfg: dict):
    """Noise-free MP (and Richardson) error vs delta with Hu et al.'s fixed estimator settings."""
    rows = []
    mat = make_kernel("matern52")
    Lhu = np.full(4, float(hcfg["hu_ell"]))
    dbar = (5.0 / 3.0) / Lhu[0] ** 2
    eta_ours = hcfg["hu_eta"] / dbar                # their ridge eta*N*L <-> our eta*n*dbar
    deltas = hcfg["deltas"]
    times = sorted(set(deltas) | {2 * d for d in deltas})
    bank = ctx.bank("hh", 0, hcfg["N"], times)
    ev = {"test": ctx.test.X}
    for d in deltas:
        for plan, est, params, B in (("SI", "MP", {"h": d}, 2 * hcfg["N"]),
                                     ("SA", "RICH", {"h1": d, "h2": 2 * d, "r": 2.0}, 3 * hcfg["N"])):
            ds = build_dataset(plan, params, B, bank, 0.0)
            pr, st = evaluate(ds, [est], mat, 1.0, [eta_ours], ev, L=Lhu)
            rows.append(dict(setting="hu_fixed_matern52", plan=plan, est=est, delta=d, ell=hcfg["hu_ell"],
                             eta=hcfg["hu_eta"], test_err=rel_err_path(pr[est]["test"], ctx.test.F)[0],
                             status=int(st[est][0])))
            # our tuned SE estimator (oracle validation over a small grid), for comparison
            best = (np.inf, None, None)
            for ell in hcfg["se_ells"]:
                pv, sv = evaluate(ds, [est], ctx.kernel, ell, hcfg["se_etas"], {"val": ctx.val.X, "test": ctx.test.X})
                ve = rel_err_path(pv[est]["val"], ctx.val.F)
                te = rel_err_path(pv[est]["test"], ctx.test.F)
                j = int(np.nanargmin(ve))
                if ve[j] < best[0]:
                    best = (ve[j], te[j], (ell, hcfg["se_etas"][j]))
            rows.append(dict(setting="tuned_se", plan=plan, est=est, delta=d, ell=best[2][0], eta=best[2][1],
                             test_err=best[1], status=0))
        log(f"hh delta={d}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir(), f"{ctx.cfg['name']}_hh_anchor.csv"), index=False)
    return df


def hu_fixed_dp(ctx, cfg_hf: dict):
    """Hu et al.'s fixed estimator (Matern-5/2, ell=3 in raw coordinates, eta=1e-9) on the double pendulum."""
    mat = make_kernel("matern52")
    Lhu = np.full(4, float(cfg_hf["hu_ell"]))
    dbar = (5.0 / 3.0) / Lhu[0] ** 2
    eta_ours = cfg_hf["hu_eta"] / dbar
    times = required_times(ctx.cfg)
    B, a = cfg_hf["cell"]
    rows = []
    for rep in range(cfg_hf["reps"]):
        bank = ctx.bank("test", rep, n_bank(ctx.cfg), times)
        for h in cfg_hf["h"]:
            ds = build_dataset("SI", {"h": h}, B, bank, a * ctx.zrms)
            pr, st = evaluate(ds, ["MP"], mat, 1.0, [eta_ours], {"test": ctx.test.X}, L=Lhu)
            rows.append(dict(B=B, a=a, rep=rep, plan="SI", est="MP", h1=h, setting="hu_fixed_matern52",
                             test_err=rel_err_path(pr["MP"]["test"], ctx.test.F)[0], status=int(st["MP"][0])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir(), f"{ctx.cfg['name']}_hufixed.csv"), index=False)
    return df
