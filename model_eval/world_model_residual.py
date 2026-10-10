#!/usr/bin/env python3
"""Residual robot world model on top of the MuJoCo simulator, with the commanded kp/kd scaling.

    real state s_t=(q,dq) ──> MuJoCo one step (true gains) ──> sim_next = (q_sim, dq_sim)(t+1)
    world model f(history, a, sim_next) ──> residual r_hat ≈ real_next - sim_next
    corrected prediction  =  sim_next + r_hat

Validation (same runs as world_model_qdq.py; never used in training, no checkpoint selection):
    A : 0924 runs (2)                       nominal gains
    B : 1001 runs with kp,kd x1.5 on ankle pitch (4)
Training: every clean run (known-bad runs already excluded by deploy_constants.list_runs()),
minus the validation runs.  MuJoCo uses the TRUE gains of each run (x1.5 on joints 4,10 in the 4
scaled 1001 runs), and the model also sees the gain-scaled PD torque.

Three terms are compared on each validation set, per joint / group, for q (deg) and dq (rad/s):
    1. pure_sim      : sim_next          vs real_next          (sim2real gap)
    2. persist       : real_t            vs real_next          (nothing changes)
    3. sim + model   : sim_next + r_hat  vs real_next          (gap after the world-model correction)

Sim predictions are cached in model_eval/cache_sim_next/ (one npz per run).

Usage:
  .venv_sim/bin/python model_eval/world_model_residual.py
  .venv_sim/bin/python model_eval/world_model_residual.py --epochs 60 --min-date 20260920   # after-0919 training only
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model_eval"))
sys.path.insert(0, os.path.join(ROOT, "sim2real"))
import sim2real_phaseC1b_onestep_qdq as M  # noqa: E402
import world_model_qdq as W  # noqa: E402
from deploy_constants import CONTROL_DT, JOINT_NAMES, KD, KP, load_csv, load_run  # noqa: E402

CACHE = os.path.join(ROOT, "model_eval", "cache_sim_next")
KP0, KD0 = KP.copy(), KD.copy()
_STATE = {}


def run_gain(path):
    g = np.ones(29)
    if W.run_date(path) == 20261001 and W.run_id(path) in W.SCALED_1001:
        g[W.ANKLE_PITCH] = 1.5
    return g


def load_run_full(path):
    """aligned arrays for one run (moving window), incl. the IMU quaternion for the sim. None if unusable."""
    d = load_run(path)
    q, dq, qt = d["q"], d["dq"], d["q_target"]
    if any(np.isnan(v).any() for v in (q, dq, qt)):
        return None
    mv = np.abs(dq).max(axis=1) > 0.05
    if mv.sum() < 200:
        return None
    lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
    try:
        bq, _ = load_csv(os.path.join(path, "base_quat.csv"), "base_q")
        bw, _ = load_csv(os.path.join(path, "base_ang_vel.csv"), "base_w")
    except Exception:  # noqa: BLE001
        return None
    hi = min(hi, len(q), len(bq), len(bw))
    if hi - lo < 100:
        return None
    s = slice(lo, hi)
    T = hi - lo
    return dict(q=q[s], dq=dq[s], qt=qt[s], bq=bq[s], w=bw[s], g=W.gravity_body(bq[s]),
                gain=np.tile(run_gain(path), (T, 1)), name=f"{W.run_date(path)}/{W.run_id(path)}")


def _cache_path(path):
    return os.path.join(CACHE, f"{W.run_date(path)}_{W.run_id(path)}_{hashlib.md5(path.encode()).hexdigest()[:8]}.npz")


def _init_worker():
    _STATE["model"] = M.build_model(True, weld_base=False)
    _STATE["data"] = M.mujoco.MjData(_STATE["model"])
    _STATE["n_sub"] = int(round(CONTROL_DT / _STATE["model"].opt.timestep))


def _sim_run(path):
    cp = _cache_path(path)
    if os.path.exists(cp):
        return path, "cached"
    r = load_run_full(path)
    if r is None:
        return path, "skip"
    gain = r["gain"][0]
    M.KP, M.KD = KP0 * gain, KD0 * gain            # true commanded gains used by the sim for this run
    model, data, n_sub = _STATE["model"], _STATE["data"], _STATE["n_sub"]
    T = len(r["q"])
    qs, ds = np.zeros((T - 1, 29)), np.zeros((T - 1, 29))
    for t in range(T - 1):
        M.mujoco.mj_resetData(model, data)
        qn, dqn, _ = M.one_step(model, data, r["q"][t], r["dq"][t], r["qt"][t], n_sub, weld_base=False,
                                base_quat=r["bq"][t], base_ang_vel=r["w"][t])
        qs[t], ds[t] = qn, dqn
    np.savez(cp, q_sim=qs, dq_sim=ds)
    return path, "done"


def ensure_sim_cache(paths, workers):
    os.makedirs(CACHE, exist_ok=True)
    todo = [p for p in paths if not os.path.exists(_cache_path(p))]
    print(f"sim cache: {len(paths) - len(todo)} cached, {len(todo)} to simulate with {workers} workers", flush=True)
    if todo:
        with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
            for i, (p, st) in enumerate(pool.imap_unordered(_sim_run, todo), 1):
                print(f"  [{i}/{len(todo)}] {st:6s} {p.split('g1_robot_data/')[1]}", flush=True)


def build_windows(paths, k):
    """windows over all runs; also returns the sim next state and the real next state."""
    runs = []
    for p in paths:
        r = load_run_full(p)
        if r is None or not os.path.exists(_cache_path(p)):
            continue
        c = np.load(_cache_path(p))
        r["q_sim"], r["dq_sim"] = c["q_sim"], c["dq_sim"]
        runs.append(r)
    win = W.make_windows(runs, k)
    ts_all, rid = win["tidx"], win["rid"]
    qs = np.concatenate([runs[i]["q_sim"][ts_all[rid == i]] for i in range(len(runs))])
    ds = np.concatenate([runs[i]["dq_sim"][ts_all[rid == i]] for i in range(len(runs))])
    win["q_sim"], win["dq_sim"] = (torch.as_tensor(qs, dtype=torch.float32), torch.as_tensor(ds, dtype=torch.float32))
    win["q_next"] = win["q"][:, 0] + win["dq_t"]          # dq_t holds the q CHANGE (see world_model_qdq)
    win["dq_next"] = win["dq"][:, 0] + win["ddq_t"]
    return win, runs


def features(win):
    base = W.windows_to_X(win)
    sim_step = torch.cat([win["q_sim"] - win["q"][:, 0], win["dq_sim"] - win["dq"][:, 0]], 1)
    return torch.cat([base, sim_step], 1)


def fit_residual(train, epochs, lr, wd, bs, hidden, layers, dropout, seed, val, dev):
    torch.manual_seed(seed)
    X = features(train)
    Y = torch.cat([train["q_next"] - train["q_sim"], train["dq_next"] - train["dq_sim"]], 1)
    xn, yn = W.Norm(X), W.Norm(Y)
    Xn, Yn = xn(X).to(dev), yn(Y).to(dev)
    net = W.MLP(Xn.shape[1], Yn.shape[1], hidden, layers, dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    N = len(Xn)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(N, device=dev)
        tot = 0.0
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            loss = ((net(Xn[b]) - Yn[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            msg = f"    epoch {ep:3d}  train loss {tot / N:.4f}"
            net.eval()
            with torch.no_grad():
                for n, w in val.items():
                    Yv = yn(torch.cat([w["q_next"] - w["q_sim"], w["dq_next"] - w["dq_sim"]], 1))
                    msg += f"  | {n} {float(((net(xn(features(w)).to(dev)).cpu() - Yv) ** 2).mean()):.4f}"
            print(msg + "  (monitor only; final epoch used)", flush=True)
    net.eval()

    @torch.no_grad()
    def predict(win):
        r = yn.inv(net(xn(features(win)).to(dev)).cpu())
        return win["q_sim"] + r[:, :29], win["dq_sim"] + r[:, 29:]

    return predict, (net, xn, yn)


def stats(eq, ed):
    return {g: (float(np.degrees(np.abs(eq[:, js])).mean()), float(np.degrees(np.sqrt((eq[:, js] ** 2).mean()))),
                float(np.abs(ed[:, js]).mean()), float(np.sqrt((ed[:, js] ** 2).mean())))
            for g, js in W.GROUPS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-date", type=int, default=0, help="train only on runs dated >= YYYYMMDD (0 = all)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--motion-thresh", type=float, default=0.3,
                    help="a step counts as MOVING if any joint has |dq| above this (rad/s); the rest are STILL")
    ap.add_argument("--out", default=os.path.join(ROOT, "model_eval", "world_model_residual_metrics.json"))
    a = ap.parse_args()

    tr_p, va_p, vb_p = W.split_runs(a.min_date)
    print(f"clean runs (list_runs excludes the 9 known-bad runs): train {len(tr_p)}  valA {len(va_p)}  valB {len(vb_p)}")
    ensure_sim_cache(tr_p + va_p + vb_p, a.workers)

    k = a.hist
    train, tr_runs = build_windows(tr_p, k)
    vals, vruns = {}, {}
    for n, p in (("valA_0924", va_p), ("valB_1001_x1.5", vb_p)):
        vals[n], vruns[n] = build_windows(p, k)
    print(f"windows: train {len(train['q'])} ({len(tr_runs)} runs)  " +
          "  ".join(f"{n} {len(w['q'])} ({len(vruns[n])} runs)" for n, w in vals.items()))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("training residual model ...", flush=True)
    predict, _ = fit_residual(train, a.epochs, a.lr, a.wd, a.bs, a.hidden, a.layers, a.dropout, a.seed, vals, dev)

    out = {}
    for n, w in vals.items():
        qn, dn = w["q_next"].numpy(), w["dq_next"].numpy()
        qm, dm = predict(w)
        terms = {
            "1 pure_sim   (sim_next vs real_next)": (w["q_sim"].numpy() - qn, w["dq_sim"].numpy() - dn),
            "2 persist    (real_t vs real_next)": (w["q"][:, 0].numpy() - qn, w["dq"][:, 0].numpy() - dn),
            "3 sim+model  (sim_next+resid vs real_next)": (qm.numpy() - qn, dm.numpy() - dn),
        }
        st = {t: stats(*e) for t, e in terms.items()}
        out[n] = {"steps": int(len(qn)), "groups": st,
                  "per_joint": {t: {"q_mae_deg": np.degrees(np.abs(e[0]).mean(0)).tolist(),
                                    "dq_mae": np.abs(e[1]).mean(0).tolist()} for t, e in terms.items()}}
        print(f"\n=== {n}  ({len(qn)} steps)  q in deg, dq in rad/s ===")
        print(f"  {'term':46s}{'group':13s}{'q MAE':>8s}{'q RMSE':>8s}{'dq MAE':>8s}{'dq RMSE':>9s}")
        for t, s in st.items():
            for g in ("legs", "waist", "arms", "ankle_pitch", "all"):
                x = s[g]
                print(f"  {t:46s}{g:13s}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}{x[3]:9.3f}")
        print(f"\n  per joint MAE  [1 pure_sim | 2 persist | 3 sim+model]    q deg  /  dq rad/s")
        pj = out[n]["per_joint"]
        ks = list(pj)
        for j in range(29):
            print(f"  {JOINT_NAMES[j]:22s} q {pj[ks[0]]['q_mae_deg'][j]:6.3f} {pj[ks[1]]['q_mae_deg'][j]:6.3f} {pj[ks[2]]['q_mae_deg'][j]:6.3f}"
                  f"   dq {pj[ks[0]]['dq_mae'][j]:6.3f} {pj[ks[1]]['dq_mae'][j]:6.3f} {pj[ks[2]]['dq_mae'][j]:6.3f}")

        # ---- split by motion: a step is "moving" if ANY joint moves faster than --motion-thresh rad/s ----
        speed = w["dq"][:, 0].abs().max(1).values.numpy()
        out[n]["subsets"] = {}
        for sname, mask in (("still", speed <= a.motion_thresh), ("moving", speed > a.motion_thresh)):
            if mask.sum() < 50:
                continue
            sub = {t: stats(e[0][mask], e[1][mask]) for t, e in terms.items()}
            pjs = {t: {"q_mae_deg": np.degrees(np.abs(e[0][mask]).mean(0)).tolist(),
                       "dq_mae": np.abs(e[1][mask]).mean(0).tolist()} for t, e in terms.items()}
            out[n]["subsets"][sname] = {"steps": int(mask.sum()), "groups": sub, "per_joint": pjs}
            print(f"\n  --- {n}: {sname.upper()} steps (max|dq| {'<=' if sname == 'still' else '>'} {a.motion_thresh} rad/s): "
                  f"{int(mask.sum())} / {len(mask)} = {100 * mask.mean():.0f}% ---")
            print(f"  {'term':46s}{'group':13s}{'q MAE':>8s}{'q RMSE':>8s}{'dq MAE':>8s}{'dq RMSE':>9s}")
            for t, s in sub.items():
                for g in ("legs", "waist", "arms", "ankle_pitch", "all"):
                    x = s[g]
                    print(f"  {t:46s}{g:13s}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}{x[3]:9.3f}")
            kk = list(pjs)
            print(f"  joints where sim+model beats persist ({sname}): q {sum(pjs[kk[2]]['q_mae_deg'][j] < pjs[kk[1]]['q_mae_deg'][j] for j in range(29))}/29"
                  f"  dq {sum(pjs[kk[2]]['dq_mae'][j] < pjs[kk[1]]['dq_mae'][j] for j in range(29))}/29;  "
                  f"pure_sim beats persist: q {sum(pjs[kk[0]]['q_mae_deg'][j] < pjs[kk[1]]['q_mae_deg'][j] for j in range(29))}/29"
                  f"  dq {sum(pjs[kk[0]]['dq_mae'][j] < pjs[kk[1]]['dq_mae'][j] for j in range(29))}/29")
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
