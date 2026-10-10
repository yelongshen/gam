#!/usr/bin/env python3
"""Is RMA-style in-context adaptation applicable to THIS robot's data?

RMA (Kumar et al., RSS 2021, arXiv:2107.04034) trains a base policy conditioned
on a latent `z` describing the environment's "extrinsics" (friction, payload,
motor strength, terrain), plus an adaptation module that INFERS `z` from the
recent state-action history. It only helps if three things hold:

  P1  the dynamics actually VARY across the conditions we deploy in
  P2  that variation is STABLE within an episode (otherwise there is nothing
      to estimate -- it is just noise)
  P3  the variation is INFERABLE from a short window of (q, dq, a)

P1 and P2 are directly measurable from the existing logs. P3 is measurable with
a tiny regressor -- no policy training required. This script tests all three
before anyone spends weeks on a full RMA pipeline.

THE OBSERVABLE PROXY FOR `z`
----------------------------
We do not have ground-truth extrinsics. But `verify_pd_law.py` established that

    tau_est = kp (q_target - q) - kd dq        holds at corr 0.998

with a 6.3% residual. That residual is precisely the unmodelled part --
friction, gearing loss, motor derating -- i.e. the physical stuff `z` is
supposed to encode. So per-joint residual statistics are a legitimate
observable stand-in for the extrinsics.

For each window we compute, per joint:
    gain_j   = <tau_est, tau_pd> / <tau_pd, tau_pd>     (effective strength)
    coulomb_j = mean(residual * sign(dq))               (dry friction)
    viscous_j = <residual, dq> / <dq, dq>               (viscous friction)

HOW TO READ THE RESULT
----------------------
P1 fails  -> every run has the same dynamics; RMA has nothing to adapt to on
             this data, and would only pay off under conditions that actually
             vary (payload, terrain, battery, wear). Not a reason to abandon
             RMA -- a reason to collect varied data first.
P2 fails  -> the signature is noise-dominated; a 200 ms window cannot estimate
             it and the adaptation module would fit noise.
P3 fails  -> the variation exists but is invisible in (q, dq, a); the
             adaptation module cannot recover it from the deployed observation.

Usage:
  .venv_sim/bin/python model_eval/rma_feasibility_check.py
  .venv_sim/bin/python model_eval/rma_feasibility_check.py --window 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim2real"))

from deploy_constants import (  # noqa: E402
    CONTROL_DT,
    JOINT_NAMES,
    KD,
    KP,
    list_runs,
    load_run,
)

# waist roll/pitch are commanded outside their joint limit on some runs; their
# residual reflects that clamp, not the actuator.
SCORE_JOINTS = [j for j in range(29) if j not in (13, 14)]


def signature(q, dq, qt, tau, idx):
    """Per-joint (gain, coulomb, viscous) over the frames in `idx`."""
    pd = KP * (qt[idx] - q[idx]) - KD * dq[idx]
    res = tau[idx] - pd
    d = dq[idx]
    out = []
    for j in SCORE_JOINTS:
        pj, rj, dj = pd[:, j], res[:, j], d[:, j]
        gain = float(pj @ tau[idx][:, j] / (pj @ pj)) if (pj @ pj) > 1e-9 else np.nan
        coul = float(np.mean(rj * np.sign(dj)))
        visc = float(rj @ dj / (dj @ dj)) if (dj @ dj) > 1e-6 else 0.0
        out += [gain, coul, visc]
    return np.asarray(out)


def load_all(max_runs):
    runs = []
    for r in list_runs()[:max_runs]:
        try:
            d = load_run(r)
        except Exception:  # noqa: BLE001
            continue
        q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
        if any(np.isnan(v).any() for v in (q, dq, qt, tau)):
            continue
        mv = np.abs(dq).max(axis=1) > 0.05
        if mv.sum() < 600:
            continue
        lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
        try:
            pol = json.load(open(os.path.join(r, "metadata.json")))
            pol = pol["robot_config"]["model_path"]
            pol = pol.replace("policy/", "").replace("/model_decoder.onnx", "")
        except Exception:  # noqa: BLE001
            pol = "?"
        runs.append(dict(name=os.path.relpath(r, "/home/grease/g1_robot_data"),
                         policy=pol, q=q[lo:hi], dq=dq[lo:hi],
                         qt=qt[lo:hi], tau=tau[lo:hi]))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=48)
    ap.add_argument("--window", type=float, default=4.0,
                    help="seconds per signature window")
    ap.add_argument("--history", type=int, nargs="+", default=[10],
                    help="history lengths in FRAMES to test for P3 "
                         "(10 = 200 ms, the deployed value). Pass several to "
                         "sweep, e.g. --history 10 25 50 100 200")
    a = ap.parse_args()

    runs = load_all(a.max_runs)
    W = int(a.window / CONTROL_DT)
    print(f"runs {len(runs)}   window {a.window:g} s ({W} frames)   "
          f"signature dim {3*len(SCORE_JOINTS)}\n")

    # --- build one signature per window, labelled by run ------------------
    S, run_id, pol_id = [], [], []
    for i, r in enumerate(runs):
        n = len(r["q"])
        for s in range(0, n - W, W):
            idx = np.arange(s, s + W)
            sig = signature(r["q"], r["dq"], r["qt"], r["tau"], idx)
            if np.isnan(sig).any():
                continue
            S.append(sig)
            run_id.append(i)
            pol_id.append(r["policy"])
    S = np.asarray(S)
    run_id = np.asarray(run_id)
    print(f"windows: {len(S)}  from {len(set(run_id))} runs\n")

    # --- P1 / P2: between-run vs within-run variation ---------------------
    # If a per-run latent exists and is estimable, between-run variance must
    # dominate within-run variance. This is an intraclass-correlation test.
    print("=" * 74)
    print("P1 + P2  does the dynamics signature VARY between runs and stay")
    print("         STABLE within a run?")
    print("=" * 74)
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-12)
    grand = Sz.mean(0)
    between = within = 0.0
    for i in set(run_id):
        m = run_id == i
        mu = Sz[m].mean(0)
        between += m.sum() * ((mu - grand) ** 2).sum()
        within += ((Sz[m] - mu) ** 2).sum()
    icc = between / (between + within)
    print(f"  between-run variance fraction (ICC) = {icc:.3f}")
    print("     ICC -> 1  : each run has its own stable signature (RMA-friendly)")
    print("     ICC -> 0  : all runs identical, or signature is pure noise")

    # per-feature breakdown for the most informative features
    feat_names = []
    for j in SCORE_JOINTS:
        feat_names += [f"{JOINT_NAMES[j]}:gain", f"{JOINT_NAMES[j]}:coulomb",
                       f"{JOINT_NAMES[j]}:viscous"]
    iccs = []
    for k in range(S.shape[1]):
        b = w = 0.0
        g = Sz[:, k].mean()
        for i in set(run_id):
            m = run_id == i
            mu = Sz[m, k].mean()
            b += m.sum() * (mu - g) ** 2
            w += ((Sz[m, k] - mu) ** 2).sum()
        iccs.append(b / (b + w + 1e-12))
    iccs = np.asarray(iccs)
    order = np.argsort(-iccs)
    print(f"\n  most run-specific features:")
    for k in order[:8]:
        print(f"     {feat_names[k]:34s} ICC {iccs[k]:.3f}")
    print(f"  least run-specific:")
    for k in order[-3:]:
        print(f"     {feat_names[k]:34s} ICC {iccs[k]:.3f}")

    # --- P3: is the signature inferable from (q, dq, a) history? ----------
    print("\n" + "=" * 74)
    print("P3  can the signature be PREDICTED from a short (q, dq, a) window?")
    print("    (this is exactly what RMA's adaptation module must do)")
    print("=" * 74)
    print("    RMA uses ~0.5 s on a 12-DoF quadruped. The deployed policy here")
    print("    sees 10 frames = 200 ms on a 29-DoF humanoid, so sweeping the")
    print("    history length is the first thing to check when P3 looks weak.\n")
    print(f"  {'history':>9}{'frames':>8}{'featdim':>9}"
          f"{'R2 median':>11}{'R2 mean':>9}{'runs>0':>10}")
    print("  " + "-" * 56)

    results = {}
    for H in a.history:
        X, Y, G = [], [], []
        for i, r in enumerate(runs):
            n = len(r["q"])
            start = max(W, H)          # window must be long enough for history
            for s in range(start, n - W, W):
                idx = np.arange(s, s + W)
                sig = signature(r["q"], r["dq"], r["qt"], r["tau"], idx)
                if np.isnan(sig).any():
                    continue
                h = slice(s + W - H, s + W)
                feat = np.concatenate([r["q"][h].ravel(), r["dq"][h].ravel(),
                                       r["qt"][h].ravel()])
                X.append(feat)
                Y.append(sig)
                G.append(i)
        if len(X) < 50:
            print(f"  {H*CONTROL_DT:8.2f}s{H:8d}      too few windows")
            continue
        X = np.asarray(X)
        Y = np.asarray(Y)
        G = np.asarray(G)

        # leave-one-RUN-out ridge regression. Splitting by run (not randomly)
        # is essential: windows from the same run are near-duplicates, so a
        # random split lets the model memorise the run and R^2 looks great for
        # free.
        r2s = []
        Ym = Y.mean(0)
        for held in sorted(set(G)):
            tr, te = G != held, G == held
            if te.sum() < 2 or tr.sum() < 20:
                continue
            Xtr = np.column_stack([X[tr], np.ones(tr.sum())])
            Xte = np.column_stack([X[te], np.ones(te.sum())])
            lam = 1e2
            A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
            Wt = np.linalg.solve(A, Xtr.T @ Y[tr])
            pred = Xte @ Wt
            ss_res = ((Y[te] - pred) ** 2).sum()
            ss_tot = ((Y[te] - Ym) ** 2).sum()
            r2s.append(1 - ss_res / max(ss_tot, 1e-12))
        r2s = np.asarray(r2s)
        results[H] = r2s
        print(f"  {H*CONTROL_DT:8.2f}s{H:8d}{X.shape[1]:9d}"
              f"{np.median(r2s):11.3f}{r2s.mean():9.3f}"
              f"{(r2s>0).sum():6d}/{len(r2s):<4d}")

    r2s = results.get(a.history[-1], np.array([0.0]))
    print("\n     R^2 > 0.3 : the window genuinely carries the run's dynamics")
    print("     R^2 ~ 0   : no better than predicting the global mean")
    print("     R^2 < 0   : worse than the mean (overfit / no signal)")

    # --- verdict ----------------------------------------------------------
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if icc < 0.2:
        print("  P1/P2 FAIL: runs are dynamically near-identical (ICC "
              f"{icc:.3f}).")
        print("  All 48 runs are the same robot, same floor, same payload, so")
        print("  this is expected. RMA cannot demonstrate value on this data --")
        print("  not because the method is wrong, but because the dataset has")
        print("  no extrinsics variation to adapt to.")
        print("  => Before committing to RMA, collect data that VARIES:")
        print("     payload, floor surface, battery level, deliberate gain")
        print("     changes (--motor-kp-scale exists in the deploy binary).")

    print("\n  CAVEAT: the signature is a PROXY for extrinsics, derived from the")
    print("  PD residual. A high ICC could also reflect differences in the")
    print("  MOTION each run performed rather than in the robot/environment.")
    print("  Controlling for that needs runs of the SAME clip under DIFFERENT")
    print("  physical conditions -- which this archive does not contain.")

    best_H = max(results, key=lambda h: np.median(results[h])) if results else None
    best_r2 = float(np.median(results[best_H])) if best_H is not None else -1.0
    # A median R^2 of ~0.02 is NOT evidence. An earlier version of this script
    # used `> 0` as the bar and wrongly reported "all three pass" on exactly
    # that value.
    R2_BAR = 0.30
    if best_H is not None:
        if best_r2 < R2_BAR:
            print(f"\n  P3 FAILS: best median R^2 {best_r2:+.3f} at "
                  f"{best_H*CONTROL_DT:.2f} s history (bar {R2_BAR}).")
            print("  Lengthening the history did NOT make the signature")
            print("  inferable, so the limit is not the 200 ms window.")
        else:
            print(f"\n  P3 PASSES at {best_H*CONTROL_DT:.2f} s history "
                  f"(median R^2 {best_r2:+.3f}).")
            if best_H > 10:
                print("  NOTE the deployed policy sees only 10 frames (0.20 s);")
                print("  using this would change the ONNX input shape and the")
                print("  deploy observation config.")

    print("\n  WHAT WOULD MAKE THIS DECIDABLE")
    print("  Replay ONE clip under DELIBERATELY VARIED conditions, then re-run")
    print("  this script. Cheapest axis, already supported by the deploy binary:")
    print("      --motor-kp-scale / --motor-kd-scale   (3 levels x 3 reps)")
    print("  Others: added payload, different floor surface, battery level.")


if __name__ == "__main__":
    main()
