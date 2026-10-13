#!/usr/bin/env python3
"""Actuator-model test on the 1001 runs.

Target: measured motor torque tau_est (motor_torque.csv) per joint. Inputs: q, dq, q_target.
Four runs ran with kp,kd x1.5 on joints 4 and 10 (ankle pitch); four matched runs ran nominal.

Models (per joint, least squares):
  M0  tau = pd(gains)                       pure PD law, NO fitted parameters
  M1  tau = g * pd
  M2  tau = g * pd + b
  M3  tau = g * pd + b + c*tanh(dq/0.05) + v*dq      (offset, Coulomb, viscous)
  M3d same as M3 but pd computed from commands delayed by d steps (best d in 0..3)

Tests:
  A. Does the PD law with the TRUE gains explain the scaled runs (M0 with 1.5x on joints 4,10) vs with nominal gains?
  B. Fit on the 4 unscaled runs, predict the 4 SCALED runs when the model is told the 1.5x gains (zero-shot).
  C. Leave-one-clip-out inside the unscaled runs (generalisation to a new clip, same hardware state).
  D. Fit g on scaled runs only with nominal pd: should recover ~1.5 on joints 4,10 and ~1 elsewhere.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim2real"))
from deploy_constants import KP, KD, JOINT_NAMES, load_run  # noqa: E402

RUNS = "/home/grease/g1_robot_data/g1_run_1001/20261001_"
CLIPS = {"walk_180": ("083356", "084808"), "jog": ("083950", "085817"),
         "dance": ("085157", "085443"), "high_jump": ("090040", "090302")}
J = [4, 10]
SC = np.ones(29); SC[J] = 1.5


def load(rid):
    d = load_run(RUNS + rid)
    q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
    mv = np.abs(dq).max(axis=1) > 0.05
    lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
    return q[lo:hi], dq[lo:hi], qt[lo:hi], tau[lo:hi]


def pd(q, dq, qt, gain, delay=0):
    """PD torque; delay d: use command/state from t-d (first d rows dropped by caller)."""
    kp, kd = KP * gain, KD * gain
    if delay:
        return kp * (qt[:-delay] - q[:-delay]) - kd * dq[:-delay]
    return kp * (qt - q) - kd * dq


def feats(model, p, dq):
    cols = [p]
    if model in ("M2", "M3"):
        cols.append(np.ones_like(p))
    if model == "M3":
        cols += [np.tanh(dq / 0.05), dq]
    return np.stack(cols, axis=-1)


def fit(model, P, DQ, T):
    """per joint least squares. P,DQ,T: (N,29). returns list of coef arrays."""
    out = []
    for j in range(29):
        X = feats(model, P[:, j], DQ[:, j])
        c, *_ = np.linalg.lstsq(X, T[:, j], rcond=None)
        out.append(c)
    return out


def predict(model, coefs, P, DQ):
    return np.stack([feats(model, P[:, j], DQ[:, j]) @ coefs[j] for j in range(29)], axis=1)


def stack(rids, scaled, delay=0):
    Ps, Ds, Ts = [], [], []
    for rid in rids:
        q, dq, qt, tau = load(rid)
        gain = SC if scaled else np.ones(29)
        p = pd(q, dq, qt, gain, delay)
        Ps.append(p); Ds.append(dq[:len(p)]); Ts.append(tau[delay:] if delay else tau)
    return np.concatenate(Ps), np.concatenate(Ds), np.concatenate(Ts)


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean(0))


unsc = [v[0] for v in CLIPS.values()]
scal = [v[1] for v in CLIPS.values()]
np.set_printoptions(precision=3, suppress=True)

print("=== A. PD law, no fitted parameters: torque RMSE [Nm], ankle pitch L/R and mean of other 27 joints ===")
for name, rids, scaled in (("unscaled runs", unsc, False), ("scaled runs", scal, True)):
    for gtag, gain_scaled in (("nominal gains", False), ("TRUE gains (x1.5 on 4,10)", scaled)):
        P, D, T = stack(rids, gain_scaled)
        e = rmse(P, T)
        print(f"{name:14s} {gtag:28s} L {e[4]:.3f}  R {e[10]:.3f}  other {np.delete(e, J).mean():.3f}   "
              f"slope tau/pd L {((P[:,4]*T[:,4]).sum()/(P[:,4]**2).sum()):.2f} R {((P[:,10]*T[:,10]).sum()/(P[:,10]**2).sum()):.2f}")

print("\n=== B. fit on UNSCALED runs, predict SCALED runs (model given the true x1.5 gains), RMSE [Nm] ===")
Pu, Du, Tu = stack(unsc, False)
Ps, Ds, Ts = stack(scal, True)
print(f"{'model':8s}{'L ankle p':>11s}{'R ankle p':>11s}{'other 27':>10s}{'  all 29':>9s}")
base = rmse(Ps, Ts)
print(f"{'M0 PD':8s}{base[4]:11.3f}{base[10]:11.3f}{np.delete(base, J).mean():10.3f}{base.mean():9.3f}")
for m in ("M1", "M2", "M3"):
    c = fit(m, Pu, Du, Tu)
    e = rmse(predict(m, c, Ps, Ds), Ts)
    print(f"{m:8s}{e[4]:11.3f}{e[10]:11.3f}{np.delete(e, J).mean():10.3f}{e.mean():9.3f}")

print("\n=== C. leave-one-clip-out inside UNSCALED runs, RMSE [Nm] (mean over 4 folds) ===")
print(f"{'model':8s}{'L ankle p':>11s}{'R ankle p':>11s}{'other 27':>10s}{'  all 29':>9s}")
for m in ("M0", "M1", "M2", "M3"):
    es = []
    for k in range(4):
        tr = [r for i, r in enumerate(unsc) if i != k]
        Pt, Dt, Tt = stack(tr, False)
        Pk, Dk, Tk = stack([unsc[k]], False)
        pred = Pk if m == "M0" else predict(m, fit(m, Pt, Dt, Tt), Pk, Dk)
        es.append(rmse(pred, Tk))
    e = np.mean(es, 0)
    print(f"{m:8s}{e[4]:11.3f}{e[10]:11.3f}{np.delete(e, J).mean():10.3f}{e.mean():9.3f}")

print("\n=== D. fitted gain g (tau = g*pd_nominal) on scaled vs unscaled runs ===")
Pn_s, _, Tn_s = stack(scal, False)
Pn_u, _, Tn_u = stack(unsc, False)
gs = (Pn_s * Tn_s).sum(0) / (Pn_s ** 2).sum(0)
gu = (Pn_u * Tn_u).sum(0) / (Pn_u ** 2).sum(0)
for j in J:
    print(f"{JOINT_NAMES[j]:20s} g(unscaled)={gu[j]:.3f}  g(scaled, nominal pd)={gs[j]:.3f}  ratio={gs[j]/gu[j]:.3f}")
print(f"other 27 joints: g(unscaled) mean {np.delete(gu, J).mean():.3f}  g(scaled) mean {np.delete(gs, J).mean():.3f}")

print("\n=== E. M3 per-joint fit on unscaled runs (ankle pitch and examples): g, bias, coulomb, viscous ===")
c3 = fit("M3", Pu, Du, Tu)
for j in (4, 10, 12, 13, 14, 15, 18):
    print(f"{JOINT_NAMES[j]:22s} g={c3[j][0]:.3f} b={c3[j][1]:+.3f} Nm  coulomb={c3[j][2]:+.3f} Nm  visc={c3[j][3]:+.3f} Nm/(rad/s)")

print("\n=== F. command delay for M3 (fit unscaled, test scaled), RMSE [Nm] ===")
for d in (0, 1, 2, 3):
    Pu_d, Du_d, Tu_d = stack(unsc, False, d)
    Ps_d, Ds_d, Ts_d = stack(scal, True, d)
    e = rmse(predict("M3", fit("M3", Pu_d, Du_d, Tu_d), Ps_d, Ds_d), Ts_d)
    print(f"delay {d} step(s): L {e[4]:.3f} R {e[10]:.3f} other {np.delete(e, J).mean():.3f} all {e.mean():.3f}")

print("\n=== G. M3 with the viscous (damping) term scaled by the commanded gain ratio (kd-proportional), fit unscaled -> test scaled ===")
def stack_s(rids, scaled):
    out = []
    for rid in rids:
        q, dq, qt, tau = load(rid)
        gain = SC if scaled else np.ones(29)
        out.append((pd(q, dq, qt, gain), dq, np.broadcast_to(gain, dq.shape), tau))
    return [np.concatenate(x) for x in zip(*out)]

def fit_s(P, D, G, T):
    cs = []
    for j in range(29):
        X = np.stack([P[:, j], np.ones(len(P)), np.tanh(D[:, j] / 0.05), D[:, j] * G[:, j]], axis=1)
        cs.append(np.linalg.lstsq(X, T[:, j], rcond=None)[0])
    return cs

def pred_s(cs, P, D, G):
    return np.stack([np.stack([P[:, j], np.ones(len(P)), np.tanh(D[:, j] / 0.05), D[:, j] * G[:, j]], axis=1) @ cs[j] for j in range(29)], axis=1)

Pu_, Du_, Gu_, Tu_ = stack_s(unsc, False)
Ps_, Ds_, Gs_, Ts_ = stack_s(scal, True)
cs = fit_s(Pu_, Du_, Gu_, Tu_)
e = rmse(pred_s(cs, Ps_, Ds_, Gs_), Ts_)
print(f"M3 (viscous x gain)  L {e[4]:.3f}  R {e[10]:.3f}  other {np.delete(e, J).mean():.3f}  all {e.mean():.3f}")
print("   compare M3 (viscous not scaled): L 0.511  R 0.905  other 0.111 ; M0 PD: L 0.594  R 0.757 other 0.276")
