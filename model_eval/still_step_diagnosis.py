#!/usr/bin/env python3
"""Still-step diagnosis: why does MuJoCo move a robot that is standing still?

At a REAL rest state (all |dq| small, real next-step change ~0) the sim, started from the real
(q, dq, IMU) and driven by the real PD torque, still predicts an acceleration. Its net unbalanced
generalised force on joint j is

    r_j = (M qacc)_j = tau_pd_j - g_hold_j ,      g_hold = qfrc_bias - qfrc_constraint - qfrc_passive
                                                   (gravity + base/contact reaction the sim needs to hold the pose)

In the real robot the pose is held, so r_real ~ 0 up to static friction.  Decompose the sim's r by
regression on the REAL rest steps of the training runs:

    r_j = alpha_j  +  beta_j * g_hold_j  +  c_j * d_j  +  eps
          |offset|     |mass/gravity scale|   |friction: d_j = direction of the last motion|

  alpha  constant offset  -> joint zero-offset / COM / constant torque bias
  beta   proportional to the gravity load the sim computes -> mass / COM scale error
  c*d    sign follows the direction the joint last moved -> static (Coulomb) friction / stiction

and evaluate how much of the sim's still-step torque imbalance each term removes on the held-out
validation runs (A: 0924, B: 1001 gain x1.5).  Torque is converted to a dq error with
dq_err ~ dt * r / M_jj (good for arms/waist; legs are coupled through the base, so approximate).

Usage:  .venv_sim/bin/python model_eval/still_step_diagnosis.py
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model_eval"))
sys.path.insert(0, os.path.join(ROOT, "sim2real"))
import sim2real_phaseC1b_onestep_qdq as M  # noqa: E402
import world_model_qdq as W  # noqa: E402
import world_model_residual as R  # noqa: E402
from deploy_constants import CONTROL_DT, EFFORT_LIMIT, JOINT_NAMES, KD, KP  # noqa: E402

mujoco = M.mujoco


def still_samples(path, model, data, thr, max_n, look_back=150):
    r = R.load_run_full(path)
    if r is None:
        return None
    q, dq, qt, bq, w, gain = r["q"], r["dq"], r["qt"], r["bq"], r["w"], r["gain"][0]
    T = len(q)
    still = np.where(np.abs(dq[:-1]).max(1) <= thr)[0]
    still = still[still >= look_back]
    if len(still) == 0:
        return None
    if len(still) > max_n:
        still = still[np.linspace(0, len(still) - 1, max_n).astype(int)]
    kp, kd = KP * gain, KD * gain
    out = dict(r=[], g=[], d=[], a=[], Mjj=[], q=[], dq_next=[], dq=[])
    nv = model.nv
    dense = np.zeros((nv, nv))
    for t in still:
        data.qpos[7:] = q[t]; data.qvel[6:] = dq[t]
        data.qpos[0:3] = 0.0; data.qpos[3:7] = bq[t]
        mujoco.mj_forward(model, data)
        zl = min(data.geom_xpos[g, 2] for g in range(model.ngeom) if model.geom_bodyid[g] != 0)
        data.qpos[2] = -zl
        data.qvel[0:3] = 0.0; data.qvel[3:6] = w[t]
        tau = np.clip(kp * (qt[t] - q[t]) - kd * dq[t], -EFFORT_LIMIT, EFFORT_LIMIT)
        data.ctrl[:] = tau
        mujoco.mj_forward(model, data)
        r_net = data.qfrc_actuator[6:] - data.qfrc_bias[6:] + data.qfrc_constraint[6:] + data.qfrc_passive[6:]
        g_hold = data.qfrc_bias[6:] - data.qfrc_constraint[6:] - data.qfrc_passive[6:]
        mujoco.mj_fullM(model, dense, data.qM)
        # direction of the last significant motion of each joint within the look-back window
        d = np.zeros(29)
        for j in range(29):
            mv = np.where(np.abs(dq[t - look_back:t, j]) > 0.05)[0]
            if len(mv):
                d[j] = np.sign(dq[t - look_back + mv[-1], j])
        out["r"].append(r_net); out["g"].append(g_hold); out["d"].append(d)
        out["a"].append(data.qacc[6:].copy()); out["Mjj"].append(np.diag(dense)[6:].copy())
        out["q"].append(q[t]); out["dq_next"].append(dq[t + 1]); out["dq"].append(dq[t])
    return {k: np.asarray(v) for k, v in out.items()}


def collect(paths, model, data, thr, max_n):
    parts = [s for s in (still_samples(p, model, data, thr, max_n) for p in paths) if s is not None]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, len(parts)


def ols(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.10, help="still = max|dq| <= thr rad/s over all joints")
    ap.add_argument("--train-per-run", type=int, default=400)
    ap.add_argument("--val-per-run", type=int, default=1500)
    a = ap.parse_args()
    tr_p, va_p, vb_p = W.split_runs(0)
    model = M.build_model(True, weld_base=False)
    data = mujoco.MjData(model)
    print(f"still = every joint |dq| <= {a.thr} rad/s ; sim = MuJoCo (floating base, contact, real armature), true gains per run")
    tr, ntr = collect(tr_p, model, data, a.thr, a.train_per_run)
    va, nva = collect(va_p, model, data, a.thr, a.val_per_run)
    vb, nvb = collect(vb_p, model, data, a.thr, a.val_per_run)
    print(f"still samples: train {len(tr['r'])} ({ntr} runs)   valA {len(va['r'])} ({nva} runs)   valB {len(vb['r'])} ({nvb} runs)")

    # fit alpha, beta, c per joint on TRAIN
    coef = np.zeros((29, 3)); has_d = np.zeros(29, bool)
    for j in range(29):
        y = tr["r"][:, j]
        dj = tr["d"][:, j]
        X = np.stack([np.ones(len(y)), tr["g"][:, j], dj], 1)
        has_d[j] = (dj != 0).sum() > 30
        coef[j] = ols(X, y)

    def after(val, j, stage):
        y = val["r"][:, j]; g = val["g"][:, j]; d = val["d"][:, j]
        al, be, c = coef[j]
        if stage == 0:
            return y
        if stage == 1:
            return y - al
        if stage == 2:
            return y - al - be * g
        return y - al - be * g - c * d

    names = ["0 sim as is", "1 + offset alpha", "2 + gravity scale beta*g", "3 + friction c*dir"]
    val = {k: np.concatenate([va[k], vb[k]]) for k in va}
    print("\nValidation A+B: RMS of the sim's unbalanced torque r [Nm] after removing each term (fit on training still steps)")
    print(f"  {'joint':22s}{'alpha':>8s}{'beta':>7s}{'c':>7s}" + "".join(f"{n:>26s}" for n in names) +
          f"{'  dq err now->after [rad/s]':>28s}")
    rows = []
    for j in range(29):
        rms = [np.sqrt((after(val, j, s) ** 2).mean()) for s in range(4)]
        Mj = np.median(val["Mjj"][:, j])
        dq0 = CONTROL_DT * rms[0] / Mj; dq3 = CONTROL_DT * rms[3] / Mj
        rows.append((j, rms, dq0, dq3))
        print(f"  {JOINT_NAMES[j]:22s}{coef[j, 0]:+8.3f}{coef[j, 1]:+7.2f}{coef[j, 2]:+7.3f}" +
              "".join(f"{x:26.3f}" for x in rms) + f"{dq0:14.3f} ->{dq3:7.3f}")

    print("\nGroup summary (mean over joints of RMS torque r, Nm) and share of the imbalance variance removed by each term")
    groups = {"legs": range(0, 12), "waist": range(12, 15), "arms": range(15, 29), "ankle_pitch": [4, 10], "all": range(29)}
    for g, js in groups.items():
        js = list(js)
        v = [np.mean([rows[j][1][s] ** 2 for j in js]) for s in range(4)]
        share = [(v[s - 1] - v[s]) / v[0] * 100 for s in (1, 2, 3)]
        print(f"  {g:12s} RMS r: " + "  ".join(f"{np.sqrt(x):.3f}" for x in v) +
              f"   variance removed: offset {share[0]:5.1f}%  gravity-scale {share[1]:5.1f}%  friction {share[2]:5.1f}%"
              f"   remaining {100 - sum(share):5.1f}%")

    # raw hysteresis check on training: mean r conditioned on approach direction
    print("\nHysteresis check (training still steps): mean r for joints whose last motion was UP (+) vs DOWN (-)  [Nm]")
    print(f"  {'joint':22s}{'n(+)':>7s}{'r(+)':>9s}{'n(-)':>7s}{'r(-)':>9s}{'offset=(+ + -)/2':>20s}{'friction=(+ - -)/2':>22s}")
    for j in range(29):
        up = tr["d"][:, j] > 0; dn = tr["d"][:, j] < 0
        if up.sum() < 20 or dn.sum() < 20:
            continue
        ru, rd = tr["r"][up, j].mean(), tr["r"][dn, j].mean()
        print(f"  {JOINT_NAMES[j]:22s}{up.sum():7d}{ru:+9.3f}{dn.sum():7d}{rd:+9.3f}{(ru + rd) / 2:+20.3f}{(ru - rd) / 2:+22.3f}")

    print("\nCheck: sim acceleration vs unbalanced torque, arms (should be ~ r/M):  corr(a_sim, r/M) per group")
    for gname, js in (("arms", range(15, 29)), ("waist", range(12, 15)), ("legs", range(0, 12))):
        cs = [np.corrcoef(val["a"][:, j], val["r"][:, j] / val["Mjj"][:, j])[0, 1] for j in js]
        print(f"  {gname:6s} mean corr {np.nanmean(cs):.3f}")


if __name__ == "__main__":
    main()
