#!/usr/bin/env python3
"""Would ankle-pitch gain scaling move the PREDICTED real robot toward the
TRAINING simulator?   (framing: sim2real/phaseC1_damping_scan.md section 6.2)

Scoring a scaled-gain sim against the logged real next-state is the documented
WRONG framing (section 6.1): the logged trajectory was produced under unscaled
gains. Instead, every distance is taken to the training ground truth:

  error_before = rms(real_next      - sim_normal_next)    observed gap
  error_after  = rms(real*_pred_next - sim_normal_next)    predicted gap if scaled

where real*_pred is a model of the real hardware (here: nominal MuJoCo + the M6
torque fit, in place of the document's Phase B dof_damping) driven by the
SCALED PD law, teacher-forced from the same measured state.

The honest control for "does scaling help" is the UNSCALED proxy (`fit only`),
not the real robot: the proxy reproduces only part of the real gap, so ANY proxy
looks closer to the training sim than the real robot does.

For every step t of the held-out session, start sim and robot from the SAME
measured state (q_t, dq_t) and the same command q_target_t, advance one control
period, and compare next-state.

  sim_normal   tau = kp (q* - q) - kd dq                      nominal gains
  sim_fit      tau = tau_pd + M6(q, dq, tau_pd, ddq)           "tau_fit"
  sim_scale    tau = PD with kp, kd x1.5 on L/R ankle pitch    (joints 4, 10)
  sim_both     tau = scaled PD + M6

M6 is the model from model_eval/fit_friction_model.py, fitted on EARLIER
sessions only and applied to the held-out one. It is fitted on the residual of
the NOMINAL PD, so in `sim_both` it is added on top of the scaled PD; any
overlap between the two corrections (both add velocity-proportional torque on
these joints) is not removed and is a known caveat of that row.

Two quantities are compared per joint, each for q and for dq:

  GAP    g = real_next - sim_normal_next      what is wrong with the baseline
  SHIFT  s = sim_cfg_next - sim_normal_next   how far the change moved the sim

  * RMSE before = rms(g)        RMSE after = rms(g - s) = rms(real - sim_cfg)
  * corr(s, g), slope(g on s)   does the move point the right way, and how far

RMSE-after < RMSE-before is the actual test of improvement. A high corr with a
tiny SHIFT means the change is directionally right but far too weak to matter.

BASE STATE CAVEAT: base position and linear velocity are not logged. The sim uses
a floating base standing on the floor (height set so the lowest geom touches it),
IMU orientation/angular velocity, and ZERO base linear velocity. Ankle pitch is a
ground-contact joint, so it is the joint this setup is LEAST able to reproduce.
Read the ankle rows as "relative change between configs", not as absolute
fidelity.
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "sim2real"), os.path.join(ROOT, "model_eval")):
    sys.path.insert(0, p)

from deploy_constants import (  # noqa: E402
    CONTROL_DT,
    EFFORT_LIMIT,
    JOINT_NAMES,
    KD,
    KP,
    load_csv,
    load_run,
)
from fit_friction_model import design, load_data, smooth  # noqa: E402
from sim2real_phaseC1b_onestep_qdq import build_model  # noqa: E402

DATA_ROOT = "/home/grease/g1_robot_data"
ANKLE_PITCH = [4, 10]
SCALE = 1.5
PHASE_B_DAMPING = {4: 1.19, 10: 1.09, 14: 0.98}   # phaseC1_damping_scan.md section 1
M6 = 6
LAM = 1e-3


def session_of(name):
    part = name.split("/")[0]
    return part[len("g1_run_"):] if part.startswith("g1_run_") else "0901"


def load_test_run(name):
    """Same trimming as fit_friction_model.load_data, plus IMU channels."""
    run = os.path.join(DATA_ROOT, name)
    d = load_run(run)
    q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
    mv = np.abs(dq).max(axis=1) > 0.05
    lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
    q, dq, qt, tau = q[lo:hi], dq[lo:hi], qt[lo:hi], tau[lo:hi]
    bq, _ = load_csv(os.path.join(run, "base_quat.csv"), "base_q")
    bw, _ = load_csv(os.path.join(run, "base_ang_vel.csv"), "base_w")
    bq, bw = bq[lo:hi], bw[lo:hi]
    n = min(len(q), len(bq), len(bw))
    q, dq, qt, tau, bq, bw = q[:n], dq[:n], qt[:n], tau[:n], bq[:n], bw[:n]
    ddq = np.gradient(smooth(dq, 5), CONTROL_DT, axis=0)
    tau_pd = KP * (qt - q) - KD * dq
    return dict(name=name, q=q, dq=dq, qt=qt, tau=tau, bq=bq, bw=bw,
                ddq=ddq, tau_pd=tau_pd, res=tau - tau_pd)


def fit_m6(train):
    coefs = {}
    for j in range(29):
        X = np.vstack([design(d, j, M6) for d in train])
        y = np.concatenate([d["res"][:, j] for d in train])
        coefs[j] = np.linalg.solve(X.T @ X + LAM * np.eye(X.shape[1]), X.T @ y)
    return coefs


def m6_residual(d, coefs):
    return np.column_stack([design(d, j, M6) @ coefs[j] for j in range(29)])


def simulate(model, run, taus, n_sub):
    """One control step from each measured state, for each torque config.

    taus: dict name -> (T, 29) torque array. Returns dict name -> (q_next, dq_next).
    """
    data = mujoco.MjData(model)
    body_mask = np.array([model.geom_bodyid[g] != 0 for g in range(model.ngeom)])
    T = len(run["q"]) - 1
    out = {k: (np.zeros((T, 29)), np.zeros((T, 29))) for k in taus}
    for t in range(T):
        for k, tau in taus.items():
            mujoco.mj_resetData(model, data)
            data.qpos[0:3] = [0.0, 0.0, 0.0]
            data.qpos[3:7] = run["bq"][t]
            data.qpos[7:] = run["q"][t]
            mujoco.mj_forward(model, data)
            data.qpos[2] = -float(data.geom_xpos[body_mask, 2].min())
            data.qvel[0:3] = 0.0
            data.qvel[3:6] = run["bw"][t]
            data.qvel[6:] = run["dq"][t]
            mujoco.mj_forward(model, data)
            data.ctrl[:] = np.clip(tau[t], -EFFORT_LIMIT, EFFORT_LIMIT)
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            out[k][0][t] = data.qpos[7:]
            out[k][1][t] = data.qvel[6:]
    return out


def rms(x):
    return float(np.sqrt((np.asarray(x) ** 2).mean()))


def compare(real, base, cfg):
    g = real - base
    s = cfg - base
    before, after = rms(g), rms(g - s)
    c = np.corrcoef(s, g)[0, 1] if s.std() > 1e-12 and g.std() > 1e-12 else np.nan
    slope = float(s @ g / (s @ s)) if (s @ s) > 1e-18 else np.nan
    return rms(s), before, after, c, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="0924")
    ap.add_argument("--proxy", choices=["fit", "damping"], default="fit",
                    help="model of the REAL hardware's extra dynamics: 'fit' = M6 "
                         "torque term; 'damping' = Phase B dof_damping "
                         "{4:1.19, 10:1.09, 14:0.98} (phaseC1_damping_scan.md 6.3)")
    a = ap.parse_args()
    plabel = "fit" if a.proxy == "fit" else "damping"

    allrun = load_data(80, 5)
    for d in allrun:
        d["sess"] = session_of(d["name"])
    train = [d for d in allrun if d["sess"] < a.test]
    test_names = [d["name"] for d in allrun if d["sess"] == a.test]
    print(f"proxy for real hardware: {a.proxy}")
    print(f"held-out {a.test}: {len(test_names)} runs"
          + (f"; M6 fitted on {len(train)} runs from earlier sessions"
             if a.proxy == "fit" else ""))
    coefs = fit_m6(train) if a.proxy == "fit" else None

    model = build_model(False, weld_base=False)          # training ground truth
    model_cal = (build_model(False, PHASE_B_DAMPING, weld_base=False)
                 if a.proxy == "damping" else model)
    n_sub = int(round(CONTROL_DT / model.opt.timestep))
    if a.proxy == "damping":
        print("dof_damping  baseline model -> calibrated model:")
        for j, b in PHASE_B_DAMPING.items():
            dof = model.jnt_dofadr[model.actuator_trnid[j, 0]]
            print(f"    {JOINT_NAMES[j]:20s} {model.dof_damping[dof]:.4f} -> "
                  f"{model_cal.dof_damping[dof]:.4f}")

    kp_s, kd_s = KP.copy(), KD.copy()
    kp_s[ANKLE_PITCH] *= SCALE
    kd_s[ANKLE_PITCH] *= SCALE
    print(f"scaled joints: {[JOINT_NAMES[j] for j in ANKLE_PITCH]}  "
          f"kp {KP[4]:.1f} -> {kp_s[4]:.1f}, kd {KD[4]:.2f} -> {kd_s[4]:.2f}\n")

    R = {k: [] for k in ("q_real", "dq_real")}
    S = {c: {"q": [], "dq": []} for c in ("normal", "fit", "both", "kdonly")}
    q_t, dq_t = [], []
    for nm in test_names:
        run = load_test_run(nm)
        res = m6_residual(run, coefs) if a.proxy == "fit" else 0.0
        qq, dd, qt = run["q"], run["dq"], run["qt"]
        kp_k = KP.copy()                       # kd-only variant (Phase B: Kp ~ nominal)
        kd_k = KD.copy()
        kd_k[ANKLE_PITCH] *= SCALE
        base_taus = {"normal": KP * (qt - qq) - KD * dd}      # training ground truth
        proxy_taus = {
            # proxy for REAL hardware, UNscaled
            "fit": KP * (qt - qq) - KD * dd + res,
            # predicted REAL*, scaled kp AND kd x1.5
            "both": kp_s * (qt - qq) - kd_s * dd + res,
            # predicted REAL*, scaled kd only x1.5 (the Phase B measurement)
            "kdonly": kp_k * (qt - qq) - kd_k * dd + res,
        }
        base_taus = {k: v[:-1] for k, v in base_taus.items()}
        proxy_taus = {k: v[:-1] for k, v in proxy_taus.items()}
        sim = simulate(model, run, base_taus, n_sub)
        sim.update(simulate(model_cal, run, proxy_taus, n_sub))
        for c in S:
            S[c]["q"].append(sim[c][0])
            S[c]["dq"].append(sim[c][1])
        R["q_real"].append(qq[1:])
        R["dq_real"].append(dd[1:])
        q_t.append(qq[:-1])
        dq_t.append(dd[:-1])
        print(f"  simulated {nm}: {len(qq)-1} steps")

    q_real = np.vstack(R["q_real"])
    dq_real = np.vstack(R["dq_real"])
    q_t = np.vstack(q_t)
    dq_t = np.vstack(dq_t)
    Sq = {c: np.vstack(v["q"]) for c, v in S.items()}
    Sd = {c: np.vstack(v["dq"]) for c, v in S.items()}
    print(f"\nsteps compared: {len(q_real)}   real per-step change: "
          f"ankle_pitch dq {rms(dq_real[:,4]-dq_t[:,4]):.3f} / "
          f"{rms(dq_real[:,10]-dq_t[:,10]):.3f} rad/s\n")

    for label, real, sim, f in (
        ("dq_next (rad/s)", dq_real, Sd, 1.0),
        ("q_next (deg)", q_real, Sq, 180.0 / np.pi),
    ):
        print("=" * 100)
        print(f"{label}  --  distance to sim_normal_next (the TRAINING ground truth)")
        print("  before = rms(real_next        - sim_normal_next)   observed, unscaled")
        print("  after  = rms(sim_cfg_next     - sim_normal_next)   cfg = predicted real")
        print("=" * 100)
        print(f"{'joint':20s}{'before (real)':>15}{plabel+' only':>14}"
              f"{plabel+'+kp,kd x1.5':>20}{plabel+'+kd x1.5':>17}")
        print("-" * 100)
        for j in ANKLE_PITCH + ["ALL"]:
            if j == "ALL":
                js = list(range(29))
                name = "ALL-29 MEAN"
                rr = lambda x, y: np.mean([rms((x[:, k] - y[:, k]) * f) for k in js])  # noqa: E731
            else:
                name = JOINT_NAMES[j]
                rr = lambda x, y, j=j: rms((x[:, j] - y[:, j]) * f)  # noqa: E731
            b = rr(real, sim["normal"])
            vals = {c: rr(sim[c], sim["normal"]) for c in ("fit", "both", "kdonly")}
            W = (("fit", 14), ("both", 20), ("kdonly", 17))
            print(f"{name:20s}{b:15.4f}" + "".join(f"{vals[c]:>{w}.4f}" for c, w in W))
            print(f"{'  vs before (real)':20s}{'':15s}"
                  + "".join(f"{100*(vals[c]-b)/b:>+{w-1}.1f}%" for c, w in W))
            p = vals["fit"]
            print(f"{'  vs '+plabel+' only':20s}{'':15s}{'':>14}"
                  + "".join(f"{100*(vals[c]-p)/p:>+{w-1}.1f}%" for c, w in W[1:])
                  + "   <- effect of the scaling itself")
        print()


if __name__ == "__main__":
    main()
