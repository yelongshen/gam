#!/usr/bin/env python3
"""Explain the inverse-dynamics gap by decomposing the torque, term by term.

WHERE THE TWO NUMBERS COME FROM
-------------------------------
tau_real   `motor_torque.csv` = `tau_est` read back from each motor driver.
           This is the torque the motor ACTUALLY produced. It is a
           measurement, not a command -- the deploy binary never sets
           feed-forward torque (`tau_ff = 0`), the motor's internal PD loop
           produces it from the position target.

tau_model  computed here from the motion the robot actually performed:

               tau_model = M(q) qddot  +  C(q, qdot) qdot  +  G(q)
                           \_________/    \_______________________/
                            inertial        MuJoCo's `qfrc_bias`
                            term            (Coriolis + centrifugal + gravity)

           i.e. "according to the rigid-body model, how much torque does it
           take to produce exactly this acceleration, at this configuration,
           at this velocity?"

Both are in Nm. The difference is what the rigid-body model cannot explain.

WHY WOULD THE MEASURED TORQUE BE LARGER?
----------------------------------------
Because a real motor spends torque on things that produce NO acceleration,
and inverse dynamics only ever sees acceleration:

  friction / gearing loss  a harmonic drive can burn several Nm just turning.
                           Invisible to M qddot + C + G.
  contact forces           a foot pressing the floor needs large torque while
                           barely accelerating. This is the leg story.
  model error              wrong link mass / inertia / COM.

This script separates those by testing each claim rather than asserting it:

  TEST 1  near-static frames. If the robot is barely moving, qddot ~ 0 and
          qdot ~ 0, so tau_model collapses to G(q) alone. Comparing tau_real
          against G(q) there tests the model's MASS DISTRIBUTION with no
          friction-vs-inertia confound... except for static friction, which
          is exactly what the residual then measures.
  TEST 2  torque split into |gravity| / |inertial| / |Coriolis| so it is
          visible which term actually dominates.
  TEST 3  sensitivity of the whole result to the ddq smoothing window, which
          is a free parameter I chose and which directly scales the inertial
          term.
  TEST 4  friction model fit: residual vs sign(qdot) (Coulomb) and vs qdot
          (viscous). If friction is the story, this should fit.

Usage:
  .venv_sim/bin/python model_eval/sim2real_torque_decomposition.py
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim2real"))

from deploy_constants import (  # noqa: E402
    CONTROL_DT,
    JOINT_NAMES,
    list_runs,
    load_csv,
    load_run,
)

XML = os.path.join(ROOT, "gear_sonic_deploy/g1/scene_29dof.xml")
ARMS = list(range(15, 29))
LEGS = list(range(0, 12))
GRAVITY = 9.81
SHOW = [15, 16, 18, 20, 0, 3, 4]   # a few arm joints then a few leg joints


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.stack([np.convolve(x[:, j], k, mode="same")
                     for j in range(x.shape[1])], axis=1)


def quat_rotate(q, v):
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def build():
    m = mujoco.MjModel.from_xml_path(XML)
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if nm and "floor" in nm.lower():
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
    return m


def collect(model, runs, win, use_base_acc=True):
    data = mujoco.MjData(model)
    Mb = np.zeros((model.nv, model.nv))
    out = {k: [] for k in ("tau_real", "tau_model", "grav", "inert",
                           "bias", "dq", "ddq", "q")}
    for r in runs:
        d = load_run(r)
        q, dq, tau = d["q"], d["dq"], d["tau"]
        if any(np.isnan(v).any() for v in (q, dq, tau)):
            continue
        mv = np.abs(dq).max(axis=1) > 0.05
        if mv.sum() < 300:
            continue
        lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
        q, dq, tau = q[lo:hi], dq[lo:hi], tau[lo:hi]
        try:
            bq, _ = load_csv(os.path.join(r, "base_quat.csv"), "base_q")
            ba, _ = load_csv(os.path.join(r, "base_accel.csv"), "base_a")
            bw, _ = load_csv(os.path.join(r, "base_ang_vel.csv"), "base_w")
            bq, ba, bw = bq[lo:hi], ba[lo:hi], bw[lo:hi]
        except Exception:  # noqa: BLE001
            continue

        ddq = np.gradient(smooth(dq, win), CONTROL_DT, axis=0)
        dbw = np.gradient(smooth(bw, win), CONTROL_DT, axis=0)

        idx = np.linspace(1, len(q) - 2, min(2500, len(q) - 2)).astype(int)
        for t in idx:
            data.qpos[0:3] = [0, 0, 0.8]
            data.qpos[3:7] = bq[t]
            data.qpos[7:] = q[t]
            data.qvel[0:3] = 0.0
            data.qvel[3:6] = bw[t]
            data.qvel[6:] = dq[t]
            mujoco.mj_forward(model, data)
            bias = data.qfrc_bias.copy()          # C + G

            # gravity-only: same pose, zero velocity => bias reduces to G(q)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            grav = data.qfrc_bias.copy()          # G

            data.qvel[3:6] = bw[t]
            data.qvel[6:] = dq[t]
            mujoco.mj_forward(model, data)

            qacc = np.zeros(model.nv)
            if use_base_acc:
                qacc[0:3] = quat_rotate(bq[t], ba[t]) + np.array([0, 0, -GRAVITY])
                qacc[3:6] = dbw[t]
            qacc[6:] = ddq[t]
            mujoco.mj_fullM(model, Mb, data.qM)
            inert = Mb @ qacc

            out["tau_real"].append(tau[t])
            out["tau_model"].append((inert + bias)[6:])
            out["grav"].append(grav[6:])
            out["inert"].append(inert[6:])
            out["bias"].append(bias[6:])
            out["dq"].append(dq[t])
            out["ddq"].append(ddq[t])
            out["q"].append(q[t])
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=6)
    ap.add_argument("--smooth", type=int, default=5)
    a = ap.parse_args()

    model = build()
    runs = list_runs()[: a.max_runs]
    D = collect(model, runs, a.smooth)
    tr, tm = D["tau_real"], D["tau_model"]
    print(f"runs {len(runs)}   steps {len(tr)}   smoothing window {a.smooth}\n")

    # ---------------------------------------------------------------- TEST 2
    print("=" * 84)
    print("TEST 2  what the model torque is MADE OF  [Nm, RMS]")
    print("=" * 84)
    print(f"{'joint':22s}{'tau_real':>10}{'gravity':>10}{'inertial':>10}"
          f"{'Coriolis':>10}{'tau_model':>11}{'residual':>10}")
    print("-" * 84)
    cor = D["bias"] - D["grav"]
    for j in SHOW:
        rms = lambda x: float(np.sqrt((x[:, j] ** 2).mean()))  # noqa: E731
        print(f"{JOINT_NAMES[j]:22s}{rms(tr):10.2f}{rms(D['grav']):10.2f}"
              f"{rms(D['inert']):10.2f}{rms(cor):10.2f}{rms(tm):11.2f}"
              f"{float(np.sqrt(((tr-tm)[:, j]**2).mean())):10.2f}")

    # ---------------------------------------------------------------- TEST 1
    print("\n" + "=" * 84)
    print("TEST 1  NEAR-STATIC frames: is the model's GRAVITY torque right?")
    print("=" * 84)
    slow = (np.abs(D["dq"]).max(axis=1) < 0.15) & (np.abs(D["ddq"]).max(axis=1) < 1.0)
    print(f"frames with |dq|<0.15 rad/s and |ddq|<1 rad/s^2 : "
          f"{slow.sum()} of {len(tr)} ({100*slow.mean():.1f}%)")
    if slow.sum() > 200:
        print(f"\n{'joint':22s}{'tau_real':>10}{'gravity':>10}{'diff':>9}"
              f"{'corr':>8}{'gain':>7}")
        print("-" * 60)
        for j in SHOW:
            x, y = D["grav"][slow, j], tr[slow, j]
            c = np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 else np.nan
            g = float(x @ y / (x @ x)) if (x @ x) > 1e-9 else np.nan
            print(f"{JOINT_NAMES[j]:22s}{np.sqrt((y**2).mean()):10.2f}"
                  f"{np.sqrt((x**2).mean()):10.2f}"
                  f"{np.sqrt(((y-x)**2).mean()):9.2f}{c:8.3f}{g:7.2f}")
        print("\n  high corr + gain~1  => mass distribution is right, gap is friction")
        print("  low corr            => the model's gravity torque itself is wrong")

    # ---------------------------------------------------------------- TEST 4
    print("\n" + "=" * 84)
    print("TEST 4  can the residual be explained as FRICTION?")
    print("=" * 84)
    print("  fitting  residual ~ a*sign(dq) + b*dq   (Coulomb + viscous)")
    print(f"\n{'joint':22s}{'Coulomb a':>11}{'viscous b':>11}{'R2':>8}"
          f"{'resid before':>14}{'after':>9}")
    print("-" * 78)
    res = tr - tm
    for j in SHOW:
        X = np.column_stack([np.sign(D["dq"][:, j]), D["dq"][:, j]])
        y = res[:, j]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        print(f"{JOINT_NAMES[j]:22s}{coef[0]:11.3f}{coef[1]:11.3f}{r2:8.3f}"
              f"{np.sqrt((y**2).mean()):14.2f}{np.sqrt(((y-pred)**2).mean()):9.2f}")
    print("\n  a = torque lost to Coulomb friction regardless of speed [Nm]")
    print("  R2 near 1 would mean friction explains the whole residual.")

    # ---------------------------------------------------------------- TEST 3
    print("\n" + "=" * 84)
    print("TEST 3  sensitivity to the ddq smoothing window (a free parameter)")
    print("=" * 84)
    print(f"{'window':>8}{'ddq RMS':>10}{'arm inert':>11}{'arm resid':>11}"
          f"{'leg resid':>11}")
    print("-" * 52)
    for w in (1, 3, 5, 9, 15):
        Dw = collect(model, runs[:2], w)
        rw, mw = Dw["tau_real"], Dw["tau_model"]
        print(f"{w:>8}{float(np.sqrt((Dw['ddq']**2).mean())):10.2f}"
              f"{float(np.sqrt((Dw['inert'][:, ARMS]**2).mean())):11.3f}"
              f"{float(np.sqrt(((rw-mw)[:, ARMS]**2).mean())):11.3f}"
              f"{float(np.sqrt(((rw-mw)[:, LEGS]**2).mean())):11.3f}")
    print("\n  if 'arm resid' barely moves, the conclusion is robust to this choice")


if __name__ == "__main__":
    main()
