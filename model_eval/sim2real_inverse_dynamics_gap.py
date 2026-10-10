#!/usr/bin/env python3
"""Sim-vs-real in PHYSICAL UNITS: inverse-dynamics torque residual.

WHY THIS INSTEAD OF THE ONE-STEP (q, dq) TEST
----------------------------------------------
The one-step forward test (`sim2real_phaseC1b_onestep_qdq.py`) needs the base
position and linear velocity to place the robot and resolve ground contact.
Neither is logged (`body_pos.csv` exists but is all-zero in 51/51 files -- it
is the streamer's reference, not robot state). With the base welded, the
effective inertia at a stance ankle is ~51x too small, so the leg numbers
measure the missing contact rather than the dynamics.

Inverse dynamics avoids that. It asks a question that needs only what we have:

    "Given the motion the robot ACTUALLY performed, how much torque does the
     model say that motion required -- and how much did the robot actually
     use?"

        tau_model = InverseDynamics(q, dq, ddq, a_base)      [Nm]
        tau_real  = motor_torque.csv (tau_est)               [Nm]
        residual  = tau_real - tau_model                     [Nm]

    q, dq        measured
    ddq          finite-differenced from dq (filtered)
    a_base       from the IMU: base_accel (specific force, includes gravity)
                 + d/dt base_ang_vel
    position / linear velocity   NOT NEEDED

Everything is reported in Nm, deg, deg/s and kg*m^2 -- no percentages of
percentages.

WHAT THE RESIDUAL CONTAINS
--------------------------
    residual = (unmodelled friction + gearing loss)
             + (contact forces, for legs only)
             + (model mass/inertia error)
             + (ddq differentiation noise)

Arms are never in contact, so for them the residual is friction + model error.
Legs carry the contact term as well, and the ARM-vs-LEG difference is itself
the measurement of how much contact matters.

Usage:
  .venv_sim/bin/python model_eval/sim2real_inverse_dynamics_gap.py
  .venv_sim/bin/python model_eval/sim2real_inverse_dynamics_gap.py --no-base-accel
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
    EFFORT_LIMIT,
    JOINT_NAMES,
    KD,
    KP,
    list_runs,
    load_csv,
    load_run,
)

XML = os.path.join(ROOT, "gear_sonic_deploy/g1/scene_29dof.xml")
ARMS = list(range(15, 29))
LEGS = list(range(0, 12))
WAIST = [12, 13, 14]
GRAVITY = 9.81


def smooth(x, w):
    """Zero-phase moving average along axis 0 (differentiation noise control)."""
    if w <= 1:
        return x
    k = np.ones(w) / w
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(x[:, j], k, mode="same")
    return out


def quat_rotate(q, v):
    """Rotate v from body into world frame. q = (w, x, y, z)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_rotate_inv(q, v):
    """Rotate v from world into body frame. q = (w, x, y, z)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--smooth", type=int, default=5,
                    help="moving-average window for dq before differentiating")
    ap.add_argument("--no-base-accel", action="store_true",
                    help="set base acceleration to zero instead of using the IMU "
                         "(sensitivity check)")
    a = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(XML)
    # Disable floor contact. The base position is unknown, so any height we
    # pick puts the feet either floating or buried in the floor; a buried foot
    # generates enormous contact forces and `qfrc_inverse` comes back in the
    # hundreds of Nm (the first version of this script did exactly that).
    # Contact is precisely the term we want to MEASURE as the residual, so it
    # must not be in the model side of the comparison.
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if nm and "floor" in nm.lower():
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0
    data = mujoco.MjData(model)
    M_buf = np.zeros((model.nv, model.nv))

    TAU_R, TAU_M, DDQ, DQ, Q = [], [], [], [], []
    runs = list_runs()[: a.max_runs]
    for r in runs:
        d = load_run(r)
        q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
        if any(np.isnan(v).any() for v in (q, dq, qt, tau)):
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

        dqs = smooth(dq, a.smooth)
        ddq = np.gradient(dqs, CONTROL_DT, axis=0)
        dbw = np.gradient(smooth(bw, a.smooth), CONTROL_DT, axis=0)

        T = len(q)
        idx = np.arange(1, T - 1)
        if len(idx) > a.max_steps:
            idx = np.linspace(1, T - 2, a.max_steps).astype(int)

        for t in idx:
            data.qpos[0:3] = [0, 0, 0.8]
            data.qpos[3:7] = bq[t]
            data.qpos[7:] = q[t]
            data.qvel[0:3] = 0.0
            data.qvel[3:6] = bw[t]
            data.qvel[6:] = dq[t]

            mujoco.mj_forward(model, data)

            if a.no_base_accel:
                qacc = np.zeros(model.nv)
            else:
                qacc = np.zeros(model.nv)
                # FRAME CONVENTIONS:
                #   IMU `base_accel` is SPECIFIC FORCE in the BODY frame:
                #       f = R^T (a_world - g_world),  g_world = (0, 0, -9.81)
                #   so   a_world = R f + g_world.
                #   MuJoCo free joint: qvel/qacc[0:3] are LINEAR in the WORLD
                #   frame, qvel/qacc[3:6] are ANGULAR in the BODY frame.
                # Sanity check: at rest f = (0, 0, +9.81) and a_world = 0.
                qacc[0:3] = quat_rotate(bq[t], ba[t]) + np.array([0.0, 0.0, -GRAVITY])
                qacc[3:6] = dbw[t]
            qacc[6:] = ddq[t]

            # Build the inverse dynamics explicitly rather than calling
            # `mj_inverse`. Its handling of a caller-supplied qacc was not
            # reproducible here (writing qacc produced no change in
            # qfrc_inverse, and a finite-difference probe disagreed with the
            # mass matrix by a constant), so the terms are assembled directly:
            #     tau = M(q) qddot + c(q, qdot)
            # where `qfrc_bias` is MuJoCo's Coriolis + centrifugal + gravity.
            mujoco.mj_fullM(model, M_buf, data.qM)
            tau_model = M_buf @ qacc + data.qfrc_bias
            TAU_M.append(tau_model[6:].copy())
            TAU_R.append(tau[t])
            DDQ.append(ddq[t])
            DQ.append(dq[t])
            Q.append(q[t])

    TAU_R = np.asarray(TAU_R)
    TAU_M = np.asarray(TAU_M)
    DDQ = np.asarray(DDQ)
    res = TAU_R - TAU_M

    print(f"runs {len(runs)}   steps {len(TAU_R)}   "
          f"({len(TAU_R)*CONTROL_DT:.0f} s of robot time)")
    print(f"base acceleration: {'ZERO (ablation)' if a.no_base_accel else 'from IMU'}")
    print(f"ddq: central difference of dq smoothed over {a.smooth} frames\n")

    print("=" * 78)
    print("TORQUE THE MODEL SAYS THE MOTION NEEDED, vs WHAT THE ROBOT USED  [Nm]")
    print("=" * 78)
    print(f"{'joint':22s}{'real RMS':>10}{'model RMS':>11}{'residual':>10}"
          f"{'resid/real':>11}{'corr':>8}{'gain':>7}")
    print("-" * 78)
    rows = []
    for j in range(29):
        rr = float(np.sqrt((TAU_R[:, j] ** 2).mean()))
        mm = float(np.sqrt((TAU_M[:, j] ** 2).mean()))
        rs = float(np.sqrt((res[:, j] ** 2).mean()))
        c = (np.corrcoef(TAU_R[:, j], TAU_M[:, j])[0, 1]
             if TAU_R[:, j].std() > 1e-9 and TAU_M[:, j].std() > 1e-9 else np.nan)
        g = (float(TAU_M[:, j] @ TAU_R[:, j] / (TAU_M[:, j] @ TAU_M[:, j]))
             if (TAU_M[:, j] @ TAU_M[:, j]) > 1e-9 else np.nan)
        rows.append((j, rr, mm, rs, c, g))
        print(f"{JOINT_NAMES[j]:22s}{rr:10.2f}{mm:11.2f}{rs:10.2f}"
              f"{rs/max(rr,1e-9):11.2f}{c:8.3f}{g:7.2f}")

    def grp(name, js):
        rr = float(np.sqrt((TAU_R[:, js] ** 2).mean()))
        rs = float(np.sqrt((res[:, js] ** 2).mean()))
        print(f"  {name:22s} real {rr:6.2f} Nm   residual {rs:6.2f} Nm   "
              f"({rs/max(rr,1e-9):.2f}x)")

    print("\n--- grouped ---")
    grp("arms (never contact)", ARMS)
    grp("legs (contact)", LEGS)
    grp("waist", WAIST)

    print("\n" + "=" * 78)
    print("EFFECTIVE INERTIA SEEN AT EACH JOINT  [kg m^2]")
    print("=" * 78)
    print("  real  = slope of tau_est vs ddq (what the motor had to push)")
    print("  model = diagonal of the mass matrix in the nominal pose")
    print(f"\n{'joint':22s}{'real':>10}{'model':>10}{'real/model':>12}")
    print("-" * 56)
    mujoco.mj_resetData(model, data)
    data.qpos[7:] = np.mean(Q, axis=0)
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    for j in range(29):
        dd = DDQ[:, j]
        i_real = float(dd @ TAU_R[:, j] / (dd @ dd)) if (dd @ dd) > 1e-9 else np.nan
        i_sim = float(M[6 + j, 6 + j])
        tag = "  <- contact" if j in LEGS else ""
        print(f"{JOINT_NAMES[j]:22s}{i_real:10.3f}{i_sim:10.3f}"
              f"{i_real/max(i_sim,1e-9):12.1f}{tag}")

    print("\nHOW TO READ THIS")
    print("  residual [Nm]      torque the model cannot account for.")
    print("  gain ~1.0          model magnitude is right.")
    print("  real/model inertia >> 1 means the joint was pushing against")
    print("                     something the model does not know about --")
    print("                     for the legs that is the ground.")


if __name__ == "__main__":
    main()
