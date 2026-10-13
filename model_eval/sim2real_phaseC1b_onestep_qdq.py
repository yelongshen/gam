#!/usr/bin/env python3
"""Phase C.1b — one-step sim-vs-real for BOTH q and dq, on the cleaned run set.

    same real state ──┬──> MuJoCo, one control step ──> q_sim(t+dt), dq_sim(t+dt)
      q_real(t)       │
      dq_real(t)      └──> the robot itself          ──> q_real(t+dt), dq_real(t+dt)
      q_target(t)

Teacher-forced: the sim is reset to the measured state every step, so errors
cannot compound and each step is judged independently. This isolates "is one
step of the dynamics right?" from "does error accumulate?" (the latter is
Phase C.2's job -- see `sim2real/phaseC2_discussion.md`).

WHAT THIS ADDS over `sim2real_phaseC1_onestep_prediction.py`:
  * reports **dq** error, not only q. At dt = 0.02 s a torque/inertia error
    shows up in dq an order of magnitude more clearly than in q, because q is
    the *integral* of the error and is dominated by the (exactly correct)
    initial condition.
  * uses `sim2real/deploy_constants.py::list_runs()`, which drops the 9
    known-bad runs documented in `sim2real/robot_log_data_quality.md`
    (constant waist torque, NaN actions, NUL-truncated CSVs, ...).
  * per-joint breakdown with the normalising scale stated explicitly.

READ THIS BEFORE INTERPRETING THE LEG NUMBERS
---------------------------------------------
The base is welded, exactly as the earlier Phase C.1 scripts do. The docstring
there argues this is harmless "since we overwrite qpos/qvel every step and only
integrate ONE dt". **That argument does not hold for joints in contact.**
Measured on this data, the effective inertia seen at a joint is:

    left_ankle_pitch    real 0.650 kg m^2   welded-base sim 0.013   -> 51x
    left_hip_pitch      real 1.739          welded-base sim 0.860   ->  2x
    left_elbow          real 0.116          welded-base sim 0.045   -> 2.6x

A stance ankle pushes against the ground and therefore against the robot's
whole mass; with the base welded and no contact, the same torque only has to
move the foot. So for the LEGS this test measures "how wrong is the model when
contact is missing", which is large and not very informative. ARM joints never
touch the ground and are the meaningful comparison here.

Fixing this properly needs base position + linear velocity + contact forces,
none of which are logged. See the "what would make this valid" note at the end
of the output.

Usage:
  .venv_sim/bin/python model_eval/sim2real_phaseC1b_onestep_qdq.py
  .venv_sim/bin/python model_eval/sim2real_phaseC1b_onestep_qdq.py --real-armature
  .venv_sim/bin/python model_eval/sim2real_phaseC1b_onestep_qdq.py --max-runs 6 --max-steps 1500
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
    ACTION_SCALE,
    CONTROL_DT,
    DEFAULT_ANGLES,
    EFFORT_LIMIT,
    ISAACLAB_TO_MUJOCO,
    JOINT_NAMES,
    KD,
    KP,
    MOTOR_TYPE,
    list_runs,
    load_run,
)

XML = os.path.join(ROOT, "gear_sonic_deploy/g1/scene_29dof.xml")

ARMS = list(range(15, 29))
LEGS = list(range(0, 12))
WAIST = [12, 13, 14]


def build_model(real_armature=False, extra_damping=None, weld_base=True):
    m = mujoco.MjModel.from_xml_path(XML)
    if weld_base:
        # No contact when the base is welded: the feet would be at an
        # arbitrary height relative to the floor.
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if nm and "floor" in nm.lower():
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
    if real_armature:
        from deploy_constants import (
            ARMATURE_4010,
            ARMATURE_5020,
            ARMATURE_7520_14,
            ARMATURE_7520_22,
        )
        A = {"5020": ARMATURE_5020, "7520_14": ARMATURE_7520_14,
             "7520_22": ARMATURE_7520_22, "4010": ARMATURE_4010}
        for i in range(m.nu):
            dof = m.jnt_dofadr[m.actuator_trnid[i, 0]]
            m.dof_armature[dof] = A[MOTOR_TYPE[i]]
    for j, b in (extra_damping or {}).items():
        dof = m.jnt_dofadr[m.actuator_trnid[j, 0]]
        m.dof_damping[dof] = b
    return m


def one_step(model, data, q, dq, q_target, n_sub, weld_base=True,
             base_quat=None, base_ang_vel=None):
    """Teacher-forced single control step. Returns (q_next, dq_next, tau)."""
    data.qpos[7:] = q
    data.qvel[6:] = dq
    if weld_base:
        data.qpos[0:3] = [0.0, 0.0, 1.0]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[0:6] = 0.0
    else:
        # Floating base WITH ground contact. Orientation comes from the IMU;
        # the height is set so the lowest geom of THIS pose rests on z = 0.
        # Base linear velocity is not logged and is left at zero -- a known
        # bias, but far smaller than deleting contact entirely.
        data.qpos[0:3] = [0.0, 0.0, 0.0]
        data.qpos[3:7] = base_quat if base_quat is not None else [1.0, 0, 0, 0]
        mujoco.mj_forward(model, data)
        z_low = min(data.geom_xpos[g, 2] for g in range(model.ngeom)
                    if model.geom_bodyid[g] != 0)
        data.qpos[2] = -z_low
        data.qvel[0:3] = 0.0
        data.qvel[3:6] = base_ang_vel if base_ang_vel is not None else 0.0
    mujoco.mj_forward(model, data)

    tau = KP * (q_target - q) - KD * dq
    tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)
    data.ctrl[:] = tau
    for _ in range(n_sub):
        if weld_base:
            data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)
    return data.qpos[7:].copy(), data.qvel[6:].copy(), tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-armature", action="store_true",
                    help="replace the XML's uniform 0.01 armature with the "
                         "per-motor-type values the deploy binary assumes")
    ap.add_argument("--max-runs", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--base", choices=["weld", "float"], default="weld",
                    help="'weld': base fixed, no contact (legs invalid). "
                         "'float': base free at the pose's standing height, "
                         "ground contact ON, IMU orientation, zero base "
                         "linear velocity (not logged).")
    a = ap.parse_args()

    weld = a.base == "weld"
    model = build_model(a.real_armature, weld_base=weld)
    data = mujoco.MjData(model)
    n_sub = int(round(CONTROL_DT / model.opt.timestep))

    runs = list_runs()[: a.max_runs]
    print(f"model      : {os.path.basename(XML)}")
    print(f"base       : {'welded, no contact' if weld else 'floating, contact ON'}")
    print(f"armature   : {'per-motor (real)' if a.real_armature else 'XML uniform 0.01'}")
    print(f"runs       : {len(runs)} (9 known-bad runs already excluded)")
    print(f"substeps   : {n_sub} x {model.opt.timestep} s = {CONTROL_DT} s\n")

    QE, DE, QR, DR, DDQR, DDQS = [], [], [], [], [], []
    for r in runs:
        d = load_run(r)
        q, dq, qt = d["q"], d["dq"], d["q_target"]
        if any(np.isnan(v).any() for v in (q, dq, qt)):
            continue
        mv = np.abs(dq).max(axis=1) > 0.05
        if mv.sum() < 200:
            continue
        lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
        q, dq, qt = q[lo:hi], dq[lo:hi], qt[lo:hi]

        bq = bw = None
        if not weld:
            try:
                from deploy_constants import load_csv
                bq, _ = load_csv(os.path.join(r, "base_quat.csv"), "base_q")
                bw, _ = load_csv(os.path.join(r, "base_ang_vel.csv"), "base_w")
                bq, bw = bq[lo:hi], bw[lo:hi]
            except Exception:  # noqa: BLE001
                continue

        T = len(q) - 1
        idx = np.arange(T) if T <= a.max_steps else \
            np.linspace(0, T - 1, a.max_steps).astype(int)
        for t in idx:
            mujoco.mj_resetData(model, data)
            qn, dqn, _ = one_step(model, data, q[t], dq[t], qt[t], n_sub,
                                  weld_base=weld,
                                  base_quat=None if weld else bq[t],
                                  base_ang_vel=None if weld else bw[t])
            QE.append(qn - q[t + 1])
            DE.append(dqn - dq[t + 1])
            QR.append(q[t + 1] - q[t])          # real per-step change in q
            DR.append(dq[t + 1] - dq[t])        # real per-step change in dq
            DDQR.append((dq[t + 1] - dq[t]) / CONTROL_DT)
            DDQS.append((dqn - dq[t]) / CONTROL_DT)

    QE, DE = np.asarray(QE), np.asarray(DE)
    QR, DR = np.asarray(QR), np.asarray(DR)
    DDQR, DDQS = np.asarray(DDQR), np.asarray(DDQS)
    print(f"steps      : {len(QE)}\n")

    def grp(name, js):
        q_rmse = float(np.sqrt((QE[:, js] ** 2).mean()))
        dq_rmse = float(np.sqrt((DE[:, js] ** 2).mean()))
        qs = np.sqrt((QR[:, js] ** 2).mean())
        ds = np.sqrt((DR[:, js] ** 2).mean())
        cr = np.nanmean([np.corrcoef(DDQR[:, j], DDQS[:, j])[0, 1]
                         for j in js if DDQR[:, j].std() > 1e-6])
        print(f"{name:20s} q RMSE {np.degrees(q_rmse):8.4f} deg   "
              f"dq RMSE {dq_rmse:7.4f} rad/s   "
              f"(real step: {np.degrees(qs):.4f} deg, {ds:.4f} rad/s)   "
              f"ddq corr {cr:6.3f}")

    print("=== ONE-STEP RMSE:  q_sim(t+dt) vs q_real(t+dt),  "
          "dq_sim(t+dt) vs dq_real(t+dt) ===")
    grp("arms (no contact)", ARMS)
    grp("legs (CONTACT!)", LEGS)
    grp("waist", WAIST)
    print(f"{'ALL 29 JOINTS':20s} q RMSE "
          f"{np.degrees(np.sqrt((QE**2).mean())):8.4f} deg   "
          f"dq RMSE {np.sqrt((DE**2).mean()):7.4f} rad/s")

    print(f"\n{'joint':22s}{'q RMSE[deg]':>13}{'dq RMSE[rad/s]':>16}"
          f"{'q step[deg]':>13}{'dq step[rad/s]':>16}{'ddq_corr':>10}")
    print("-" * 92)
    for j in range(29):
        q_rmse = np.sqrt((QE[:, j] ** 2).mean())
        dq_rmse = np.sqrt((DE[:, j] ** 2).mean())
        qs = np.sqrt((QR[:, j] ** 2).mean())
        ds = np.sqrt((DR[:, j] ** 2).mean())
        c = (np.corrcoef(DDQR[:, j], DDQS[:, j])[0, 1]
             if DDQR[:, j].std() > 1e-6 and DDQS[:, j].std() > 1e-6 else np.nan)
        tag = "  contact" if j in LEGS else ""
        print(f"{JOINT_NAMES[j]:22s}{np.degrees(q_rmse):13.4f}{dq_rmse:16.4f}"
              f"{np.degrees(qs):13.4f}{ds:16.4f}{c:10.3f}{tag}")

    print("\n'step' columns = the real robot's own change over one dt, i.e. the")
    print("scale the RMSE should be judged against.")
    print("\nCAVEAT  the base is welded and the floor is off, so the LEG rows")
    print("        include the missing ground reaction, not just model error.")
    print("        Arm rows are the clean sim-vs-real comparison.")


if __name__ == "__main__":
    main()
