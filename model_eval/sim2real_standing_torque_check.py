#!/usr/bin/env python3
"""Does the standing-torque discrepancy disappear once contact is modelled?

The inverse-dynamics analysis (`sim2real_inverse_dynamics_gap.py`) reported
that the model needs only ~0.16 Nm at the ankle while the robot actually used
~5-9 Nm, with a NEGATIVE correlation on the leg joints. That looks like a
broken model, which would be alarming -- the policy keeps the robot standing,
so the physics clearly works.

It is not a broken model. It is a broken ANALYSIS SETUP: that script disables
floor contact and places the base at an arbitrary 0.8 m, so the model describes
a robot FLOATING IN THE AIR. A floating robot's ankle only has to hold up its
own foot. A standing robot's ankle has to hold up the whole machine through the
contact chain.

This script tests that explanation directly: stand the model on the floor in
the pose the robot actually held, let it settle under the same PD law, and read
off the actuator torques. If the explanation is right, the leg torques should
jump from ~0.2 Nm to the several-Nm range the robot really used.

Note this is also why the policy is unaffected by any of it: the deploy loop
never evaluates a dynamics model. The policy emits joint position targets and
the motor-level PD loop tracks them; the ground supplies the reaction force.
Inverse dynamics is only ever used here, offline, by us.

Usage:
  .venv_sim/bin/python model_eval/sim2real_standing_torque_check.py
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim2real"))

from deploy_constants import (  # noqa: E402
    EFFORT_LIMIT,
    JOINT_NAMES,
    KD,
    KP,
    list_runs,
    load_run,
)

XML = os.path.join(ROOT, "gear_sonic_deploy/g1/scene_29dof.xml")
LEGS = list(range(0, 12))
SHOW = [0, 3, 4, 5, 9, 10, 15, 18]


def measured_standing_pose():
    """Mean pose + mean |tau| over near-static frames of real standing data."""
    Q, T = [], []
    for r in list_runs()[:6]:
        d = load_run(r)
        q, dq, tau = d["q"], d["dq"], d["tau"]
        if any(np.isnan(v).any() for v in (q, dq, tau)):
            continue
        still = np.abs(dq).max(axis=1) < 0.10
        if still.sum() < 100:
            continue
        Q.append(q[still])
        T.append(tau[still])
    Q = np.concatenate(Q)
    T = np.concatenate(T)
    return Q.mean(axis=0), np.abs(T).mean(axis=0), len(Q)


def run_sim(q_hold, with_contact, settle_s=3.0):
    m = mujoco.MjModel.from_xml_path(XML)
    if not with_contact:
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if nm and "floor" in nm.lower():
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
    d = mujoco.MjData(m)

    # Place the base so the lowest body geom rests exactly on z = 0 for THIS
    # pose. Using a fixed 0.80 m instead (as the first version did) leaves the
    # feet 6 mm in the air, and the resulting drop makes the robot collapse to
    # a 0.089 m heap -- which then reports near-zero leg torque and looks like
    # a modelling failure rather than a setup error.
    d.qpos[0:3] = [0.0, 0.0, 0.0]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:] = q_hold
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    z_low = min(d.geom_xpos[g, 2] for g in range(m.ngeom) if m.geom_bodyid[g] != 0)
    d.qpos[2] = -z_low
    mujoco.mj_forward(m, d)

    n = int(settle_s / m.opt.timestep)
    tau_hist = []
    for _ in range(n):
        tau = KP * (q_hold - d.qpos[7:]) - KD * d.qvel[6:]
        tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        tau_hist.append(tau.copy())
    tau_hist = np.asarray(tau_hist)
    # average over the last 0.5 s, by which point it has settled
    k = int(0.5 / m.opt.timestep)
    return (np.abs(tau_hist[-k:]).mean(axis=0),
            float(d.qpos[2]),
            float(np.abs(d.qvel[6:]).max()))


def main():
    q_hold, tau_real, n = measured_standing_pose()
    print(f"real standing pose averaged over {n} near-static frames "
          f"(|dq| < 0.10 rad/s)\n")

    tau_air, z_air, v_air = run_sim(q_hold, with_contact=False)
    tau_gnd, z_gnd, v_gnd = run_sim(q_hold, with_contact=True)

    print(f"sim, contact OFF : final base height {z_air:6.3f} m "
          f"(free fall -- meaningless, shown for contrast)")
    print(f"sim, contact ON  : final base height {z_gnd:6.3f} m, "
          f"max |dq| {v_gnd:.3f} rad/s "
          f"({'settled / standing' if v_gnd < 0.2 else 'NOT settled'})\n")

    print("=" * 76)
    print("HOLDING TORQUE WHILE STANDING  [Nm, mean |tau|]")
    print("=" * 76)
    print(f"{'joint':22s}{'real robot':>12}{'sim contact ON':>16}"
          f"{'sim contact OFF':>17}")
    print("-" * 76)
    for j in SHOW:
        print(f"{JOINT_NAMES[j]:22s}{tau_real[j]:12.2f}{tau_gnd[j]:16.2f}"
              f"{tau_air[j]:17.2f}")

    print("-" * 76)
    for nm, js in (("legs", LEGS), ("all joints", list(range(29)))):
        print(f"{nm:22s}{tau_real[js].mean():12.2f}{tau_gnd[js].mean():16.2f}"
              f"{tau_air[js].mean():17.2f}")

    rg = np.corrcoef(tau_real[LEGS], tau_gnd[LEGS])[0, 1]
    ra = np.corrcoef(tau_real[LEGS], tau_air[LEGS])[0, 1]
    print(f"\ncorrelation with the real leg torques:")
    print(f"   contact ON  : {rg:+.3f}")
    print(f"   contact OFF : {ra:+.3f}")

    print("\nCONCLUSION")
    if rg > 0.8 and rg > ra + 0.3:
        print("  Modelling contact reproduces the real standing torques. The")
        print("  earlier 'model is 40x too small / negatively correlated' result")
        print("  was an artifact of the contact-free analysis setup, not evidence")
        print("  of a wrong model.")
    else:
        print("  Contact alone does NOT reconcile the two. Something else is")
        print("  also wrong -- do not treat the leg model as validated.")
    print("\n  Either way the policy is unaffected: the deploy loop contains no")
    print("  dynamics model at all, only the motor-level PD law.")


if __name__ == "__main__":
    main()
