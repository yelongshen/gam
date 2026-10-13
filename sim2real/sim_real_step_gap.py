#!/usr/bin/env python
"""Single-step sim-vs-real dynamics gap: (q_t, dq_t, a_t) -> (q_{t+1}, dq_{t+1}).

For every logged control step we re-create the measured state in MuJoCo, apply
the *same* torque the real PD loop applied, advance exactly one control period,
and compare against what the robot actually did.

    real:  (q_t, dq_t) --[robot]--> (q_{t+1},  dq_{t+1})      from the CSV logs
    sim:   (q_t, dq_t) --[mujoco]-> (q'_{t+1}, dq'_{t+1})     one control step

Torque is NOT taken from the model's own PD -- the MuJoCo actuators here are
direct-torque (gainprm=1, biasprm=0), so we inject
`tau = kp(q_target - q) - kd*dq` ourselves, exactly as the motor firmware does
(verified to corr 0.998 against `motor_torque.csv` by `verify_pd_law.py`).

WHY SINGLE STEP: a free-running rollout diverges within a few hundred ms and
the resulting error says nothing about *where* the model is wrong. Re-setting
the state every step isolates the one-step dynamics error, which is what a
system-identification loop actually needs.

THE BASE-STATE PROBLEM (read before trusting any number here):
  The logs contain IMU orientation and angular velocity but **no base position
  and no base linear velocity**. A floating-base robot's joint accelerations
  depend on both, plus on the contact forces they determine. So we cannot
  reproduce the real base state, and three regimes are reported separately:

    --base free    base at nominal height, zero linear velocity, ground ON.
                   Legs get *some* contact but with a wrong base state.
    --base pinned  base welded. No contact at all; pure articulated dynamics.
    --base air     base free, robot lifted clear of the ground, ground OFF.

  Arms never touch the ground, so their numbers are meaningful in all three.
  Stance-leg numbers are only interpretable in the sense of "how wrong does the
  model get when contact is mis-specified" -- which is itself worth measuring,
  but is not a clean dynamics comparison.

Usage:
  .venv_sim/bin/python sim2real/sim_real_step_gap.py --max-runs 6
  .venv_sim/bin/python sim2real/sim_real_step_gap.py --base pinned --max-steps 3000
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_constants import (  # noqa: E402
    CONTROL_DT,
    JOINT_NAMES,
    KD,
    KP,
    list_runs,
    load_run,
)

XML = "/home/grease/gam/gear_sonic_deploy/g1/scene_29dof.xml"

ARM_JOINTS = list(range(15, 29))          # shoulders -> wrists, never in contact
LEG_JOINTS = list(range(0, 12))           # hips -> ankles
WAIST_JOINTS = [12, 13, 14]
# waist roll/pitch are commanded far outside their +-0.52 rad limit by the
# `low_latency` policy (24% of frames); excluded from headline numbers.
WAIST_BAD = [13, 14]


def build_model(base_mode, use_real_armature):
    spec_path = XML
    m = mujoco.MjModel.from_xml_path(spec_path)

    if base_mode in ("pinned", "air"):
        # disable ground contact by removing the floor geom's contype/conaffinity
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and "floor" in name.lower():
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0

    if use_real_armature:
        # The shipped XML uses a uniform armature of 0.01, but the real
        # actuators differ by motor type (0.0036 / 0.0102 / 0.0251).
        # verify_pd_law.py's constants are the authoritative ones.
        from deploy_constants import (
            ARMATURE_4010,
            ARMATURE_5020,
            ARMATURE_7520_14,
            ARMATURE_7520_22,
            MOTOR_TYPE,
        )
        A = {"5020": ARMATURE_5020, "7520_14": ARMATURE_7520_14,
             "7520_22": ARMATURE_7520_22, "4010": ARMATURE_4010}
        for i in range(m.nu):
            dof = m.jnt_dofadr[m.actuator_trnid[i, 0]]
            m.dof_armature[dof] = A[MOTOR_TYPE[i]]
    return m


def step_once(m, d, q, dq, tau, base_quat, base_ang_vel, base_mode, n_sub):
    """Reset to the measured state, apply tau, advance one control period."""
    mujoco.mj_resetData(m, d)
    if base_mode == "pinned":
        d.qpos[0:3] = [0, 0, 1.0]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qvel[0:6] = 0
    else:
        d.qpos[0:3] = [0, 0, 1.0 if base_mode == "air" else 0.793]
        d.qpos[3:7] = base_quat
        d.qvel[0:3] = 0.0                 # base linear velocity is NOT logged
        d.qvel[3:6] = base_ang_vel
    d.qpos[7:] = q
    d.qvel[6:] = dq

    if base_mode == "pinned":
        # freeze the base by zeroing its velocity every substep
        for _ in range(n_sub):
            d.ctrl[:] = tau
            d.qvel[0:6] = 0
            mujoco.mj_step(m, d)
    else:
        d.ctrl[:] = tau
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

    return d.qpos[7:].copy(), d.qvel[6:].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["free", "pinned", "air"], default="pinned")
    ap.add_argument("--real-armature", action="store_true",
                    help="override the XML's uniform 0.01 armature with the "
                         "per-motor values used by the deploy binary")
    ap.add_argument("--max-runs", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=2000,
                    help="steps sampled per run")
    ap.add_argument("--root", default="/home/grease/g1_robot_data")
    a = ap.parse_args()

    m = build_model(a.base, a.real_armature)
    d = mujoco.MjData(m)
    n_sub = int(round(CONTROL_DT / m.opt.timestep))
    print(f"model {os.path.basename(XML)}  base={a.base}  "
          f"real_armature={a.real_armature}  substeps={n_sub}\n")

    runs = [r for r in list_runs(a.root)][: a.max_runs]

    dq_err_all, q_err_all, ddq_real_all, ddq_sim_all = [], [], [], []
    n_used = 0
    for r in runs:
        try:
            dat = load_run(r)
        except Exception:  # noqa: BLE001
            continue
        q, dq, qt, tau_meas = dat["q"], dat["dq"], dat["q_target"], dat["tau"]
        if any(np.isnan(v).any() for v in (q, dq, qt, tau_meas)):
            continue
        mv = np.abs(dq).max(axis=1) > 0.05
        if mv.sum() < 200:
            continue
        lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
        q, dq, qt = q[lo:hi], dq[lo:hi], qt[lo:hi]

        bq, _ = load_csv_safe(os.path.join(r, "base_quat.csv"), "quat_")
        bw, _ = load_csv_safe(os.path.join(r, "base_ang_vel.csv"), "ang_vel_")
        if bq is None or len(bq) < hi:
            bq = np.tile([1.0, 0, 0, 0], (hi, 1))
            bw = np.zeros((hi, 3))
        bq, bw = bq[lo:hi], bw[lo:hi]

        T = len(q) - 1
        idx = np.arange(T)
        if T > a.max_steps:
            idx = np.linspace(0, T - 1, a.max_steps).astype(int)

        for t in idx:
            tau = KP * (qt[t] - q[t]) - KD * dq[t]
            qn, dqn = step_once(m, d, q[t], dq[t], tau, bq[t], bw[t],
                                a.base, n_sub)
            q_err_all.append(qn - q[t + 1])
            dq_err_all.append(dqn - dq[t + 1])
            ddq_real_all.append((dq[t + 1] - dq[t]) / CONTROL_DT)
            ddq_sim_all.append((dqn - dq[t]) / CONTROL_DT)
        n_used += 1

    q_err = np.asarray(q_err_all)
    dq_err = np.asarray(dq_err_all)
    ddq_r = np.asarray(ddq_real_all)
    ddq_s = np.asarray(ddq_sim_all)
    print(f"runs used {n_used}, steps {len(q_err)}\n")

    def block(name, js):
        qe = np.degrees(q_err[:, js])
        de = dq_err[:, js]
        cr = [np.corrcoef(ddq_r[:, j], ddq_s[:, j])[0, 1]
              for j in js if ddq_r[:, j].std() > 1e-6]
        print(f"{name:16s} |dq err| RMS {np.sqrt((de**2).mean()):7.3f} rad/s   "
              f"|q err| RMS {np.sqrt((qe**2).mean()):7.4f} deg   "
              f"ddq corr {np.mean(cr):6.3f}")

    print("=== one-step error, grouped ===")
    block("arms (no contact)", ARM_JOINTS)
    block("legs", LEG_JOINTS)
    block("waist_yaw", [12])
    block("waist roll/pitch", WAIST_BAD)

    print(f"\n{'joint':24s}{'ddq_real':>10}{'ddq_sim':>10}{'ratio':>8}"
          f"{'ddq_corr':>10}{'dq_err':>9}{'dq_step':>9}{'rel%':>8}")
    print("-" * 88)
    for j in range(29):
        de = dq_err[:, j]
        dr = ddq_r[:, j]
        ds = ddq_s[:, j]
        c = np.corrcoef(dr, ds)[0, 1] if dr.std() > 1e-6 and ds.std() > 1e-6 else np.nan
        r_rms = float(np.sqrt((dr**2).mean()))
        s_rms = float(np.sqrt((ds**2).mean()))
        # the real per-step change in dq is the scale the error should be
        # judged against -- not dq itself.
        step_rms = r_rms * CONTROL_DT
        rel = 100.0 * float(np.sqrt((de**2).mean())) / max(step_rms, 1e-9)
        mark = "  <- over-limit cmd" if j in WAIST_BAD else ""
        print(f"{JOINT_NAMES[j]:24s}{r_rms:10.2f}{s_rms:10.2f}"
              f"{s_rms/max(r_rms,1e-9):8.2f}{c:10.3f}"
              f"{np.sqrt((de**2).mean()):9.3f}{step_rms:9.4f}{rel:8.0f}{mark}")


def load_csv_safe(path, prefix):
    try:
        from deploy_constants import load_csv
        return load_csv(path, prefix)
    except Exception:  # noqa: BLE001
        return None, None


if __name__ == "__main__":
    main()
