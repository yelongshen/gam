"""
Phase C.1 -- frictionloss (Coulomb friction) scan, on top of the Phase B
dof_damping calibration.

WHY THIS EXISTS
---------------
The damping scan (sim2real_phaseC1_damping_scan.py) showed that a linear
viscous term (`dof_damping`, contributing `qfrc_passive = -b*qvel`) only
explains ~10-13% of the variance in `residual = tau_real - tau_cmd` on the
ankle-pitch joints (correlation with `dq` only -0.34 to -0.42), and ~2% on
waist_pitch. That means no choice of `b` can close most of the gap -- the
remaining ~87-90% needs a DIFFERENT functional form, most plausibly Coulomb
(dry) friction: a constant-magnitude force that opposes the SIGN of velocity,
independent of its magnitude. MuJoCo models this via `dof_frictionloss`.

HOW dof_frictionloss SHOWS UP IN MuJoCo (empirically verified, NOT qfrc_passive)
---------------------------------------------------------------------------------
Unlike `dof_damping` (a simple analytic term in `qfrc_passive`), MuJoCo's
Coulomb friction is solved via the CONSTRAINT solver and shows up in
`data.qfrc_constraint`, not `qfrc_passive`:

    >>> m.dof_frictionloss[dofadr] = 2.0
    >>> qfrc_passive[dofadr]    == 0.0        # untouched
    >>> qfrc_constraint[dofadr] != 0.0        # friction lives here

COMPLICATION: this model's ankle joints already have a nonzero
`qfrc_constraint` at frictionloss=0 (from the `<equality>` ankle-linkage
constraint in the XML, noted in `sim2real/phaseB_actuator.md` section 3b/3.3),
so the RAW `qfrc_constraint` value cannot be used directly -- it conflates
frictionloss with whatever else is already constrained at that DOF. This
script isolates the frictionloss-only contribution via a DIFFERENCE:

    friction_force[t] = qfrc_constraint(model_with_frictionloss, state[t])
                       - qfrc_constraint(model_frictionloss=0,   state[t])

(both models share the same `dof_damping`, so this delta is exactly the
incremental effect of adding Coulomb friction, independent of the ankle
linkage's baseline constraint force.)

FINAL SIM-SIDE TOTAL
--------------------
    tau_sim_total = tau_cmd + qfrc_passive(damping) + friction_force

USAGE
-----
  .venv_sim/bin/python model_eval/sim2real_phaseC1_friction_scan.py \\
      --joints 4 10 14 --lo 0.0 --hi 3.0 --steps 7 --duration 439.6
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_process'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_sim2real_session import load_session          # noqa: E402
from g1_params import action_to_q_target, KPS, KDS, JOINT_NAMES  # noqa: E402
from sim2real_phaseC1_onestep_prediction import (        # noqa: E402
    build_model, FS, MJ_JOINT_NAMES, MJ_ACTUATOR_NAMES, EFFORT_LIMIT,
)


def build_model_with_friction(extra_damping, extra_friction):
    """Same as build_model(), plus dof_frictionloss overrides."""
    m = build_model(extra_damping)
    for j, f in extra_friction.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, MJ_JOINT_NAMES[j])
        m.dof_frictionloss[m.jnt_dofadr[jid]] = f
    return m


def tau_with_friction(model_fric, model_ref, q_real, dq_real, q_target):
    """For every t, compute tau_sim_total including the isolated frictionloss
    contribution (model_fric vs model_ref, same dof_damping, differing only
    in dof_frictionloss). Returns tau_sim_total (T,29)."""
    qadr = np.array([model_fric.jnt_qposadr[mujoco.mj_name2id(model_fric, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in MJ_JOINT_NAMES])
    dofadr = np.array([model_fric.jnt_dofadr[mujoco.mj_name2id(model_fric, mujoco.mjtObj.mjOBJ_JOINT, n)]
                        for n in MJ_JOINT_NAMES])

    d_fric = mujoco.MjData(model_fric)
    d_ref = mujoco.MjData(model_ref)

    T = len(q_real)
    tau_cmd = np.zeros((T, 29))
    tau_sim_total = np.zeros((T, 29))

    for t in range(T):
        d_fric.qpos[qadr] = q_real[t]
        d_fric.qvel[dofadr] = dq_real[t]
        mujoco.mj_forward(model_fric, d_fric)
        passive = d_fric.qfrc_passive[dofadr].copy()
        constraint_fric = d_fric.qfrc_constraint[dofadr].copy()

        d_ref.qpos[qadr] = q_real[t]
        d_ref.qvel[dofadr] = dq_real[t]
        mujoco.mj_forward(model_ref, d_ref)
        constraint_ref = d_ref.qfrc_constraint[dofadr].copy()

        friction_force = constraint_fric - constraint_ref

        tau = KPS * (q_target[t] - q_real[t]) - KDS * dq_real[t]
        tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)

        tau_cmd[t] = tau
        tau_sim_total[t] = tau + passive + friction_force

    return tau_sim_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='aug11')
    ap.add_argument('--duration', type=float, default=439.6)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--joints', type=int, nargs='+', default=[4, 10, 14])
    ap.add_argument('--lo', type=float, default=0.0)
    ap.add_argument('--hi', type=float, default=3.0)
    ap.add_argument('--steps', type=int, default=7)
    ap.add_argument('--damping', type=str, default='4:1.19,10:1.09,14:0.98',
                     help='fixed dof_damping to hold while scanning frictionloss')
    args = ap.parse_args()

    damping = {}
    for pair in args.damping.split(','):
        j, b = pair.split(':')
        damping[int(j)] = float(b)

    S = load_session(args.session)
    r = S['robot']
    lo, hi = S['overlap']
    grid = np.arange(lo + args.start, min(hi, lo + args.start + args.duration), 1 / FS)
    print(f"=== Phase C.1 frictionloss scan - session {args.session} ===")
    print(f"testing {len(grid)/FS:.1f}s ({len(grid)} steps), fixed damping = {damping}\n")

    def resample(t, v):
        return np.stack([np.interp(grid, t, v[:, j]) for j in range(v.shape[1])], 1)

    q_real = resample(r['t'], r['q'])
    dq_real = resample(r['t'], r['dq'])
    tau_real = resample(r['t'], r['motor_torque'])
    qt_real = resample(r['t'], action_to_q_target(r['action']))

    f_values = np.linspace(args.lo, args.hi, args.steps)

    # reference: friction=0 everywhere, damping fixed as given
    model_ref = build_model_with_friction(damping, {})

    for j in args.joints:
        jname = JOINT_NAMES[j]
        print(f"--- scanning frictionloss on joint {j} ({jname}) ---")
        print(f"{'frictionloss':>14s} {'RMS(N*m)':>10s}")
        best_f, best_rms = None, np.inf
        for f in f_values:
            model_fric = build_model_with_friction(damping, {j: float(f)})
            tau_sim_total = tau_with_friction(model_fric, model_ref, q_real, dq_real, qt_real)
            err2 = (tau_sim_total[:, j] - tau_real[:, j]) ** 2
            n_bad = int(np.isnan(err2).sum())
            rms = np.sqrt(np.nanmean(err2))
            flag = f'  ({n_bad} nan steps dropped)' if n_bad else ''
            if rms < best_rms:
                best_rms, best_f = rms, f
                flag += '  <-- best so far'
            print(f"{f:14.3f} {rms:10.4f}{flag}")
        print(f"==> best frictionloss for {jname}: {best_f:.3f} (RMS={best_rms:.4f})\n")


if __name__ == '__main__':
    main()
