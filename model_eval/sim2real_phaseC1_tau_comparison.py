"""
Phase C.1 (tau variant) -- sim-side tau_estimate vs. tau_measured, instead of the
q(t+dt) position-prediction metric.

WHY THIS DIFFERS FROM THE q1step METRIC
----------------------------------------
The original `one_step_predict()` in sim2real_phaseC1_onestep_prediction.py
computes `tau_sim[t]` as just the *commanded* PD torque:

    tau_cmd = Kp*(q_target - q_real) - Kd*dq_real

This is the actuator's `ctrl` signal (MuJoCo <motor> = direct torque
actuator), and Phase B already showed this matches real `tau_est` almost
tautologically (R^2=0.999) for most joints -- it's not new information, and
adding extra `dof_damping` (the C.1 calibration) does NOT change `tau_cmd` at
all, since the PD law is identical regardless of damping.

What DOES change with `dof_damping` is MuJoCo's `qfrc_passive`: the passive
generalized force contributed by joint damping/friction (`qfrc_passive[j] =
-dof_damping[j] * qvel[j]`, plus any `frictionloss` term). This is the
missing piece that makes `tau_sim_total = tau_cmd + qfrc_passive` a fairer
analogue of the *real* `motor_torque` (`tau_est`), which is a HARDWARE
measurement that already includes whatever real friction/back-EMF the motor
experiences -- friction the sim's actuator command alone cannot capture.

WHAT THIS SCRIPT COMPUTES
-------------------------
For each candidate damping config (baseline / Phase B / best-scan):
  tau_cmd[t]        = clip(Kp*(q_target[t]-q_real[t]) - Kd*dq_real[t])   (actuator command only)
  qfrc_passive[t]    = MuJoCo's passive force at (q_real[t], dq_real[t]) under this model's damping
  tau_sim_total[t]  = tau_cmd[t] + qfrc_passive[t]                       (total generalized force sim implies)

then compares BOTH `tau_cmd` (old-style, damping-insensitive) and
`tau_sim_total` (damping-sensitive) against the real `motor_torque` log via
RMS error, correlation, and linear-fit gain, per joint.

Expectation: if the Phase B damping calibration is correct, `tau_sim_total`
should track real `tau_est` measurably better than plain `tau_cmd` on the
calibrated joints (ankle pitch, waist pitch), closing part of the gain > 1
anomaly documented in `sim2real/phaseB_actuator.md` section 3b.

USAGE
-----
  .venv_sim/bin/python model_eval/sim2real_phaseC1_tau_comparison.py \\
      --duration 439.6
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
    build_model, FS, MJ_JOINT_NAMES, MJ_ACTUATOR_NAMES, TEST_JOINTS, EFFORT_LIMIT,
)


def tau_estimate_vs_measured(model, q_real, dq_real, q_target, dt):
    """For every t, set sim state to (q_real[t], dq_real[t]) [teacher forcing,
    same as one_step_predict], but instead of stepping forward and reading
    q(t+dt), read the INSTANTANEOUS generalized forces implied by this exact
    state under this model's damping/friction settings:

      tau_cmd[t]       = clipped PD command (damping-insensitive)
      qfrc_passive[t]  = passive force from dof_damping/frictionloss at this state
      tau_sim_total[t] = tau_cmd[t] + qfrc_passive[t]

    Returns (tau_cmd, tau_sim_total), both (T,29)."""
    data = mujoco.MjData(model)
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in MJ_JOINT_NAMES])
    dofadr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                        for n in MJ_JOINT_NAMES])
    actadr = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in MJ_ACTUATOR_NAMES])

    T = len(q_real)
    tau_cmd = np.zeros((T, 29))
    tau_sim_total = np.zeros((T, 29))

    for t in range(T):
        data.qpos[qadr] = q_real[t]
        data.qvel[dofadr] = dq_real[t]
        mujoco.mj_forward(model, data)   # computes qfrc_passive from CURRENT state/damping,
                                          # independent of ctrl
        passive = data.qfrc_passive[dofadr].copy()

        tau = KPS * (q_target[t] - q_real[t]) - KDS * dq_real[t]
        tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)

        tau_cmd[t] = tau
        tau_sim_total[t] = tau + passive

    return tau_cmd, tau_sim_total


def fit_stats(tau_sim, tau_real):
    """Per-joint RMS error, correlation, and linear-fit gain of tau_sim vs tau_real."""
    n = tau_sim.shape[1]
    rms = np.sqrt(((tau_sim - tau_real) ** 2).mean(0))
    corr = np.zeros(n)
    gain = np.zeros(n)
    for j in range(n):
        corr[j] = np.corrcoef(tau_sim[:, j], tau_real[:, j])[0, 1]
        # least-squares gain: tau_real ~= gain * tau_sim (through origin, matches Phase B convention loosely)
        denom = (tau_sim[:, j] ** 2).sum()
        gain[j] = (tau_sim[:, j] * tau_real[:, j]).sum() / denom if denom > 1e-9 else np.nan
    return rms, corr, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='aug11')
    ap.add_argument('--duration', type=float, default=439.6)
    ap.add_argument('--start', type=float, default=0.0)
    args = ap.parse_args()

    S = load_session(args.session)
    r = S['robot']
    lo, hi = S['overlap']
    grid = np.arange(lo + args.start, min(hi, lo + args.start + args.duration), 1 / FS)
    print(f"=== Phase C.1 tau_estimate vs tau_measured - session {args.session} ===")
    print(f"testing {len(grid)/FS:.1f}s ({len(grid)} steps)\n")

    def resample(t, v):
        return np.stack([np.interp(grid, t, v[:, j]) for j in range(v.shape[1])], 1)

    q_real = resample(r['t'], r['q'])
    dq_real = resample(r['t'], r['dq'])
    tau_real = resample(r['t'], r['motor_torque'])
    qt_real = resample(r['t'], action_to_q_target(r['action']))

    cand_sets = {
        'baseline (b=0)': {},
        'calibrated (Phase B)': {4: 1.19, 10: 1.09, 14: 0.98},
        'best-scan': {4: 3.0, 10: 3.0, 14: 1.0},
    }
    calibrated_joints = set(cand_sets['calibrated (Phase B)'].keys())

    results = {}
    for name, extra in cand_sets.items():
        model = build_model(extra)
        tau_cmd, tau_sim_total = tau_estimate_vs_measured(model, q_real, dq_real, qt_real, 1 / FS)
        rms_cmd, corr_cmd, gain_cmd = fit_stats(tau_cmd, tau_real)
        rms_tot, corr_tot, gain_tot = fit_stats(tau_sim_total, tau_real)
        results[name] = dict(rms_cmd=rms_cmd, corr_cmd=corr_cmd, gain_cmd=gain_cmd,
                              rms_tot=rms_tot, corr_tot=corr_tot, gain_tot=gain_tot)
        print(f"  done: {name}")

    print()
    print(f"{'joint':16s} " + "".join(f"{name+' tau_cmd RMS':>24s}{name+' tau_tot RMS':>24s}" for name in cand_sets))
    for jname, j, _ in TEST_JOINTS:
        row = f"{jname:16s} "
        for name in cand_sets:
            row += f"{results[name]['rms_cmd'][j]:24.4f}{results[name]['rms_tot'][j]:24.4f}"
        print(row)

    print("\n--- gain (tau_real ~= gain * tau_sim), calibrated joints only ---")
    print(f"{'joint':16s} " + "".join(f"{name+' cmd-gain':>20s}{name+' tot-gain':>20s}" for name in cand_sets))
    for jname, j, _ in TEST_JOINTS:
        if j not in calibrated_joints:
            continue
        row = f"{jname:16s} "
        for name in cand_sets:
            row += f"{results[name]['gain_cmd'][j]:20.4f}{results[name]['gain_tot'][j]:20.4f}"
        print(row)

    print("\nRMS = |tau_sim - tau_real| RMS error (N*m). tau_cmd = pure PD command")
    print("(damping-insensitive, matches Phase B's near-tautological result).")
    print("tau_tot = tau_cmd + qfrc_passive (includes this model's dof_damping);")
    print("should move gain closer to 1.0 on the calibrated joints if the")
    print("Phase B Kd-residual hypothesis (extra viscous friction/back-EMF) is correct.")


if __name__ == '__main__':
    main()
