"""
Phase C.2 -- closed-loop rollout test (world-modeling style, NO teacher forcing).

See sim2real/phaseC2_discussion.md for the full discussion of why this is a
DIFFERENT test from Phase C.1's one-step-ahead prediction:

  Phase C.1: reset sim state to REAL state every step, apply real action,
             step ONE dt, compare q_sim(t+dt) vs q_real(t+dt). Errors CANNOT
             compound (teacher forcing) -- tests local dynamics fidelity only.

  Phase C.2 (this script): initialize sim ONCE from the real state at t0,
             then integrate CONTINUOUSLY -- q_sim(t+dt) becomes the input to
             the next step, never reset to q_real. Errors CAN compound. This
             is the test that actually exercises "is this a usable world
             model of the real robot", and is what failed catastrophically in
             the earlier open-loop attempt (sim2real_phaseB2_compensation_test.py,
             robot fell by t=30-50s) when the sim model was NOT calibrated.

WHAT THIS SCRIPT REPORTS
------------------------
Per-joint MSE (and RMS, in degrees) between q_real(t) and q_sim(t) over the
autoregressive rollout, for one or more candidate damping/friction configs,
so the effect of calibration on ROLLOUT stability (not just one-step
accuracy) can finally be measured. If the sim state ever goes non-finite
(diverges), the rollout is truncated there and the truncation time is
reported -- this is itself an important result (does calibration delay or
prevent divergence?).

USAGE
-----
  .venv_sim/bin/python model_eval/sim2real_phaseC2_closed_loop_rollout.py \\
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
from sim2real_phaseC1_friction_scan import build_model_with_friction  # noqa: E402


def closed_loop_rollout(model, q_real, dq_real, q_target, dt):
    """Initialize sim state ONCE from (q_real[0], dq_real[0]), then integrate
    continuously: at each step, compute the PD command from the SIM's OWN
    current state (q_sim, dq_sim) against q_target[t] (the real robot's
    commanded target, still fed in open-loop since we don't re-run the
    policy), step forward, and record q_sim[t]. NEVER reset to q_real.

    Returns (q_sim, valid_mask, first_invalid_t):
      q_sim          (T,29) sim's own trajectory (NaN after divergence)
      valid_mask     (T,) bool, True while sim state stayed finite
      first_invalid_t index of first non-finite step, or T if never diverged
    """
    data = mujoco.MjData(model)
    n_sub = max(1, int(round(dt / model.opt.timestep)))
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in MJ_JOINT_NAMES])
    dofadr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                        for n in MJ_JOINT_NAMES])
    actadr = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in MJ_ACTUATOR_NAMES])

    T = len(q_real)
    q_sim = np.full((T, 29), np.nan)
    valid_mask = np.zeros(T, dtype=bool)

    # initialize ONCE from real state at t=0
    data.qpos[qadr] = q_real[0]
    data.qvel[dofadr] = dq_real[0]
    mujoco.mj_forward(model, data)

    first_invalid_t = T
    for t in range(T):
        q_sim_now = data.qpos[qadr].copy()
        dq_sim_now = data.qvel[dofadr].copy()

        if not np.all(np.isfinite(q_sim_now)) or not np.all(np.isfinite(dq_sim_now)):
            first_invalid_t = t
            break

        q_sim[t] = q_sim_now
        valid_mask[t] = True

        tau = KPS * (q_target[t] - q_sim_now) - KDS * dq_sim_now
        tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)
        data.ctrl[actadr] = tau

        for _ in range(n_sub):
            mujoco.mj_step(model, data)

    return q_sim, valid_mask, first_invalid_t


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
    print(f"=== Phase C.2 - closed-loop rollout - session {args.session} ===")
    print(f"testing {len(grid)/FS:.1f}s ({len(grid)} steps), NO reset (autoregressive)\n")

    def resample(t, v):
        return np.stack([np.interp(grid, t, v[:, j]) for j in range(v.shape[1])], 1)

    q_real = resample(r['t'], r['q'])
    dq_real = resample(r['t'], r['dq'])
    qt_real = resample(r['t'], action_to_q_target(r['action']))

    cand_sets = {
        'baseline (b=0)': ({}, {}),
        'calibrated (Phase B)': ({4: 1.19, 10: 1.09, 14: 0.98}, {}),
        'optimal (damping+friction)': ({4: 0.806, 10: 0.870, 14: 0.537}, {14: 0.25}),
    }

    results = {}
    for name, (extra_damping, extra_friction) in cand_sets.items():
        if extra_friction:
            model = build_model_with_friction(extra_damping, extra_friction)
        else:
            model = build_model(extra_damping)
        q_sim, valid_mask, first_invalid_t = closed_loop_rollout(model, q_real, dq_real, qt_real, 1 / FS)
        n_valid = valid_mask.sum()
        err = np.degrees(q_sim[valid_mask] - q_real[valid_mask])   # (n_valid, 29)
        mse = (err ** 2).mean(0)
        rms = np.sqrt(mse)
        results[name] = dict(rms=rms, mse=mse, n_valid=n_valid, first_invalid_t=first_invalid_t)
        survived_s = first_invalid_t / FS if first_invalid_t < len(grid) else len(grid) / FS
        print(f"  done: {name:24s}  survived {survived_s:.1f}s / {len(grid)/FS:.1f}s "
              f"({'DIVERGED at t=%.2fs' % survived_s if first_invalid_t < len(grid) else 'no divergence'})")

    print()
    print(f"{'joint':16s} " + "".join(f"{name+' MSE(deg^2)':>26s}{name+' RMS(deg)':>18s}" for name in cand_sets))
    for jname, j, _ in TEST_JOINTS:
        row = f"{jname:16s} "
        for name in cand_sets:
            row += f"{results[name]['mse'][j]:26.4f}{results[name]['rms'][j]:18.4f}"
        print(row)

    print(f"\n{'ALL-29 MEAN':16s} " + "".join(
        f"{results[name]['mse'].mean():26.4f}{results[name]['rms'].mean():18.4f}" for name in cand_sets))

    print("\nMSE/RMS computed only over the VALID (finite) portion of the rollout for each config")
    print("(they may cover different durations if one config diverges earlier than the other --")
    print("see the 'survived Xs' line above for how much of the session each config completed).")
    print("Unlike Phase C.1 (teacher-forced, cannot diverge), THIS metric reflects both local")
    print("dynamics fidelity AND autoregressive stability -- a genuine world-model-style test.")


if __name__ == '__main__':
    main()
