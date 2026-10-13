#!/usr/bin/env python3
"""Refit the Phase-B `extra_damping` calibration (currently {4: 1.19, 10: 1.09,
14: 0.98} in `sim2real_phaseC1_onestep_prediction.py`, joints 4/10 =
L/R_ankle_pitch) using data from the NEW robot (`g1_run_0918`), to check
whether the same calibration still holds or whether this robot's actual
ankle-pitch dynamics differ.

Method: same one-step teacher-forced MuJoCo prediction as
`sim2real_kd_scale_sweep.py`, but this time sweeping `dof_damping` (additive,
mirrors `build_model`'s `extra_damping` argument) at NOMINAL Kp/Kd (no gain
scaling), for joints 4 and 10 independently (coordinate sweep, holding the
other joint + waist_pitch at their old calibrated values), then a small joint
grid around the two independent optima.

Usage:
    .venv_sim/bin/python model_eval/sim2real_damping_sweep.py \
        --run-dirs RUN_DIR [RUN_DIR ...]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES, KDS, KPS  # noqa: E402
from model_eval.sim2real_kd_scale_sweep import load_run  # noqa: E402
from model_eval.sim2real_phaseC1_onestep_prediction import (  # noqa: E402
    build_model, one_step_predict,
)

FS = 50.0
OLD_CALIBRATION = {4: 1.19, 10: 1.09, 14: 0.98}  # from Phase C.1 (aug11 session)


def rmse_deg(q_sim_next, q_real_next, j):
    return np.degrees(np.sqrt(((q_sim_next[:, j] - q_real_next[:, j]) ** 2).mean()))


def sweep_1d(joint_idx, joint_name, candidates, base_damping, q_real, dq_real, qt_real, q_real_next):
    print(f"\n--- sweeping dof_damping[{joint_idx}] ({joint_name}), "
          f"others held at {base_damping} ---")
    print(f"{'damping':>10s} {'RMSE_'+joint_name:>20s}")
    best = None
    for d in candidates:
        damping = dict(base_damping)
        damping[joint_idx] = d
        model = build_model(damping)
        q_sim_next, _ = one_step_predict(model, q_real, dq_real, qt_real, 1 / FS, kp=KPS, kd=KDS)
        e = rmse_deg(q_sim_next, q_real_next, joint_idx)
        flag = ""
        if best is None or e < best[1]:
            best = (d, e)
            flag = "  <- best so far"
        print(f"{d:10.3f} {e:20.4f}{flag}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--all-samples", action="store_true")
    args = ap.parse_args()

    q_list, dq_list, qt_list, qn_list = [], [], [], []
    for rd in args.run_dirs:
        out = load_run(rd, only_mode2=not args.all_samples)
        if out is None:
            print(f"[skip] {os.path.basename(rd)}: too few usable samples")
            continue
        q_t, dq_t, qt_t, q_next = out
        print(f"[ok]   {os.path.basename(rd)}: {len(q_t)} samples")
        q_list.append(q_t); dq_list.append(dq_t); qt_list.append(qt_t); qn_list.append(q_next)

    if not q_list:
        print("no usable data")
        return 1

    q_real = np.concatenate(q_list)
    dq_real = np.concatenate(dq_list)
    qt_real = np.concatenate(qt_list)
    q_real_next = np.concatenate(qn_list)
    print(f"\npooled: {len(q_real)} samples across {len(q_list)} runs")

    # Baseline: nominal dynamics (no extra damping at all) and the OLD
    # calibration, for reference.
    for label, damping in [("nominal (no extra damping)", {}),
                            ("OLD calibration (aug11)", OLD_CALIBRATION)]:
        model = build_model(damping)
        q_sim_next, _ = one_step_predict(model, q_real, dq_real, qt_real, 1 / FS, kp=KPS, kd=KDS)
        e4 = rmse_deg(q_sim_next, q_real_next, 4)
        e10 = rmse_deg(q_sim_next, q_real_next, 10)
        e14 = rmse_deg(q_sim_next, q_real_next, 14)
        print(f"\n[{label}] L_ankle_pitch={e4:.4f} deg  R_ankle_pitch={e10:.4f} deg  "
              f"waist_pitch={e14:.4f} deg")

    candidates = [1.19, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    base = {10: OLD_CALIBRATION[10], 14: OLD_CALIBRATION[14]}
    best4 = sweep_1d(4, "L_ankle_pitch", candidates, base, q_real, dq_real, qt_real, q_real_next)

    base = {4: OLD_CALIBRATION[4], 14: OLD_CALIBRATION[14]}
    best10 = sweep_1d(10, "R_ankle_pitch", candidates, base, q_real, dq_real, qt_real, q_real_next)

    waist_candidates = [0.4, 0.8, 1.2, 1.8, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    base = {4: OLD_CALIBRATION[4], 10: OLD_CALIBRATION[10]}
    best14 = sweep_1d(14, "waist_pitch", waist_candidates, base, q_real, dq_real, qt_real, q_real_next)

    print("\n=== Refit vs old calibration ===")
    print(f"L_ankle_pitch (joint 4):  old=1.19  new_best={best4[0]:.3f}  "
          f"(RMSE {best4[1]:.4f} deg)")
    print(f"R_ankle_pitch (joint 10): old=1.09  new_best={best10[0]:.3f}  "
          f"(RMSE {best10[1]:.4f} deg)")
    print(f"waist_pitch (joint 14):   old=0.98  new_best={best14[0]:.3f}  "
          f"(RMSE {best14[1]:.4f} deg)")

    # Confirm with the joint best-of-all combo
    model = build_model({4: best4[0], 10: best10[0], 14: best14[0]})
    q_sim_next, _ = one_step_predict(model, q_real, dq_real, qt_real, 1 / FS, kp=KPS, kd=KDS)
    e4 = rmse_deg(q_sim_next, q_real_next, 4)
    e10 = rmse_deg(q_sim_next, q_real_next, 10)
    e14 = rmse_deg(q_sim_next, q_real_next, 14)
    print(f"\n[joint new-best combo] L_ankle_pitch={e4:.4f} deg  R_ankle_pitch={e10:.4f} deg  "
          f"waist_pitch={e14:.4f} deg")


if __name__ == "__main__":
    raise SystemExit(main())
