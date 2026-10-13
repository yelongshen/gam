#!/usr/bin/env python3
"""Direct one-step q_next validation: is a Kd scale of ~1.5x (the pitch
subgroup correction in `gear_sonic/utils/mujoco_sim/sim2real_gain_correction.py`)
still the best predictor of the REAL robot's next-tick state on the NEW robot
(`g1_run_0918`)?

Unlike `sim2real_phaseC1_onestep_prediction.py` (which compares candidate
predictions against a *training-distribution ground truth* because no
scaled-gain hardware log existed), this script compares directly against the
REAL, logged q_real_next -- valid because we don't need real hardware run
under scaled gains: MuJoCo's <motor> actuators are direct torque actuators,
so `tau = Kp_scaled*e - Kd_scaled*dq` computed at the REAL measured (q, dq)
already tells us what the REAL robot's next state *should* look like if that
Kd is the right one for its actual dynamics.

Method: teacher-forced one-step MuJoCo integration (reset to real q/dq every
tick, apply the real q_target with a candidate Kp/Kd scale, step 1 dt),
repeated for a sweep of Kd scales, RMSE against the REAL logged q(t+1).
Whichever scale minimizes RMSE is empirically "best" for this robot's data.

Usage:
    .venv_sim/bin/python model_eval/sim2real_kd_scale_sweep.py \
        --run-dirs RUN_DIR [RUN_DIR ...] --joints L_ankle_pitch R_ankle_pitch waist_pitch \
        --kd-scales 1.0 1.2 1.4 1.5 1.6 1.8 2.0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import (  # noqa: E402
    JOINT_NAMES, KDS, KPS, action_to_q_target,
)
from model_eval.sim2real_phaseC1_onestep_prediction import (  # noqa: E402
    build_model, one_step_predict,
)
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402

FS = 50.0


def load_run(run_dir, only_mode2):
    q, t = load(run_dir, "q")
    dq, _ = load(run_dir, "dq")
    act, _ = load(run_dir, "action")
    mode, _ = load(run_dir, "encoder_mode", ncols=1)
    n = min(len(q), len(dq), len(act), len(mode))
    q, dq, act, mode, t = q[:n], dq[:n], act[:n], mode[:n, 0], t[:n]

    qt = action_to_q_target(act)
    finite = np.all(np.isfinite(q), 1) & np.all(np.isfinite(dq), 1) & np.all(np.isfinite(qt), 1)
    if only_mode2:
        finite &= (mode == 2)
    q, dq, qt = q[finite], dq[finite], qt[finite]
    if len(q) < 10:
        return None
    # need q(t+1): use the raw (unfiltered-by-finite) array's next row where
    # possible; simplest robust approach is to just require the finite mask
    # to be True for consecutive rows too, so re-derive from indices.
    idx = np.nonzero(finite)[0]
    idx = idx[idx < n - 1]  # need a following raw sample
    idx_next = idx + 1
    q_t = load(run_dir, "q")[0][idx]
    dq_t = load(run_dir, "dq")[0][idx]
    qt_t = action_to_q_target(load(run_dir, "action")[0])[idx]
    q_next = load(run_dir, "q")[0][idx_next]
    ok = np.all(np.isfinite(q_t), 1) & np.all(np.isfinite(dq_t), 1) & \
         np.all(np.isfinite(qt_t), 1) & np.all(np.isfinite(q_next), 1)
    return q_t[ok], dq_t[ok], qt_t[ok], q_next[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--joints", nargs="+",
                     default=["L_ankle_pitch", "R_ankle_pitch", "waist_pitch"])
    ap.add_argument("--kd-scales", nargs="+", type=float,
                     default=[1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0])
    ap.add_argument("--kp-scale", type=float, default=1.0,
                     help="Kp scale to hold fixed while sweeping Kd (nominal Kp already"
                          " showed no shift in the free-fit check)")
    ap.add_argument("--all-samples", action="store_true",
                     help="use all samples, not just encoder_mode==2")
    ap.add_argument("--extra-damping", action="store_true",
                     help="apply the Phase-B calibrated dof_damping (joints 4,10,14) "
                          "so this is apples-to-apples with the existing 1.5-1.6x "
                          "correction in sim2real_gain_correction.py, which was derived "
                          "WITH that calibration already applied")
    args = ap.parse_args()

    jidx = [JOINT_NAMES.index(n) for n in args.joints]

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
        print("no usable data across all run dirs")
        return 1

    q_real = np.concatenate(q_list)
    dq_real = np.concatenate(dq_list)
    qt_real = np.concatenate(qt_list)
    q_real_next = np.concatenate(qn_list)
    print(f"\npooled: {len(q_real)} samples across {len(q_list)} runs\n")

    model = build_model({4: 1.19, 10: 1.09, 14: 0.98} if args.extra_damping else {})
    kp = KPS.copy()
    kp[jidx] *= args.kp_scale

    print(f"{'kd_scale':>10s} " + " ".join(f"{n:>16s}" for n in args.joints) + f" {'ALL-29 mean':>14s}")
    best = None
    for s in args.kd_scales:
        kd = KDS.copy()
        kd[jidx] *= s
        q_sim_next, _ = one_step_predict(model, q_real, dq_real, qt_real, 1 / FS, kp=kp, kd=kd)
        rmse_deg = np.degrees(np.sqrt(((q_sim_next - q_real_next) ** 2).mean(0)))
        row_vals = [rmse_deg[j] for j in jidx]
        target_mean = np.mean(row_vals)  # mean over just the swept joints, the decision metric
        all_mean = rmse_deg.mean()
        print(f"{s:10.2f} " + " ".join(f"{v:16.4f}" for v in row_vals) + f" {all_mean:14.4f}"
              + ("   <- best so far (swept-joint mean)" if best is None or target_mean < best[1] else ""))
        if best is None or target_mean < best[1]:
            best = (s, target_mean)

    print(f"\nbest Kd scale (min RMSE averaged over {args.joints}): {best[0]:.2f} "
          f"(mean RMSE {best[1]:.4f} deg)")


if __name__ == "__main__":
    raise SystemExit(main())
