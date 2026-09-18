#!/usr/bin/env python3
"""Quick-look analysis of a single-directory G1 deploy log (e.g. g1_run_0917).

Unlike sim2real_phaseB_* this makes no Mode-2 assumption, so it still says
something useful about logs where the policy never engaged.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import (  # noqa: E402
    DEFAULT_ANGLES, G1_ACTION_SCALE, ISAACLAB_TO_MUJOCO, JOINT_NAMES, KDS, KPS,
)

NUM_META_COLS = 5  # index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp


def load(run_dir, name, ncols=29):
    path = os.path.join(run_dir, f"{name}.csv")
    arr = np.genfromtxt(path, delimiter=",", skip_header=1, invalid_raise=False)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, NUM_META_COLS:NUM_META_COLS + ncols], arr[:, 1]


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/grease/g1_robot_data/g1_run_0917"
    print(f"=== {run_dir} ===")

    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    rc = meta["robot_config"]
    print(f"policy      : {rc['model_path']}")
    print(f"encoder     : {rc['encoder_file']}")
    print(f"ctrl freq   : {rc['control_frequency']} Hz   dt={meta['logging']['dt']}")

    q, t = load(run_dir, "q")
    dq, _ = load(run_dir, "dq")
    tau, _ = load(run_dir, "motor_torque")
    act, _ = load(run_dir, "action")
    mode, _ = load(run_dir, "encoder_mode", ncols=1)
    playing, _ = load(run_dir, "motion_playing", ncols=1)
    temp, _ = load(run_dir, "motor_temperature")
    err, _ = load(run_dir, "motor_error")

    # Independent per-channel ring buffers can end up with slightly different
    # row counts (async writers) -- align everything to the shortest channel.
    n = min(len(q), len(dq), len(tau), len(act), len(mode), len(playing),
            len(temp), len(err))
    q, dq, tau, act = q[:n], dq[:n], tau[:n], act[:n]
    mode, playing, temp, err, t = mode[:n], playing[:n], temp[:n], err[:n], t[:n]

    dur = (t[-1] - t[0]) / 1000.0
    print(f"samples     : {len(q)}  duration={dur:.1f}s  "
          f"({len(q) / dur:.1f} Hz effective)")
    print(f"encoder_mode: {dict(zip(*np.unique(mode, return_counts=True)))}")
    print(f"playing     : {dict(zip(*np.unique(playing, return_counts=True)))}")
    print(f"motor_error : {int((err != 0).sum())} nonzero entries")
    print(f"temp range  : {np.nanmin(temp):.0f}-{np.nanmax(temp):.0f} C")

    # Did the robot actually move?
    print("\n--- motion check (is the robot doing anything?) ---")
    q_range = np.nanmax(q, 0) - np.nanmin(q, 0)
    print(f"max |dq| over all joints : {np.nanmax(np.abs(dq)):.3f} rad/s")
    print(f"max joint travel         : {np.nanmax(q_range):.4f} rad "
          f"({JOINT_NAMES[int(np.nanargmax(q_range))]})")
    print(f"mean |tau|               : {np.nanmean(np.abs(tau)):.2f} N.m")
    print(f"raw action max|.|        : {np.nanmax(np.abs(act)):.3f}")

    top = np.argsort(-q_range)[:6]
    print("\n  joint            travel(rad)  max|dq|   mean|tau|")
    for j in top:
        print(f"  {JOINT_NAMES[j]:15s} {q_range[j]:10.4f} "
              f"{np.nanmax(np.abs(dq[:, j])):8.3f} {np.nanmean(np.abs(tau[:, j])):10.2f}")

    # Holding position? compare q against default pose
    dev = np.abs(np.nanmean(q, 0) - DEFAULT_ANGLES)
    print(f"\nmean |q - default_angles|: {dev.mean():.4f} rad "
          f"(max {dev.max():.4f} on {JOINT_NAMES[int(dev.argmax())]})")

    # PD consistency, using reconstructed q_target (valid only if policy engaged)
    q_target = DEFAULT_ANGLES[None, :] + act[:, ISAACLAB_TO_MUJOCO] * G1_ACTION_SCALE[None, :]
    e = q_target - q
    tau_sim = e * KPS[None, :] - dq * KDS[None, :]
    m = np.all(np.isfinite(e), 1) & np.all(np.isfinite(dq), 1) & np.all(np.isfinite(tau), 1)
    if m.sum() > 20:
        rmse = np.sqrt(np.nanmean((tau[m] - tau_sim[m]) ** 2, axis=0))
        print(f"\n--- nominal PD fit (all samples, mode-agnostic) ---")
        print(f"mean RMSE over 29 joints : {rmse.mean():.3f} N.m")
        wj = int(np.nanargmax(rmse))
        print(f"worst joint              : {JOINT_NAMES[wj]} ({rmse[wj]:.3f})")
        j = JOINT_NAMES.index("waist_roll")
        print(f"waist_roll RMSE          : {rmse[j]:.3f}")

        # Per-joint (Kp, Kd) free-fit -- same method as
        # model_eval/sim2real_phaseE_gain_refit_per_run.py (see phaseE doc S13).
        print("\n--- (Kp,Kd) free-fit vs nominal, selected joints ---")
        print(f"  {'joint':15s} {'Kp_fit':>8s} {'Kp/nom':>7s} {'Kd_fit':>8s} "
              f"{'Kd/nom':>7s} {'R2':>6s} {'RMSEnom':>8s} {'RMSEfit':>8s}")
        watch = ["waist_roll", "waist_pitch", "L_ankle_roll", "R_ankle_roll",
                 "L_ankle_pitch", "L_hip_pitch", "R_knee"]
        for nm in watch:
            j = JOINT_NAMES.index(nm)
            A = np.stack([e[m, j], -dq[m, j]], axis=1)
            t_ = tau[m, j]
            coef, *_ = np.linalg.lstsq(A, t_, rcond=None)
            resid = t_ - A @ coef
            ss = ((t_ - t_.mean()) ** 2).sum()
            r2 = 1 - (resid ** 2).sum() / ss if ss > 0 else np.nan
            rn = np.sqrt(np.mean((t_ - tau_sim[m, j]) ** 2))
            print(f"  {nm:15s} {coef[0]:8.3f} {coef[0] / KPS[j]:7.3f} "
                  f"{coef[1]:8.4f} {coef[1] / KDS[j]:7.3f} {r2:6.3f} "
                  f"{rn:8.3f} {np.sqrt(np.mean(resid ** 2)):8.3f}")

        # dq aliasing check (phaseE S6): logged dq vs finite-difference of q
        print("\n--- dq aliasing: corr(dq_logged, dq_fd) ---")
        dt_s = np.diff(t) / 1000.0
        ok = dt_s > 1e-6
        dq_fd = np.diff(q, axis=0) / np.where(ok, dt_s, np.nan)[:, None]
        print(f"  {'joint':15s} {'corr':>7s} {'std_log':>8s} {'std_fd':>8s} {'ratio':>7s}")
        for nm in watch:
            j = JOINT_NAMES.index(nm)
            a, b = dq[:-1, j], dq_fd[:, j]
            g = np.isfinite(a) & np.isfinite(b)
            if g.sum() < 20 or a[g].std() < 1e-9 or b[g].std() < 1e-9:
                continue
            c = np.corrcoef(a[g], b[g])[0, 1]
            print(f"  {nm:15s} {c:7.3f} {a[g].std():8.3f} {b[g].std():8.3f} "
                  f"{a[g].std() / max(b[g].std(), 1e-9):7.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
