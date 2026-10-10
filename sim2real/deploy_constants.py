#!/usr/bin/env python
"""Deploy-side constants and log loaders, mirrored from the C++ deploy binary.

Every constant here is transcribed from
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`.
Joint-order handling follows `src/state_logger.cpp` exactly -- getting this
wrong silently corrupts every downstream number, so the reasoning is spelled
out rather than assumed:

  q.csv, dq.csv      HARDWARE order. `state_logger.cpp:296` writes
                       body_q_measured[i] = body_q[isaaclab_to_mujoco[i]] + default_angles[i]
                     i.e. absolute measured angle, already converted back.
  action.csv         ISAACLAB order, raw policy output, NOT scaled
                     (`state_logger.cpp:302` writes `e.last_action` verbatim,
                      and `g1_deploy_onnx_ref.cpp:3145` sets
                      `last_action[i] = floatarr[i]`).
  motor_torque.csv   HARDWARE order, `tau_est` straight off the motor
                     (`g1_deploy_onnx_ref.cpp:2864`, indexed by `[i]` with no
                     remap -- unlike body_q two lines above, which IS remapped).

The commanded target, in HARDWARE order (`g1_deploy_onnx_ref.cpp:3144-3146`):

    q_target[i] = default_angles[i] + action_isaaclab[isaaclab_to_mujoco[i]] * action_scale[i]

and the low-level loop closes at the motor with tau_ff == 0, dq_target == 0:

    tau[i] = kp[i] * (q_target[i] - q[i]) + kd[i] * (0 - dq[i])
"""
from __future__ import annotations

import csv
import os

import numpy as np

# --- actuator constants (policy_parameters.hpp:40-46) ----------------------
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFF = {
    "5020": ARMATURE_5020 * NATURAL_FREQ**2,
    "7520_14": ARMATURE_7520_14 * NATURAL_FREQ**2,
    "7520_22": ARMATURE_7520_22 * NATURAL_FREQ**2,
    "4010": ARMATURE_4010 * NATURAL_FREQ**2,
}
DAMP = {
    "5020": 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ,
    "7520_14": 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ,
    "7520_22": 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ,
    "4010": 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ,
}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}

# motor type per HARDWARE joint index, and the x2 gain boost on ankles/waist
# roll+pitch (policy_parameters.hpp:143-205)
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
MOTOR_TYPE = [
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
]
GAIN_X2 = [False] * 29
for _i in (4, 5, 10, 11, 13, 14):  # ankles + waist roll/pitch
    GAIN_X2[_i] = True

KP = np.array([STIFF[t] * (2.0 if x2 else 1.0)
               for t, x2 in zip(MOTOR_TYPE, GAIN_X2)])
KD = np.array([DAMP[t] * (2.0 if x2 else 1.0)
               for t, x2 in zip(MOTOR_TYPE, GAIN_X2)])
# action_scale = 0.25 * effort_limit / stiffness  -- note: uses the UNBOOSTED
# stiffness even for the x2 joints (policy_parameters.hpp:109-140)
ACTION_SCALE = np.array([0.25 * EFFORT[t] / STIFF[t] for t in MOTOR_TYPE])
EFFORT_LIMIT = np.array([EFFORT[t] for t in MOTOR_TYPE])

DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])

# policy_parameters.hpp:100-104
ISAACLAB_TO_MUJOCO = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                               11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
MUJOCO_TO_ISAACLAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                               16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])

CONTROL_DT = 0.02  # 50 Hz (metadata.json: logging.dt)

# ---------------------------------------------------------------------------
# Excluded runs -- see `sim2real/robot_log_data_quality.md` for the full write-up.
#
# Every entry was found by data-driven QC, not by reading the session reports:
# none of these is mentioned in any `online_eval_*_report.md`. They are keyed by
# path substring and matched against the full run directory path.
# ---------------------------------------------------------------------------
EXCLUDED_RUNS = {
    # waist_roll / waist_pitch `tau_est` is a constant (slope 0.000, corr nan),
    # i.e. those two motors reported no torque at all. Earliest logs in the
    # archive (2026-08-07), most likely recorded before the waist actuators
    # were commissioned. Together these are 29723 frames and they alone made
    # the whole `low_latency` policy group read as nan.
    "g1_deploy_run002": "waist tau_est constant (uninstrumented waist)",
    "g1_deploy_run": "waist tau_est constant (uninstrumented waist)",
    # 8700 NaNs in action.csv -- the policy output itself is corrupt.
    "g1_run_0909/g1_deploy_run_09082026_run1": "action.csv contains 8700 NaNs",
    # CSVs contain NUL bytes: logging was interrupted mid-write.
    "g1_run_0925": "CSV files contain NUL bytes (truncated logs)",
    # unreadable / zero-length CSVs
    "g1_run_0919/20260919_073126": "unreadable CSV",
    "g1_run_0922/20260922_083850": "unreadable CSV",
    # waist_roll disagrees with the PD law while every other joint in the same
    # run is clean, so this is not a gain or joint-order problem. Root cause
    # NOT diagnosed -- excluded as unexplained rather than as understood-bad.
    "g1_deploy_run_09182026_081125": "waist_roll corr 0.297, undiagnosed",
    "g1_deploy_run_09142026_run8": "waist_roll slope 1.497, undiagnosed",
}


def is_excluded(run_dir):
    """True if `run_dir` matches a known-bad run. Returns (bool, reason).

    Matching is on EXACT path components, never substrings. Two traps this
    avoids, both of which silently deleted most of the archive while being
    developed:
      * `g1_deploy_run` is a substring of `g1_deploy_run_09082026_run4`
      * `..._run1` is a prefix of `..._run10`, `..._run11`, `..._run14`
    """
    norm = os.path.normpath(run_dir).replace(os.sep, "/")
    parts = norm.split("/")
    for key, reason in EXCLUDED_RUNS.items():
        key_parts = key.split("/")
        n = len(key_parts)
        # the key must appear as a contiguous run of WHOLE components
        for i in range(len(parts) - n + 1):
            if parts[i:i + n] == key_parts:
                return True, reason
    return False, ""


def load_csv(path, prefix):
    """Read a deploy CSV into (values[T,N], t_ms[T])."""
    vals, t = [], []
    with open(path) as fh:
        r = csv.reader(fh)
        hdr = next(r)
        cols = [i for i, h in enumerate(hdr) if h.startswith(prefix)]
        ti = hdr.index("time_realtime_ms")
        for row in r:
            if len(row) <= cols[-1]:
                continue  # truncated final row (logger killed mid-write)
            vals.append([float(row[i]) for i in cols])
            t.append(float(row[ti]))
    return np.asarray(vals), np.asarray(t)


def load_run(run_dir):
    """Load one deploy run, returning everything in HARDWARE joint order."""
    q, t = load_csv(os.path.join(run_dir, "q.csv"), "q_")
    dq, _ = load_csv(os.path.join(run_dir, "dq.csv"), "dq_")
    act, _ = load_csv(os.path.join(run_dir, "action.csv"), "act_")
    tau, _ = load_csv(os.path.join(run_dir, "motor_torque.csv"), "tau_")

    n = min(len(q), len(dq), len(act), len(tau))
    q, dq, act, tau, t = q[:n], dq[:n], act[:n], tau[:n], t[:n]

    # action.csv is IsaacLab-ordered; convert to hardware order the same way
    # CreatePolicyCommand() does, then apply the per-joint scale.
    act_hw = act[:, ISAACLAB_TO_MUJOCO]
    q_target = DEFAULT_ANGLES[None, :] + act_hw * ACTION_SCALE[None, :]

    return dict(q=q, dq=dq, action_isaaclab=act, action_hw=act_hw,
                q_target=q_target, tau=tau, t_ms=t, n=n)


def list_runs(root="/home/grease/g1_robot_data", include_excluded=False):
    """Every deploy run directory that has the four CSVs we need.

    Known-bad runs (`EXCLUDED_RUNS`) are dropped by default -- pass
    `include_excluded=True` to get the raw list, e.g. to re-verify the QC.
    """
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if {"q.csv", "dq.csv", "action.csv", "motor_torque.csv"} <= set(filenames):
            if not include_excluded and is_excluded(dirpath)[0]:
                continue
            out.append(dirpath)
    return sorted(out)
