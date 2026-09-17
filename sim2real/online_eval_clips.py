#!/usr/bin/env python
"""Real-robot online deployment evaluation for streamed SMPL clips.

Metric suite mirrors GR00T-WholeBodyControl's eval_agent_trl.py / im_eval_callback.py
(smpl_sim.smpllib.smpl_eval.compute_metrics_lite: mpjpe_g / mpjpe_l / mpjpe_pa /
vel_dist / accel_dist, same body subsets), plus the real-robot terms from
sim2real/online_deployment_eval_plan.md (saturation, shaking, swing, fall).

Ground truth
------------
PRIMARY (sim protocol -- same reference the sim eval uses):
    ego_dataset/eval_subset/robot/<clip>.pkl -> `dof` (T,29), the retargeted G1 joint
    trajectory, FK'd to link positions and compared against FK(q_measured).

SECONDARY / diagnostics only:
    cmd           : FK(q_target) vs FK(q)       -> control-loop fidelity (plan 1.1)
    smpl_diag     : FK(q) vs streamed SMPL      -> G1 links vs raw SMPL joints over a
                    hand-built 14-joint map; inflated by morphology mismatch, NOT
                    comparable to sim.
    retarget_diag : FK(dof) vs streamed SMPL    -> isolates the retargeting term that
                    contaminates smpl_diag.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoupled_wbc.control.robot_model.instantiation.g1 import (  # noqa: E402
    instantiate_g1_robot_model,
)

# Hardware / MuJoCo joint order, from gear_sonic_deploy policy_parameters.hpp
HW_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

E5020, E7520_14, E7520_22, E4010 = 25.0, 88.0, 139.0, 5.0
MOTOR_TYPE = (
    ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2
    + ["7520_14", "5020", "5020"]
    + ["5020"] * 5 + ["4010", "4010"]
    + ["5020"] * 5 + ["4010", "4010"]
)
_EFF = {"5020": E5020, "7520_14": E7520_14, "7520_22": E7520_22, "4010": E4010}
_ARM = {"5020": 0.003609725, "7520_14": 0.010177520,
        "7520_22": 0.025101925, "4010": 0.00425}
_W = 10 * 2 * np.pi
EFFORT_LIMITS = np.array([_EFF[m] for m in MOTOR_TYPE])
ACTION_SCALE = np.array([0.25 * _EFF[m] / (_ARM[m] * _W * _W) for m in MOTOR_TYPE])
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])
# "mujoco order in isaaclab index" (policy_parameters.hpp). action.csv is logged RAW
# (IsaacLab order, unscaled) despite the state_logger docstring, so q_target is
# reconstructed as in zmq_output_handler.hpp:348.
ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
])
ACTION_LAG_FRAMES = 2  # measured command -> achieved q lag @50Hz

BODY_LINKS = [
    "pelvis",
    "left_hip_yaw_link", "right_hip_yaw_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]
SUBSETS = {
    "legs": ["left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
             "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link"],
    "vr_3points": ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"],
    "other_upper_bodies": ["pelvis", "left_shoulder_roll_link", "left_elbow_link",
                           "right_shoulder_roll_link", "right_elbow_link"],
    "foot": ["left_ankle_roll_link", "right_ankle_roll_link"],
}
# SMPL(24) index -> G1 link, used ONLY for the secondary SMPL diagnostics.
SMPL_TO_LINK = {
    0: "pelvis",
    1: "left_hip_yaw_link", 2: "right_hip_yaw_link",
    4: "left_knee_link", 5: "right_knee_link",
    7: "left_ankle_roll_link", 8: "right_ankle_roll_link",
    12: "torso_link",
    16: "left_shoulder_roll_link", 17: "right_shoulder_roll_link",
    18: "left_elbow_link", 19: "right_elbow_link",
    20: "left_wrist_yaw_link", 21: "right_wrist_yaw_link",
}
STRESS_JOINTS = ["left_wrist_roll_joint", "right_wrist_roll_joint",
                 "left_shoulder_yaw_joint", "right_shoulder_yaw_joint"]


def action_to_q_target(action):
    """Raw logged policy action -> commanded joint target in hardware/MuJoCo order."""
    return action[:, ISAACLAB_TO_MUJOCO] * ACTION_SCALE + DEFAULT_ANGLES


def read_csv(path, ncols=None):
    """Tolerant CSV reader: skips the truncated final row every log has."""
    with open(path) as fh:
        header = next(fh).rstrip("\n").split(",")
        n = ncols or len(header)
        rows = [
            [float(x) for x in line.rstrip("\n").split(",")]
            for line in fh
            if len(line.rstrip("\n").split(",")) == n
        ]
    return header, np.asarray(rows, dtype=np.float64)


def load_run_signal(run_dir, name, t0_ms, t1_ms):
    """Load <name>.csv from a g1 run, sliced to [t0_ms, t1_ms) on `time_ms`."""
    header, arr = read_csv(os.path.join(run_dir, name + ".csv"))
    t = arr[:, header.index("time_ms")]
    data_cols = [i for i, h in enumerate(header)
                 if h not in ("index", "time_ms", "time_realtime_ms",
                              "time_monotonic_ms", "ros_timestamp")]
    m = (t >= t0_ms) & (t < t1_ms)
    return t[m], arr[np.ix_(m, data_cols)]


def load_session_smpl(session_dir, off, nframes):
    _, arr = read_csv(os.path.join(session_dir, "smpl_joint.csv"), ncols=72)
    return arr[off:off + nframes].reshape(-1, 24, 3)


def load_reference(robot_root, clip):
    """Retargeted G1 reference: eval_subset/robot/<clip>.pkl."""
    d = joblib.load(os.path.join(robot_root, clip + ".pkl"))
    if isinstance(d, dict) and clip in d:
        d = d[clip]
    return {
        "dof": np.asarray(d["dof"], dtype=np.float64),
        "root_rot": np.asarray(d["root_rot"], dtype=np.float64),
        "root_trans": np.asarray(d["root_trans_offset"], dtype=np.float64),
        "smpl_joints": np.asarray(d["smpl_joints"], dtype=np.float64),
        "fps": float(d.get("fps", 30.0)),
    }


class FK:
    def __init__(self):
        self.rm = instantiate_g1_robot_model()
        self.qdim = self.rm.num_dofs
        self.idx = [self.rm.dof_index(j) for j in HW_JOINTS]

    def __call__(self, q_hw):
        """q_hw: (T,29) hardware order -> (T, len(BODY_LINKS), 3) link positions."""
        out = np.zeros((len(q_hw), len(BODY_LINKS), 3))
        q = np.zeros(self.qdim)
        for t, qt in enumerate(q_hw):
            q[:] = 0.0
            q[self.idx] = qt
            self.rm.cache_forward_kinematics(q, auto_clip=False)
            for k, link in enumerate(BODY_LINKS):
                out[t, k] = self.rm.frame_placement(link).translation
        return out


def resample(arr, n):
    """Linear resample along axis 0 to n samples (both cover the same clip span)."""
    if len(arr) == n:
        return arr
    src = np.linspace(0, 1, len(arr))
    dst = np.linspace(0, 1, n)
    flat = arr.reshape(len(arr), -1)
    out = np.stack([np.interp(dst, src, flat[:, c]) for c in range(flat.shape[1])], 1)
    return out.reshape(n, *arr.shape[1:])


def p_mpjpe(pred, gt):
    """Procrustes-aligned MPJPE (scale+rot+trans), as in smpl_sim.smpl_eval."""
    muX, muY = gt.mean(1, keepdims=True), pred.mean(1, keepdims=True)
    X0, Y0 = gt - muX, pred - muY
    nX = np.sqrt((X0 ** 2).sum((1, 2), keepdims=True))
    nY = np.sqrt((Y0 ** 2).sum((1, 2), keepdims=True))
    X0, Y0 = X0 / nX, Y0 / nY
    U, s, Vt = np.linalg.svd(X0.transpose(0, 2, 1) @ Y0)
    V = Vt.transpose(0, 2, 1)
    R = V @ U.transpose(0, 2, 1)
    sign = np.sign(np.expand_dims(np.linalg.det(R), 1))
    V[:, :, -1] *= sign
    s[:, -1] *= sign.flatten()
    R = V @ U.transpose(0, 2, 1)
    a = np.expand_dims(s.sum(1, keepdims=True), 2) * nX / nY
    t = muX - a * (muY @ R)
    return np.linalg.norm(a * (pred @ R) + t - gt, axis=-1)


def compute_metrics_lite(pred, gt, root_idx=0):
    """Port of smpl_sim.smpllib.smpl_eval.compute_metrics_lite (mm).

    NOTE mpjpe_g is emitted for interface parity only. The real robot has no external
    mocap, so FK runs with the root pinned at the origin and mpjpe_g is numerically
    identical to mpjpe_l here. It is NOT comparable to a sim mpjpe_g.
    """
    mpjpe_g = np.linalg.norm(gt - pred, axis=-1) * 1000
    vel = np.linalg.norm(np.diff(pred, axis=0) - np.diff(gt, axis=0), axis=-1) * 1000
    acc = np.linalg.norm(np.diff(pred, 2, axis=0) - np.diff(gt, 2, axis=0), axis=-1) * 1000
    p = pred - pred[:, [root_idx]]
    g = gt - gt[:, [root_idx]]
    return {
        "mpjpe_g": float(mpjpe_g.mean()),
        "mpjpe_l": float((np.linalg.norm(p - g, axis=-1) * 1000).mean()),
        "mpjpe_pa": float((p_mpjpe(p, g) * 1000).mean()),
        "vel_dist": float(vel.mean()),
        "accel_dist": float(acc.mean()),
    }


def with_subsets(pred, gt):
    """compute_metrics_lite + the per-body-subset mpjpe_l used by im_eval_callback."""
    li = {ln: i for i, ln in enumerate(BODY_LINKS)}
    res = compute_metrics_lite(pred, gt)
    for name, links in SUBSETS.items():
        idx = [li[x] for x in links]
        res["mpjpe_l_" + name] = compute_metrics_lite(pred[:, idx], gt[:, idx])["mpjpe_l"]
    return res


def joint_space_stats(a, b):
    """Per-joint |a-b| in degrees: mean, worst joint, and the plan 0.1 stress axes."""
    err = np.abs(a - b) * 180 / np.pi
    per_joint = err.mean(0)
    return {
        "joint_err_deg_mean": float(err.mean()),
        "joint_err_deg_worst": [HW_JOINTS[int(per_joint.argmax())],
                                float(per_joint.max())],
        "stress_joints_deg": {j: float(per_joint[HW_JOINTS.index(j)])
                              for j in STRESS_JOINTS},
    }


def highpass_rms(x, fs=50.0, fc=4.0):
    """RMS of the high-frequency residual (shaking term, plan section 4)."""
    k = max(1, int(round(fs / fc)))
    ker = np.ones(k) / k
    low = np.stack([np.convolve(x[:, j], ker, mode="same") for j in range(x.shape[1])], 1)
    return np.sqrt(((x - low) ** 2).mean(0))


def quat_rpy(q, wxyz=True):
    """(T,4) -> roll, pitch, yaw."""
    if wxyz:
        w, x, y, z = q.T
    else:
        x, y, z, w = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def heading_error(base_quat, root_rot):
    """Yaw tracking error (deg) between robot IMU and the reference root rotation.

    root_rot's quaternion convention is undocumented, so both are tried and the one
    whose heading *change* best correlates with the robot's is used. Absolute yaw is
    meaningless (arbitrary world origin), so everything is relative to frame 0.
    """
    _, _, yaw_r = quat_rpy(base_quat, wxyz=True)
    yaw_r = np.unwrap(yaw_r) - yaw_r[0]
    best = None
    for wxyz in (False, True):
        _, _, y = quat_rpy(root_rot, wxyz=wxyz)
        y = np.unwrap(y) - y[0]
        corr = float(np.corrcoef(y, yaw_r)[0, 1]) if y.std() > 1e-9 else 0.0
        cand = {"quat_convention": "wxyz" if wxyz else "xyzw",
                "heading_corr": corr,
                "heading_err_deg": float(np.abs(np.degrees(y - yaw_r)).mean()),
                "ref_total_turn_deg": float(np.degrees(np.abs(y).max()))}
        if best is None or corr > best["heading_corr"]:
            best = cand
    return best


def evaluate(ep, fk, sessions_root, runs_root, robot_root, fps=50.0):
    sess_dir = os.path.join(sessions_root, ep["session"])
    run_dir = os.path.join(runs_root, ep["run"])
    n = ep["nframes"]

    ref = load_reference(robot_root, ep["clip"])
    smpl_stream = load_session_smpl(sess_dir, ep["offset"], n)

    _, q = load_run_signal(run_dir, "q", ep["t0_ms"], ep["t1_ms"])
    _, act = load_run_signal(run_dir, "action", ep["t0_ms"], ep["t1_ms"])
    _, dq = load_run_signal(run_dir, "dq", ep["t0_ms"], ep["t1_ms"])
    _, tau = load_run_signal(run_dir, "motor_torque", ep["t0_ms"], ep["t1_ms"])
    _, bq = load_run_signal(run_dir, "base_quat", ep["t0_ms"], ep["t1_ms"])

    # Compensate the actuation lag so cmd[t] is compared with the q it produced.
    lag = ACTION_LAG_FRAMES
    tgt = action_to_q_target(act)[:len(act) - lag]
    q, dq, tau, bq = q[lag:], dq[lag:], tau[lag:], bq[lag:]

    q, tgt = resample(q, n), resample(tgt, n)
    dq, tau, bq = resample(dq, n), resample(tau, n), resample(bq, n)
    dof_ref = resample(ref["dof"], n)          # 30 fps reference -> clip frames
    root_rot = resample(ref["root_rot"], n)

    fk_q, fk_cmd, fk_ref = fk(q), fk(tgt), fk(dof_ref)
    li = {ln: i for i, ln in enumerate(BODY_LINKS)}
    s_idx = sorted(SMPL_TO_LINK)
    l_idx = [li[SMPL_TO_LINK[i]] for i in s_idx]

    # PRIMARY: imitation vs retargeted G1 reference (sim protocol)
    imitation = with_subsets(fk_q, fk_ref)
    imitation.update(joint_space_stats(q, dof_ref))
    imitation.update(heading_error(bq, root_rot))

    # SECONDARY: control-loop fidelity, FK(q_target) vs FK(q)  (plan 1.1)
    cmd = with_subsets(fk_q, fk_cmd)
    cmd.update(joint_space_stats(tgt, q))

    # robustness: saturation, shaking, swing, fall  (plan 2-4)
    sat = (np.abs(tau) > 0.9 * EFFORT_LIMITS).mean(0)
    roll, pitch, _ = quat_rpy(bq, wxyz=True)
    tilt = np.degrees(np.maximum(np.abs(roll), np.abs(pitch)))

    return {
        "clip": ep["clip"], "category": ep.get("category", ""),
        "run": ep["run"], "session": ep["session"],
        "frames": n, "duration_s": n / fps,
        "ref_fps": ref["fps"], "ref_frames": len(ref["dof"]),
        "imitation": imitation,
        "cmd": cmd,
        "smpl_diag": compute_metrics_lite(fk_q[:, l_idx], smpl_stream[:, s_idx]),
        "retarget_diag": compute_metrics_lite(fk_ref[:, l_idx], smpl_stream[:, s_idx]),
        "robustness": {
            "saturation_mean": float(sat.mean()),
            "saturation_worst": [HW_JOINTS[int(sat.argmax())], float(sat.max())],
            "shaking_rms_rad_s": float(highpass_rms(dq, fps).mean()),
            "swing_roll_deg_rms": float(np.sqrt((roll ** 2).mean()) * 180 / np.pi),
            "swing_pitch_deg_rms": float(np.sqrt((pitch ** 2).mean()) * 180 / np.pi),
            "max_tilt_deg": float(tilt.max()),
            "non_fall": bool(tilt.max() < 35.0),
        },
    }


EPISODES = [
    dict(clip="walk_sideway_045_stop_005__A042_M", category="Test-Repetition (ID)",
         session="streamed_092422", offset=5547, nframes=277,
         run="g1_deploy_run_09042026_run5", t0_ms=127560.0, t1_ms=133100.0),
    dict(clip="walk_sideway_045_stop_005__A042_M", category="Test-Repetition (ID)",
         session="streamed_092026", offset=104, nframes=277,
         run="g1_deploy_run_09042026_run5", t0_ms=38520.0, t1_ms=44060.0),
    dict(clip="walk_180_R_003__A332_M", category="Test-Repetition (ID)",
         session="streamed_091340", offset=105, nframes=560,
         run="g1_deploy_run_09042026_run3", t0_ms=18020.0, t1_ms=29220.0),
    dict(clip="walk_180_R_003__A332_M", category="Test-Repetition (ID)",
         session="streamed_091541", offset=104, nframes=560,
         run="g1_deploy_run_09042026_run4", t0_ms=37200.0, t1_ms=48400.0),
    dict(clip="walk_180_R_003__A332_M", category="Test-Repetition (ID)",
         session="streamed_091118", offset=105, nframes=560,
         run="g1_deploy_run_09042026_run2", t0_ms=20500.0, t1_ms=31700.0),
]


def fmt(r):
    im, c, s = r["imitation"], r["cmd"], r["smpl_diag"]
    rt, rb = r["retarget_diag"], r["robustness"]
    stress = "  ".join(k.replace("_joint", "") + "=%.2f" % v
                       for k, v in im["stress_joints_deg"].items())
    return (
        "\n### %s   [%s / %s]\n" % (r["clip"], r["run"], r["session"]) +
        "    %d frames (%.2f s), ref %d fr @%gfps\n"
        % (r["frames"], r["duration_s"], r["ref_frames"], r["ref_fps"]) +
        "  == PRIMARY: imitation vs retargeted G1 reference ==\n" +
        "     mpjpe_l %7.2f mm   mpjpe_pa %6.2f mm   vel_dist %6.2f   accel_dist %6.2f\n"
        % (im["mpjpe_l"], im["mpjpe_pa"], im["vel_dist"], im["accel_dist"]) +
        "     legs %6.2f  vr_3points %6.2f  upper %6.2f  foot %6.2f\n"
        % (im["mpjpe_l_legs"], im["mpjpe_l_vr_3points"],
           im["mpjpe_l_other_upper_bodies"], im["mpjpe_l_foot"]) +
        "     joint err %.2f deg (worst %s %.2f)\n"
        % (im["joint_err_deg_mean"], im["joint_err_deg_worst"][0],
           im["joint_err_deg_worst"][1]) +
        "     stress: %s\n" % stress +
        "     heading err %.2f deg (ref turns %.0f deg, corr %+.2f, conv %s)\n"
        % (im["heading_err_deg"], im["ref_total_turn_deg"], im["heading_corr"],
           im["quat_convention"]) +
        "  -- control-loop fidelity: FK(q_target) vs FK(q) --\n" +
        "     mpjpe_l %7.2f mm   joint err %.2f deg (worst %s %.2f)\n"
        % (c["mpjpe_l"], c["joint_err_deg_mean"], c["joint_err_deg_worst"][0],
           c["joint_err_deg_worst"][1]) +
        "  -- diagnostics vs raw SMPL (morphology-contaminated, NOT sim-comparable) --\n" +
        "     robot-vs-SMPL %7.2f mm (pa %6.2f)   retargeting alone %7.2f mm (pa %6.2f)\n"
        % (s["mpjpe_l"], s["mpjpe_pa"], rt["mpjpe_l"], rt["mpjpe_pa"]) +
        "  -- robustness --\n" +
        "     saturation mean %.2f%%  worst %s %.2f%%\n"
        % (rb["saturation_mean"] * 100, rb["saturation_worst"][0],
           rb["saturation_worst"][1] * 100) +
        "     shaking %.4f rad/s   swing r/p %.2f/%.2f deg   max tilt %.2f deg   non_fall=%s\n"
        % (rb["shaking_rms_rad_s"], rb["swing_roll_deg_rms"],
           rb["swing_pitch_deg_rms"], rb["max_tilt_deg"], rb["non_fall"])
    )


def aggregate(results):
    """Per-clip mean +/- std of the primary metric (plan section 5)."""
    out = {}
    for clip in sorted({r["clip"] for r in results}):
        rs = [r for r in results if r["clip"] == clip]
        entry = {
            "n_episodes": len(rs),
            "non_fall_rate": float(np.mean([r["robustness"]["non_fall"] for r in rs])),
        }
        for k in ("mpjpe_l", "mpjpe_pa", "vel_dist", "accel_dist", "joint_err_deg_mean"):
            vals = [r["imitation"][k] for r in rs]
            entry[k] = [float(np.mean(vals)), float(np.std(vals))]
        out[clip] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="/home/grease/g1_robot_data/20260905")
    ap.add_argument("--runs", default="/home/grease/g1_robot_data/g1_run_0905")
    ap.add_argument("--robot-ref", default="/home/grease/ego_dataset/eval_subset/robot")
    ap.add_argument("--episodes", default=None,
                    help="JSON file with an episode list; defaults to the built-in "
                         "EPISODES (the 2026-09-04 session).")
    ap.add_argument("--out", default="sim2real/online_eval_results.json")
    args = ap.parse_args()

    episodes = EPISODES
    if args.episodes:
        with open(args.episodes) as fh:
            episodes = json.load(fh)
        print("[episodes] %d loaded from %s" % (len(episodes), args.episodes))

    fk = FK()
    results = [evaluate(ep, fk, args.sessions, args.runs, args.robot_ref)
               for ep in episodes]
    for r in results:
        print(fmt(r))

    agg = aggregate(results)
    print("\n=== per-clip aggregate (primary imitation metric, mean +/- std) ===")
    for clip, a in agg.items():
        print("  %s  (n=%d, non-fall %.0f%%)"
              % (clip, a["n_episodes"], a["non_fall_rate"] * 100))
        print("     mpjpe_l  %6.2f +/- %5.2f mm    mpjpe_pa %6.2f +/- %5.2f mm"
              % (a["mpjpe_l"][0], a["mpjpe_l"][1], a["mpjpe_pa"][0], a["mpjpe_pa"][1]))
        print("     vel_dist %6.2f +/- %5.2f      joint err %5.2f +/- %4.2f deg"
              % (a["vel_dist"][0], a["vel_dist"][1],
                 a["joint_err_deg_mean"][0], a["joint_err_deg_mean"][1]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"episodes": results, "aggregate": agg}, fh, indent=2, default=str)
    print("\n[saved] " + args.out)


if __name__ == "__main__":
    main()
