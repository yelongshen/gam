"""
Parse a g1_deploy_onnx_ref CSV log directory into simulation-evaluation metrics
(as defined in EVAL_METRICS_DRAFT.md, Section 1.B).

Metrics per run/clip:
  n_frames            number of control steps logged
  duration_s          n_frames * dt
  track_mae_rad       mean |action - q| over joints/time  (policy target vs achieved pose)
  track_mae_deg       same in degrees
  root_angvel_mean    mean |base angular velocity| (rad/s)  -> spinning/instability
  max_tilt_deg        max deviation of base up-axis from world-up (fall indicator)
  final_tilt_deg      tilt at the last frame
  non_fall            1 if final_tilt < fall_thresh and max_tilt < 90 else 0
  action_jerk         std of 3rd time-derivative of action (normalized smoothness)
  mean_torque_Nm      mean |joint torque|
  peak_torque_Nm      max |joint torque|

Usage (standalone on any run):
  .venv_sim/bin/python sim_eval_metrics.py gear_sonic_deploy/logs/sim_deploy_run
"""
import os
import sys
import json
import numpy as np


def _load_csv(path):
    """Load a deploy CSV; return (data[T, C]) of the trailing numeric columns.
    The first 5 columns are index,time_ms,time_realtime_ms,time_monotonic_ms,
    ros_timestamp -> we drop them and keep the payload columns."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path) as f:
        header = f.readline().strip().split(',')
    ncol = len(header)
    if ncol <= 5:
        return None
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if data.size == 0:
        return None
    return data[:, 5:]  # drop the 5 timing columns


def _quat_tilt_deg(quat):
    """quat: (T,4) [w,x,y,z]. Return tilt angle (deg) of body up-axis vs world up."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # world-up (0,0,1) rotated into body frame -> body up-axis z-column of R
    # R[:,2] = (2(xz+wy), 2(yz-wx), 1-2(x^2+y^2))
    up_z = 1 - 2 * (x**2 + y**2)
    up_z = np.clip(up_z, -1.0, 1.0)
    return np.degrees(np.arccos(up_z))


def compute_metrics(log_dir, dt=None, fall_thresh_deg=45.0):
    meta_path = os.path.join(log_dir, 'metadata.json')
    if dt is None:
        dt = 0.02
        if os.path.exists(meta_path):
            try:
                dt = json.load(open(meta_path))['logging']['dt']
            except Exception:
                pass

    action = _load_csv(os.path.join(log_dir, 'action.csv'))
    q = _load_csv(os.path.join(log_dir, 'q.csv'))
    tau = _load_csv(os.path.join(log_dir, 'motor_torque.csv'))
    angvel = _load_csv(os.path.join(log_dir, 'base_ang_vel.csv'))
    quat = _load_csv(os.path.join(log_dir, 'base_quat.csv'))

    if q is None or q.shape[0] < 3:
        return None

    T = q.shape[0]
    m = {'n_frames': int(T), 'duration_s': round(T * dt, 2)}

    # tracking error: policy target action vs achieved q (align joint count)
    if action is not None:
        n = min(action.shape[1], q.shape[1], action.shape[0], q.shape[0])
        err = np.abs(action[:n, :n] - q[:n, :n])
        m['track_mae_rad'] = float(err.mean())
        m['track_mae_deg'] = float(np.degrees(err.mean()))
    else:
        m['track_mae_rad'] = m['track_mae_deg'] = float('nan')

    # root angular velocity magnitude
    if angvel is not None:
        m['root_angvel_mean'] = float(np.linalg.norm(angvel, axis=1).mean())
    else:
        m['root_angvel_mean'] = float('nan')

    # fall / tilt
    if quat is not None and quat.shape[1] >= 4:
        tilt = _quat_tilt_deg(quat)
        m['max_tilt_deg'] = float(tilt.max())
        m['final_tilt_deg'] = float(tilt[-1])
        m['non_fall'] = int(tilt[-1] < fall_thresh_deg and tilt.max() < 90.0)
    else:
        m['max_tilt_deg'] = m['final_tilt_deg'] = float('nan')
        m['non_fall'] = -1

    # action smoothness: normalized jerk (std of 3rd derivative)
    if action is not None and action.shape[0] >= 4:
        j = np.diff(action, n=3, axis=0) / (dt ** 3)
        m['action_jerk'] = float(np.linalg.norm(j, axis=1).std())
    else:
        m['action_jerk'] = float('nan')

    # torque
    if tau is not None:
        at = np.abs(tau)
        m['mean_torque_Nm'] = float(at.mean())
        m['peak_torque_Nm'] = float(at.max())
    else:
        m['mean_torque_Nm'] = m['peak_torque_Nm'] = float('nan')

    # --- validity guards -------------------------------------------------
    # The policy only tracks the streamed SMPL motion when it is in encoder
    # mode 2 AND the streamed motion is actually playing. If these are not
    # observed, the robot was idling / tracking the default reference motion
    # and the metrics are NOT a measurement of our clip.
    enc = _load_csv(os.path.join(log_dir, 'encoder_mode.csv'))
    play = _load_csv(os.path.join(log_dir, 'motion_playing.csv'))
    m['saw_mode2'] = int(enc is not None and bool((enc == 2).any()))
    m['saw_playing'] = int(play is not None and bool((play >= 1).any()))
    m['tracked_frac'] = float(((enc == 2).mean()) if enc is not None else 0.0)

    # The simulator is NOT reset between clips, so the robot can begin an
    # episode already lying on the floor from a previous failure. Measure the
    # settle phase (before tracking starts): if the robot was not standing
    # upright there, the episode says nothing about tracking quality.
    m['start_tilt_deg'] = float('nan')
    m['settle_max_tilt_deg'] = float('nan')
    m['clean_start'] = 0
    if quat is not None and quat.shape[1] >= 4:
        tilt = _quat_tilt_deg(quat)
        m['start_tilt_deg'] = float(tilt[0])
        if enc is not None and bool((enc == 2).any()):
            ts = int(np.argmax(enc == 2))
            if ts > 0:
                m['settle_max_tilt_deg'] = float(tilt[:ts].max())
                m['clean_start'] = int(tilt[:ts].max() < 20.0)
        else:
            m['settle_max_tilt_deg'] = float(tilt[0])
            m['clean_start'] = int(tilt[0] < 20.0)

    m['valid'] = int(m['saw_mode2'] and m['saw_playing'] and m['clean_start'])

    return m


def main():
    if len(sys.argv) < 2:
        print("usage: sim_eval_metrics.py <log_dir>")
        sys.exit(1)
    d = sys.argv[1]
    m = compute_metrics(d)
    if m is None:
        print(f"No usable logs in {d}")
        sys.exit(2)
    print(f"Metrics for {d}:")
    for k, v in m.items():
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
