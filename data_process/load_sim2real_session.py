"""
Load an Aug-6 / Aug-11 sim2real session (PICO human capture + robot state_logger
CSV bundle) onto a common absolute wall-clock time base, and cache the result.

Both sides carry wall-clock timestamps:
  - PICO  `pose_*.npz`      -> `timestamp_realtime`  (seconds or ms since epoch)
  - robot `state_logger` CSV -> `time_realtime_ms`   (column index 2)

so they can be aligned on absolute time rather than frame index. This matters:
the human stream STARTS BEFORE the robot in both sessions (+1.99 s on Aug 6,
+96 s on Aug 11), so index 0 != index 0.

See `model_eval/SIM2REAL_GAP_STUDY_PLAN.md` §5 for the alignment pitfalls this
module deliberately avoids (clock drift, assumed frame rates, time-warping).

Usage:
  from data_process.load_sim2real_session import load_session
  S = load_session('aug11')
  S['robot']['q']          # (T,29) measured joint position
  S['robot']['t']          # (T,)   absolute epoch seconds
  S['human']['smpl_joints']# (N,24,3)
  S['human']['t']          # (N,)   absolute epoch seconds
  S['mode2']               # (t0,t1) absolute-time Mode-2 window
  S['overlap']             # (t0,t1) absolute-time human/robot overlap
"""
import csv
import glob
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, 'sim2real', 'cache')

SESSIONS = {
    'aug6': dict(
        human='paired_smpl_g1_deploy',
        robot='/home/grease/g1_robot_data/g1_deploy_run',
    ),
    'aug11': dict(
        human='logs/smpl_raw_real_robot',
        robot='/home/grease/g1_robot_data/g1_real_deploy_logs',
    ),
}

# state_logger CSVs: cols 0..4 = index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp
_TCOL, _DATA0 = 2, 5

ROBOT_SIGNALS = {
    'q': 29, 'dq': 29, 'action': 29, 'motor_torque': 29,
    'base_quat': 4, 'base_ang_vel': 3, 'encoder_mode': 1, 'token_state': 64,
}


def _read_csv(path, ncol):
    """Robust CSV read: skips malformed/truncated rows (the recorder is killed
    mid-write, so the tail of some files is ragged)."""
    t, v = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < _DATA0 + ncol:
                continue
            try:
                ts = float(row[_TCOL]) / 1000.0
                vals = [float(x) for x in row[_DATA0:_DATA0 + ncol]]
            except ValueError:
                continue
            t.append(ts)
            v.append(vals)
    return np.asarray(t), np.asarray(v)


def _load_robot(d):
    out = {}
    base_t = None
    for name, n in ROBOT_SIGNALS.items():
        p = os.path.join(d, name + '.csv')
        if not os.path.exists(p):
            continue
        t, v = _read_csv(p, n)
        if base_t is None:
            base_t = t
        # each signal is written by its own sink and can be truncated at a
        # slightly different row -> resample everything onto base_t
        if len(t) != len(base_t) or not np.array_equal(t, base_t):
            v = np.stack([np.interp(base_t, t, v[:, j]) for j in range(v.shape[1])], 1)
        out[name] = v
    out['t'] = base_t
    return out


def _load_human(d):
    fs = sorted(glob.glob(os.path.join(REPO, d, 'pose_*.npz')))
    if not fs:
        fs = sorted(glob.glob(os.path.join(d, 'pose_*.npz')))
    t, J, P, Q = [], [], [], []
    for f in fs:
        z = np.load(f)
        t.append(float(z['timestamp_realtime'][0]))
        J.append(z['smpl_joints'][0])
        P.append(z['smpl_pose'][0])
        Q.append(z['body_quat_w'][0])
    t = np.asarray(t, dtype=np.float64)
    if t[0] > 1e12:          # stored in ms
        t /= 1000.0
    return dict(t=t, smpl_joints=np.asarray(J), smpl_pose=np.asarray(P),
                body_quat_w=np.asarray(Q))


def load_session(name, use_cache=True):
    cfg = SESSIONS[name]
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f'{name}.npz')

    if use_cache and os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        robot = {k[2:]: z[k] for k in z.files if k.startswith('r_')}
        human = {k[2:]: z[k] for k in z.files if k.startswith('h_')}
    else:
        robot = _load_robot(cfg['robot'])
        human = _load_human(cfg['human'])
        np.savez_compressed(cp,
                            **{f'r_{k}': v for k, v in robot.items()},
                            **{f'h_{k}': v for k, v in human.items()})

    em = robot['encoder_mode'][:, 0]
    idx = np.where(em == 2)[0]
    mode2 = (robot['t'][idx[0]], robot['t'][idx[-1]]) if len(idx) else (None, None)
    overlap = (max(human['t'][0], mode2[0]), min(human['t'][-1], mode2[1]))

    return dict(name=name, robot=robot, human=human, mode2=mode2, overlap=overlap)


if __name__ == '__main__':
    import datetime as dt
    import sys
    for s in (sys.argv[1:] or ['aug11']):
        S = load_session(s)
        r, h = S['robot'], S['human']
        f = lambda x: dt.datetime.fromtimestamp(x).strftime('%H:%M:%S')
        print(f"=== {s} ===")
        print(f"  robot  {len(r['t']):6d} fr  {f(r['t'][0])}-{f(r['t'][-1])}  "
              f"{r['t'][-1]-r['t'][0]:.1f}s  {len(r['t'])/(r['t'][-1]-r['t'][0]):.1f} Hz")
        print(f"  human  {len(h['t']):6d} fr  {f(h['t'][0])}-{f(h['t'][-1])}  "
              f"{h['t'][-1]-h['t'][0]:.1f}s  {len(h['t'])/(h['t'][-1]-h['t'][0]):.1f} Hz")
        print(f"  mode2  {f(S['mode2'][0])}-{f(S['mode2'][1])}  "
              f"{S['mode2'][1]-S['mode2'][0]:.1f}s")
        print(f"  usable {f(S['overlap'][0])}-{f(S['overlap'][1])}  "
              f"{S['overlap'][1]-S['overlap'][0]:.1f}s  <- Mode-2 AND human present")
