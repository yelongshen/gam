"""recover_transl.py — estimate the missing root translation in PICO captures
from foot-contact odometry, and validate it against ground truth.

The problem
-----------
The PICO capture stores `smpl_joints` pelvis-pinned and has no world pelvis
position, so `transl` is all-zero. After retargeting that becomes
`root_trans_offset ~ 0`, i.e. every clip is "in place" (measured: 0.01 m median
displacement vs 0.26 m for `eval_subset`).

The idea
--------
A planted foot does not move in the world. So if we know the pose (we do), the
root displacement between consecutive frames is the NEGATIVE of the apparent
displacement of whichever foot is currently planted:

    droot[t] = -(p_stance[t] - p_stance[t-1])

Stance is chosen per frame as the foot that is both LOW (near the floor) and
SLOW (small apparent velocity). Blending the two feet by a soft contact weight
avoids a discontinuity at each stance switch.

Validation
----------
`eval_subset/smpl` HAS real `transl`, so running the estimator on those clips
and comparing gives an honest error number. Run with `--validate` to get it.
This is deliberately the first thing to do: if the estimator cannot reproduce
known translation, it should not be applied to the PICO clips.

Usage
-----
    # honest accuracy check against ground truth (do this first)
    .venv_sim/bin/python data_process/recover_transl.py --validate --n 40

    # single clip, verbose
    .venv_sim/bin/python data_process/recover_transl.py \
        --clip ~/ego_dataset/eval_subset/smpl/walk_180_R_003__A332_M.pkl
"""
import argparse
import glob
import os

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R

LANK, RANK, LTOE, RTOE = 7, 8, 10, 11
CONTACT_PAIRS = [(LANK, LTOE), (RANK, RTOE)]


def world_joints(smpl_joints, pose_aa):
    """Pelvis-pinned joints -> world-ORIENTED (still pelvis-pinned) joints.

    `smpl_filtered` keeps the root rotation in `pose_aa[:, :3]` and pins only the
    root translation, so the stored joints already carry world orientation. PICO
    raw captures are de-rotated instead and need `body_quat_w` re-applied; the
    caller is responsible for handing us consistently-oriented joints.
    """
    return np.asarray(smpl_joints, dtype=np.float64)


def estimate_transl(J, fps, floor_pct=10.0, vel_ref=0.35, height_ref=0.06):
    """Foot-contact odometry.

    Args:
        J: (T,24,3) pelvis-pinned, world-ORIENTED joints.
        fps: sample rate.
        floor_pct: percentile of foot height used as the floor.
        vel_ref / height_ref: soft-contact scales (m/s, m).

    Returns:
        transl (T,3) with transl[0] = 0, and the per-frame contact weights.
    """
    T = len(J)
    dt = 1.0 / fps
    foot = np.stack([J[:, [a, b]].mean(axis=1) for a, b in CONTACT_PAIRS], axis=1)  # (T,2,3)

    floor = np.percentile(foot[..., 2], floor_pct)
    height = foot[..., 2] - floor                                   # (T,2)

    vel = np.zeros_like(foot)
    vel[1:] = (foot[1:] - foot[:-1]) / dt
    speed = np.linalg.norm(vel[..., :2], axis=-1)                   # horizontal only

    # Soft contact weight: low AND slow -> planted.
    w = np.exp(-(height / height_ref) ** 2) * np.exp(-(speed / vel_ref) ** 2)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-6, None)

    # Root displacement = -(weighted mean apparent foot displacement)
    dfoot = np.zeros_like(foot)
    dfoot[1:] = foot[1:] - foot[:-1]
    droot = -(w[..., None] * dfoot).sum(axis=1)                     # (T,3)
    droot[:, 2] = 0.0                                               # height handled separately

    transl = np.cumsum(droot, axis=0)
    transl -= transl[0]
    return transl, w


def validate(n, seed=0):
    """Compare the estimate against ground truth.

    IMPORTANT -- coordinate conventions differ *within* a single smpl_filtered
    file, which is easy to get wrong:

        smpl_joints : Z-up  (head-foot offset lands in axis 2; horizontal = x,y)
        transl      : Y-up  (axis 1 is a constant ~1.25 m = body height;
                             travel appears in x,z)

    The estimator runs in `smpl_joints` space, so the correct reference is the
    ROBOT `root_trans_offset`, which is also Z-up (height ~0.79 m in axis 2).
    Comparing against `transl[:, :2]` silently measures horizontal-vs-vertical
    and makes a 7 m walk look like 0.19 m.

    The robot is a retarget of the human, so its travel is scaled by roughly the
    leg-length ratio (~0.78). We therefore report the est/true RATIO and the
    correlation, which are scale-robust, alongside the raw error.
    """
    fs = sorted(glob.glob(os.path.expanduser('~/ego_dataset/eval_subset/smpl/*.pkl')))
    rng = np.random.default_rng(seed)
    fs = [fs[i] for i in rng.permutation(len(fs))[:n]]
    rows = []
    for f in fs:
        name = os.path.basename(f)[:-4]
        rf = f.replace('/smpl/', '/robot/')
        if not os.path.exists(rf):
            continue
        d = joblib.load(f)
        r = joblib.load(rf)
        r = r if 'dof' in r else list(r.values())[0]
        J = np.asarray(d['smpl_joints'], dtype=np.float64)
        fps = float(d['fps'])
        gt = np.asarray(r['root_trans_offset'], dtype=np.float64)     # Z-up
        gt = gt - gt[0]
        est, _ = estimate_transl(J, fps)                              # Z-up

        # robot runs at its own fps; compare net displacement + path length only
        true_d = np.linalg.norm(gt[-1, :2])
        true_path = np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1).sum()
        est_d = np.linalg.norm(est[-1, :2])
        est_path = np.linalg.norm(np.diff(est[:, :2], axis=0), axis=1).sum()
        rows.append((name, len(J) / fps, true_d, est_d, true_path, est_path))
    rows.sort(key=lambda r: -r[2])

    print(f"{'clip':42s} {'dur':>5s} {'true_d':>7s} {'est_d':>7s} "
          f"{'true_path':>10s} {'est_path':>9s} {'ratio':>6s}")
    for r in rows[:22]:
        ratio = r[3] / r[2] if r[2] > 1e-6 else float('nan')
        print(f"{r[0][:42]:42s} {r[1]:5.1f} {r[2]:7.2f} {r[3]:7.2f} "
              f"{r[4]:10.2f} {r[5]:9.2f} {ratio:6.2f}")

    a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
    mv = a[a[:, 0] > 0.5]
    print(f"\n{len(rows)} clips, {len(mv)} with true displacement > 0.5 m")
    if len(mv):
        rd = mv[:, 1] / mv[:, 0]
        rp = mv[:, 3] / np.clip(mv[:, 2], 1e-6, None)
        print(f"  net-displacement ratio est/true : p50 {np.median(rd):.2f}  "
              f"p10 {np.percentile(rd, 10):.2f}  p90 {np.percentile(rd, 90):.2f}")
        print(f"  path-length      ratio est/true : p50 {np.median(rp):.2f}  "
              f"p10 {np.percentile(rp, 10):.2f}  p90 {np.percentile(rp, 90):.2f}")
        print(f"  corr(true_d, est_d)             : {np.corrcoef(mv[:, 0], mv[:, 1])[0, 1]:+.2f}")
        print(f"  corr(true_path, est_path)       : {np.corrcoef(mv[:, 2], mv[:, 3])[0, 1]:+.2f}")
    st = a[a[:, 0] <= 0.5]
    if len(st):
        print(f"  {len(st)} near-static clips: est displacement p50 {np.median(st[:, 1]):.2f} m  "
              f"p90 {np.percentile(st[:, 1], 90):.2f} m   (should be ~0 = false-motion rate)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--clip')
    args = ap.parse_args()

    if args.validate:
        validate(args.n)
        return
    if not args.clip:
        raise SystemExit('pass --validate or --clip')

    d = joblib.load(os.path.expanduser(args.clip))
    J = np.asarray(d['smpl_joints'], dtype=np.float64)
    fps = float(d['fps'])
    est, w = estimate_transl(J, fps)
    print(f"{os.path.basename(args.clip)}  T={len(J)} @ {fps:g}fps")
    print(f"  estimated net displacement : {np.linalg.norm(est[-1, :2]):.2f} m")
    print(f"  estimated path length      : {np.linalg.norm(np.diff(est[:, :2], axis=0), axis=1).sum():.2f} m")
    if 'transl' in d and not np.allclose(d['transl'], 0):
        gt = np.asarray(d['transl']) - np.asarray(d['transl'])[0]
        print(f"  TRUE net displacement      : {np.linalg.norm(gt[-1, :2]):.2f} m")


if __name__ == '__main__':
    main()
