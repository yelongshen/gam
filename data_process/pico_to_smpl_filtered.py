"""
Convert a raw PICO VR SMPL-streaming capture (a directory of 4-frame
`pose_*.npz` chunks, e.g. `logs/yelong_cliptest_0/`, `paired_smpl_raw/`,
`logs/smpl_raw_real_robot/`) into a single `smpl_filtered`-style `.pkl` clip,
matching the schema documented in `SMPL_FILTERED_DATA_FORMAT.md`:

    pose_aa           (T, 72)  float32  SMPL pose, 24 joints x 3 axis-angle
    transl            (T, 3)   float32  global root translation
    smpl_joints       (T, 24, 3) float32  root-local joint positions (Z-up)
    fps               scalar   float    target rate (default 50.0)
    original_pose_aa  (T0, 72) float32  pre-resample poses
    original_fps      scalar   float    source rate (~89-90 Hz for PICO)

Field provenance (see raw npz schema, confirmed by inspection):

    smpl_pose    (4, 21, 3)  body-joint axis-angle, no root  (per-chunk, we use frame 0)
    body_quat_w  (4, 4)      global root orientation quaternion, [w, x, y, z]
    smpl_joints  (4, 24, 3)  ALREADY root-local, Z-up joint positions -- this
                             matches the target `smpl_joints` convention
                             directly, so we reuse it as-is (resampled) instead
                             of re-deriving it via forward kinematics.
    vr_position  (9,)        3 anchor points (head, L-hand, R-hand), xyz each

pose_aa construction:
    pose_aa[:, 0:3]   = rotvec(body_quat_w)            (root / global_orient)
    pose_aa[:, 3:66]  = smpl_pose.reshape(T, 63)       (21 body joints)
    pose_aa[:, 66:72] = 0                               (2 hand joints, untracked)

transl: the raw capture does not store a per-frame pelvis/root world
position directly (only head/hand VR anchors). By default we emit
`transl = 0` for every frame (safe, but loses world-frame locomotion --
fine for in-place / stationary clips). Pass `--transl_mode head` to use
the head anchor from `vr_position` as an approximate root trajectory
(rough placeholder, not verified against ground truth).

Usage:
  .venv_sim/bin/python data_process/pico_to_smpl_filtered.py \\
      --dir logs/yelong_cliptest_0 --out /tmp/yelong_cliptest_0.pkl

  # convert only a frame sub-range (array-index based, like detect_action_clips.py):
  .venv_sim/bin/python data_process/pico_to_smpl_filtered.py \\
      --dir paired_smpl_raw --start 3200 --end 3900 \\
      --out logs/paired_smpl_raw_walk0.pkl
"""
import argparse
import glob
import os

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from scipy.spatial.transform import Slerp

# SMPL-X rest pelvis position J[0]; identical to
# `normalize_split_test.PELVIS_OFFSET`. Every smpl_filtered clip pins its root
# TRANSLATION here (root rotation is kept), so a correct conversion lands the
# pelvis exactly on this constant for every frame.
PELVIS_OFFSET = np.array([0.003, -0.351, 0.012])


def load_pico_sequence(d, start=None, end=None):
    """Load frame-0-of-each-4-frame-chunk arrays needed for conversion, same
    convention as `detect_action_clips.load_sequence` / `visualize_pico.py`."""
    files = sorted(glob.glob(os.path.join(d, "pose_*.npz")))
    if not files:
        raise SystemExit(f"no pose_*.npz in {d}")
    if start is not None or end is not None:
        files = files[start:end]
        if not files:
            raise SystemExit(f"empty frame range [{start},{end}) for {d}")

    smpl_pose = []
    body_quat_w = []
    smpl_joints = []
    vr_position = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        smpl_pose.append(z['smpl_pose'][0])        # (21, 3)
        body_quat_w.append(z['body_quat_w'][0])     # (4,)  [w, x, y, z]
        smpl_joints.append(z['smpl_joints'][0])     # (24, 3)
        vr_position.append(z['vr_position'])        # (9,)
    fps = float(np.load(files[0])['pico_fps'][0])

    return dict(
        smpl_pose=np.asarray(smpl_pose, dtype=np.float64),
        body_quat_w=np.asarray(body_quat_w, dtype=np.float64),
        smpl_joints=np.asarray(smpl_joints, dtype=np.float32),
        vr_position=np.asarray(vr_position, dtype=np.float64),
        fps=fps,
        n=len(files),
    )


def load_pico_clip_npz(path, start=None, end=None):
    """Load a CONSOLIDATED clip .npz (produced by
    `gear_sonic/scripts/chunk_pico_session.py`), where every key is already
    stacked over time as (T, ...) instead of a directory of 4-frame chunks.

    Unlike the legacy chunked captures, these clips carry `body_pos_w` -- the
    real world root position recorded via `--record_root_pos` -- which is what
    finally makes a non-zero `transl` possible.
    """
    z = np.load(path, allow_pickle=True)
    sl = slice(start, end)
    # FRAME RATE.  A consolidated clip keeps ONE frame per recorded 4-frame message (the newest), and messages are written
    # at ~50 Hz.  `pico_fps` (~86-90 Hz) is the SDK capture rate, NOT the spacing of these frames: using it made every
    # clip play ~1.75-1.9x too fast (measured on 0928/0929/0930/1002: dataset duration 0.49-0.55x the real one).  The real
    # spacing is in `timestamp_monotonic`, so the rate is measured from it.
    pico_fps = float(np.asarray(z['pico_fps']).reshape(-1)[0])
    fps = pico_fps
    if 'timestamp_monotonic' in z.files:
        ts = np.asarray(z['timestamp_monotonic'], dtype=np.float64).reshape(-1)[sl]
        if len(ts) >= 2 and ts[-1] > ts[0]:
            meas = (len(ts) - 1) / (ts[-1] - ts[0])
            if 20.0 <= meas <= 120.0:
                fps = float(meas)
    if abs(fps - pico_fps) > 1.0:
        print(f"  [fps] measured {fps:.2f} Hz from timestamps (pico_fps field says {pico_fps:.2f}) -> using {fps:.2f}")
    out = dict(
        smpl_pose=np.asarray(z['smpl_pose'][sl], dtype=np.float64),
        body_quat_w=np.asarray(z['body_quat_w'][sl], dtype=np.float64),
        smpl_joints=np.asarray(z['smpl_joints'][sl], dtype=np.float32),
        vr_position=np.asarray(z['vr_position'][sl], dtype=np.float64),
        fps=fps,
    )
    out['body_pos_w'] = (np.asarray(z['body_pos_w'][sl], dtype=np.float64)
                         if 'body_pos_w' in z.files else None)
    out['n'] = out['smpl_pose'].shape[0]
    return out


def yup_to_zup(p):
    """PICO Y-up (x, y, z) -> Z-up (x, -z, y), preserving handedness."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=-1)


def build_pose_aa(smpl_pose, body_quat_w):
    """smpl_pose: (T,21,3) axis-angle, body_quat_w: (T,4) [w,x,y,z] quat
    -> pose_aa (T,72): root(3) + 21 body joints(63) + 2 hand joints(6, zero).

    IMPORTANT: `body_quat_w` in the raw PICO npz is NOT a raw/unprocessed SMPL
    Y-up root rotation -- it is already the fully-processed result of the
    live-capture pipeline (`pico_manager_thread_server.py::process_smpl_joints`):
    Y-up->Z-up (`smpl_root_ytoz_up`) followed by base-rest-pose removal
    (`remove_smpl_base_rot`). `stream_clip_mode2.py`'s `.pkl` loader, however,
    expects `pose_aa[:, :3]` to be a RAW/unprocessed SMPL root and applies
    that exact same Y-up->Z-up + base-rot-removal itself
    (`official_root_quat_w`) before using it. Storing `body_quat_w` verbatim
    as `pose_aa[:, :3]` therefore gets the transform applied TWICE.

    Fix: invert the transform here so that
    official_root_quat_w(pose_aa[:, :3]) == body_quat_w exactly, i.e.

        quat(root_aa) = conj(YTOZ) * body_quat_w * BASE_ROT

    (using the same w-first quaternion convention / constants as
    `stream_clip_mode2._YTOZ` / `_BASE_CONJ`)."""
    T = smpl_pose.shape[0]

    # Same constants as stream_clip_mode2.py (_YTOZ, _BASE_CONJ), w-first.
    YTOZ = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # aa[pi/2,0,0]
    BASE_ROT = np.array([0.5, 0.5, 0.5, 0.5])                          # conj(_BASE_CONJ)
    conj_YTOZ = YTOZ * np.array([1.0, -1.0, -1.0, -1.0])

    def qmul(a, b):
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], -1)

    q = qmul(np.broadcast_to(conj_YTOZ, (T, 4)), body_quat_w)
    q = qmul(q, np.broadcast_to(BASE_ROT, (T, 4)))
    root_quat_xyzw = q[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w] for scipy
    root_rotvec = sRot.from_quat(root_quat_xyzw).as_rotvec()  # (T, 3)

    pose_aa = np.zeros((T, 72), dtype=np.float64)
    pose_aa[:, 0:3] = root_rotvec
    pose_aa[:, 3:66] = smpl_pose.reshape(T, 63)
    # pose_aa[:, 66:72] left as zero: SMPL-X joints 22/23 (jaw/eyes) are never
    # tracked by PICO and are zero-filled by compute_human_joints() too.
    return pose_aa


def build_smpl_joints_for_pkl(raw_smpl_joints, body_quat_w, pelvis_offset=None):
    """Convert the raw PICO `smpl_joints` into the `smpl_filtered` convention.

    Ground truth (traced through the actual capture code):
      `pico_manager_thread_server.py` records `smpl_joints` =
      `process_smpl_joints()["smpl_joints_local"]`, which is

          joints_local = quat_apply(quat_inv(global_orient_quat), joints)

      where `joints` comes from `compute_human_joints()` (SMPL-X FK, pelvis
      left at the constant rest position J[0]) and `global_orient_quat` is the
      Z-up + base-rot-removed root quat that is ALSO recorded as `body_quat_w`.

    `smpl_filtered` clips instead store joints with the root ROTATION still
    baked in and only the root TRANSLATION pinned to `PELVIS_OFFSET`
    (see `normalize_split_test.to_local_zup`). So we simply undo the
    de-rotation:

          pkl_joints = quat_apply(body_quat_w, raw_smpl_joints)

    This lands the pelvis exactly on PELVIS_OFFSET = [0.003, -0.351, 0.012]
    automatically (verified numerically: std == 0 across all frames), because
    SMPL-X's rest pelvis J[0] IS that constant.

    `stream_clip_mode2.official_encoder_joints()` then re-applies the
    de-rotation at stream time, exactly reproducing the raw PICO joints.
    """
    def qapply(q, v):
        w = q[..., 0:1]
        u = q[..., 1:]
        return v + 2 * np.cross(u, np.cross(u, v) + w * v)

    q = np.repeat(body_quat_w[:, None, :], raw_smpl_joints.shape[1], axis=1)
    joints = qapply(q, raw_smpl_joints)

    if pelvis_offset is not None:
        # Pin the root TRANSLATION only (root rotation stays baked in), exactly
        # like `normalize_split_test.to_local_zup`. Re-pinning also removes the
        # ~5e-5 m chord-vs-arc drift introduced by linearly interpolating the
        # (rotating) raw joints during fps resampling.
        joints = joints - joints[:, 0:1, :] + pelvis_offset
    return joints


def build_transl(n, vr_position, mode='zero', body_pos_w=None, world_floor_z=None,
                 recenter_xy=True):
    if mode == 'zero':
        return np.zeros((n, 3), dtype=np.float64)
    if mode == 'head':
        # vr_position layout assumed [head_xyz, left_hand_xyz, right_hand_xyz].
        # Rough placeholder: use the head anchor, shifted down by an
        # approximate head-to-pelvis height offset. NOT verified against
        # ground truth -- use with caution for anything beyond quick checks.
        head_xyz = vr_position[:, 0:3].copy()
        head_xyz[:, 2] -= 0.6
        return head_xyz
    if mode == 'body_pos_w':
        # REAL root translation, recorded by pico_manager_thread_server.py
        # under --record_root_pos. This is the fix for the long-standing
        # "transl is entirely zero / every clip retargets in-place" problem
        # documented in PICO_TELEOP_RETARGETING_PIPELINE.md.
        if body_pos_w is None:
            raise ValueError("transl_mode='body_pos_w' but the clip has no body_pos_w")
        t = yup_to_zup(body_pos_w)
        if world_floor_z is not None:
            # Put the floor at z = 0 so pelvis height is physical (~0.9 m).
            t[:, 2] -= world_floor_z
        if recenter_xy:
            # Start each clip at the world origin in the horizontal plane;
            # the SHAPE of the trajectory (the locomotion) is preserved.
            t[:, 0:2] -= t[0, 0:2]
        return t
    raise ValueError(f"unknown transl_mode: {mode}")


def estimate_world_floor_z(smpl_joints_local, body_quat_w, body_pos_w,
                           foot_idx=(10, 11), pct=5.0):
    """Robust floor height (z) in the Z-up world frame.

    Reconstructs world joints as R(body_quat_w) * joints_local + zup(body_pos_w)
    and takes a low percentile of the lowest-foot height. The percentile (not
    the raw minimum) keeps a single foot-tracking spike from dragging the whole
    clip into the air -- the exact failure mode seen on clip_018.
    """
    def qapply(q, v):
        w = q[..., 0:1]
        u = q[..., 1:]
        return v + 2 * np.cross(u, np.cross(u, v) + w * v)

    q = np.repeat(body_quat_w[:, None, :], smpl_joints_local.shape[1], axis=1)
    world = qapply(q, np.asarray(smpl_joints_local, dtype=np.float64))
    world = world + yup_to_zup(body_pos_w)[:, None, :]
    low = world[:, list(foot_idx), 2].min(axis=1)
    return float(np.percentile(low, pct))


def floor_envelope_z(smpl_joints_local, body_quat_w, body_pos_w, fps, win_s=1.5,
                     foot_idx=(10, 11), stand_lo=0.88, stand_hi=1.00):
    """PER-FRAME floor height (z, Z-up world), (T,) array.

    `estimate_world_floor_z` returns ONE number per clip. That is wrong when the
    tracker's height scale differs between postures: in the 2026-10-02 sitting
    session the SEATED feet sit on the floor but the STANDING feet float 0.10-0.24 m
    above it (pelvis 1.07 m instead of 0.94 m, head-to-pelvis unchanged), so any
    single floor value leaves either the sit or the walk off the ground.

    A person always has at least one foot on the floor while walking, standing or
    sitting, so the lowest-foot height, followed by a rolling MINIMUM over `win_s`
    seconds (long enough to cover a stride) and then a moving average (so the correction
    ramps smoothly through the stand<->sit transitions), is the local floor. Subtracting
    it puts the supporting foot at z = 0 in every posture.  NOT suitable for clips with
    flight phases (jumps) -- there the envelope would follow the airborne feet.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    def qapply(q, v):
        w = q[..., 0:1]
        u = q[..., 1:]
        return v + 2 * np.cross(u, np.cross(u, v) + w * v)

    q = np.repeat(body_quat_w[:, None, :], smpl_joints_local.shape[1], axis=1)
    world = qapply(q, np.asarray(smpl_joints_local, dtype=np.float64))
    world = world + yup_to_zup(body_pos_w)[:, None, :]
    low = world[:, list(foot_idx), 2].min(axis=1)
    n = max(3, int(round(win_s * fps)))
    env = minimum_filter1d(low, size=n, mode='nearest')
    env = uniform_filter1d(env, size=n, mode='nearest')
    # Apply the envelope ONLY while STANDING / walking. While seated the feet may legitimately
    # dangle (a high stool): a rolling minimum would then follow the lifted feet and sink the whole
    # body (measured: pelvis 0.34 m instead of 0.57 m on two clips). Seated frames keep the plain
    # clip-level floor (p5), which is the old behaviour and is correct where the feet rest on it.
    p5 = float(np.percentile(low, 5.0))
    pel = uniform_filter1d(world[:, 0, 2] - p5, size=n, mode='nearest')
    w = np.clip((pel - stand_lo) / (stand_hi - stand_lo), 0.0, 1.0)
    return p5 + w * (env - p5)


def _resample_times(T, src_fps, tgt_fps):
    """Target sample times (in source-frame index space) for a linear/slerp
    resample of a length-T sequence from src_fps to tgt_fps."""
    if T < 2:
        return np.zeros(1), np.zeros(1, dtype=int), np.zeros(1, dtype=int), np.zeros(1)
    duration = (T - 1) / src_fps
    n_new = int(np.floor(duration * tgt_fps)) + 1
    times = np.arange(n_new) / tgt_fps
    times = times[times <= duration + 1e-9]
    frame_idx = times * src_fps
    idx0 = np.floor(frame_idx).astype(int)
    idx1 = np.minimum(idx0 + 1, T - 1)
    blend = frame_idx - idx0
    return frame_idx, idx0, idx1, blend


def resample_linear(arr, src_fps, tgt_fps):
    """Linearly resample the leading (time) axis of `arr`. Only valid for
    POSITIONS / translations -- never for rotations (see resample_slerp)."""
    T = arr.shape[0]
    if T < 2:
        return arr.copy()
    _, idx0, idx1, blend = _resample_times(T, src_fps, tgt_fps)
    orig_shape = arr.shape[1:]
    flat = arr.reshape(T, -1)
    out = (1 - blend[:, None]) * flat[idx0] + blend[:, None] * flat[idx1]
    return out.reshape(-1, *orig_shape)


def resample_rotvec_slerp(rotvec, src_fps, tgt_fps):
    """SLERP-resample axis-angle rotations of shape (T,3) or (T,J,3).

    CRITICAL: axis-angle must NEVER be linearly interpolated. The PICO root
    rotation sits near |aa| ~= 2.8 rad (pi = 3.14), so it repeatedly crosses
    the +/-pi wrap-around where the axis-angle representation flips sign --
    producing consecutive-frame jumps of ~2*pi. Linearly blending across those
    yields a completely wrong rotation (measured: 1.38 max quaternion error,
    0.88 m joint error) even though every individual frame is valid.
    """
    T = rotvec.shape[0]
    if T < 2:
        return rotvec.copy()
    frame_idx, _, _, _ = _resample_times(T, src_fps, tgt_fps)
    src_times = np.arange(T)

    if rotvec.ndim == 2:                       # (T, 3)
        slerp = Slerp(src_times, sRot.from_rotvec(rotvec))
        return slerp(frame_idx).as_rotvec()

    out = np.zeros((len(frame_idx), rotvec.shape[1], 3))
    for j in range(rotvec.shape[1]):           # (T, J, 3)
        slerp = Slerp(src_times, sRot.from_rotvec(rotvec[:, j]))
        out[:, j] = slerp(frame_idx).as_rotvec()
    return out


def resample_quat_slerp(quat_wfirst, src_fps, tgt_fps):
    """SLERP-resample (T,4) quaternions given/returned in [w,x,y,z] order."""
    T = quat_wfirst.shape[0]
    if T < 2:
        return quat_wfirst.copy()
    frame_idx, _, _, _ = _resample_times(T, src_fps, tgt_fps)
    rot = sRot.from_quat(quat_wfirst[:, [1, 2, 3, 0]])   # -> [x,y,z,w]
    out = Slerp(np.arange(T), rot)(frame_idx).as_quat()  # [x,y,z,w]
    return out[:, [3, 0, 1, 2]]                          # -> [w,x,y,z]


def convert_clip(src_dir, out_path, start=None, end=None, target_fps=50.0, transl_mode='zero',
                 verify=True, floor_mode='clip_p5', floor_win=1.5):
    if os.path.isfile(src_dir) and src_dir.endswith('.npz'):
        seq = load_pico_clip_npz(src_dir, start=start, end=end)
    else:
        seq = load_pico_sequence(src_dir, start=start, end=end)
    print(f"  loaded {seq['n']} frames @ {seq['fps']:.2f} fps from {src_dir}"
          + (f" [{start},{end})" if (start is not None or end is not None) else ""))

    original_pose_aa = build_pose_aa(seq['smpl_pose'], seq['body_quat_w']).astype(np.float32)
    original_fps = seq['fps']

    # ---- Resample the RAW quantities FIRST, then build the pkl from them ----
    # Rotations go through SLERP (axis-angle/quaternions cannot be linearly
    # blended); only positions/translations are linearly interpolated. Doing it
    # in this order keeps pose_aa, body_quat_w and smpl_joints mutually
    # consistent at the target fps.
    smpl_pose_rs   = resample_rotvec_slerp(seq['smpl_pose'], original_fps, target_fps)
    body_quat_rs   = resample_quat_slerp(seq['body_quat_w'], original_fps, target_fps)
    raw_joints_rs  = resample_linear(seq['smpl_joints'].astype(np.float64),
                                     original_fps, target_fps)

    # Resample in the RAW (de-rotated) frame and rotate afterwards. This is the
    # ordering that reproduces `pico_replay_server.py` to ~1e-7; interpolating
    # in the pelvis-pinned world frame instead drifts ~5e-4 m from it.
    pose_aa = build_pose_aa(smpl_pose_rs, body_quat_rs).astype(np.float32)
    # Re-pin the pelvis to PELVIS_OFFSET. Resampling ~87fps -> 50fps linearly
    # interpolates joints that are ROTATING, so the pelvis picks up a
    # chord-vs-arc error of up to ~0.7mm and no longer sits exactly on the
    # constant the `smpl_filtered` spec requires. Re-pinning removes it.
    # This does NOT flatten squats: the pelvis' WORLD height lives in `transl`,
    # and the pose lives in the pelvis-relative joint offsets -- neither is
    # touched here.
    smpl_joints = build_smpl_joints_for_pkl(raw_joints_rs, body_quat_rs,
                                            pelvis_offset=PELVIS_OFFSET).astype(np.float32)

    if transl_mode == 'body_pos_w' and seq.get('body_pos_w') is not None:
        if floor_mode == 'foot_envelope':
            world_floor = floor_envelope_z(seq['smpl_joints'], seq['body_quat_w'],
                                           seq['body_pos_w'], original_fps, win_s=floor_win)
            print(f"     floor[foot_envelope win={floor_win}s]: z range "
                  f"[{world_floor.min():.3f}, {world_floor.max():.3f}] m (per-frame)")
        else:
            world_floor = estimate_world_floor_z(seq['smpl_joints'], seq['body_quat_w'],
                                                 seq['body_pos_w'])
    else:
        world_floor = None
    transl_full = build_transl(
        seq['n'], seq['vr_position'], mode=transl_mode,
        body_pos_w=seq.get('body_pos_w'),
        world_floor_z=world_floor,
    )
    transl = resample_linear(transl_full, original_fps, target_fps).astype(np.float32)

    if transl_mode != 'zero':
        disp = float(np.linalg.norm(transl[-1] - transl[0]))
        path = float(np.linalg.norm(np.diff(transl, axis=0), axis=1).sum())
        print(f"     transl[{transl_mode}]: path={path:.2f}m disp={disp:.2f}m "
              f"pelvis_z mean={transl[:, 2].mean():.3f}m "
              f"[{transl[:, 2].min():.3f}, {transl[:, 2].max():.3f}]")

    out = dict(
        pose_aa=pose_aa,
        transl=transl,
        smpl_joints=smpl_joints,
        fps=float(target_fps),
        original_pose_aa=original_pose_aa,
        original_fps=float(original_fps),
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    joblib.dump(out, out_path)
    print(f"  -> wrote {out_path}")
    print(f"     pose_aa {pose_aa.shape}  transl {transl.shape}  "
          f"smpl_joints {smpl_joints.shape}  fps={target_fps}  "
          f"original_pose_aa {original_pose_aa.shape}  original_fps={original_fps:.2f}")

    if verify:
        verify_roundtrip(out, body_quat_rs, raw_joints_rs)
    return out


def verify_roundtrip(out, body_quat_rs, raw_joints_rs, tol=1e-5):
    """Assert that streaming this clip reproduces the raw PICO capture exactly.

    Applies `stream_clip_mode2.official_root_quat_w/official_encoder_joints`
    (re-implemented here to avoid importing zmq) to the generated pkl and
    compares against the (resampled) raw capture values that
    `pico_replay_server.py` would publish verbatim. These MUST match, otherwise
    the clip will drive the policy with a different pose than the validated
    raw-replay path."""
    YTOZ = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    BASE_CONJ = np.array([0.5, -0.5, -0.5, -0.5])

    def qmul(a, b):
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], -1)

    def qconj(q):
        c = q.copy()
        c[..., 1:] *= -1
        return c

    def qapply(q, v):
        w = q[..., 0:1]
        u = q[..., 1:]
        return v + 2 * np.cross(u, np.cross(u, v) + w * v)

    aa = out['pose_aa'][:, :3].astype(np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    ax = np.where(th < 1e-8, 0.0, aa / np.maximum(th, 1e-12))
    q = np.concatenate([np.cos(th / 2), ax * np.sin(th / 2)], -1)
    T = len(q)
    root_q = qmul(qmul(np.broadcast_to(YTOZ, (T, 4)), q), np.broadcast_to(BASE_CONJ, (T, 4)))

    joints_local = qapply(np.repeat(qconj(root_q)[:, None, :], 24, axis=1),
                          out['smpl_joints'].astype(np.float64))

    sgn = np.sign(np.sum(root_q * body_quat_rs, axis=1, keepdims=True))
    q_err = np.abs(root_q * sgn - body_quat_rs).max()

    # Compare the POSE, i.e. joints relative to the pelvis. The pelvis' own
    # absolute position is deliberately re-pinned to PELVIS_OFFSET during
    # conversion (see convert_clip), so comparing it against the un-pinned raw
    # capture would just re-measure the resampling error we intentionally
    # corrected. Everything that carries actual motion is pelvis-relative.
    j_err = np.abs((joints_local - joints_local[:, 0:1, :])
                   - (raw_joints_rs - raw_joints_rs[:, 0:1, :])).max()

    # Pelvis must stay PINNED. The residual tolerance here is set by the raw
    # capture itself: `smpl_joints`/`body_quat_w` are stored as float32, so
    # reconstructing the world frame from them has a ~1e-5 m noise floor, plus a
    # small chord-vs-arc term from fps resampling. Both are far below anything
    # physically meaningful (<0.05 mm).
    pel = out['smpl_joints'][:, 0, :]
    pel_drift = pel.std(axis=0).max()
    pel_off = np.abs(pel.mean(axis=0) - PELVIS_OFFSET).max()

    ok = (q_err < tol) and (j_err < tol) and (pel_drift < 1e-4) and (pel_off < 1e-3)
    print(f"     [verify] body_quat_w err={q_err:.2e}  pose(rel) err={j_err:.2e}  "
          f"pelvis drift={pel_drift:.2e} off={pel_off:.2e}  -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("VERIFY FAILED: streamed clip would NOT match the raw PICO replay path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True,
                    help='directory of pose_*.npz files, OR a consolidated clip .npz')
    ap.add_argument('--out', required=True, help='output .pkl path')
    ap.add_argument('--start', type=int, default=None, help='start frame (array index, inclusive)')
    ap.add_argument('--end', type=int, default=None, help='end frame (array index, exclusive)')
    ap.add_argument('--target_fps', type=float, default=50.0)
    ap.add_argument('--transl_mode', choices=['zero', 'head', 'body_pos_w'], default='zero',
                     help="'zero': static root translation (safe default); "
                          "'head': rough placeholder derived from the VR head anchor; "
                          "'body_pos_w': REAL recorded world root position (needs "
                          "a capture made with --record_root_pos)")
    ap.add_argument('--floor_mode', choices=['clip_p5', 'foot_envelope'], default='clip_p5',
                    help="'clip_p5': one floor height per clip (5th percentile of the lowest foot; "
                         "default, unchanged); 'foot_envelope': per-frame floor from the lowest foot "
                         "(rolling min + smoothing) -- for sit/stand sessions where the tracker's "
                         "height differs between postures. Do not use for jumps.")
    ap.add_argument('--floor_win', type=float, default=1.5,
                    help='window (s) of the foot_envelope rolling minimum / smoothing')
    ap.add_argument('--no_verify', action='store_true',
                     help='skip the built-in round-trip check against the raw PICO values')
    args = ap.parse_args()

    print(f"Converting {args.dir} -> {args.out}")
    convert_clip(args.dir, args.out, start=args.start, end=args.end,
                 target_fps=args.target_fps, transl_mode=args.transl_mode,
                 verify=not args.no_verify, floor_mode=args.floor_mode, floor_win=args.floor_win)


if __name__ == '__main__':
    main()
