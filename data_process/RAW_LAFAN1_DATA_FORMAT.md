# Raw LAFAN1 Data Format

Reference for the **raw** LAFAN1 `.bvh` motion-capture files and how they
are converted into the `smpl_filtered` representation documented in
`SMPL_FILTERED_DATA_FORMAT.md`.

- **Location:** `/home/grease/egodata/downloads/lafan1_extracted/*.bvh`
- **Format:** standard BioVision Hierarchy (BVH) text format
- **Parsed by:** `classify_motions.parse_bvh()`

---

## 1. File structure (standard BVH)

A BVH file has two sections: `HIERARCHY` (skeleton definition: joint names,
parent/child nesting, rest-pose offsets, channel layout) followed by
`MOTION` (per-frame channel values).

```text
HIERARCHY
ROOT Hips
{
    OFFSET 272.094208 92.886261 -190.072601
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT LeftUpLeg
    {
        OFFSET 0.103458 1.857832 10.548514
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT LeftLeg
        {
            OFFSET 43.500000 -0.000027 0.000013
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftFoot
            {
                OFFSET 42.372200 0.000008 -0.000018
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT LeftToe
                {
                    OFFSET 17.299995 -0.000003 0.000021
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    End Site { OFFSET 0.000000 0.000000 0.000000 }
                }
            }
        }
    }
    ...
}
MOTION
Frames: 6771
Frame Time: 0.033333
<6771 rows of channel values>
```

- **Root joint (`Hips`)** has 6 channels: 3 position + 3 rotation.
- **All other joints** have 3 channels: rotation only (position is implied by
  the fixed `OFFSET` rest-pose translation from the parent).
- **`End Site`** blocks terminate a chain (fingertip/toe-tip); they carry no
  rotation channels and are not separate tracked joints. A parser must
  consume the entire `End Site { ... }` block (including its own closing
  brace) without popping the enclosing joint off its stack.

---

## 2. Joint list (22 total)

```text
Hips, LeftUpLeg, LeftLeg, LeftFoot, LeftToe,
RightUpLeg, RightLeg, RightFoot, RightToe,
Spine, Spine1, Spine2, Neck, Head,
LeftShoulder, LeftArm, LeftForeArm, LeftHand,
RightShoulder, RightArm, RightForeArm, RightHand
```

Note: real LAFAN1 files name the toe joints `LeftToe` / `RightToe` (no
`Base` suffix).

## 3. Motion section

```text
Frames: 6771
Frame Time: 0.033333   ->  fps = 1 / 0.033333 = 30.0 Hz
```

LAFAN1 is native **30 fps**. Every `smpl_filtered` clip is uniformly
**50 fps**, so the conversion upsamples via linear interpolation.

## 4. Units and axis convention

Raw BVH `OFFSET`/position values are in **centimeters, Y-up**. `parse_bvh()`
converts to **meters, Z-up** via:

```text
parsed = (raw.X, -raw.Z, raw.Y) / 100
```

i.e. scale by 1/100 (cm -> m), then remap axes so BVH's `Y` (up) becomes
world `Z` (up) and BVH's `Z` becomes `-Y`. This is a proper right-handed
rotation (det = +1), matching the axis-remap convention used elsewhere in
the SMPL pipeline (`SMPL_FILTERED_DATA_FORMAT.md` §5.3).

---

## 5. How LAFAN1 connects to the `smpl_filtered` format

`smpl_filtered` (see `SMPL_FILTERED_DATA_FORMAT.md`) expects, per clip:

| Field | Shape | Notes |
|---|---|---|
| `pose_aa` | `(T, 72)` | only `[:, :3]` (root axis-angle) is populated for LAFAN; body joints have no SMPL pose params, so the rest stays zero |
| `transl` | `(T, 3)` | pelvis world position |
| `smpl_joints` | `(T, 24, 3)` | Z-up, meters, root-local (pelvis pinned at `PELVIS_OFFSET`) |
| `fps` | scalar | always 50.0 |

LAFAN1's 22 raw joints map onto the 24 SMPL joint indices used everywhere
else in this pipeline (`classify_motions.BVH_TO_SMPL`):

```text
0 Hips, 1 LeftUpLeg, 2 RightUpLeg, 3 Spine, 4 LeftLeg, 5 RightLeg,
6 Spine1, 7 LeftFoot, 8 RightFoot, 9 Spine2, 10 LeftToe, 11 RightToe,
12 Neck, 13 LeftShoulder, 14 RightShoulder, 15 Head, 16 LeftArm,
17 RightArm, 18 LeftForeArm, 19 RightForeArm, 20 LeftHand, 21 RightHand
```

SMPL joints 22/23 (hand-tip end-effectors) have no LAFAN1 equivalent and are
**extrapolated** from the forearm->hand vector:

```python
smpl_joints[:, 22] = smpl_joints[:, 20] + (smpl_joints[:, 20] - smpl_joints[:, 18]) * 0.3
smpl_joints[:, 23] = smpl_joints[:, 21] + (smpl_joints[:, 21] - smpl_joints[:, 19]) * 0.3
```

## 6. Conversion pipeline (`.bvh` -> `smpl_filtered`-style `.pkl`)

1. **Parse BVH -> world-space joints.** `classify_motions.parse_bvh()` walks
   the BVH hierarchy, does forward kinematics per frame, and returns
   `(joints (T,24,3), root_rot (T,3,3))` already in meters/Z-up (§4).
2. **Extrapolate hand-tips** (joints 22/23) as shown above.
3. **Resample 30 fps -> 50 fps** via linear interpolation of joint positions,
   matching every other `smpl_filtered` clip's frame rate.
4. **Canonicalize frame-0 orientation.** Build an orthonormal body-frame
   basis at frame 0 directly from the skeleton geometry:
   - `up = normalize(Neck(12) - Pelvis(0))`
   - `right = normalize(L_hip(1) - R_hip(2))`, Gram-Schmidt'd against `up`
   - `fwd = up x right`

   This gives `F0 = [right | fwd | up]`, the rotation implied by the actual
   pose (not any dataset-specific "rest" convention). A single rigid
   correction `D = F0_target @ F0^T` is computed once and applied to every
   frame's joints, re-anchoring the whole clip so frame 0 stands upright and
   faces a canonical heading, without altering any relative motion between
   joints or frames.
5. **Derive `pose_aa[:, :3]` (root axis-angle) from the now-canonicalized
   joints**, per frame, using the same `up`/`right`/`fwd` construction above,
   then converting that basis into the root-quaternion convention
   `official_root_quat_w()` expects (`stream_clip_mode2._YTOZ`,
   `_BASE_CONJ`) and back to axis-angle. This guarantees `pose_aa` is always
   geometrically consistent with `smpl_joints`, independent of LAFAN1's own
   BVH rest-pose convention (which is not the same as SMPL's rest pose, so
   the raw root-rotation channel out of the BVH file cannot be used as-is).
6. **Store** `pose_aa`, `transl` (pelvis position), `smpl_joints`
   (root-relative, `PELVIS_OFFSET`-pinned), and `fps=50.0` in a `.pkl`,
   identical in shape/convention to native `smpl_filtered` clips.

The result streams correctly through `stream_clip_mode2.py` (Mode 2 / `smpl`
encoder) with a stable, upright starting pose and physically valid joint
distances throughout the clip.

