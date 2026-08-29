# Plan: `smpl_filtered` -> BVH (SOMA skeleton) Conversion

Goal: convert our `smpl_filtered`-style `.pkl` clips (`pose_aa`, `transl`,
`smpl_joints`, `fps` -- see `SMPL_FILTERED_DATA_FORMAT.md`) into `.bvh` files
that the external **SOMA Retargeter**
(https://github.com/NVIDIA/soma-retargeter) can consume as input, so we can
retarget our own LAFAN1/AMASS-derived `smpl_filtered` clips onto the Unitree
G1 the same way BONES-SEED's G1 data was produced.

This is the *reverse* direction of `convert_lafan_to_smpl_filtered.py` /
`convert_amass_to_smpl_filtered.py` (which go BVH/npz -> `smpl_filtered`).

---

## 1. Ground truth, verified directly from the retargeter's own source

Rather than guess BVH conventions, we read
`soma-retargeter/soma_retargeter/assets/bvh.py` directly (the actual parser
the retargeter uses):

### 1.1 Skeleton structure (from a real sample, `assets/motions/bvh/*.bvh`)

```text
ROOT Root                      <- dummy, ALWAYS zero position+rotation
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Hips                    <- the REAL root (has its own 6 channels!)
  {
    OFFSET -0.0046 101.28 0.0   <- fallback only, NOT used when channels present (see 1.2)
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Spine1 { ... }        <- Spine1 -> Spine2 -> Chest -> Neck1 -> Neck2 -> Head -> HeadEnd/Jaw/L+REye
                                    Chest -> Left/RightShoulder -> Arm -> ForeArm -> Hand -> fingers
    JOINT LeftLeg { ... }        <- Hips -> Left/RightLeg -> Shin -> Foot -> ToeBase -> ToeEnd
  }
}
```

26 body joints match `classify_motions.py`'s `SOMA_JOINTS` naming
(`Hips, Spine1, Spine2, Chest, Neck1, Head, Left/RightShoulder, Arm,
ForeArm, Hand, Left/RightLeg, Shin, Foot, ToeBase`), plus extra
non-body detail (`Neck2`, `Jaw`, `L/REye`, 5 fingers/hand) that we
cannot reconstruct from SMPL-24 and will leave at rest (identity rotation).

### 1.2 CRITICAL: position channels REPLACE `OFFSET`, they do not add to it

`soma_retargeter/assets/bvh.py` lines ~78-88:

```python
if positions_exists[frame, joint_index]:
    positions = wp.vec3(anim_x, anim_y, anim_z) * 0.01   # cm -> m, used AS-IS
else:
    positions = reference_local_transforms[...].p          # OFFSET, fallback only
```

Verified empirically: our own generic `fk()` (used throughout this repo,
e.g. `data_visual_script/visualize_soma_retarget.py`) **adds** `OFFSET +
position_channel` for any joint with position channels, which is WRONG for
this file: it produces `Hips.y ~= 201 cm` (adds `OFFSET.y=101.28` +
`channel.y=100.39`) instead of the anatomically plausible `~100 cm`. This is
a latent bug in the shared visualization FK for "true root with its own
OFFSET" skeletons like this one (LAFAN/AMASS-derived BVH never hit this
because their Hips OFFSET is `0` or their root joint has no parent). **Our
BVH *writer* must therefore write `Hips` position channels as the raw
world-space translation directly (no OFFSET subtraction needed)**, matching
how the retargeter's own parser will read it back.

### 1.3 Rotation composition order (from `bvh.py::euler_to_quaternion`)

```python
quaternion = identity
for axis_char, angle in zip(rotation_order, euler_angles):   # e.g. 'zyx'
    quaternion *= quat_from_axis_angle(axis_char, angle)
```

i.e. `R = Rz(a0) @ Ry(a1) @ Rx(a2)` for channel order `Zrotation Yrotation
Xrotation` -- **identical** to the convention our own `_rot()` /
`classify_motions._rot()` already use for *reading* BVH. Good: only the
*writer* is new, the rotation-composition math doesn't need re-deriving.

---

## 2. The hard part: SMPL-24 <-> SOMA-26 joint mapping

Both skeletons represent a similar human topology but are **different rigs**
with different joint counts, names, and (crucially) different **local
rest-pose bone axis conventions**. A rotation value that means "elbow bends
90 deg" in SMPL's local frame does not directly transplant into SOMA's local
frame unless the two skeletons' rest-pose local axes happen to already agree
for that joint chain.

### 2.1 Anatomical joint mapping (used for BOTH position derivation checks and rotation copy)

| SMPL-24 idx | SMPL name | -> | SOMA joint | Notes |
|---|---|---|---|---|
| 0 | pelvis | -> | `Hips` | root: translation + rotation |
| 1 / 2 | L/R hip | -> | `Left/RightLeg` | SOMA names the *upper* leg segment "Leg" |
| 3 | spine1 | -> | `Spine1` | |
| 4 / 5 | L/R knee | -> | `Left/RightShin` | |
| 6 | spine2 | -> | `Spine2` | |
| 7 / 8 | L/R ankle | -> | `Left/RightFoot` | |
| 9 | spine3 | -> | `Chest` | |
| 10 / 11 | L/R foot (toe) | -> | `Left/RightToeBase` | |
| 12 | neck | -> | `Neck1` | `Neck2` left at rest (no SMPL equivalent) |
| 13 / 14 | L/R collar | -> | `Left/RightShoulder` | SOMA's "Shoulder" = clavicle |
| 15 | head | -> | `Head` | |
| 16 / 17 | L/R shoulder | -> | `Left/RightArm` | SOMA's "Arm" = upper arm |
| 18 / 19 | L/R elbow | -> | `Left/RightForeArm` | |
| 20 / 21 | L/R wrist | -> | `Left/RightHand` | |
| 22 / 23 | L/R hand-tip | -> | *(unmapped)* | reduced-24-joint end-effectors only, no BVH-rotatable equivalent |

Unmapped SOMA joints (identity rotation every frame): `Neck2`, `Jaw`,
`LeftEye`, `RightEye`, all finger joints, `HeadEnd`/`ToeEnd`/etc. (`End
Site`-equivalent leaves, no channels anyway).

### 2.2 The local-axis calibration problem (RESOLVED, `convert_smpl_filtered_to_bvh.py`)

SMPL's rest pose has legs pointing along local `-Y`, spine along local `+Y`
(see `classify_motions.SMPL_REST`). SOMA's rest pose (from the template
BVH's own `OFFSET` vectors) has legs/spine pointing along **different local
axes** (e.g. `LeftLeg` offset is `[-8.44, 2.57, 10.04]` -- not purely along
one axis). Directly copying `SMPL_local_rotation` into the corresponding
SOMA joint's Euler channels, frame for frame, bends the limb in the *wrong
direction* by whatever constant misalignment exists between the two
skeletons' rest local frames for that joint.

**Fix implemented:** for every joint `p` that has at least one mapped child
(this **includes the root/`Hips`**), compute a constant calibration
rotation `A_p` from the two skeletons' rest-pose `parent -> child` bone
directions (`SMPL_REST[child] - SMPL_REST[p]` vs the template BVH's
`offsets[soma_name(child)]`). For a single child, use the minimal
(shortest-arc) rotation aligning the two directions; for a parent with
**multiple** mapped children (e.g. `Chest -> Neck1, LeftShoulder,
RightShoulder`), use an SVD/Procrustes best-fit rotation
(`scipy.spatial.transform.Rotation.align_vectors`) across all of that
parent's child bone-direction pairs simultaneously, since a single rotation
cannot perfectly satisfy multiple non-collinear constraints at once.

Then, per frame: `Rlocal_soma(j) = A[parent(j)] @ Rlocal_smpl(j) @ A[j]^T`
(root: `Rlocal_soma(Hips) = Rlocal_smpl(Hips) @ A[Hips]^T`, no ancestor
term). Leaf joints in our mapped set (`Head`, `LeftHand`/`RightHand`,
`LeftToeBase`/`RightToeBase` -- no further mapped children) get `A[j] =
Identity` (no additional twist correction beyond what `A[parent(j)]`
already provides).

**Bug found and fixed during testing:** the first implementation assigned
the computed calibration to `A[child]` instead of `A[parent]`. This is
wrong because a `parent -> child` bone segment is oriented, in BVH-style
FK, by the **parent's** accumulated world rotation
(`R_world(child) = R_world(parent) @ R_local(child)`; the OFFSET vector is
rotated by `R_world(parent)`, not by the child's own rotation) -- so the
alignment correction belongs to `A[parent]`. This bug also meant the
root's own calibration (`A[Hips]`) was silently skipped entirely (always
Identity). Symptom: one leg's whole chain reached the floor correctly
across the clip while the other leg's foot never got more than a few cm
below hip height throughout the entire sequence (verified with an explicit
min/max-per-frame check across all 13,065 frames, not just a single-frame
spot check) -- a clear asymmetric, systematic miscalibration rather than a
normal stride-phase pose. Fixed by re-indexing to the parent and adding the
missing root term; re-verified: both feet/shins now show matching
symmetric height ranges across the whole clip.

**Remaining, accepted limitation:** an unavoidable **twist/roll ambiguity**
around each single-DOF limb's own bone axis (a direction-only alignment
has no information about rotation *around* that direction). Acceptable for
qualitative/visual validation; a production-quality retarget would need
either the real SOMA Retargeter's own IK (which resolves this via joint
limits and multi-objective solving) or explicit twist calibration from a
second reference direction per joint (e.g. shoulder-plane normal).

---

## 3. Conversion steps (v1 script: `convert_smpl_filtered_to_bvh.py`)

1. **Load a reference/template SOMA BVH** (any file from
   `soma-retargeter/assets/motions/bvh/*.bvh`) to get the exact joint
   hierarchy, `CHANNELS` order, and rest `OFFSET`s -- reuse verbatim, only
   the `MOTION` section is replaced.
2. **Load the `smpl_filtered` `.pkl`**: `pose_aa (T,72)`, `transl (T,3)`.
3. **Per frame, per mapped joint**: `R = axis_angle_to_matrix(pose_aa[:,
   j*3:j*3+3])`, extract Euler angles in `Rz @ Ry @ Rx` order (`scipy
   Rotation.from_matrix(R).as_euler('zyx', degrees=True)`, verified via
   round-trip self-test) and write into the corresponding SOMA joint's 3
   rotation channels, in the file's own listed channel order.
4. **Root translation**: `transl` (`smpl_filtered`'s world pelvis position,
   Z-up meters) -> raw BVH frame (Y-up, cm), inverting the documented
   `RAW_LAFAN1_DATA_FORMAT.md` Sec.4 mapping:
   `raw.X = zup.x`, `raw.Y = zup.z`, `raw.Z = -zup.y`, all `* 100`.
   Written directly into `Hips`'s position channels (§1.2: no OFFSET math
   needed).
5. **Unmapped joints**: write `0 0 0` for all 3 rotation channels every
   frame (rest pose).
6. **Frame rate**: write the clip's own `fps` as `Frame Time = 1/fps`
   (retargeter reads this from the file; no forced resampling needed).
7. **Save** the resulting `.bvh`, verify round-trip via FK (reusing
   `data_visual_script/visualize_soma_retarget.py`'s `parse_bvh`/`fk`) and
   visually compare against the original `smpl_joints`.

---

## 4. Validation plan

1. Run the v1 script on one `data/lafan1_smpl_filtered/*.pkl` clip.
2. FK the generated BVH (our own `parse_bvh`/`fk`, NOT the buggy shared one
   for double-position-root files -- see §1.2) and check:
   - Root/Hips world trajectory matches `transl` (should be near-exact).
   - Overall body silhouette is recognizable, even if limb bend directions
     are off per the known §2.2 limitation.
3. (Once available) run the actual `soma-retargeter` batch conversion on the
   generated BVH and inspect the resulting G1 CSV with
   `visualize_soma_retarget.py` for a real end-to-end sanity check.
