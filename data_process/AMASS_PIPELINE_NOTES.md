# AMASS → G1 Motion Pipeline: Scripts, Stages, and Bug Fixes

Full pipeline to convert raw AMASS `.npz` mocap into G1 robot `motion_lib` PKLs
for SONIC training/eval.

## Pipeline Stages

```
Stage 0: Raw AMASS .npz
   │  (SMPL-H "poses" schema A, or SMPL-X "_stageii" schema B)
   ▼
Stage 1: convert_amass_to_smpl_filtered.py
   Script: /home/grease/gam/data_process/convert_amass_to_smpl_filtered.py
   Input:  raw .npz (e.g. /home/grease/egodata/downloads/amass/extracted/**/*.npz)
   Output: smpl_filtered .pkl  (pose_aa, transl, smpl_joints, fps, original_pose_aa, original_fps)
   Key steps:
     - Schema-aware parse (handles both SMPL-H and SMPL-X "_stageii" layouts)
     - Schema-aware framerate detection (mocap_framerate vs mocap_frame_rate)
     - Resample source fps -> --target_fps (default 50, supports --target_fps N)
     - Canonicalize frame-0 root rotation (zero heading at t=0)
     - FK via smpl_fk() -> smpl_joints (Z-up, pelvis-pinned to PELVIS_OFFSET)
     - transl: Z-up (FIXED 2026-08-24, see Bug #3 below)
   Usage:
     python3 convert_amass_to_smpl_filtered.py \
       --input /home/grease/egodata/downloads/amass/extracted \
       --output /home/grease/ego_dataset/amass_smpl_filtered_v2 \
       --target_fps 50 --num_workers 8
   ▼
Stage 2: convert_smpl_filtered_to_bvh.py
   Script: /home/grease/gam/data_process/convert_smpl_filtered_to_bvh.py
   Env:    /home/grease/gam/.venv_sim/bin/python
   Input:  smpl_filtered .pkl directory (Stage 1 output)
   Output: .bvh files on the SOMA skeleton (one per clip)
           Batch mode prepends "{input_dir_basename}__" to each output filename
           (e.g. "amass_smpl_filtered_FPS30" -> "amass_FPS30__<name>.bvh")
   Key steps:
     - Derives body joint rotations from smpl_joints (3D positions), NOT pose_aa
       (pose_aa[:,3:72] is unpopulated for LAFAN-derived data; using smpl_joints
       works universally for both LAFAN and AMASS sources)
     - Computes scale_ratio = ll_soma/ll_smpl (leg-length calibration) per-clip
   Usage:
     .venv_sim/bin/python convert_smpl_filtered_to_bvh.py \
       --input_dir /home/grease/ego_dataset/amass_smpl_filtered_FPS30 \
       --output_dir /home/grease/gam/data/amass_bvh_FPS30 \
       --template /home/grease/soma-retargeter/assets/motions/bvh/Neutral_walk_forward_002__A057.bvh \
       --num_workers 8
   ▼
Stage 3: SOMA Retargeter (Newton physics-based IK)
   Script: /home/grease/soma-retargeter/app/bvh_to_csv_converter.py
   Env:    /home/grease/soma-retargeter/.venv/bin/python  (Python 3.12, has `newton` package)
   Input:  BVH directory (Stage 2 output) + JSON config
   Output: Flat "Bones-SEED"-style CSV per clip (root_translate*, root_rotate*, *_joint_dof columns)
   Config (JSON, --config flag):
     {
       "import_folder": "<bvh dir>",
       "export_folder": "<csv output dir>",
       "batch_size": 100,
       "retargeter": "Newton",
       "retarget_source": "soma",
       "retarget_target": "unitree_g1",
       "retarget_source_facing_direction": "Mujoco"
     }
   Usage:
     .venv/bin/python app/bvh_to_csv_converter.py --viewer null --headless --config <config.json>
   Throughput: ~0.87s/motion (~4 hours for the full 16,753-clip corpus)
   ▼
Stage 4: convert_soma_csv_to_motion_lib.py
   Script: /home/grease/GR00T-WholeBodyControl/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py
   Input:  CSV directory (Stage 3 output)
   Output: motion_lib .pkl (root_trans_offset, pose_aa, dof, root_rot, smpl_joints[zeros], fps)
   IMPORTANT FLAGS:
     --floor_clamp                 : enable per-frame floor-penetration fix (see Bug #4 below)
     --floor_clamp_mjcf <path>     : G1 MJCF, default gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml
     --floor_clamp_epsilon_m 0.003 : safety clearance above floor (meters)
   Usage:
     python3 convert_soma_csv_to_motion_lib.py \
       --input /home/grease/gam/data/amass_csv_FPS30 \
       --output /home/grease/gam/data/amass_motion_lib_FPS30 \
       --fps 30 --individual --num_workers 8 \
       --floor_clamp --floor_clamp_mjcf gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml
```

## Bugs Found & Fixed (2026-08-24)

### Bug #1: Off-by-one frame-drop (LAFAN + AMASS converters)
`np.arange(0, duration, step)` silently drops the last frame whenever
`duration * TARGET_FPS` is an exact integer. Fixed in both
`convert_lafan_to_smpl_filtered.py` and `convert_amass_to_smpl_filtered.py`
by switching to `np.linspace(0, duration, round(duration*fps)+1, endpoint=True)`.
Affects ~43-66% of clips depending on source framerate.

### Bug #2: motion_lib/smpl_filtered filename prefix mismatch
`convert_smpl_filtered_to_bvh.py`'s batch mode prepends
`"{input_dir_basename}__"` to output filenames (e.g. `amass_FPS30__...`),
which then propagates through Stages 3-4 into the final `motion_lib` PKL
keys. But the raw `smpl_filtered` directory (used as `smpl_motion_file` in
eval configs) never had this prefix -> exact-filename lookup in
`motion_lib_base.py` silently returns `smpl_data=None` for every motion,
producing degenerate `NaN`/near-zero eval metrics with no error/warning.
**Workaround** (not a code fix): create a symlink farm with matching
prefixed names, e.g. `amass_smpl_filtered_v2_prefixed/`, pointing back at
the real files (negligible disk cost, no data duplication).

### Bug #3: transl Y-up/Z-up convention mismatch (AMASS only)
`convert_amass_to_smpl_filtered.py` saved `transl` in AMASS's native Y-up
convention (only heading-rotated, never axis-remapped), while
`convert_smpl_filtered_to_bvh.py` assumes ALL `smpl_filtered` `transl`
arrays are Z-up (true for LAFAN1's converter, which already produces
genuine Z-up `transl`). This caused every AMASS clip's true height and
forward-walk axes to be swapped -> BVH/G1 output appeared to continuously
"climb" while the source human was just walking forward (~6m rise over a
6s clip). **Fixed**: added `transl = transl_yup @ ZUP.T` in
`convert_amass_to_smpl_filtered.py::convert_one()`, matching the same
remap already applied to `smpl_joints`.

### Bug #4: Foot/floor penetration (frame-0 AND persistent across clip)
Even with Bug #3 fixed (stable, non-drifting height), retargeted G1 feet
were found 4.7-9cm underground throughout an entire test clip (not
explained by leg-scale mismatch -- `scale_ratio` for the test clip was
1.072, which raises height, not lowers it). Root cause not fully isolated
(possibly Newton retargeter IK/floor-constraint imprecision), but a
principled, general fix already exists: `foot_floor_clamp.py`
(`/home/grease/gam/data_process/foot_floor_clamp.py`), which:
  - Parses the G1 MJCF's own foot collision-sphere geometry programmatically
    (no hardcoded offsets)
  - Runs full per-frame FK to compute TRUE foot-sole world height
  - Applies a PER-FRAME (not global-constant) vertical root correction,
    only lifting frames that actually penetrate, preserving real vertical
    motion (jumps/crouches) elsewhere
Already wired into `convert_soma_csv_to_motion_lib.py` via `--floor_clamp`.
**Verified fix**: test clip's worst-case penetration (-90mm) -> now
consistently +7-8mm above floor (matching requested epsilon) across all
sampled frames.

## Verified end-to-end on test clip
`ACCAD__Male2Walking_c3d__B10_-__Walk_turn_left_45_poses` (180 frames @ 30fps):
- Stage 1 height: 0.787-0.886m (was: drifting 6+ meters, Bug #3)
- Stage 2 BVH height: 84.4-95.0cm (stable)
- Stage 3 CSV height: 66.2-74.4cm (stable)
- Stage 4 motion_lib (with --floor_clamp): feet +7.2 to +8.0mm above floor,
  all sampled frames (was: -37mm to -90mm penetration, Bug #4)
