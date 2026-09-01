# PICO SMPL Streaming Datasets — Inventory Note

Catalog of all raw PICO VR SMPL-streaming recordings found on disk (searched via the
`pose_*.npz` per-frame signature format). To be studied further later (e.g. axis-convention
verification, root-orientation correction work, deploy-vs-raw comparison).

## Dataset Table

| Directory | Frames | Frame-index range | Schema | Notes |
|---|---|---|---|---|
| `logs/smpl_raw_real_robot/` | 51,438 | `000000` – `051437` | basic (`smpl_pose`, `smpl_joints`, `body_quat_w`, `joint_pos`, `joint_vel`, ...) | Explicitly labeled `_real_robot` — captured from an actual hardware PICO session (not sim). Largest single recording found. |
| `logs/yelong_cliptest_0/` | 2,000 | `144143` – `146142` | basic | Yelong's first raw PICO take. Source of the many `gamc/storage/test/pkl_playback_yelong_*` axis-correction debugging variants (NOT independent recordings — all derived from this + `yelong_cliptest_1`). |
| `logs/yelong_cliptest_1/` | 1,000 | `146143` – `147142` | basic | Yelong's second raw PICO take. |
| `reuben_testclip_0/` | 10,000 | `000000` – `009999` | basic | Reuben's raw PICO take. |
| `paired_smpl_raw/` | 13,908 | `011572` – `025479` | **rich** (adds `vr_position`, `vr_orientation`, `frame_index`, `left/right_trigger`, `left/right_grip`, `pico_dt`, `pico_fps`, `timestamp_realtime`, `timestamp_monotonic`, `left/right_hand_joints`, `toggle_data_collection`, `toggle_data_abort`, `heading_increment`) | Richer/later-generation capture format vs. the 4 above. Frame-index range does NOT overlap with `paired_smpl_g1_deploy` below. |
| `paired_smpl_g1_deploy/` | 33,596 | `081019` – `114614` | rich (identical schema to `paired_smpl_raw`) | **Human-side input of the `/home/grease/g1_deploy_run` real-robot session** (Aug 6, 18:30, matched to +1.99 s — see "Pairing With Real-Robot Deploy Runs" below). The `"paired"` prefix refers to *that* robot-side pairing, NOT to `paired_smpl_raw`, which is a separate earlier session. |

### Derived Action Segments (Symlinked Subsets)

The following datasets are clean, single-action sub-segments systematically extracted from the raw recordings above using `detect_action_clips.py` (which detects basic locomotion, jumping, waving, etc., based on kinematics thresholding):

| Directory | Frames | Segment Range | Source | Label |
|---|---|---|---|---|
| `logs/reuben_cliptest_0/` | 600 | `[0, 600)` | `reuben_testclip_0` | walking |
| `logs/reuben_cliptest_1/` | 1,500 | `[3000, 4500)` | `reuben_testclip_0` | walking |
| `logs/reuben_cliptest_2/` | 1,500 | `[7000, 8500)` | `reuben_testclip_0` | walking |
| `logs/paired_smpl_g1_deploy_clipreal_0/` | 1,500 | `[1000, 2500)` | `paired_smpl_g1_deploy` | (manually requested range) |
| `logs/paired_smpl_g1_deploy_clipreal_1/` | 800 | `[3000, 3800)` | `paired_smpl_g1_deploy` | (manually requested range) |
| `logs/paired_smpl_g1_deploy_clipreal_2/` | 1,700 | `[4200, 5900)` | `paired_smpl_g1_deploy` | (manually requested range) |
| `logs/paired_smpl_g1_deploy_clipreal_3/` | 1,800 | `[6000, 7800)` | `paired_smpl_g1_deploy` | (manually requested range) |
| `logs/paired_smpl_g1_deploy_clipreal_4/` | 700 | `[8800, 9500)` | `paired_smpl_g1_deploy` | (manually requested range) |
| `logs/paired_smpl_g1_deploy_clipreal_5/` | 1,200 | `[9800, 11000)` | `paired_smpl_g1_deploy` | (manually requested range) |
| `logs/paired_smpl_g1_deploy_clipreal_6/` | 1,500 | `[13500, 15000)` | `paired_smpl_g1_deploy` | (manually requested range) |
| `logs/paired_smpl_g1_deploy_clipreal_7/` | 1,600 | `[19300, 20900)` | `paired_smpl_g1_deploy` | (manually requested range) |
| `logs/paired_smpl_raw_clip_0/` | 2,500 | `[0, 2500)` | `paired_smpl_raw` | (manually requested range) |
| `logs/paired_smpl_raw_clip_1/` | 2,700 | `[2800, 5500)` | `paired_smpl_raw` | (manually requested range) |
| `logs/paired_smpl_raw_clip_2/` | 1,900 | `[5600, 7500)` | `paired_smpl_raw` | (manually requested range) |
| `logs/paired_smpl_raw_clip_3/` | 700 | `[7600, 8300)` | `paired_smpl_raw` | (manually requested range) |
| `logs/paired_smpl_raw_clip_4/` | 2,100 | `[8600, 10700)` | `paired_smpl_raw` | (manually requested range) |
| `logs/paired_smpl_raw_clip_5/` | 3,000 | `[10900, 13900)` | `paired_smpl_raw` | (manually requested range) |
| `logs/smpl_raw_real_robot_clip_0/` | 2,000 | `[0, 2000)` | `smpl_raw_real_robot` | (manually requested range) |

*Note: Frame ranges for derived sets correspond to their array indices in the sorted source sequence, not literal filename suffix boundaries.*

## Converted `smpl_filtered` Datasets (`logs_pkl/`, `logs_pkl_FPS30/`)

Every clip directory above is also available as a single `smpl_filtered`-format `.pkl`,
produced by `data_process/pico_to_smpl_filtered.py` (see `SMPL_FILTERED_DATA_FORMAT.md`).
These are **generated artifacts** (gitignored) — regenerate with:

```bash
for d in logs/*clip*/; do
  .venv_sim/bin/python data_process/pico_to_smpl_filtered.py \
      --dir "$d" --out "logs_pkl/$(basename $d).pkl" --target_fps 50
done
```

| Directory | Clips | Target fps | Total frames | Notes |
|---|---|---|---|---|
| `logs_pkl/` | 20 | 50 | 17,901 (358.1 s) | Matches the `stream_clip_mode2.py` default rate. |
| `logs_pkl_FPS30/` | 20 | 30 | 10,744 (358.1 s) | Source for the GMR retargeting run below. |

### Conversion conventions (verified against the capture source)

The raw capture (`pico_manager_thread_server.py` → `process_smpl_joints()`) stores fields in a
**different frame** than `smpl_filtered` expects, so the converter inverts them:

- `body_quat_w` is **already** Z-up + base-rot-removed (`smpl_root_ytoz_up` then
  `remove_smpl_base_rot`). `pose_aa[:, :3]` is therefore the *inverse* of that, so
  `stream_clip_mode2.official_root_quat_w(pose_aa[:, :3])` reproduces `body_quat_w` exactly.
- Raw `smpl_joints` are **already de-rotated** (`quat_apply(quat_inv(body_quat_w), joints)`),
  whereas `smpl_filtered` keeps root rotation and pins only root *translation*. The converter
  therefore re-applies `quat_apply(body_quat_w, raw)`, which lands the pelvis exactly on
  `PELVIS_OFFSET` (SMPL-X rest `J[0]`).
- Joints come from **SMPL-X** FK (`compute_human_joints`, 55-joint tree, output indices
  `[0..21] + [39, 54]`) — *not* the standard SMPL 24-joint tree, so `fix_amass.smpl_fk` is
  the wrong model for validating this data.
- Rotations are resampled with **SLERP**, never linearly: the PICO root sits at `|aa| ≈ 2.8 rad`
  and crosses the ±π wrap-around (measured jumps of ~2π), which linear interpolation turns into
  garbage (1.38 quaternion / 0.88 m joint error).

`--verify` (on by default) hard-fails if a converted clip would not stream identically to the
validated `pico_replay_server.py` raw path; all 20 clips pass at ~1e-8.

**Known limitation:** `transl` is all-zero (`--transl_mode zero`) because the raw capture stores
no pelvis world position — only head/hand VR anchors. Pose fidelity is unaffected (`smpl_joints`
is pelvis-pinned by design), but there is no world locomotion, so walking clips play *in place*.

## Retargeted G1 `motion_lib` Data (`/home/grease/GMR/teleop_retargeted_g1_motion_lib/`)

`logs_pkl_FPS30/` retargeted to the Unitree G1 via GMR — 20 `.pkl` files, one per clip, in
`motion_lib` format (`root_trans_offset` (T,3), `pose_aa` (T,30,3), `dof` (T,29),
`root_rot` (T,4), `smpl_joints` (zeros), `fps=30`). Frame counts match the 30 fps sources
exactly, and root height is a sensible 0.76–0.83 m (no foot-through-floor).

Because the source `transl` is zero (above), **root travel is ~0.00–0.03 m in every clip** —
the retargeted robot articulates correctly but never moves across the floor.

Visualize with:

```bash
.venv_sim/bin/python data_visual_script/visualize_robot_pkl.py \
    --pkl_dir /home/grease/GMR/teleop_retargeted_g1_motion_lib \
    --out_dir data_visualization/gmr_retarget_check --frame_step 4
```

## Evaluation Results

`logs_pkl/` (smpl) + the retargeted `motion_lib` data (robot) were combined into
`pico_evalset` and evaluated across three checkpoints — see
**`notes_pico_evalset_3model_comparison.md`** in this folder. Headline findings:

- No checkpoint universally dominates on this teleop-specific set (unlike the AMASS
  evalset, where `RELEASED` wins — see `notes_amass_108clips_3model_comparison.md`).
- `paired_smpl_raw_clip_2` and `paired_smpl_raw_clip_5` fail across **all** checkpoints
  and both `num_envs` settings (progress ~0.10–0.20).
- `_raw_clip` clips are systematically harder than the `_deploy_clipreal_*` ones.
- Results are sensitive to `num_envs`; fix it when comparing checkpoints.

Also checked but currently **empty** (no recorded frames yet):
- `logs/smpl_raw/`
- `gear_sonic_deploy/reference/evaluation_set_raw_smpl/`

## Separately: Full C++ Deploy-Side Recording (CSV bundle, not `pose_*.npz`)

| Directory | Frames | Format |
|---|---|---|
| `gear_sonic_deploy/reference/recorded_motion/20260806/streamed_180732/` | 13,270 | CSV bundle: `smpl_joint.csv`, `smpl_pose.csv` (human side) + `joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`, `body_quat.csv`, `body_lin_vel.csv`, `body_ang_vel.csv` (robot side). Session timestamp = Aug 6, 2026, 18:07:32. |

This is the C++ deploy binary's own recorder output (different format from the raw per-frame
`.npz` captures above), and uniquely captures **both** the human SMPL stream and the robot's
tracked state simultaneously in one bundle.

## Rich-Schema Field Reference (`paired_smpl_raw` / `paired_smpl_g1_deploy`)

```text
smpl_pose               (4, 21, 3)   per-4-frame-window SMPL pose (axis-angle?)
smpl_joints             (4, 24, 3)   per-4-frame-window SMPL joint positions
body_quat_w             (4, 4)       per-4-frame-window root world quaternion
joint_pos               (4, 29)      G1 joint positions
joint_vel               (4, 29)      G1 joint velocities
vr_position             (9,)         raw VR controller/headset positions
vr_orientation          (12,)        raw VR controller/headset orientations
frame_index             (4,)
left_trigger            (1,)
right_trigger           (1,)
left_grip               (1,)
right_grip              (1,)
pico_dt                 (1,)
pico_fps                (1,)
timestamp_realtime      (1,)
timestamp_monotonic     (1,)
left_hand_joints        (7,)
right_hand_joints       (7,)
toggle_data_collection  (1,)
toggle_data_abort       (1,)
heading_increment       (1,)
```

## Pairing With Real-Robot Deploy Runs (`/home/grease/g1_deploy_run*`)

Real-robot deployment logs are `state_logger` CSV bundles (`q.csv`, `dq.csv`, `base_quat.csv`,
`action.csv`, `token_state.csv`, `motor_torque.csv`, `motor_temperature.csv`, ... +
`metadata.json`). They carry `time_realtime_ms` (Unix epoch), and the rich PICO captures carry
`timestamp_realtime`, so the two sides can be matched on **wall clock**.

### Session timeline (all Aug 6, 2026)

```text
15:03:53  yelong_cliptest_0     ( 40 s)
15:04:33  yelong_cliptest_1     ( 20 s)
18:03:20  reuben_testclip_0     (209 s)
18:07:20  paired_smpl_raw       (279 s)  ── pairs with recorded_motion/20260806/streamed_180732 (18:07:32)
18:30:46  paired_smpl_g1_deploy (505 s)  ── pairs with g1_deploy_run       (18:30:48, +1.99 s)
                                 ~184 s gap
18:42:15  g1_deploy_run002      ( 78 s)  ── NO PICO pair (reference-motion run, see below)

(separately: logs/smpl_raw_real_robot is Aug 11 17:14, 731 s — no deploy-run counterpart)
```

### ✅ `g1_deploy_run` ↔ `paired_smpl_g1_deploy`

| | PICO (human input) | Robot (deploy output) |
|---|---|---|
| Start | 18:30:46.948 | 18:30:48.938 (**+1.99 s**) |
| End | 18:39:11.585 | 18:39:26.758 (+15.17 s) |
| Frames | 33,596 @ 66.6 Hz | 25,892 @ 50.0 Hz |
| Overlap | **99.6%** of PICO | **97.1%** of robot |

The streamer was started ~2 s before the deploy binary attached; the 15 s tail is the robot
still running after streaming stopped. **This also resolves the `"paired"` naming**: the prefix
does NOT mean it is a twin of `paired_smpl_raw` — it means *SMPL paired with the g1 deploy run*.
Non-overlapping frame indices with `paired_smpl_raw` are expected (separate sessions).

### ❌ `g1_deploy_run002` has no corresponding PICO SMPL data — by design

It wasn't a teleop session at all.

| | `g1_deploy_run` | `g1_deploy_run002` |
|---|---|---|
| `motion_name` | `"streamed"` + `"squat_001__A359"` | `"squat_001__A359"` only |
| `encoder_mode` | `0` and **`2`** (Mode-2 = live SMPL stream) | `0` only — never Mode-2 |
| `motion_playing` | `0` and `1` | `0` only |
| Duration | 517.8 s | 78.2 s |
| PICO pair | ✅ `paired_smpl_g1_deploy` | ❌ none exists |

`encoder_mode == 2` is the Mode-2 live-SMPL path fed by `pico_manager`. `run002` never enters
it and never logs `"streamed"` — it only replays the built-in reference motion
`squat_001__A359` against `reference/real_example`. With no human in the loop there was no SMPL
stream to record, so no counterpart capture can exist. **This conclusion does not depend on
timestamps at all.**

### Verifying that `smpl_raw_real_robot` is not `run002` under a reset clock

Both PICO captures come from the same long-uptime host, so their two independent clocks
cross-check each other:

| | realtime Δ | monotonic Δ |
|---|---|---|
| `paired_smpl_g1_deploy` → `smpl_raw_real_robot` | 4.961 days | 4.947 days |

They agree to 0.28% (ordinary NTP correction), and neither capture contains a single backward
jump. A reset/relabel would break that agreement. Note the deploy runs show monotonic
~2,000–2,800 s (robot onboard PC, freshly booted) vs PICO ~948,000–1,376,000 s (VR host, 11–16
days uptime) — **different machines**, so only `realtime` is comparable across the two sides.

> ⚠️ **Do NOT use the PICO `joint_pos` field to content-match against a deploy run's `q.csv`.**
> It populates only indices 23–28 (hand/gripper DOFs from `G1GripperInverseKinematicsSolver`),
> not the 29-DOF body state. A cross-correlation on it scores ~0.05 even for the *known-good*
> `paired_smpl_g1_deploy` ↔ `g1_deploy_run` pair, i.e. the test is invalid, not the pairing.
> For content-based confirmation use `base_quat` (robot) ↔ `body_quat_w` (PICO), or
> `token_state.csv` ↔ PICO `smpl_joints` (in Mode-2 the tokens *are* the encoded SMPL input).

## Open Questions For Later Study

1. ~~What is the actual relationship between `paired_smpl_raw` and `paired_smpl_g1_deploy`?~~
   **RESOLVED** (see the pairing section above): they are two *separate* Aug 6 sessions, each
   paired with its own robot-side recording — `paired_smpl_raw` ↔
   `recorded_motion/20260806/streamed_180732/` (18:07), and `paired_smpl_g1_deploy` ↔
   `g1_deploy_run` (18:30). The `"paired"` prefix refers to the robot-side pairing, not to a
   relationship between the two PICO captures.
2. Confirm whether `logs/yelong_cliptest_0/1` and `reuben_testclip_0` use the "basic" schema
   only, or also contain the richer VR/trigger/hand-joint fields (need to actually inspect one
   file from each — this note currently assumes "basic" based on the earlier session's
   discovery of `convert_yelong_clip_to_track_npz.py`, but hasn't been directly verified).
3. Investigate why `logs/smpl_raw/` and `evaluation_set_raw_smpl/` are empty — were they meant
   to hold recordings that were never captured, or were their contents moved/deleted?
4. `logs/smpl_raw_real_robot` (Aug 11, 731 s) is labelled `_real_robot` but has **no** deploy-run
   counterpart on disk — was its robot-side log never saved, or stored elsewhere?
