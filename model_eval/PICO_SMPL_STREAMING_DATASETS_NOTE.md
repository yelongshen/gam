# PICO SMPL Streaming Datasets — Inventory Note

Catalog of all raw PICO VR SMPL-streaming recordings found on disk (searched via the
`pose_*.npz` per-frame signature format). To be studied further later (e.g. axis-convention
verification, root-orientation correction work, deploy-vs-raw comparison).

## Dataset Table

| Directory | Frames | Frame-index range | Schema | Notes |
|---|---|---|---|---|
| `logs/smpl_raw_real_robot/` | 51,438 | `000000` – ... | basic (`smpl_pose`, `smpl_joints`, `body_quat_w`, `joint_pos`, `joint_vel`, ...) | Explicitly labeled `_real_robot` — captured from an actual hardware PICO session (not sim). Largest single recording found. |
| `logs/yelong_cliptest_0/` | 2,000 | `144143` – ... | basic | Yelong's first raw PICO take. Source of the many `gamc/storage/test/pkl_playback_yelong_*` axis-correction debugging variants (NOT independent recordings — all derived from this + `yelong_cliptest_1`). |
| `logs/yelong_cliptest_1/` | 1,000 | (later range) | basic | Yelong's second raw PICO take. |
| `reuben_testclip_0/` | 10,000 | `000000` – ... | basic | Reuben's raw PICO take. |
| `paired_smpl_raw/` | 13,908 | `011572` – ... | **rich** (adds `vr_position`, `vr_orientation`, `frame_index`, `left/right_trigger`, `left/right_grip`, `pico_dt`, `pico_fps`, `timestamp_realtime`, `timestamp_monotonic`, `left/right_hand_joints`, `toggle_data_collection`, `toggle_data_abort`, `heading_increment`) | Richer/later-generation capture format vs. the 4 above. Frame-index range does NOT overlap with `paired_smpl_g1_deploy` below. |
| `paired_smpl_g1_deploy/` | 33,596 | `081019` – ... | rich (identical schema to `paired_smpl_raw`) | Despite the "paired" naming, this does NOT appear to be a frame-aligned twin of `paired_smpl_raw` (non-overlapping frame-index ranges, different total counts) — relationship between the two is still unconfirmed. |

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

## Open Questions For Later Study

1. What is the actual relationship between `paired_smpl_raw` and `paired_smpl_g1_deploy`
   (same session continued, two different sessions, or one derived from the other)? Check
   `timestamp_realtime` / `timestamp_monotonic` values to determine actual chronological
   relationship rather than relying on `frame_index` filename ranges alone.
2. Confirm whether `logs/yelong_cliptest_0/1` and `reuben_testclip_0` use the "basic" schema
   only, or also contain the richer VR/trigger/hand-joint fields (need to actually inspect one
   file from each — this note currently assumes "basic" based on the earlier session's
   discovery of `convert_yelong_clip_to_track_npz.py`, but hasn't been directly verified).
3. Investigate why `logs/smpl_raw/` and `evaluation_set_raw_smpl/` are empty — were they meant
   to hold recordings that were never captured, or were their contents moved/deleted?
