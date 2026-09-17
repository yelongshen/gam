# Real-Robot Online Deployment Evaluation — 2026-09-04 Session (report)

Execution record for **Step 1** of `sim2real/online_deployment_eval_plan.md`: measuring
real-G1 policy performance on streamed SMPL clips, using the metric suite from
GR00T-WholeBodyControl's `eval_agent_trl.py`.

- **Data**: `/home/grease/g1_robot_data` (SMPL stream logs `20260905/`, policy logs `g1_run_0905/`)
- **Reference**: `ego_dataset/eval_subset/robot/<clip>.pkl` (retargeted G1 `dof`)
- **Clips**: `walk_180_R_003__A332_M` (3 reps), `walk_sideway_045_stop_005__A042_M` (2 reps)
- **Tooling**: `sim2real/locate_clip_in_session.py`, `sim2real/online_eval_clips.py`
- **Raw output**: `sim2real/online_eval_results.json`
- **Nature of the numbers**: **online** (real hardware telemetry), computed **post-hoc**
  from CSV logs. No simulator involved. See §7.

> **Revision note (v2).** The first version of this report used raw streamed SMPL joints
> as ground truth via a hand-built 14-joint SMPL→G1 map, which inflated the headline
> tracking number by ~3× (126 mm vs the correct 45 mm) because SMPL/G1 morphology
> mismatch was being counted as tracking error. It also reported only n=1 per clip.
> All numbers below now use the retargeted G1 reference (`dof`) — the same ground truth
> the sim eval uses — over all 5 episodes. §5.4 records what changed.

---

## 1. Problem: three data sources, no shared key

| source | path | contents | missing |
|---|---|---|---|
| **Reference (retargeted)** | `eval_subset/robot/<clip>.pkl` | `dof` (T,29) @30fps, `root_rot`, `root_trans_offset`, `smpl_joints` | not time-linked to any run |
| **Stream side** | `20260905/streamed_<HHMMSS>/` | `smpl_joint.csv` (24×3 SMPL on the wire) | no robot feedback, **no clip name**, no timestamps |
| **Robot side** | `g1_run_0905/g1_deploy_run_09042026_run{1..5}/` | `q`, `dq`, `action`, `motor_torque`, `base_quat`, `motion_name` + `time_ms` | reference motion only labelled `"streamed"` |

Neither log family alone supports the plan's metrics, and `motion_name.csv` labels the
relevant segments only as the opaque string `"streamed"`. §2 reconstructs the join.

### 1.1 Data inventory

**Robot-side signals** (`g1_run_0905/g1_deploy_run_*/<signal>.csv`, one row per control
tick at **50 Hz**). Every file carries 5 leading meta columns
(`index`, `time_ms`, `time_realtime_ms`, `time_monotonic_ms`, `ros_timestamp`); the widths
below are the payload columns after those. All widths verified against `run3`.

| signal | width | role in this study |
|---|---|---|
| `q` | 29 | measured joint position — **the quantity being scored** (pred side of every MPJPE) |
| `action` | 29 | policy output = position target — cmd metric (§1.1 of the plan); **raw IsaacLab order**, see §3 |
| `dq` | 29 | measured joint velocity — shaking term (§4) |
| `motor_torque` | 29 | realized torque — saturation rate (§2), actuator-ID target |
| `token_state` | 64 | policy input (encoded SMPL) — **separates obs gap from dynamics gap** (not yet used, see §8) |
| `base_quat` / `base_ang_vel` / `base_accel` | 4 / 3 / 3 | root state — swing, tilt, non-fall (§3-4) |
| `torso_quat` / `torso_ang_vel` / `torso_accel` | 4 / 3 / 3 | torso IMU — secondary sway reference (unused) |
| `motor_temperature` | 58 | 2 per motor (winding, driver) — thermal / stall detection (unused) |
| `motor_error` | 29 | per-motor fault code, 0 = healthy (unused) |
| `motion_name` / `motion_playing` / `encoder_mode` | 1 each | **segment gating** — how the `"streamed"` windows in §2C were found |
| `left/right_hand_q`, `_dq`, `_action` | 7 each | Dex3 hands — not part of the 29-DoF body eval |

Of these, `q`, `action`, `dq`, `motor_torque`, `base_quat` and `motion_name` are consumed
by `online_eval_clips.py`; the rest are logged and available but unused so far.
`token_state` is the notable one — it is the policy's actual input, so pairing it with `q`
would let an obs-side gap (encoder sees something different on real vs sim) be separated
from a dynamics-side gap (same obs, different physics). That is listed as follow-up work.

**Stream-side signals** (`20260905/streamed_*/`, 50 Hz, no timestamps, last row truncated):

| signal | width | role |
|---|---|---|
| `smpl_joint` | 72 = 24×3 | SMPL joints on the ZMQ wire — **the clip fingerprint used to locate episodes** (§2B) |
| `smpl_pose` | 63 = 21×3 | axis-angle body pose (unused) |
| `body_pos` / `body_quat` / `body_lin_vel` / `body_ang_vel` | 3 / 4 / 3 / 3 | streamed root state (unused) |
| `joint_pos` / `joint_vel` | 29 each | **all zeros** in mode-2 clip replay — `stream_clip_mode2.build_frames()` never fills them, so they are *not* a usable target |

**Reference** (`eval_subset/robot/<clip>.pkl`, **30 fps** — resampled to 50 Hz, see §7):

| field | shape | role |
|---|---|---|
| `dof` | (T, 29) | retargeted G1 joint trajectory — **primary ground truth** |
| `root_rot` | (T, 4) | root orientation — heading error (currently unreliable, §7) |
| `root_trans_offset` | (T, 3) | global root translation — unusable without robot-side mocap (§7) |
| `smpl_joints` | (T, 24, 3) | source human joints — retargeting diagnostic only |

---

## 2. Pipeline

```
 eval_subset/smpl/<clip>.pkl
        │
        │ (A) load_stream_data() + official_encoder_joints()   [stream_clip_mode2.py]
        ▼
 clip joints (T,24,3), root-de-rotated  ── exactly the ZMQ wire format
        │
        │ (B) sliding-window L2 search over each session's smpl_joint.csv
        ▼
 (session, frame_offset) ───────────────────────────────┐
                                                        │
 g1_run_0905/*/motion_name.csv                          │
        │ (C) segment "streamed" runs, match by duration │
        ▼                                               │
 (run, streamed-segment t0) ────────────────────────────┤
                                                        ▼
                                    (run, t0_ms .. t1_ms) window per clip
                                                        │
 eval_subset/robot/<clip>.pkl ──> dof @30fps ──┐        │ (D) slice q/action/dq/tau/base_quat
                                               ▼        ▼
                                     FK (instantiate_g1_robot_model)
                                               │
                                               ▼
                                 compute_metrics_lite + plan §2-4 terms
```

### (A)+(B) Locating the clip inside a stream session — `locate_clip_in_session.py`

The clip is re-encoded through the *same* function the streamer uses
(`load_stream_data` → `official_encoder_joints`), so the result is directly comparable to
`smpl_joint.csv`. A sliding window minimises mean per-joint L2 distance. Matches are
unambiguous — true matches land at **4.6–6.7 mm** (float32 CSV rounding), non-matches at
**67–162 mm**:

| clip | session | frame offset | time in session | err |
|---|---|---|---|---|
| `walk_sideway_045_stop_005__A042_M` (277 fr) | `streamed_092422` | 5547 | 110.94–116.48 s | 4.60 mm |
| `walk_sideway_045_stop_005__A042_M` | `streamed_092026` | 104 | 2.08–7.62 s | 4.70 mm |
| `walk_180_R_003__A332_M` (560 fr) | `streamed_091340` | 105 | 2.10–13.30 s | 5.59 mm |
| `walk_180_R_003__A332_M` | `streamed_091541` | 104 | 2.08–13.28 s | 6.48 mm |
| `walk_180_R_003__A332_M` | `streamed_091118` | 105 | 2.10–13.30 s | 6.67 mm |

The consistent ~104-frame (2.08 s) offset is the `--settle 2.0` first-frame hold — an
independent confirmation the alignment is real, and the reason metrics start *after* it.

**Implementation note**: every `streamed_*` CSV has a truncated final row (logger killed
mid-write), so the reader skips rows with the wrong column count.

### (C) Linking a stream session to a policy run

`streamed_*` folders carry no timestamps and their `HH` is offset ~9 h from the robot
clock (different TZ), so **duration** is the join key. Segmenting `motion_name.csv` into
`squat_001__A359` ↔ `streamed` gives an exact match for all six sessions (≤0.18 s):

| session | frames → duration | policy run | `streamed` segment (robot clock) |
|---|---|---|---|
| `streamed_090954` | 1396 → 27.9 s | run2 (seg 1) | 18:09:53 → 18:10:22 |
| `streamed_091118` | 3474 → 69.5 s | run2 (seg 2) | 18:11:18 → 18:12:27 |
| `streamed_091340` | 1699 → 34.0 s | run3 | 18:13:40 → 18:14:14 |
| `streamed_091541` | 2012 → 40.2 s | run4 | 18:15:41 → 18:16:21 |
| `streamed_092026` | 1486 → 29.7 s | run5 (seg 1) | 18:20:26 → 18:20:55 |
| `streamed_092422` | 6094 → 121.9 s | run5 (seg 2) | 18:24:22 → 18:26:25 |

`run1` is squat-only (no streamed replay). `streamed_090954` is a sixth, unidentified clip.

### Episode table (the reproducible clip list, plan §0/§6)

| # | clip | session @offset | run | run-relative window | wall clock (2026-09-04) |
|---|---|---|---|---|---|
| 1 | `walk_sideway_045_stop_005__A042_M` | `streamed_092422` @5547 | run5 | 127.56–133.10 s | 18:26:13.9–18:26:19.5 |
| 2 | `walk_sideway_045_stop_005__A042_M` | `streamed_092026` @104 | run5 | 38.52–44.06 s | 18:20:28.1–18:20:33.7 |
| 3 | `walk_180_R_003__A332_M` | `streamed_091340` @105 | run3 | 18.02–29.22 s | 18:13:42.5–18:13:53.7 |
| 4 | `walk_180_R_003__A332_M` | `streamed_091541` @104 | run4 | 37.20–48.40 s | 18:15:43.1–18:15:54.3 |
| 5 | `walk_180_R_003__A332_M` | `streamed_091118` @105 | run2 | 20.50–31.70 s | 18:11:20.2–18:11:31.4 |

All five are evaluated below.

---

## 3. Reconstructing the commanded target (`q_target`)

`state_logger.hpp` documents `action.csv` as *"Policy actions (hardware order, scaled +
offset)"*. **This is wrong** — the logger writes `entry.last_action` verbatim, i.e. raw
IsaacLab-order actions. Verified against `q` on episode 3:

| candidate | mean corr with `q` | mean abs err |
|---|---|---|
| `action` as-logged | +0.056 | 39.84° |
| `action * scale + default` | +0.056 | 16.66° |
| **`action[isaaclab_to_mujoco] * scale + default`** | **+0.436** | **7.36°** |
| ...same, with 2-frame lag compensation | **+0.603** | **6.46°** |

The correct transform is the one in `zmq_output_handler.hpp:348`:

```
q_target[i] = action[isaaclab_to_mujoco[i]] * g1_action_scale[i] + default_angles[i]
```

Per-joint sanity after the fix: wrist joints correlate **0.76–0.91** (err 1.4–3.2°), knees
**0.69–0.73**. Low-ROM joints (shoulder roll/pitch, waist) stay near zero correlation —
expected, they barely move. A **2-frame (40 ms) actuation lag** was measured and is
compensated; skipping it inflates the cmd metric ~5%.

> **Action item**: fix the `state_logger.hpp` docstring — it will mislead the next analysis.

---

## 4. Metrics

Ported verbatim from `smpl_sim.smpllib.smpl_eval.compute_metrics_lite` (the function
`eval_agent_trl.py` → `im_eval_callback.py` use): `mpjpe_g`, `mpjpe_l` (root-zeroed),
`mpjpe_pa` (Procrustes: scale+rot+trans), `vel_dist`, `accel_dist`, all in mm — plus the
same body subsets (`legs`, `vr_3points`, `other_upper_bodies`, `foot`).

| role | metric | pred | ground truth | sim-comparable? |
|---|---|---|---|---|
| **PRIMARY** | `imitation` | `FK(q_measured)` | `FK(dof)` from `eval_subset/robot` | **yes** — same reference as sim |
| diagnostic | `smpl_diag` | `FK(q_measured)` | streamed SMPL, 14-joint map | **no** — morphology-contaminated |
| diagnostic | `retarget_diag` | `FK(dof)` | streamed SMPL, 14-joint map | **no** — isolates the retargeting term |

FK uses `instantiate_g1_robot_model()` (pinocchio, `g1_29dof_with_hand.urdf`) over 14 links.
The 30 fps reference is linearly resampled to the 50 Hz log rate.

Plus the real-robot terms: torque saturation (§2, `|τ| > 0.9 × effort_limit`), non-fall
(§3, IMU tilt), shaking / swing (§4), the §0.1 wrist_roll / shoulder_yaw sub-metric, and
joint-space error against the reference.

---

## 5. Results

### 5.1 Per-clip aggregate (PRIMARY imitation metric, mean ± std)

| clip | n | `mpjpe_l` | `mpjpe_pa` | `vel_dist` | joint err | non-fall |
|---|---|---|---|---|---|---|
| `walk_180_R_003__A332_M` | 3 | **45.6 ± 16.2 mm** | 39.4 ± 18.0 | 6.8 ± 2.9 | 6.46 ± 0.91° | 100% |
| `walk_sideway_045_stop_005__A042_M` | 2 | **44.3 ± 10.7 mm** | 39.4 ± 10.1 | 4.4 ± 1.7 | 7.78 ± 0.82° | 100% |

### 5.2 Per-episode

| # | clip / run | `mpjpe_l` | `mpjpe_pa` | legs | foot | joint err (worst) | cmd `mpjpe_l` |
|---|---|---|---|---|---|---|---|
| 1 | sideway / run5 (seg2) | 33.6 | 29.3 | 38.0 | 46.9 | 6.95° (`L_shoulder_yaw` 20.2) | 67.2 |
| 2 | sideway / run5 (seg1) | 55.0 | 49.5 | 79.8 | 117.4 | 8.60° (`L_shoulder_yaw` 20.5) | 62.2 |
| 3 | walk_180 / run3 | 39.1 | 29.4 | 45.2 | 64.1 | 6.41° (`L_wrist_roll` 17.6) | 70.5 |
| 4 | walk_180 / run4 | **29.8** | 24.1 | 32.4 | 44.9 | 5.37° (`L_wrist_roll` 16.9) | 63.4 |
| 5 | walk_180 / run2 | **67.8** | 64.6 | 105.1 | 163.5 | 7.60° (`L_knee` 17.4) | 46.4 |

### 5.3 Robustness

| # | saturation mean / worst | shaking | swing r/p | max tilt | non-fall |
|---|---|---|---|---|---|
| 1 | 1.22% / `R_ankle_pitch` **21.7%** | 0.206 rad/s | 2.17 / 2.78° | 7.6° | ✅ |
| 2 | 0.40% / `R_ankle_pitch` 5.4% | 0.487 | 2.18 / 2.26° | 8.6° | ✅ |
| 3 | 0.46% / `L_ankle_pitch` 8.0% | **1.117** | 3.23 / 4.24° | 10.1° | ✅ |
| 4 | 0.68% / `L_ankle_pitch` 13.0% | 0.375 | 2.63 / 4.57° | 10.7° | ✅ |
| 5 | 0.08% / `R_ankle_pitch` 1.3% | 0.442 | 1.56 / 5.17° | 10.5° | ✅ |

### 5.4 Findings

1. **5/5 non-falls.** Max base tilt 7.6–10.7°, well inside the stability bound.
2. **Rep-to-rep variance dominates and invalidates n=1 comparisons.** `walk_180` spans
   **29.8 → 67.8 mm** across three reps (2.3×, σ=16 mm on a 46 mm mean). `run2` is the
   outlier (legs 105 mm, foot 163 mm vs 32/45 mm in `run4`). Any policy A/B test on this
   clip set needs ≥3 reps or it is measuring noise, not the policy.
3. **The §0.1 stress axes are a real problem — against the reference.** Measured against
   the *commanded target* they look clean (0.8–2.7°), but against the *reference* they are
   the worst joints in 4 of 5 episodes: `left_shoulder_yaw` **20.2/20.5°** (both sideway
   reps), `left_wrist_roll` **16.9/17.6°** (walk_180). These joints track their commands
   accurately; the commands themselves deviate from the reference. This is consistent with
   the Phase C.2 low-inertia-twist finding, and it is exactly the split the plan's
   two-flavor MPJPE was designed to expose.
4. **Imitation error is *lower* than command error** (45 mm vs 46–70 mm). The PD loop lags
   each instantaneous command but averages onto the reference trajectory. The v1 report
   claimed the opposite ordering, which was an artifact of the wrong ground truth.
5. **Ankle pitch remains the actuation hotspot** — worst saturation joint in all 5 episodes,
   peaking at **21.7%** of timesteps above 0.9×effort (episode 1). Foot is also the worst FK
   subset everywhere (45–163 mm).
6. **Retargeting accounts for essentially all of the old headline number.** `retarget_diag`
   (reference vs raw SMPL, no robot involved) is **114–126 mm** — i.e. the entire 126 mm
   that v1 reported as "MPJPE-human tracking error" was morphology/retargeting, not control.
7. **`walk_180` is dynamically harder**: highest shaking (1.12 rad/s in ep 3, 2.3–5× the
   sideway reps) despite comparable positional error.

---

## 6. Reproduction

```bash
# (1) locate a clip inside the stream sessions
.venv_teleop/bin/python sim2real/locate_clip_in_session.py \
    --clip ../ego_dataset/eval_subset/smpl/walk_sideway_045_stop_005__A042_M.pkl \
    --sessions /home/grease/g1_robot_data/20260905

# (2) run the metric suite (episode table is EPISODES in the script)
.venv_teleop/bin/python sim2real/online_eval_clips.py \
    --sessions  /home/grease/g1_robot_data/20260905 \
    --runs      /home/grease/g1_robot_data/g1_run_0905 \
    --robot-ref /home/grease/ego_dataset/eval_subset/robot \
    --out       sim2real/online_eval_results.json
```

Deployed config (from `metadata.json`): 50 Hz control,
`policy/low_latency/model_{encoder,decoder}.onnx`, encoder enabled, fp16 off, 29 joints /
29 actions.

---

## 7. Validity limits

**These are online metrics** — every input is real hardware telemetry (`q` = encoder
readback, `action` = on-robot policy output, `motor_torque` = estimated motor torque,
`base_quat` = IMU). They are *computed* offline from CSVs, which is what the plan
specifies, but nothing is simulated.

- **`mpjpe_g` is degenerate here.** No external mocap ⇒ no world-frame root, so FK pins the
  root at the origin and `mpjpe_g` is numerically identical to `mpjpe_l`. **Never compare
  this `mpjpe_g` against a sim `mpjpe_g`**, which is genuinely stricter because global
  drift is observable there. Only `mpjpe_l` / `mpjpe_pa` are sim-comparable.
- **No global drift is measured.** A policy tracking joints perfectly while walking
  off-course scores identically.
- **Heading error is currently unreliable — do not cite it.** The `root_rot` quaternion
  convention is undocumented; auto-detection returns `xyzw` with corr +0.98 on one episode
  but `wxyz` with corr +0.07/+0.13 on others, and `ref_total_turn_deg` reads only 9° for a
  clip named `walk_180`. The `root_rot` semantics need confirming before this number means
  anything.
- **The reference is resampled 30 → 50 fps** (linear). This is a mild low-pass and may
  flatter `vel_dist` / `accel_dist`; it should be matched to whatever the sim eval does
  before cross-comparing those two columns.
- **Non-fall is a proxy** (IMU tilt < 35°), not the sim env's termination condition.
- **`smpl_diag` / `retarget_diag` are diagnostics only** — they use a hand-built 14-joint
  SMPL→G1 map with unmatched morphology and must not be quoted as tracking quality.
- **n = 2–3 per clip.** Enough to show variance is large (§5.4.2), not enough for tight
  confidence intervals.

**Safe comparisons**: `mpjpe_l` / `mpjpe_pa` vs other real-robot runs on the same clip list
(plan §6 contract), and — with the resampling caveat — vs sim `eval_agent_trl.py`, since
both now use the same retargeted G1 reference.

---

## 8. Next steps

1. Confirm `root_rot` convention/semantics so heading error becomes usable.
2. Run the same 5 episodes through sim `eval_agent_trl.py`; compare `mpjpe_l`/`mpjpe_pa`
   to quantify the sim2real gap directly.
3. Investigate the `run2` outlier on `walk_180` (67.8 mm, foot 163 mm) — is it a distinct
   failure mode or just initial-condition variance?
4. Investigate ankle-pitch saturation (up to 21.7%) — the clearest fragility signal.
5. Investigate `shoulder_yaw` / `wrist_roll` reference-tracking error (17–20°) — commands
   deviate from the reference even though the actuators follow the commands.
6. Build the §0.1 wrist/shoulder-yaw stress subset for a targeted probe.
7. Fix the `state_logger.hpp` `action.csv` docstring (§3).
8. Identify `streamed_090954` (sixth clip) or drop it from the manifest.

---

## 9. Sim comparison (addresses §8 next-step #2)

Ran the same two clips through the sim evaluator (`gear_sonic/eval_agent_trl.py`) with
the `LOW_LATENCY` checkpoint (`policy/low_latency/last.pt`), filtered via
`motion_lib_cfg.filter_motion_keys` to just these two motions, `num_envs=2`,
`+manager_env/terminations=tracking/eval`, corruption disabled — i.e. the standard
offline eval harness config, same reference (`eval_subset/robot/<clip>.pkl`) and same
metric suite (`compute_metrics_lite`) as the online numbers above.

- **Sim log**: `GR00T-WholeBodyControl/logs_eval/20260908_145750-EVAL_2clips_LOW_LATENCY_sim2real_compare/metrics_eval.json`
- Both clips: **100% success, 100% progress** (no early termination) in sim.

### 9.1 Sim per-clip results

| clip | `mpjpe_l` | `mpjpe_pa` | `vel_dist` |
|---|---|---|---|
| `walk_180_R_003__A332_M` | 22.53 mm | 21.24 mm | 2.89 |
| `walk_sideway_045_stop_005__A042_M` | 30.04 mm | 26.07 mm | 3.24 |

### 9.2 Sim vs. online (§7: only `mpjpe_l` / `mpjpe_pa` / `vel_dist` are sim-comparable; `mpjpe_g` is not, per §7)

| clip | metric | **sim** | **online** (§5.1, mean ± std, n=2-3) | ratio (online / sim) |
|---|---|---|---|---|
| `walk_180_R_003__A332_M` | `mpjpe_l` | **22.5 mm** | 45.6 ± 16.2 mm | 2.0× |
| `walk_180_R_003__A332_M` | `mpjpe_pa` | **21.2 mm** | 39.4 ± 18.0 mm | 1.9× |
| `walk_180_R_003__A332_M` | `vel_dist` | **2.89** | 6.8 ± 2.9 | 2.4× |
| `walk_sideway_045_stop_005__A042_M` | `mpjpe_l` | **30.0 mm** | 44.3 ± 10.7 mm | 1.5× |
| `walk_sideway_045_stop_005__A042_M` | `mpjpe_pa` | **26.1 mm** | 39.4 ± 10.1 mm | 1.5× |
| `walk_sideway_045_stop_005__A042_M` | `vel_dist` | **3.24** | 4.4 ± 1.7 | 1.4× |

### 9.3 Findings

1. **A real, consistent sim2real gap exists**: online tracking error is **1.4-2.4×** the
   sim error across every comparable metric on both clips. This quantifies the gap that
   §8 next-step #2 asked for.
2. **The gap is larger on `walk_180`** (1.9-2.4×) **than on `walk_sideway`** (1.4-1.5×),
   consistent with §5.4 finding #7 that `walk_180` is "dynamically harder" (highest
   shaking, 1.12 rad/s in one episode) — harder dynamics amplify whatever sim2real
   discrepancy exists (actuator modeling, latency, contact/friction mismatch, etc.),
   whereas the sim rollout has no such physical-hardware disturbances.
3. **Sim numbers land below the online per-episode range, not just below the mean**:
   e.g. sim `walk_180` `mpjpe_l` (22.5 mm) is even below the best online rep (`run4`,
   29.8 mm, §5.2) — i.e. this is not simply "sim looks better than the online average
   because of averaging," sim is better than every individual online rep recorded so far.
4. **Caveat (§7 carries over)**: online is n=2-3 reps with large rep-to-rep variance
   (§5.4 finding #2, e.g. `walk_180` spans 29.8-67.8 mm across 3 reps) — the sim number
   is a single deterministic rollout with no comparable variance estimate. A larger
   online rep count (or repeated sim rollouts with `enable_corruption=True` domain
   randomization) would be needed to compare distributions, not just point estimates.
5. **Next step**: the ~1.5-2.4× multiplicative gap suggests the dominant sim2real
   contributors are likely actuator/latency/contact-model mismatches rather than a
   policy-tracking-quality issue per se, since the *same* policy weights track much more
   tightly in sim. Cross-referencing this against the §5.4 finding #5 stress joints
   (`shoulder_yaw`, `wrist_roll`) and §5.4 finding #5 ankle-pitch saturation would help
   localize which subsystem (actuator dynamics vs. contact/friction vs. latency)
   contributes most to the gap.

**Reproduction**:
```bash
cd GR00T-WholeBodyControl
python gear_sonic/eval_agent_trl.py \
  checkpoint=/home/grease/gam/gear_sonic_deploy/policy/low_latency/last.pt \
  +headless=true +num_envs=2 \
  +manager_env.commands.motion.motion_lib_cfg.motion_file=/home/grease/ego_dataset/eval_subset/robot \
  +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/home/grease/ego_dataset/eval_subset/smpl \
  "+manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[walk_180_R_003__A332_M,walk_sideway_045_stop_005__A042_M]" \
  eval_name=EVAL_2clips_LOW_LATENCY_sim2real_compare \
  +run_once=true +eval_callbacks=im_eval +eval_output_dir='${eval_log_dir}' \
  ++manager_env.observations.policy.enable_corruption=False \
  ++manager_env.observations.tokenizer.enable_corruption=False \
  +manager_env/terminations=tracking/eval
```
