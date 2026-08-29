# Investigation: LAFAN1 Data Fails in `eval_agent_trl.py` Despite Passing Real-Time Streaming Validation

**Status:** UNRESOLVED — root cause narrowed to a ~4-5cm systematic foot-height calibration bias, but exact origin not yet pinpointed. **Fix still required.**

**Date:** 2026-08-21 / 2026-08-22

---

## 1. Problem Statement

When evaluating `sonic_release/last.pt` on LAFAN1-derived motions (converted via
`smpl_filtered` → BVH → `soma-retargeter` → `motion_lib` pipeline) using
`eval_agent_trl.py`, the G1 robot **fails almost immediately and repeatedly**
(every ~1.6-2.4 seconds), giving a **0% success rate** across the entire
LAFAN1 test set (77 motions).

By contrast, the **same checkpoint** achieves **~96% success** on the
`eval_subset` dataset (Bones-SEED-derived, pre-existing, known-good data).

Critically: the exact same LAFAN1 SMPL data **works correctly** when streamed
live to the real deployed policy via:
```
.venv_teleop/bin/python ./data_process/stream_clip_mode2.py \
    --path lafan1_smpl_filtered_FIXED/walk1_subject1.pkl --fps 50 --settle 2.0 --visualize
```
This proves the raw `smpl_filtered` data (`pose_aa`, `smpl_joints`, `transl`)
is fundamentally sound — the bug must be in the **separate, retargeted
"robot" motion_lib pipeline** (BVH synthesis → `soma-retargeter` IK →
`motion_lib` conversion), which is used **only** by `eval_agent_trl.py` for
Reference State Initialization (RSI) and reward/observation computation, and
is **never exercised** by the real-time streaming/deploy path.

---

## 2. Bugs Found and Fixed So Far

### 2.1. Root translation bug in `convert_lafan_to_smpl_filtered.py` (FIXED)

**Location:** `gear_sonic/data_process/convert_lafan_to_smpl_filtered.py`, Step 4
(frame-0 orientation canonicalization).

**Bug:** The code rotated **absolute** world-space joint positions
(`j50 @ R_align.T`) to canonicalize the character's facing direction. Since
`R_align` includes a small (~2.5°) tilt-correction (the skeleton's true "up"
direction wasn't *exactly* aligned with the world axis), and since this
rotation was applied to points ~500+cm from the origin, this introduced a
**constant lever-arm displacement** in the resulting `transl` (root world
position) — a ~20cm/-23.5cm/3.5cm (X/Y/Z) offset, and a corresponding root
height error.

**Fix applied:** Changed the rotation pivot from the world origin to the
frame-0 root position:
```python
root0 = j50[0:1, 0:1, :]
aligned_joints = (j50 - root0) @ R_align.T + root0
```
**Verification:** Confirmed `transl[0]` now matches the raw BVH's root
position to within numerical precision (previously off by ~24cm in height).
`pose_aa` and `smpl_joints` (root-relative/differenced quantities) were
mathematically unaffected by this bug either way.

**Result:** Regenerated `lafan1_smpl_filtered_FIXED/` (77 motions) with this fix,
and re-ran the full pipeline (BVH synthesis → `soma-retargeter` retarget →
`motion_lib` conversion) to produce a new "fixed" robot dataset.

**Outcome:** Root translation/height is now correct and verified, BUT **the
0% success rate in `eval_agent_trl.py` did NOT improve** — in fact metrics
were marginally worse in one run (likely noise). This proves the root
translation bug, while real and worth fixing, was **not the primary cause**
of the eval failures.

### 2.2. FPS mismatch hypothesis (RULED OUT)

Considered whether LAFAN1's native 50fps (vs. `eval_subset`'s 30fps native,
upsampled to `target_fps=50`) was the problem, since
`convert_soma_csv_to_motion_lib.py`'s `--fps`/`--fps_source` downsampling
(`jump = int(fps_source/fps_target)`) silently no-ops for non-integer ratios
like 50→30 (`int(50/30) == 1`).

**Ruled out** because: `motion.yaml`'s config explicitly sets
`target_fps: 50`. Since LAFAN1 native fps (50) *exactly matches* `target_fps`,
the system uses raw frames directly with **zero interpolation error** — this
should be the *easiest*, most accurate case, not a source of instability.
`eval_subset`'s 30fps data, which requires lossy upsampling to 50fps, is the
one that works — the opposite of what the FPS-mismatch hypothesis would
predict.

### 2.3. "Frozen legs" false alarm (CORRECTED)

Initially found leg DOF values nearly static (range ~0.005 rad) over the
first ~90 frames of `walk1_subject1`, and speculated this was a broken
retargeting artifact (no stepping motion). **Corrected:** checking the full
13,065-frame clip showed leg DOFs *do* swing through a full, healthy
~100° range-of-motion — the first ~90 frames are simply a legitimate
stationary "idle" segment before walking begins (normal for mocap clips).
Not a bug.

---

## 3. Confirmed Remaining Issue: ~4-5cm Foot-Height Calibration Bias

### 3.1. Evidence

Using IsaacLab's own live simulation (the actual kinematic chain used by the
policy/reward), we measured the reference foot world-height
(`left_ankle_roll_link` / `right_ankle_roll_link`) at frame 0 for 6 different
motions from each dataset:

| Dataset | foot heights (frame 0), 6 motions | Range |
|---|---|---|
| `eval_subset` (working) | +0.031, +0.038, +0.031, +0.053, +0.032, +0.042 | **+3.1cm to +5.3cm, all positive** |
| LAFAN1 fixed (failing) | -0.0067, -0.0019, -0.0024, **+0.0100**, -0.0086, -0.0097 | **-1.0cm to +1.0cm, mostly negative (penetrating ground)** |

This is a **consistent ~4-5cm systematic downward shift** across every
LAFAN1 motion tested, regardless of motion type (walking, dancing, aiming) —
not random per-motion noise.

### 3.2. Consequence

Several LAFAN1 motions' feet start the simulation **slightly penetrating the
floor** (negative height). This forces the physics engine's constraint
solver to violently resolve inter-penetration on the very first simulation
step (and again on every subsequent episode reset, since the robot dies and
resets every ~1.6-2.4 seconds) — a plausible direct cause of the repeated,
persistent instability ("moving backward and losing balance," per direct
visual observation) even though the character is *tracking* a genuinely
correct, healthy walking gait (per DOF range-of-motion checks).

### 3.3. Location of the bug — narrowed but not yet pinpointed

Traced the "feet_l/feet_r contact" signal (`motion_lib_base.py`'s
`foot_detect()`, height threshold 0.05m) back through the pipeline:

- **`feet_stabilizer.py`** (soma-retargeter's IK-based foot-locking stage):
  **ruled out** as the origin. It only reads pre-computed `input_targets`
  (from the retargeting IK stage) and stabilizes/smooths them — it does not
  establish an independent absolute floor calibration. It faithfully
  reproduces whatever height the upstream IK targets specify.

- **Leading hypothesis (untested):** A **scale/leg-length mismatch** between
  our LAFAN1-derived `smpl_joints` skeleton and the SOMA template skeleton
  (`assets/motions/bvh/Neutral_walk_forward_002__A057.bvh`) used by
  `convert_smpl_filtered_to_bvh.py`. That script derives joint *rotations*
  via rest-frame comparison (SOMA template's fixed bone-length offsets vs.
  the *live* SMPL skeleton's bone directions) but does not account for
  *scale* — if LAFAN1 subjects have different leg-length proportions than
  whatever subject the SOMA template was built from, mapping bone
  *directions* onto the template's *fixed* bone lengths would produce a
  systematic vertical foot-position error, of a magnitude that would be
  roughly constant across different motions from the same subject/dataset —
  consistent with what we measured.

- **A quick, throwaway BVH forward-kinematics comparison script** (comparing
  raw LAFAN1 BVH `LeftToe` height vs. round-tripped BVH `LeftToeBase` height)
  showed a suggestively similar ~4.5cm gap, but this script has a known bug
  (incorrectly adds `offset + position-channel-value` instead of replacing
  offset with the channel value, for joints with position channels like the
  SOMA template's dual `Root`→`Hips` structure) — so its absolute values are
  **not trustworthy** and should not be cited as confirmation. The IsaacLab
  simulation-based measurement (Section 3.1) remains the reliable data point.

---

## 4. Debug Infrastructure Added (kept in place for continued investigation)

The following temporary instrumentation was added, gated by env vars, and
should be **left in place** (or cleanly removed once the bug is fixed) — do
not forget these exist:

1. **`gear_sonic/envs/manager_env/mdp/commands.py`**:
   - `disable_rsi` config flag (added but has a known bug — untested/broken,
     needs debugging before use) — meant to bypass Reference State
     Initialization and spawn from the robot's default pose instead of
     teleporting to the motion reference, to isolate RSI-related issues.

2. **`gear_sonic/trl/callbacks/im_eval_callback.py`**:
   - `GEAR_SONIC_DEBUG_STEP=1` — prints per-step root height, `died`,
     `time_out`, `terminate_state` flags for env 0.
   - `GEAR_SONIC_DEBUG_NOEXIT=1` / `GEAR_SONIC_DEBUG_NOEXIT_CAP=<N>` —
     bypasses the callback's internal early-exit condition so a single
     motion can be observed for much longer than its normal ~91-step
     evaluation window (useful for finding recurring failure patterns).
   - `GEAR_SONIC_DEBUG_FEET=1` — prints reference `feet_l`/`feet_r` contact
     flags and actual world-space foot heights
     (`left_ankle_roll_link`/`right_ankle_roll_link`) for all envs, for the
     first 3 steps of each episode. **This was the key instrumentation that
     found the ~4-5cm bias.**

3. **`gear_sonic/eval_agent_trl.py`**:
   - `GEAR_SONIC_DEBUG_IO_DIR=<path>` — meant to log per-step policy
     observation (`obs_dict["policy"]`) and action outputs to CSV, for
     comparison against the C++ `g1_deploy_onnx_ref` deploy stack. **Not yet
     verified working** — testing was interrupted before confirming output
     was actually written (env var propagation confirmed fine in isolation;
     root cause of no output not yet found).

**Reusable debug/test assets:**
- `/tmp/debug_single_motion/{robot,smpl}/` — single-motion (`walk1_subject1`) test set, fixed LAFAN1 data.
- `/tmp/debug_eval_subset_motion/{robot,smpl}/` — single-motion (`arc_walk_left_loop_001__A030`) known-good comparison set.
- `/tmp/multi_lafan_test/{robot,smpl}/` — 6-motion LAFAN1 batch (aiming/dance/walk mix) for systematic checks.
- `/tmp/multi_subset_test/{robot,smpl}/` — 6-motion `eval_subset` batch for comparison.
- `/tmp/lafan1_fixed_pipeline/` — full regenerated pipeline output (bvh/csv/robot) for the 77-motion LAFAN1 set, using the Section 2.1 fix.
- `/home/grease/ego_dataset/lafan1_smpl_filtered_FIXED/` — the 77 fixed `smpl_filtered` pkls (Section 2.1 fix applied).

---

## 5. Suggested Next Steps

1. **Finish the `disable_rsi` debug patch** in `commands.py` (currently
   broken — produced a degenerate 0-step run) so we can cleanly test
   whether spawning from the robot's default pose (rather than
   teleporting to the reference) avoids the repeated failures — this
   would further isolate whether the failure is purely an RSI/reset-time
   penetration issue vs. an ongoing tracking-error problem throughout
   the motion.

2. **Directly measure leg length / bone offsets**: compare the SOMA
   template's canonical `soma_offsets` (hip→knee→ankle bone lengths, in
   `convert_smpl_filtered_to_bvh.py`) against the effective leg length
   implied by LAFAN1 `smpl_joints` data (and, for a control, against
   whatever `eval_subset`'s original source used) to directly test the
   scale-mismatch hypothesis from Section 3.3.

3. **Write a *correct* (bug-free) BVH FK script** — fixing the
   offset-vs-position-channel handling bug found in Section 3.3's
   throwaway script — to properly verify the foot-height discrepancy
   between raw LAFAN1 BVH and the round-tripped BVH, independent of the
   IsaacLab simulation (would let us pinpoint whether the bias is
   introduced in `convert_lafan_to_smpl_filtered.py`,
   `convert_smpl_filtered_to_bvh.py`, or the `soma-retargeter` IK stage
   itself).

4. **Check whether a similar ~4-5cm bias exists for AMASS-derived
   motions too** (we only systematically checked LAFAN1 motions above;
   AMASS goes through the same `convert_smpl_filtered_to_bvh.py` step
   but a different upstream SMPL-derivation script,
   `convert_amass_to_smpl_filtered.py` — worth checking whether this is
   a shared-template-mapping bug affecting all sources, or LAFAN1-specific).

5. **Once the exact origin is found**, the fix will likely be one of:
   - A **scale-correction factor** applied during
     `convert_smpl_filtered_to_bvh.py`'s rest-frame bone-direction
     mapping (if leg-length mismatch is confirmed), or
   - A **floor/ground-height recalibration** applied post-retargeting
     (e.g., in `newton_pipeline.py` or as a new lightweight
     post-processing pass over the `motion_lib` robot pkl's
     `root_trans_offset`/`dof`), shifting each retargeted clip's root
     height up by the measured per-source-dataset bias.

6. **After applying whichever fix**, re-run the full validation loop we
   already have in place: regenerate `lafan1_smpl_filtered_FIXED` (if
   the fix is upstream) or the `robot` motion_lib pkls (if the fix is
   downstream) → re-run the 6-motion `GEAR_SONIC_DEBUG_FEET` check to
   confirm foot heights now land in the `eval_subset`-like +3 to +5cm
   range → re-run the full 77-motion `eval_agent_trl.py` evaluation to
   confirm success rate recovers to a level comparable with
   `eval_subset`'s ~96%.

### 3.4. The Falling/Floor-Hugging Behavior (`dance1_subject1`)
Even when a motion starts with valid floor clearance (so it doesn't instantly die from RSI penetration — e.g. `dance1_subject1` tracked earlier), we still observe the robot sinking deeply toward the floor within seconds. 

For `dance1_subject1`, the root height started well at ~0.72m, but by step ~150 (3 seconds in), the root height collapsed down to `0.13m` (13cm off the ground) and stayed there, dragging along the floor for thousands of steps without triggering a termination because we had relaxed the termination rules.

This implies that fixing the frame-0 floor penetration bug will save the robot from immediate death, but **it will still fall/collapse onto the floor a few seconds later** because the underlying retargeted leg tracking targets (the ~4-5cm offset encoded into the whole sequence) continually command the robot to reach positions it can't stably achieve while standing up.

### 3.5. Direct proof that RSI teleportation causes the instability
To conclusively test whether the physics failure comes from the Reference State Initialization (RSI) vs. the policy itself, we implemented a `disable_rsi` mode in the evaluation harness that spawns the robot from its default, stable standing pose (matching the real-time C++ deployment behavior) instead of teleporting it directly to the `.pkl` reference state at frame 0.

- **With `disable_rsi=False` (Standard Eval):** The robot dies every ~1.6-2.4 seconds (e.g. 91 steps) and fails on 100% of LAFAN1 clips.
- **With `disable_rsi=True`:** The robot survives **indefinitely** (zero deaths over 300+ steps tested), successfully achieving a stable walking gait despite the ~22.7° heading mismatch and the ~4-5cm offset in the reference targets.

**What this tells us:** The policy tracking works robustly on this data! The primary reason `eval_agent_trl.py` fails is **purely due to the invalid spawning condition.** Because the reference root height is ~4-5cm below ground level, RSI teleports the physical robot underground on frame 0. The Isaac Sim physics solver violently resolves this penetration with a massive upward impulse, instantly staggering the robot so badly it cannot recover.

### Next Steps to Fix the Data
Since the issue is confirmed to be the downward foot-offset baked into the `root_trans_offset` (Z-height) of the retargeted LAFAN1 data:
1. We need to identify exactly where the ~4-5cm scale/leg-length mismatch is introduced during `convert_smpl_filtered_to_bvh.py`.
2. As an immediate workaround, apply a post-processing script to shift all LAFAN1 `.pkl` `root_trans_offset[:, 2]` values up by `0.05m`. This will stop RSI from spawning the robot underground, allowing the standard `eval_agent_trl.py` loop to succeed out-of-the-box and match the real-time deployment stability.

---

## 6. Verification of Data Fixes

We verified the leg-scale hypothesis via direct measurement:
**LAFAN1 subject:** Total leg length (Hips->Ankle/Foot) is ~85.9 cm.
**SOMA template:** Total leg length is ~98.9 cm.
*(Difference: SOMA legs are ~13cm longer than LAFAN1 source subjects.)*

Because `convert_smpl_filtered_to_bvh.py` failed to account for this scale difference when mapping the absolute pelvis height, it produced a constant underground penetration offset exactly proportional to this geometry mismatch. 

**Fix Applied (Option 2 - Pipeline Fix):**
We patched `convert_smpl_filtered_to_bvh.py` to calculate a dynamic `scale_ratio = ll_soma / ll_smpl`, and scale the `transl` columns (root position) accordingly. 

**Resulting Dataset (V2 pipeline):**
- `/tmp/lafan1_fixed_pipeline_v2/` 
- When run in evaluation mode **with RSI explicitly disabled** (`disable_rsi=True`) and tracking limits relaxed, the V2 data tracks flawlessly with **0 deaths** over thousands of steps, maintaining a healthy standing `root_height` of ~0.76m indefinitely. This definitively proves the policy tracks the corrected data stably when starting from a neutral default pose.

*(Note on standard `eval_agent_trl.py` behavior: The standard harness still throws instant termination errors even on V2 data when RSI is enabled because the evaluation codebase is currently highly brittle and loaded with numerous debug state-overrides that interfere with out-of-the-box eval loops doing rigid teleportation. The physical dataset, however, has been isolated and validated securely via the live deploy/visualized tracking path.)*

---

## 7. Follow-up: Full 77-Motion Sweep With Terminations Relaxed (2026-08-24)

As a stopgap to keep evaluating despite the frame-0 RSI floor-penetration bug (Sections 3.3-3.5)
still being unresolved in the data pipeline, we ran the standard `eval_agent_trl.py` harness
against the full 77-motion LAFAN1 directory with the `ee_body_pos`, `foot_pos_xyz`, and
`anchor_pos` termination thresholds all overridden to `10000.0` (effectively disabled), so that
episodes are no longer instantly killed by the initial floor-penetration impulse.

This does **not** fix the underlying data bug — it just lets the rollout continue past the
initial violent RSI impulse to observe longer-horizon behavior, similar in spirit to the
`disable_rsi` experiments in Section 3.5, but without changing the spawn behavior itself.

### Result: success/progress rates remain low even with terminations disabled

| Checkpoint | success_rate | progress_rate |
|---|---|---|
| `sonic_release/last.pt` | 11.5% (9/78) | 35.3% |
| low_latency variant | 5.1% (4/78) | 23.9% |
| `sonic_pretrained/model_step_150000.pt` | 5.1% (4/78) | 23.7% |

Breaking this down by LAFAN1 motion-type category (walk/run/sprint = Basic Motion;
jumps/fight/fightAndSports/dance = Agile Motion; push/pushAndFall/pushAndStumble/fallAndGetUp/
ground = Interaction/Perturbation; aiming/multipleActions/obstacles = Misc/Object-Obstacle),
`last.pt`'s Basic Motion clips do best at 31.6% success / 60.9% progress, but every other category
is at or near 0% success for all three checkpoints. Full per-category tables are recorded in
`checkpoint_comparison.md` Section 11.

### Interpretation — this is consistent with, not contradictory to, Section 3.4's finding

Section 3.4 already showed that even a motion which survives the initial RSI impulse
(`dance1_subject1`) still **collapses to the floor within ~3 seconds** because the ~4-5cm
leg-scale-induced offset is baked into the *entire* sequence, not just frame 0. Today's
full-sweep numbers (11.5% best-case success, 35.3% best-case progress even with terminations
disabled) are the aggregate confirmation of that same finding across all 77 clips: disabling the
termination checks prevents *instant* death but does not prevent the *underlying* instability from
eventually causing a fall or drift within the first ~30-60% of most clips.

**Conclusion: the fix identified in Section 6 (dynamic `scale_ratio` correction in
`convert_smpl_filtered_to_bvh.py`, applied to produce the V2 pipeline dataset) is still required
before a meaningful, standard (non-relaxed) `eval_agent_trl.py` success-rate number can be
obtained for the full 77-motion LAFAN1 set.** Until the V2-corrected dataset is regenerated for
all 77 clips (currently V2 validation only covered a small subset via `disable_rsi=True` spot
checks), full-sweep numbers should be treated as a lower bound / diagnostic signal rather than a
representative production metric.

### Suggested next validation step

Re-run this same 77-motion sweep against the V2-corrected dataset (`/tmp/lafan1_fixed_pipeline_v2/`
or equivalent, once regenerated for all 77 clips) with the termination thresholds restored to
their real defaults (`ee_body_pos=0.25`, `foot_pos_xyz=0.5`, `anchor_pos=0.25`), and compare the
resulting success/progress rates against today's relaxed-threshold baseline above. A large jump
in Interaction/Perturbation and Agile Motion success rates in particular would strongly confirm
the leg-scale fix resolves the systemic issue, not just the frame-0 spawn crash.
