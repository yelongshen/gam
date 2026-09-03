# Systemic Plan: Real-Robot Online Deployment Evaluation

This is the concrete protocol for **Step 1** (and, unchanged, **Step 3**) of
`sim2real/longcontext_adaptation_plan.md`: measuring policy performance/robustness on
the real G1, using sampled SMPL clips replayed on hardware, MPJPE-style tracking
metrics, and reward-style robustness terms (shaking / swing / fall).

## 0. Test set

Reuse the existing categorical split from `model_eval/EVAL_METRICS_DRAFT.md` §A:

- **Test-Content (OOD):** agility/acrobatics (cartwheels, kicks), complex
  gestures/dance, unstructured motion (crawling).
- **Test-Repetition (ID):** basic locomotion (walk/run/crouch), new performances of
  known motion types.

Sample a fixed number of clips per category (e.g. 5-10) **once**, and reuse the exact
same set for every policy comparison — never resample per run, or comparisons become
noisy/unfair. Store the clip list (paths + a version hash) alongside the report so
future re-runs are guaranteed reproducible.

### 0.1 Additional targeted stress subset (new, motivated by Phase C.2)

Add a **dedicated stress-test subset** targeting the low-inertia "twist" axes
identified in this session's Phase C.2 root-cause analysis
(`wrist_roll`, `shoulder_yaw` — effective inertia ~5-20× smaller than neighboring
pitch/roll axes, hence structurally most sensitive to sim2real dynamics mismatch and
error accumulation in closed-loop/long-horizon use):

- Select or construct SMPL clips with **fast forearm rotation** (wrist twisting
  motions — e.g. tool-use, waving, spinning gestures) and **fast torso twisting**
  (shoulder-yaw-dominant reaching/turning motions).
- These clips do not need to be exotic — even short (5-10s) isolated-motion clips are
  fine, since the point is to isolate these specific DOFs' behavior, not to test
  general capability.
- Track `L/R_wrist_roll` and `L/R_shoulder_yaw` tracking error **as their own reported
  sub-metric**, not folded into the all-29-joint mean — this session's Phase C.2 result
  showed the mean can look nearly unchanged (21.55°→21.52°) while these specific joints
  swing by tens of degrees in either direction. A policy/config change that helps or
  hurts here would otherwise be invisible.

## 1. Core tracking metric: two-flavor MPJPE

The real robot has **no ground-truth external body-pose sensor** by default (unlike
sim, where `q_sim` can be forward-kinematics'd directly against the SMPL target). Two
different, non-interchangeable quantities are both called "MPJPE" in casual usage —
report both, separately, every time:

### 1.1 MPJPE-cmd ("control-loop fidelity")

```
MPJPE_cmd = mean_j || FK(q_target[t])_j - FK(q[t])_j ||_2     (joint/EE positions, meters or cm)
```

Forward-kinematics **both** the policy's commanded target (`q_target`, from
`action_to_q_target`) and the robot's actually-achieved joint state (`q`) into 3D
joint/end-effector positions, then take mean per-joint L2 distance. This is a
Cartesian-space relabeling of the Phase A angular tracking-error metric
(`phaseA_latency.md` §5) — it measures **"did the actuator do what the policy asked,"**
independent of whether the policy asked for the right thing. Comparable across policy
versions even if retargeting/tokenization changes.

### 1.2 MPJPE-human ("full pipeline fidelity")

```
MPJPE_human = mean_j || FK(q[t])_j - SMPL_joint[t]_j ||_2     (after retargeting-consistent alignment)
```

Forward-kinematics the robot's **actually-achieved** state (`q`) into 3D joint
positions, and compare directly against the **original human SMPL clip's** joint
positions (the thing that was supposed to be imitated). This is the headline
"does it look like the human" number, but **silently bakes in retargeting accuracy** —
a retargeting bug and a control-tracking bug both show up here and cannot be
distinguished without also looking at MPJPE-cmd.

**Always report both.** A regression in `MPJPE_human` with unchanged `MPJPE_cmd`
implicates retargeting; a regression in both implicates control/dynamics.

### 1.3 (Optional, higher fidelity) External motion capture

If available (e.g. the PICO-based visualization pipeline used in
`stream_clip_mode2.py --visualize`), an external camera/tracker on the physical robot
gives a genuine independent ground truth for `MPJPE_human`, removing the FK-model
assumption. Recommended for a final validation pass, not required for routine
comparisons (adds logistics overhead per trial).

## 2. Torque saturation rate

```
saturation_rate_j = fraction of timesteps where |motor_torque[t,j]| > 0.9 * actuatorfrcrange_j
```

Computed per joint from `motor_torque.csv`, reported per Test-Content/Test-Repetition
category. A joint's saturation rate rising under a new policy is a leading indicator of
fragility even if tracking error hasn't visibly degraded yet — the policy is commanding
near-limit effort and has little margin left to react to disturbance.

## 3. Task success / non-fall rate

```
non_fall_rate = (# episodes completed without entering a failure state) / (total episodes)
```

Failure state definition (reuse `EVAL_METRICS_DRAFT.md` §1B): base/pelvis height below
a minimum threshold, or base pitch/roll exceeding a stability bound, or an explicit
E-stop/recovery trigger in the logs. Report **per category** (Test-Content vs
Test-Repetition) — do not pool, since a policy can be reliable on basic locomotion
while failing badly on OOD/acrobatic content, and pooling hides that (same lesson as
the all-29-joint mean masking the wrist/shoulder-yaw problem).

## 4. Reward-style robustness terms

| term | metric | formula / source |
|---|---|---|
| **Shaking** | high-frequency joint velocity energy | RMS of `dq[t,j]` after high-pass filtering (e.g. >3-5 Hz), per joint and pooled; complements "Action Smoothness / Normalized Jerk" already in `EVAL_METRICS_DRAFT.md` §1B |
| **Swing** | base/torso sway | if base pose/IMU logged: RMS of base roll/pitch or COM lateral velocity. If not available: proxy via variance of `waist_roll`/`waist_pitch` tracking error (already flagged as elevated in this session's Phase C analysis) |
| **Fall** | binary failure event | same definition as §3's non-fall rate; report both the rate (§3) and, for clips that do fail, the **elapsed time-to-fall** as a severity indicator |

## 5. Stratified reporting format

For every policy under test, produce one report table with this structure (not a
single pooled number):

| category | clip | MPJPE-cmd | MPJPE-human | non-fall | saturation (worst joint) | shaking | swing |
|---|---|---|---|---|---|---|---|
| Test-Repetition (ID) | walk_01 | ... | ... | ... | ... | ... | ... |
| Test-Repetition (ID) | run_01 | ... | ... | ... | ... | ... | ... |
| Test-Content (OOD) | cartwheel_01 | ... | ... | ... | ... | ... | ... |
| Test-Content (OOD) | dance_01 | ... | ... | ... | ... | ... | ... |
| **Wrist/Shoulder-Yaw stress** | wrist_twist_01 | (report `L/R_wrist_roll`, `L/R_shoulder_yaw` sub-columns explicitly) | | | | | |

Plus category-level aggregates (mean ± std per category, **not** a single global mean)
and the dedicated stress-subset table with per-DOF breakdown (§0.1).

## 6. Reproducibility requirements

- Fixed clip list + version hash (§0).
- Same robot, same firmware/`policy_parameters.hpp` gains, same session length per
  clip, logged the same way (`state_logger` CSV format already in use).
- Every report references which policy checkpoint / commit was deployed.
- Old-vs-new comparisons must use **identical** clip sets and identical metric
  definitions — this document, versioned, is the contract for that.

## 7. Relationship to sim-side tooling

This protocol produces the **real-robot side** of the baseline/comparison described in
`sim2real/longcontext_adaptation_plan.md` §1 and §3. It intentionally does not require
new simulation tooling — it only reuses/extends what `EVAL_METRICS_DRAFT.md` and
`phaseA_latency.md` already established, plus the two new items motivated by this
session's analysis: the wrist-roll/shoulder-yaw stress subset (§0.1) and the two-flavor
MPJPE split (§1), which prevents retargeting error and control error from being
conflated into one number.
