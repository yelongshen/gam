# Checkpoint Evaluation Comparison

**Dataset**: `/home/grease/ego_dataset/eval_subset` (160 motions)
**Date**: August 21, 2026

## 1. Metrics Summary

| Metric | `sonic_release/last.pt` | `low_latency/last.pt` | Delta |
|---|---|---|---|
| **Success rate** | **96.25%** (154/160) | **93.13%** (149/160) | ▼ 3.12% |
| **Progress rate** | **97.77%** | **94.92%** | ▼ 2.85% |
| MPJPE Global (success) | 137.1 mm | 201.7 mm | ▼ worse (+64.6mm) |
| MPJPE Local (success) | 29.1 mm | 28.7 mm | ▲ better (-0.4mm) |
| MPJPE PA (success) | 21.2 mm | 21.9 mm | ▼ worse (+0.7mm) |
| MPJPE Global (all) | 138.2 mm | 198.7 mm | ▼ worse (+60.5mm) |
| MPJPE Local (all) | 30.0 mm | 29.3 mm | ▲ better (-0.7mm) |
| MPJPE PA (all) | 21.9 mm | 22.3 mm | ▼ worse (+0.4mm) |

### Key Observations
* **Global Tracking vs Local Articulation**: The `low_latency` checkpoint performs noticeably worse on general global tracking (MPJPE-g shifted from ~137mm to ~201mm) and success rates. However, its local joint articulation (MPJPE-l) remains highly competitive (and even marginally better on this subset). This indicates the trade-off for lower latency was a slight degradation in precise root/heading tracking but no loss in local pose matching.

---

## 2. Failure Cases

### Shared Failures (Failed in both checkpoints)
These 4 motions are highly dynamic or complex, proving difficult for both policies:
1. `flip_090_003__A304_M`
2. `flip_360_004__A415`
3. `dance_hiphop_mike_tyson_R_fast_001__A319_M`
4. `small_heavy_two_hands_behind_medium_to_behind_low_R_001__A520`

### Uniquely Failed in `sonic_release` (2 motions)
1. `injured_torso_stoop_down_R_003__A214`
2. `mohak_forward_stop_002__A036_M`

### Uniquely Failed in `low_latency` (7 motions)
1. `turn_crawl_360_003__A133`
2. `jog_forward_loop_002__A033_M`
3. `dance_vouge_boogle_180_R_003__A316`
4. `nailing_floor_R_004__A283_M`
5. `injured_R_leg_jog_ff_stop_180_R_003__A214_M`
6. `kneeling_stop_002__A051_M`
7. `idle_turn_270_002__A049_M`

---

## 3. Raw Logs / Metrics JSONs
* **`sonic_release`**: `/home/grease/GR00T-WholeBodyControl/logs_eval/metrics/EVAL_SUBSET_OFFLINE/metrics_eval.json`
* **`low_latency`**: `/home/grease/GR00T-WholeBodyControl/logs_eval/metrics/EVAL_SUBSET_LOW_LATENCY/metrics_eval.json`

---

## 4. `sonic_pretrained` Updates

| Metric | `sonic_pretrained` | `sonic_release/last.pt` | (Delta vs release) |
|---|---|---|---|
| **Success rate** | **95.63%** (153/160) | **96.25%** | ▼ 0.62% |
| **Progress rate** | **97.05%** | **97.77%** | ▼ 0.72% |
| MPJPE Global (success) | 152.6 mm | 137.1 mm | ▼ worse (+15.5mm) |
| MPJPE Local (success) | 29.2 mm | 29.1 mm | ▼ barely worse (+0.1mm) |
| MPJPE PA (success) | 20.6 mm | 21.2 mm | ▲ better (-0.6mm) |
| MPJPE Global (all) | 150.7 mm | 138.2 mm | ▼ worse (+12.5mm) |
| MPJPE Local (all) | 30.1 mm | 30.0 mm | ▼ barely worse (+0.1mm) |
| MPJPE PA (all) | 20.9 mm | 21.9 mm | ▲ better (-1.0mm) |

### Key Observations
* **Position in the Hierarchy**: The `sonic_pretrained` checkpoint occupies a middle ground. Its global tracking (MPJPE-g ~152mm) is worse than the final `sonic_release` policy (~137mm), but much better than `low_latency` (~201mm).
* **Local Pose**: Its Procrustes-Aligned error (MPJPE-PA) is actually the best of all three (~20.6mm), meaning the raw local pose output is excellent, but its ability to track the exact global root path isn't as perfectly honed as the final release checkpoint.

### Failed Motions (7 motions)
* `flip_090_003__A304_M` (fails on all 3)
* `flip_360_004__A415` (fails on all 3)
* `small_heavy_two_hands_behind_medium_to_behind_low_R_001__A520` (fails on all 3)
* `turn_crawl_360_003__A133` (also fails on `low_latency`)
* `avoid_obstacle_jump_run_ff_180_R_002__A501` (unique failure)
* `dance_western_country_one_step_270_R_003__A308_M` (unique failure)
* `step_rotate_idle_045_002__A023_M` (unique failure)

*(Note: It actually succeeds on `dance_hiphop_mike_tyson_R_fast_001__A319_M`, which both other checkpoints fail on).*


## 5. `scratch_stable_lr` (100k steps) Updates

| Metric | `scratch_stable_lr/last.pt` | `sonic_release/last.pt` | (Delta vs release) |
|---|---|---|---|
| **Success rate** | **73.75%** (118/160) | **96.25%** | ▼ 22.50% |
| **Progress rate** | **80.28%** | **97.77%** | ▼ 17.49% |
| MPJPE Global (all) | 239.3 mm | 138.2 mm | ▼ worse (+101.1mm) |
| MPJPE Local (all) | 42.2 mm | 30.0 mm | ▼ worse (+12.2mm) |
| MPJPE PA (all) | 32.6 mm | 21.9 mm | ▼ worse (+10.7mm) |

### Key Observations
* **Relative Immaturity**: This training run from scratch (even at its final 100K step checkpoint) is vastly inferior to the `sonic_release` model and the other finalized checkpoints. The success rate plummets to ~74% (failing on 42 of 160 motions) indicating that this training run either didn't converge fully, wasn't trained on the same data volume, or lacked the staged curriculum learning that the release model benefited from. 

---

## 6. Pretraining Progression (`sonic_pretrained` intermediates)

The following compares intermediate checkpoints generated over the course of the pretraining run (`model_step_050000` through the final `model_step_150000`):

| Step | Success Rate | Failed | Progress | MPJPE-G (all) | MPJPE-L (all) |
|---|---|---|---|---|---|
| **050,000** | 90.63% | 15 | 93.94% | 191.6 mm | 31.0 mm |
| **080,000** | 93.13% | 11 | 95.94% | 174.9 mm | 31.6 mm |
| **100,000** | 95.63% | 7 | 97.82% | 162.1 mm | 31.1 mm |
| **120,000** | **96.25%** | **6** | **98.13%** | **150.1 mm** | 30.1 mm |
| **150,000** | 95.63% | 7 | 97.05% | 150.7 mm | **30.1 mm** |

### Key Observations Note
* **Peak Success Rate**: The success rate actually peaks at step `120,000` (96.25%, tying the final `sonic_release`), before dropping slightly to 95.63% by step `150,000`.
* **Global Tracking Improvement**: We see a beautiful, monotonic improvement in Global MPJPE (from 191.6mm down to 150.7mm) as training progresses, confirming the model learns to better track absolute spatial coordinates later in training.
* **Local Articulation Plateau**: Notice how Local MPJPE stubbornly hovers around 30-31mm throughout this entire window. This suggests the local pose dynamics are learned almost entirely within the first 50k steps, and the next 100k steps are primarily optimizing global/root locomotion stability rather than intra-limb articulation.

---

## Appendix: MPJPE Metrics Explained

**MPJPE** stands for **Mean Per-Joint Position Error**. It measures the average 3D Euclidean distance (in millimeters) between the predicted joint positions and the ground-truth joint positions across all frames of a motion sequence.

In this codebase, it is measured in three distinct ways:

* **MPJPE-g (`mpjpe_g` - Global)**: Measures the absolute position error in the global world coordinate frame. This encapsulates *both* the robot's internal posture accuracy *and* its global path tracking (e.g., did the robot drift 5 cm to the left while walking forward?).
  * *Physical meaning*: "How far away is the robot's physical hand/foot in the room compared to where it should be in the room?"

* **MPJPE-l (`mpjpe_l` - Local / Root-Relative)**: Measures the position error *after subtracting the root (pelvis) position* from both prediction and ground truth. This removes any global spatial drifting.
  * *Physical meaning*: "Assuming the robot's torso is perfectly placed in the room, how accurate is its posture and limb positioning?"

* **MPJPE-pa (`mpjpe_pa` - Procrustes-Aligned)**: Measures the error after performing a full rigid alignment (Procrustes analysis) between the predicted pose and the ground-truth pose. This aligns translation, rotation, and scale before calculating the error.
  * *Physical meaning*: "Ignoring global position, global facing direction (heading), and uniform scaling, how well do the structural joint angles match?" This is the purest measure of localized articulation quality (e.g., did it replicate the exact shape of a pose?).

### Acceleration and Velocity Metrics (`accel_dist` / `vel_dist`)

The evaluation JSON also includes `eval/success/accel_dist` and `eval/success/vel_dist`. These metrics track the temporal smoothness and dynamic tracking quality of the 3D joint movements rather than just their static spatial positions:

* **`vel_dist` (Velocity Error)**: Measures the mean absolute error (in millimeters per second) between the ground-truth joint velocities and the predicted joint velocities over the trajectory. This reflects how well the policy matches the target speed of movements.
* **`accel_dist` (Acceleration Error)**: Measures the mean absolute error in joint accelerations. A high `accel_dist` even with a low `vel_dist` or `mpjpe` implies the presence of high-frequency "jitter" or jerky, unsmooth movements that are constantly reversing or accelerating faster than the clean ground truth. These metrics are crucial for robotic deployment because highly jittery motions (high `accel_dist`) can strip hardware gears or overheat actuators.

---

## 7. Categorized Failure Breakdown

The 160 EVAL subset motions have been broken down into four distinct categories to measure performance across different OOD (Out-of-Distribution) boundaries.

| Category | Size | `sonic_release` | `low_latency` | `sonic_pretrained` | `scratch_stable_lr` |
|---|---|---|---|---|---|
| **Agility & Acrobatics** (Flips, jumps, martial arts) | 19 | **10.5%** (2 fails) | 10.5% (2 fails) | 15.8% (3 fails) | 52.6% (10 fails) |
| **Complex Gestures / Dance** (Hip-hop, sweeping, interactions) | 36 | **2.8%** (1 fail) | 5.6% (2 fails) | 2.8% (1 fail) | 27.8% (10 fails) |
| **Unstructured Motion** (Crawls, crouches, kneeling) | 28 | **7.1%** (2 fails) | 10.7% (3 fails) | 7.1% (2 fails) | 39.3% (11 fails) |
| **Basic Locomotion** (Walking, jogging, turns) | 77 | **1.3%** (1 fail) | 5.2% (4 fails) | 1.3% (1 fail) | 14.3% (11 fails) |

### Key Categorical Takeaways:
1. **The "Flips" ceiling:** Agility & Acrobatics are definitively the hardest domain for all models. Even the best `sonic_release` model fails on `flip_090_003` and `flip_360_004`, indicating intense airborne acrobatics exceed the boundaries of current zero-shot physics tracking.
2. **`low_latency` penalizes Basic Locomotion:** While `low_latency` matches the release model on flips (only 2 fails), it struggles significantly more on normally trivial **Basic Locomotion** (4 fails vs 1, a ~4x increase in failure rate). This supports the earlier MPJPE-G finding that the shorter latency degrades the robot's ability to maintain stable root balance during standard walking/turning sequences.
3. **Dance and Gestures are mostly solved:** Despite the complexity of upper-body arm waving during hip-hop dances and elaborate gestures, the top policies solve 95-97% of these easily. The only shared dance failure is a highly aggressive Mike Tyson routine (`dance_hiphop_mike_tyson_R_fast_001__A319_M`).

---

## 8. Pretraining Progression by Category (50k to 150k)
Failure rates across the 4 categories for the `sonic_pretrained` intermediates:

| Category | Size | 50k | 80k | 100k | 120k | 150k (final) |
|---|---|---|---|---|---|---|
| **Agility & Acrobatics** | 19 | 6 fails (31.6%) | 4 fails (21.1%) | 4 fails (21.1%) | 4 fails (21.1%) | 3 fails (15.8%) |
| **Complex Gestures / Dance** | 36 | 3 fails (8.3%) | 3 fails (8.3%) | 2 fails (5.6%) | 2 fails (5.6%) | 1 fails (2.8%) |
| **Unstructured Motion** | 28 | 4 fails (14.3%) | 2 fails (7.1%) | 1 fails (3.6%) | 0 fails (0.0%) | 2 fails (7.1%) |
| **Basic Locomotion** | 77 | 2 fails (2.6%) | 2 fails (2.6%) | 0 fails (0.0%) | 0 fails (0.0%) | 1 fails (1.3%) |

---

## 9. Success Metrics Progression by Category (50k to 150k)

### Agility & Acrobatics
| Metric | 50k | 80k | 100k | 120k | 150k |
|---|---|---|---|---|---|
| **mpjpe_g** | 105.98 | 128.49 | 119.47 | 106.16 | 118.28 |
| **mpjpe_l** | 25.54 | 28.77 | 28.80 | 27.98 | 28.04 |
| **mpjpe_pa** | 19.70 | 20.94 | 19.94 | 19.99 | 19.99 |
| **accel_dist** | 1.32 | 1.55 | 1.52 | 1.47 | 1.58 |
| **vel_dist** | 3.90 | 4.42 | 4.27 | 4.13 | 4.34 |

### Complex Gestures / Dance
| Metric | 50k | 80k | 100k | 120k | 150k |
|---|---|---|---|---|---|
| **mpjpe_g** | 118.16 | 101.80 | 105.75 | 91.44 | 105.43 |
| **mpjpe_l** | 29.65 | 29.51 | 29.48 | 29.33 | 29.30 |
| **mpjpe_pa** | 21.65 | 20.52 | 20.01 | 19.70 | 20.01 |
| **accel_dist** | 1.14 | 1.03 | 1.12 | 1.09 | 1.19 |
| **vel_dist** | 3.27 | 2.94 | 3.07 | 2.96 | 3.21 |

### Unstructured Motion
| Metric | 50k | 80k | 100k | 120k | 150k |
|---|---|---|---|---|---|
| **mpjpe_g** | 144.50 | 153.57 | 151.26 | 144.78 | 150.35 |
| **mpjpe_l** | 31.05 | 32.73 | 33.37 | 36.61 | 32.01 |
| **mpjpe_pa** | 23.20 | 23.66 | 23.34 | 23.78 | 22.96 |
| **accel_dist** | 0.85 | 0.86 | 0.87 | 0.90 | 0.86 |
| **vel_dist** | 2.92 | 2.92 | 2.96 | 3.08 | 2.95 |

### Basic Locomotion
| Metric | 50k | 80k | 100k | 120k | 150k |
|---|---|---|---|---|---|
| **mpjpe_g** | 259.89 | 214.81 | 203.52 | 194.14 | 184.67 |
| **mpjpe_l** | 27.92 | 28.29 | 28.38 | 27.65 | 28.03 |
| **mpjpe_pa** | 21.88 | 20.93 | 21.01 | 20.37 | 20.21 |
| **accel_dist** | 1.40 | 1.37 | 1.39 | 1.38 | 1.38 |
| **vel_dist** | 4.46 | 4.10 | 4.20 | 4.15 | 4.04 |

---

## 9. Success Metrics Progression by Category (50k to 150k)

### Agility & Acrobatics
| Metric | 50k | 80k | 100k | 120k | 150k |
|---|---|---|---|---|---|
| **mpjpe_g** | 105.98 | 128.49 | 119.47 | 106.16 | 118.28 |
| **mpjpe_l** | 25.54 | 28.77 | 28.80 | 27.98 | 28.04 |
| **mpjpe_pa** | 19.70 | 20.94 | 19.94 | 19.99 | 19.99 |
| **accel_dist** | 1.32 | 1.55 | 1.52 | 1.47 | 1.58 |
| **vel_dist** | 3.90 | 4.42 | 4.27 | 4.13 | 4.34 |

### Complex Gestures / Dance
| Metric | 50k | 80k | 100k | 120k | 150k |
|---|---|---|---|---|---|
| **mpjpe_g** | 118.16 | 101.80 | 105.75 | 91.44 | 105.43 |
| **mpjpe_l** | 29.65 | 29.51 | 29.48 | 29.33 | 29.30 |
| **mpjpe_pa** | 21.65 | 20.52 | 20.01 | 19.70 | 20.01 |
| **accel_dist** | 1.14 | 1.03 | 1.12 | 1.09 | 1.19 |
| **vel_dist** | 3.27 | 2.94 | 3.07 | 2.96 | 3.21 |

### Unstructured Motion
| Metric | 50k | 80k | 100k | 120k | 150k |
|---|---|---|---|---|---|
| **mpjpe_g** | 144.50 | 153.57 | 151.26 | 144.78 | 150.35 |
| **mpjpe_l** | 31.05 | 32.73 | 33.37 | 36.61 | 32.01 |
| **mpjpe_pa** | 23.20 | 23.66 | 23.34 | 23.78 | 22.96 |
| **accel_dist** | 0.85 | 0.86 | 0.87 | 0.90 | 0.86 |
| **vel_dist** | 2.92 | 2.92 | 2.96 | 3.08 | 2.95 |

### Basic Locomotion
| Metric | 50k | 80k | 100k | 120k | 150k |
|---|---|---|---|---|---|
| **mpjpe_g** | 259.89 | 214.81 | 203.52 | 194.14 | 184.67 |
| **mpjpe_l** | 27.92 | 28.29 | 28.38 | 27.65 | 28.03 |
| **mpjpe_pa** | 21.88 | 20.93 | 21.01 | 20.37 | 20.21 |
| **accel_dist** | 1.40 | 1.37 | 1.39 | 1.38 | 1.38 |
| **vel_dist** | 4.46 | 4.10 | 4.20 | 4.15 | 4.04 |

---

## 10. Full Dataset Category Statistics 

Based on the 16,270 files available in the large unified dataset `/home/grease/ego_dataset/smpl_filtered_to_bvh_csv_robot_filtered`, here is the breakdown of motion types:

| Category | Size | Percentage |
|---|---:|---:|
| **Basic Locomotion** | 14,723 | 90.5% |
| **Agility & Acrobatics** | 748 | 4.6% |
| **Unstructured Motion** | 423 | 2.6% |
| **Complex Gestures / Dance** | 376 | 2.3% |
| **Total** | **16,270** | **100.0%** |

*Note: This heavily skewed distribution emphasizes normal walking/turning locomotion, which reinforces why succeeding on the Out-of-Distribution elements (Flips, Dances, Crawls) is a strong indicator of model generalization versus overfitting.*

---

## 11. Full 77-Clip LAFAN1 Sweep (2026-08-24)

Ran the full 77-motion LAFAN1 directory (`/tmp/lafan1_all/motion_lib_individual/g1_csv`, +1 stray
`walk1_subject1_chunked.pkl` = 78 total) through `eval_agent_trl.py` in a single parallel batch
(`num_envs=77`), for three checkpoints. **Note:** the `ee_body_pos`, `foot_pos_xyz`, and
`anchor_pos` termination thresholds were overridden to `10000.0` (effectively disabled) to work
around the RSI floor-penetration instant-death issue documented in `LAFAN1_eval_investigation.md`
(Sections 3.3-3.5) — so these numbers reflect *underlying tracking capability with safety
terminations relaxed*, not the standard eval harness's default pass/fail bar.

### Overall success / progress rate

| Checkpoint | success_rate | progress_rate |
|---|---|---|
| `sonic_release/last.pt` | **11.5%** (9/78) | **35.3%** |
| low_latency variant | 5.1% (4/78) | 23.9% |
| `sonic_pretrained/model_step_150000.pt` | 5.1% (4/78) | 23.7% |

`last.pt` is the clear best of the three, roughly 2x the success rate of the other two.

### Breakdown by LAFAN1 motion category

(Categorization inferred from LAFAN1's native action-type prefixes — walk/run/sprint → Basic
Motion; jumps/fight/fightAndSports/dance → Agile Motion; push/pushAndFall/pushAndStumble/
fallAndGetUp/ground → Interaction/Perturbation; aiming/multipleActions/obstacles →
Misc/Object-Obstacle. Distinct from this doc's AMASS-based
Agility-Acrobatics/Gestures-Dance/Unstructured/Basic-Locomotion taxonomy in Sections 9-10 above,
since LAFAN1's file-naming scheme differs from the AMASS `eval_subset` naming used elsewhere in
this doc.)

**`last.pt`:**

| Category | n | success% | progress% | mpjpe_l | mpjpe_g | mpjpe_pa | accel_dist | vel_dist |
|---|---|---|---|---|---|---|---|---|
| Basic Motion | 19 | 31.6% | 60.9% | 81.8 | 3581.5 | 62.7 | 6.80 | 11.74 |
| Agile Motion | 16 | 6.2% | 31.1% | 82.6 | 2203.7 | 62.6 | 7.80 | 13.36 |
| Misc/Object-Obstacle | 26 | 7.7% | 29.5% | 86.6 | 3052.9 | 65.7 | 5.15 | 10.55 |
| Interaction/Perturbation | 17 | 0.0% | 19.5% | 84.5 | 992.4 | 62.2 | 4.99 | 8.38 |

**low_latency variant:**

| Category | n | success% | progress% | mpjpe_l | mpjpe_g | mpjpe_pa | accel_dist | vel_dist |
|---|---|---|---|---|---|---|---|---|
| Basic Motion | 19 | 21.1% | 43.2% | 76.1 | 3105.6 | 66.0 | 7.54 | 12.93 |
| Misc/Object-Obstacle | 26 | 0.0% | 19.2% | 76.4 | 2403.5 | 65.3 | 5.62 | 10.73 |
| Agile Motion | 16 | 0.0% | 19.2% | 83.0 | 2025.0 | 71.5 | 8.71 | 14.06 |
| Interaction/Perturbation | 17 | 0.0% | 14.0% | 77.4 | 812.0 | 64.8 | 5.31 | 8.62 |

**`sonic_pretrained/model_step_150000.pt`:**

| Category | n | success% | progress% | mpjpe_l | mpjpe_g | mpjpe_pa | accel_dist | vel_dist |
|---|---|---|---|---|---|---|---|---|
| Basic Motion | 19 | 21.1% | 46.0% | 71.4 | 3642.5 | 57.4 | 6.93 | 12.80 |
| Misc/Object-Obstacle | 26 | 0.0% | 19.3% | 70.4 | 1972.5 | 55.8 | 6.00 | 10.99 |
| Agile Motion | 16 | 0.0% | 15.8% | 75.2 | 1349.6 | 59.2 | 9.58 | 14.06 |
| Interaction/Perturbation | 17 | 0.0% | 13.0% | 64.0 | 837.5 | 50.3 | 5.30 | 8.38 |

### Observations

- **Basic Motion is the only category with meaningfully non-zero success** across all three
  checkpoints (21-32%). Every other category is at or near 0% success rate for `low_latency`/
  `sonic_pretrained`, and only marginally better for `last.pt` (6-8%).
- **Interaction/Perturbation (falls, pushes, recoveries) is a complete failure (0%) for all three
  checkpoints** — consistent with this being the hardest category (fast, large, near-ground-contact
  motions) and likely compounded by the RSI floor-penetration issue for these already-unstable
  reference poses.
- `sonic_pretrained` has the **lowest mpjpe_l/mpjpe_pa** in most categories despite near-zero
  success rate, suggesting it tracks accurately in the brief windows before falling, but recovers/
  survives far less often than `last.pt`.
- Progress-rate gaps between checkpoints (e.g. Basic Motion: 60.9% vs 43-46%) are proportionally
  larger than the success-rate gaps, suggesting `last.pt`'s advantage is partly in *how far it gets*
  before failing, not purely a binary pass/fail difference.
- These results reinforce the root-cause finding in `LAFAN1_eval_investigation.md`: even with
  hard safety terminations disabled, tracking quality/survivability on LAFAN1 data remains far
  below the ~96% success rate reported there for `eval_subset` — consistent with the ~4-5cm
  leg-scale-induced floor-penetration bias still degrading stability throughout the rollout, not
  just at frame 0.


Evalution script:
example:

GEAR_SONIC_DEBUG_RESET=1 GEAR_SONIC_DEBUG_PERIODIC=1 GEAR_SONIC_DEBUG_PERIODIC_EVERY=200 /home/grease/miniforge3/envs/env_isaaclab/bin/python gear_sonic/eval_agent_trl.py checkpoint=/home/grease/gam/gear_sonic_deploy/policy/low_latency/last.pt +headless=true +num_envs=32 +manager_env.commands.motion.motion_lib_cfg.motion_file=/home/grease/ego_dataset/eval_subset/robot +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/home/grease/ego_dataset/eval_subset/smpl eval_name=EVAL_subset_PRETRAINED algo.config.eval.num_eval_episodes=1 +run_once=true +eval_callbacks=im_eval +eval_output_dir='${eval_log_dir}'

++manager_env.terminations.ee_body_pos.params.threshold=10000.0 ++manager_env.terminations.foot_pos_xyz.params.threshold=10000.0 ++manager_env.terminations.anchor_pos.params.threshold=10000.0 ++manager_env.terminations.anchor_ori_full.params.threshold=10000.0 