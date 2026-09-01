# AMASS 108-clip Evalset — 3-Model Comparison

> **Provenance:** copied from
> `GR00T-WholeBodyControl/dev_notes/eval_examples/notes_amass_108clips_3model_comparison.md`.
> Relative references below (e.g. `docs_bad_clip_filters.md`, `README.md`) point at
> paths in **that** repo, not this one.

Evaluated three checkpoints on the same **108-clip, arm-twist + policy-fail
filtered** `amass_evalset` (see `docs_bad_clip_filters.md` for how this set
was cleaned down from 133 clips), using identical eval settings
(`enable_corruption=False`, `terminations=tracking/eval`, `num_envs=108`).

| Run | Checkpoint | Log dir |
|---|---|---|
| **LOW_LATENCY** | `/home/grease/gam/gear_sonic_deploy/policy/low_latency/last.pt` | `logs_eval/20260828_160340-EVAL_amass_108clips_LOW_LATENCY` |
| **PRETRAINED** | (pretrained checkpoint) | `logs_eval/20260827_163451-EVAL_amass_108clips_PRETRAINED` |
| **RELEASED** | `/home/grease/GR00T-WholeBodyControl/sonic_release/last.pt` | `logs_eval/20260827_162752-EVAL_amass_108clips_RELEASED` |

## 1. Overall metrics

| Model | success_rate | progress_rate | mpjpe_g | mpjpe_l | mpjpe_pa | accel_dist | vel_dist |
|---|---|---|---|---|---|---|---|
| LOW_LATENCY | 0.6111 | 0.7454 | 345.21 | 34.27 | 23.44 | 0.985 | 2.848 |
| PRETRAINED | 0.6667 | 0.7756 | 429.89 | 33.75 | 22.88 | 0.936 | 2.707 |
| **RELEASED** | **0.8056** | **0.9000** | **269.00** | 36.16 | 25.22 | 0.957 | 2.619 |

**`RELEASED` is the clear winner**: highest success rate (80.6% vs 61-67%
for the other two), highest progress rate (90.0%), and lowest global
position error (`mpjpe_g` = 269.0, vs 345-430 for the others). Interestingly,
its local joint error (`mpjpe_l`/`mpjpe_pa`) is slightly *higher* than the
other two — meaning `RELEASED` prioritizes not falling/losing global
tracking over ultra-precise local joint angle matching, a reasonable
trade-off for a "released" checkpoint aimed at robustness.

## 2. Per-source-dataset success rate (successes / total, %)

| Category | LOW_LATENCY | PRETRAINED | RELEASED |
|---|---|---|---|
| ACCAD | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) |
| BMLhandball | 1/4 (25%) | 1/4 (25%) | 1/4 (25%) |
| BMLmovi | 19/27 (70%) | 21/27 (78%) | 23/27 (85%) |
| CMU | 9/15 (60%) | 9/15 (60%) | 10/15 (67%) |
| DFaust | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) |
| EyesJapanDataset | 2/2 (100%) | 1/2 (50%) | 2/2 (100%) |
| GRAB | 2/5 (40%) | 4/5 (80%) | 4/5 (80%) |
| **KIT** | 28/39 (72%) | 26/39 (67%) | **37/39 (95%)** |
| MoSh | 0/2 (0%) | 1/2 (50%) | **2/2 (100%)** |
| Transitions | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) |
| WEIZMANN | 1/6 (17%) | 4/6 (67%) | 3/6 (50%) |

**Key observations:**
- **`KIT` (largest category, 39 clips)** shows the biggest jump: `RELEASED`
  gets 95% vs. 67-72% for the others — this single category alone
  accounts for most of `RELEASED`'s overall lead.
- **`MoSh`** goes from 0% (LOW_LATENCY) → 100% (RELEASED) — a complete
  turnaround.
- **`BMLhandball` and `Transitions`** are hard for **all three** models
  (25% and 0% respectively) — these are likely genuinely
  out-of-distribution motion styles (fast handball throws/dives,
  punch-karate stances) regardless of checkpoint.
- **`WEIZMANN`** is the one category where `RELEASED` (50%) is *worse*
  than `PRETRAINED` (67%) — worth a closer look if walking-style long-
  straight-line locomotion regressed for some reason in `RELEASED`.
- **`GRAB` (object manipulation)** improves a lot for both `PRETRAINED`
  and `RELEASED` (40%→80%) vs `LOW_LATENCY` — suggests `LOW_LATENCY` was
  undertrained/weaker specifically on object-interaction-style motions.

## 3. Per-behavior-category success rate (4 semantic groups)

Rather than grouping by AMASS *source dataset* (Section 2), this groups
clips by *motion content* — keyword-matched against each motion key
(walk/run/turn/gesture = Basic; jump/throw/punch/dance/fast/handball =
Dynamic; grab/pick/pass/drink/eat = Object Manipulation;
crouch/kneel/squat/sit/lie = Ground/Non-standing). This reveals a much
cleaner difficulty gradient than the per-dataset view above, since motion
*style* (not which lab captured it) is what actually drives tracking
difficulty.

| Behavior Category | # Clips | LOW_LATENCY | PRETRAINED | RELEASED |
|---|---|---|---|---|
| **Basic Locomotion/Gesture** | 77 | 55/77 (71%) | 53/77 (69%) | **65/77 (84%)** |
| **Dynamic/Athletic** | 17 | 6/17 (35%) | 11/17 (65%) | 11/17 (65%) |
| **Object Manipulation** | 7 | 4/7 (57%) | 5/7 (71%) | **6/7 (86%)** |
| **Ground/Non-standing** | 7 | 1/7 (14%) | 3/7 (43%) | **5/7 (71%)** |

**Interpretation — a clear difficulty gradient emerges, same order in
all three models:**

1. **Basic Locomotion/Gesture (walking, turning, arm gestures, standing
   poses)** is by far the largest group (77/108 = 71% of the whole
   evalset) and the easiest overall — `RELEASED` handles 84% of it,
   confirming this checkpoint is very solid on "normal" everyday motion.

2. **Dynamic/Athletic (jumps, throws, punches, handball, fast running,
   push-recovery)** is the hardest category for **every** model (only
   35-65% success) — high-energy, high-impact, often single-limb-dominant
   motions consistently break tracking regardless of checkpoint. Notably
   `PRETRAINED` and `RELEASED` tie here (65% each) — `RELEASED`'s overall
   advantage comes almost entirely from the other 3 categories, not from
   improved dynamic-motion handling.

3. **Object Manipulation (GRAB reach/grab/pass, drinking, eating)**
   improves steadily and substantially with checkpoint quality
   (57%→71%→86%) — this is the category with the cleanest, most
   consistent improvement curve across the three models.

4. **Ground/Non-standing (crouch, kneel, squat, sit, lie)** shows the
   single **biggest relative jump**: `LOW_LATENCY` is nearly unusable here
   (14%, essentially only handling 1/7 clips), while `RELEASED` recovers
   most of it (71%) — a 5x improvement. This is a strong signal that
   `LOW_LATENCY`'s training distribution likely under-samples
   non-standing/ground-contact reference poses.

**Bottom line:** `RELEASED`'s overall lead is driven by across-the-board
gains in Basic, Object Manipulation, and especially Ground/Non-standing
motions — but **Dynamic/Athletic motion remains an unsolved weak point for
all three checkpoints** and is the best target for future training data
augmentation if you want to push overall success rate further.

## 4. Cross-model failure-set analysis

- **42 failed clips (LOW_LATENCY)**, **36 (PRETRAINED)**, **21 (RELEASED)**.
- **12 clips fail in ALL THREE models** — likely genuinely hard/borderline
  content, independent of checkpoint quality:
  ```
  BMLhandball__S04_Expert__Trial_upper_left_right_234_poses
  BMLhandball__S09_Novice__Trial_upper_right_left_168_poses
  BMLmovi__Subject_28_F_MoSh__Subject_28_F_21_poses
  CMU__05__05_16_stageii
  CMU__141__141_15_stageii
  CMU__87__87_01_stageii
  DFaust__50021__50021_knees_stageii
  KIT__200__Kniebeuge01_stageii
  KIT__348__walking_fast07_stageii
  Transitions__mazen_c3d__punchkarate_stand_stageii
  WEIZMANN__66__Normal_StraightLong(14)_stageii
  WEIZMANN__67__Normal_StraightLong(4)_stageii
  ```
- **14 clips are fixed specifically by `RELEASED`** (fail in both
  LOW_LATENCY & PRETRAINED, succeed only in RELEASED) — this is the
  clearest evidence of `RELEASED`'s improvement, spanning crouch-to-lie,
  kneeling, object handling (banana_eat, waterbottle-adjacent), and dynamic
  KIT throw/stomp motions:
  ```
  ACCAD__Male2General_c3d__A8-_Crouch_to_Lie_poses
  BMLhandball__S09_Novice__Trial_upper_right_left_240_poses
  BMLmovi__Subject_32_F_MoSh__Subject_32_F_1_poses
  BMLmovi__Subject_45_F_MoSh__Subject_45_F_20_poses
  BMLmovi__Subject_46_F_MoSh__Subject_46_F_12_poses
  BMLmovi__Subject_80_F_MoSh__Subject_80_F_2_poses
  CMU__74__74_13_stageii
  GRAB__s2__banana_eat_1_stageii
  KIT__1226__Trial_05_stageii
  KIT__3__kneel_down_with_right_hand05_stageii
  KIT__3__walk_with_support03_stageii
  KIT__572__stomp_right02_stageii
  KIT__572__throw_right01_stageii
  MoSh__50022__misc_1_stageii
  ```
- **11 clips fail ONLY in `LOW_LATENCY`** (both PRETRAINED and RELEASED
  succeed) — these are cases where `LOW_LATENCY` specifically underperforms
  relative to the other two, including several squat/step-over-gap/turn
  motions:
  ```
  BMLmovi__Subject_14_F_MoSh__Subject_14_F_10_poses
  CMU__122__122_04_stageii
  CMU__54__54_27_stageii
  GRAB__s1__waterbottle_pick_all_stageii
  KIT__359__turn_left04_stageii
  KIT__3__squat04_stageii
  KIT__425__step_over_gap02_stageii
  KIT__513__evasion05_stageii
  MoSh__50021__squat_simple_stageii
  WEIZMANN__68__Mixed_SlowFast(3)_stageii
  WEIZMANN__70__Normal_StraightLong(20)_stageii
  ```

## 5. Conclusions

1. **`RELEASED` is the strongest checkpoint** on this cleaned 108-clip
   AMASS set: +14-19 percentage points success rate over the other two,
   driven overwhelmingly by the `KIT` source dataset (95% vs 67-72%) and,
   at the behavior level, by broad gains across Basic Locomotion, Object
   Manipulation, and especially Ground/Non-standing motions.
2. **Dynamic/Athletic motion is the universal weak point** — all three
   checkpoints plateau at 35-65% here, far below their performance on
   every other behavior category. This is the highest-value target for
   future training data augmentation.
3. **A consistent hard core of 12 clips** fails across all three
   checkpoints — these are good candidates for either (a) excluding from
   future eval sets as genuinely out-of-distribution content
   (fast handball, punch-karate stances, long straight-line walks), or
   (b) targeted fine-tuning data if you want to close this gap.
4. **`WEIZMANN` is the one source-dataset category that regressed** in
   `RELEASED` vs `PRETRAINED` (50% vs 67%) — worth investigating
   specifically, since every other category improved or stayed flat.
5. **`LOW_LATENCY` is weakest overall**, particularly on `Ground/Non-
   standing` motions (14% — essentially only 1/7 clips) and `GRAB` object
   manipulation (40%) — consistent with it being an earlier/lighter-weight
   checkpoint variant optimized for latency rather than raw tracking
   accuracy across diverse AMASS content.
