# ICL settle study: why hold-failure clips fall at the settle pose (2026-10-09)

## 1. Question

`ICL_hardset_settle2` (= `ICL_hardset_aligned` with frame 0 held 2 s before the motion) showed 10-12 of 35 clips failing
**during the hold** (t < 2 s, failure at 0.12-0.38 s) for every checkpoint (`icl_settle_results.md`). Why can the policy
not stay at a static start pose, even though the spawn velocity is zero, and can the set be fixed so the settle stage is
a meaningful test?

Checkpoints: `base100k` (`sonic_no_vr_low_latency_with_amass_envs_16K/model_step_100000.pt`) and
`mixv3b_010000 / 014000 / 020000` (`sonic_release_no_teleop_mixpico_v3b-20261005_164106`).

## 2. Findings

### 2.1 The failures are caused by the start pose, not the policy
- The failure second with the hold is about the same as without it (e.g. 0.2 vs 0.22), so a hold only exposes a bad start.
- Spawn velocity is already zero. The motion library (`motion_lib_base.py`, `torch_humanoid_batch.py`) gets root velocity by
  `np.gradient` + Gaussian filter (sigma 2) and dof velocity by forward difference. `build_settle_set.py` repeats frame 0 for
  60 frames, so on `run1_subject5__w035s` root speed is 0.0 m/s for frames 0-51 and joint velocity is 0.0 through frame 59.
  The velocity only rises with the real motion from about frame 55.
- The eval resets the robot to the reference pose and velocity (RSI) in `gear_sonic/envs/manager_env/mdp/commands.py`;
  `disable_rsi` would spawn from the default standing pose instead. With a frozen reference, any airborne frame 0 cannot be
  tracked whatever the spawn velocity.

### 2.2 Frame-0 foot contact (real foot collision geometry, `diag_frame0_contact.py`)
Sole heights are above the ground in m; "CoM off" is the distance of the CoM (xy) outside the support hull (negative = inside).

| Clip | Sole L / R | Drop | Free-fall time | Root tilt | CoM off |
|---|---|---|---|---|---|
| run1_subject5 | 0.55 / 0.60 | 0.55 | 0.33 s | 28° | -0.005 |
| fight1_subject3 | 0.30 / 0.38 | 0.30 | 0.25 s | 7° | 0.21 |
| jumps1_subject1 | 0.24 / 0.31 | 0.24 | 0.22 s | 13° | 0.22 |
| dance1_subject2 | 0.16 / 0.56 | 0.16 | 0.18 s | 17° | 0.06 |
| fight1_subject2 | 0.16 / 0.14 | 0.14 | 0.17 s | 11° | -0.01 |
| dance2_subject4 | 0.15 / 0.13 | 0.13 | 0.16 s | 4° | 0.08 |
| jumps1_subject2 | 0.11 / 0.27 | 0.11 | 0.15 s | 10° | 0.26 |
| run2_subject4 | 0.14 / 0.11 | 0.11 | 0.15 s | 28° | 0.24 |
| multipleActions1_subject3 | 0.10 / 0.10 | 0.10 | 0.14 s | 7° | -0.02 |
| fightAndSports1_subject1 | 0.07 / 0.02 | 0.02 | 0.07 s | 15° | 0.24 |
| multipleActions1_subject1 | 0.01 / 0.25 | 0.01 | 0.04 s | 12° | 0.01 |

- None of the 11 frame-0 poses is standable. Most are airborne (10-55 cm); the rest are single-foot mid-stride poses
  (foot pitch down to -34°), and 6 have the CoM outside the support. Root speed between frames 0 and 1 is 0.7-2.3 m/s.
- In the settle set the **reference is frozen in that airborne pose for 2 s**, so the robot falls and diverges from it.
  Failure before the free-fall time (e.g. `run1_subject5`, 0.16 s) is reference-vs-robot divergence; after it (e.g.
  `multipleActions1_subject1`, 0.28 s) is impact or tilt recovery.
- `mixv3b_020000` survived the hold on `dance2_subject4` and `fight1_subject2` only: 13-14 cm drops, flat feet, little
  tilt, and a reference whose feet come down to about 5 cm within 4-5 frames (about the 0.16 s fall time).
- The earlier "ankle link z <= 0.08 m" test used the ankle origin, not the sole. For `multipleActions1_subject1` the ankle
  origin reads 0.116 m while the sole is at 0.007 m, so it misclassified clips.

### 2.3 Corrections to `icl_settle_results.md` (2 s set)
- Section 2 lists `dance2_subject1__w088s` as a hold failure but the table has it as a motion failure (+0.46 s), and it omits
  `fightAndSports1_subject1__w062s`, which fails the hold in every run.
- `dance1_subject1__w100s` fails the hold on `mixv3b_014000` and `mixv3b_020000` (0.86 s, 1.22 s) although it passes on the
  other two checkpoints. This is a standing-stability regression, not a spawn glitch.
- The "18 motion failures" list names 17 clips and omits `dance1_subject1`.

## 3. Interventions

### 3.1 Re-cut at a clean start frame (`find_clean_start_window.py`, `build_recut_settle_set.py`)
Clean frame = a sole on the ground, CoM inside the support, low tilt and root speed. The window is located in the full
`robot_v2` sequence by exact dof match, the same root-z shift is reapplied, and only the start is slid (window length is kept).

| Clip | Shift | Result |
|---|---|---|
| dance2_subject4 | +0.57 s | clean (soles 2 cm, CoM inside, tilt 6.6°) |
| fightAndSports1_subject1 | -1.10 s | clean (needs a ±3 s search) |
| other 9 | up to +1.43 s within range | no clean frame at relaxed thresholds (sole <= 3 cm / 15 cm, CoM <= 3 cm, tilt <= 12°, speed <= 2.5 m/s) |

Built: `~/ego_dataset/ICL_hardset_recut_aligned` and `ICL_hardset_recut_settle2` (the two clean clips). Fast runs, jumps and
fights almost always have a swinging foot, so a static clean pose rarely exists near these windows.

### 3.2 Grounded settle (`build_grounded_settle_set.py`)
Lower the held pose so the lowest sole is at z = 0 (robot root z and SMPL transl / joints), and fade the offset out
over the first 1 s of motion. Joints, orientation, xy and zero velocity are unchanged. Eval of the 11 clips alone
(base100k, default terminations):

| Failure time | Clips |
|---|---|
| passes a 2 s hold | fight1_subject2 4.67, dance2_subject4 3.33, fightAndSports1_subject1 2.63, multipleActions1_subject3 2.30, dance1_subject2 2.24, fight1_subject3 2.24, multipleActions1_subject1 2.12 |
| still fails the hold | jumps1_subject1 0.98, run1_subject5 0.42, run2_subject4 0.36, jumps1_subject2 0.34 |

The four that still fail have tilt (about 28° on run1/run2) or the CoM 22-26 cm outside the support, so a lowered pose is
still not a standable one.

### 3.3 `ICL_hardset_settle` (final set, 35 clips)
- 7 hold clips: grounded, 2 s hold. 4 unholdable clips (`run1_subject5`, `run2_subject4`, `jumps1_subject1`,
  `jumps1_subject2`): grounded, **0.2 s** settle. 24 other clips: copied from `ICL_hardset_settle2` (2 s hold, unchanged).
- `settle_info.json` has `settle_s`, `motion_start_s` and `grounded` per clip. SMPL length equals the loader-resampled robot
  length for all 35.
- Report: `sim2real/icl_settle_report.py` (`SET`, `PREFIX`, `OUT` env vars), output `icl_settle_results_ICL_hardset_settle.md`.

Results (35 clips, default terminations; the hold is each clip's own `motion_start_s`):

| Run | Fail in hold | Fail in motion | OK |
|---|---|---|---|
| base100k | 2 | 27 | 6 |
| mixv3b_010000 | 3 | 28 | 4 |
| mixv3b_014000 | 4 | 26 | 5 |
| mixv3b_020000 | 5 | 26 | 4 |

The 11 targeted clips (failure second):

| Clip | Hold | base100k | mix 010000 | mix 014000 | mix 020000 |
|---|---|---|---|---|---|
| fightAndSports1_subject1 | 2.0 | 0.20 HOLD | 0.22 HOLD | 0.24 HOLD | 0.20 HOLD |
| multipleActions1_subject1 | 2.0 | 0.26 HOLD | 0.24 HOLD | 0.22 HOLD | 0.22 HOLD |
| dance1_subject2 | 2.0 | 2.59 | 2.12 | 2.14 | 0.32 HOLD |
| dance2_subject4 | 2.0 | 2.89 | 2.93 | 2.89 | 2.89 |
| fight1_subject2 | 2.0 | 4.11 | 4.65 | 4.67 | 4.17 |
| fight1_subject3 | 2.0 | 2.22 | 2.24 | 2.24 | 2.22 |
| multipleActions1_subject3 | 2.0 | 2.10 | 2.26 | 2.26 | 2.24 |
| run1_subject5 | 0.2 | 0.24 | 0.14 HOLD | 0.16 HOLD | 0.18 HOLD |
| run2_subject4 | 0.2 | 0.42 | 0.28 | 0.24 | 0.24 |
| jumps1_subject1 | 0.2 | 0.52 | 0.52 | 0.52 | 0.46 |
| jumps1_subject2 | 0.2 | 0.60 | 0.54 | 0.62 | 0.60 |

The four short-settle clips fail 0.04-0.42 s into the motion, about as early as without a hold (0.2-0.34 s), so the short settle
only moves them from "hold" to "motion" failure.

## 4. Caveats / open questions
- **Batch effect:** `fightAndSports1_subject1` and `multipleActions1_subject1` survive the hold in the 11-clip eval
  (2.6 s, 2.1 s) but fail at 0.2 s in the 35-clip eval with byte-identical data (md5 checked). Two identical 35-clip base100k
  runs give identical results, so it is deterministic but depends on batch composition / env slot. Not yet traced. A test would
  be to run the 11 clips alone and inside the 35, or each clip with a single env.
- The grounded offset fades out over 1 s of motion, which adds a vertical reference velocity; several clips fail within 0.3 s
  of the motion starting. Whether the ramp contributes has not been checked.
- Videos (base100k, failure terminations relaxed, one env, 10 s): `~/GR00T-WholeBodyControl/model_eval/`
  `vis_iclsettle_base100k_FULL/` (original 2 s set), `vis_iclsettle_base100k_grounded_FULL/` (grounded) and
  `vis_iclsettle_base100k_settlev2_FULL/` (the two clips that fail on `ICL_hardset_settle`). They were not watched when this
  was written, so the description of how the robot falls is inferred from the frame-0 numbers.
- Hold-clip ordering in sections 3.2 / 3.3 is the same 11 clips throughout; `ICL_hardset_settle` supersedes `ICL_hardset_settle2`
  only for these 11.

## 5. Reproduce
```bash
cd ~/gam
.venv_sim/bin/python sim2real/diag_frame0_contact.py                    # frame-0 foot contact table
.venv_sim/bin/python sim2real/find_clean_start_window.py --sole-lo 0.03 --sole-hi 0.15 --com-tol 0.03 --tilt 12 \
    --root-speed 2.5 --max-back-s 3 --max-fwd-s 1.5                      # clean start frames -> clean_start_frames.csv
.venv_sim/bin/python sim2real/build_recut_settle_set.py                  # re-cut + settle set for the clean clips
.venv_sim/bin/python sim2real/build_grounded_settle_set.py               # ICL_hardset_settle (grounded + short settle)
./model_eval/run_icl_settle_evals.sh                                     # 4 checkpoints on ICL_hardset_settle
.venv_sim/bin/python sim2real/icl_settle_report.py                       # -> icl_settle_results_ICL_hardset_settle.md
DS=~/ego_dataset/ICL_hardset_settle TAG=base100k ./model_eval/vis_icl_settle_fail.sh <clip> ...   # render videos (FULL=0 keeps terminations)
```

## 6. Files
| File | Purpose |
|---|---|
| `sim2real/diag_frame0_contact.py` | frame-0 sole height, foot angles, CoM vs support, reference trace |
| `sim2real/find_clean_start_window.py` | search for the nearest clean start frame in the full sequence |
| `sim2real/build_recut_settle_set.py` | re-cut clean clips and build their settle set |
| `sim2real/build_grounded_settle_set.py` | grounded settle + short settle for unholdable clips -> `ICL_hardset_settle` |
| `sim2real/icl_settle_report.py` | per-clip hold/motion classification with per-clip hold length |
| `model_eval/run_icl_settle_evals.sh` | evals of the four checkpoints on a settle set |
| `model_eval/vis_icl_settle_fail.sh` | render per-clip videos with the policy |
