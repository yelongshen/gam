# Online Evaluation — `llam_080k` on real G1, 2026-10-01

First hardware evaluation of `llam_080k` (`policy/sonic_no_vr_llam_080k`, confirmed from
`metadata.json`). It closes the open item in `0928_official_report.md` §5.

**Data.** Robot logs `~/g1_robot_data/g1_run_1001` (10 runs) + streamed SMPL
`~/g1_robot_data/20261001` (10 `streamed_*` sessions). Pipeline: `sim2real/eval_online_runs.py`
(identify clip from SMPL stream and from executed `q`, pair, score with
`online_eval_clips.evaluate`). Raw results: `sim2real/online_eval_results_20261001.json`.
Metrics are defined as in `0928_official_report.md` §0. Reference dir was `reference/example/`.

**Bottom line.** On the four manifest clips where the other policies also ran, `llam_080k`
is **statistically tied with `ll_062k`** (30.6 vs 30.1 mm on the 3-clip macro-mean; 29.3 vs
30.3 on 4 clips). Differences are far below the ~10 mm resolution of this data. No falls
in 10 episodes. The offline win of `llam_080k` therefore **does not hurt**, but also
**does not show up as an online gain on in-distribution locomotion**. Its offline advantage
was on OOD (`amass_evalset`), which these clips barely probe.

**Sim vs real (§4b).** Against the simulator evaluation of the *same checkpoint on the same clips*
(`logs_eval/20260921_121122-EVAL_subset_LLAM080k`), tracking error on the robot is higher:
`mpjpe_l` is **27 % above sim over all six clips (26.9 -> 34.1 mm) and 46 % above on the four
confirmed locomotion clips (20.1 -> 29.3 mm)**. The gap is in the **legs** (1.45 x sim on
locomotion, 29.0 -> 42.0 mm); the upper body matches (0.98 x over all clips, 1.13 x on locomotion).
`accel_dist` is 1.55-1.65 x and `vel_dist` 1.2-1.3 x sim. Per-clip gaps are noisy (1-2 real
reps per clip); the macro-mean is the number to use. So the sim2real gap on tracking is **not
small**, even though `llam_080k` is no worse than `ll_062k` on hardware.

---

## 1. Episodes

| run | clip | identification | `mpjpe_l` | `mpjpe_pa` | joint err | heading err | shaking (rad/s) | max tilt |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 083356 | `walk_180_R_003__A332_M` | confirmed | 37.5 | 35.2 | 5.8° | 3.4° | 0.32 | 9.6° |
| 083624 | `walk_backward_start_001__A030_M` | confirmed | 25.5 | 21.8 | 4.9° | 2.5° | 0.48 | 5.8° |
| 083950 | `jog_ff_start_180_R_002__A192_M` | confirmed | 38.3 | 30.7 | 6.3° | 2.9° | 0.68 | 13.5° |
| 084400 | `walk_sideway_045_stop_005__A042_M` | confirmed | 24.4 | 17.3 | 4.5° | 1.8° | 0.24 | 9.2° |
| 084808 | `walk_180_R_003__A332_M` | confirmed | 22.7 | 20.8 | 4.7° | 2.1° | 0.32 | 7.9° |
| 085817 | `jog_ff_start_180_R_002__A192_M` | confirmed | 36.4 | 27.3 | 5.9° | 2.5° | 0.71 | 13.1° |
| 085157 | `dance_vouge_shake_it_babe…` | **unconfirmed** | 44.0 | 33.6 | 11.4° | 5.3° | 1.33 | 20.0° |
| 085443 | `dance_vouge_shake_it_babe…` | **unconfirmed** | 43.5 | 31.8 | 11.4° | 4.9° | 1.35 | 18.6° |
| 090040 | `high_jump_R_103__A389_M` | **unconfirmed** | 41.1 | 32.0 | 8.7° | 4.3° | 1.35 | 25.3° |
| 090302 | `high_jump_R_103__A389_M` | **unconfirmed** | 46.0 | 37.6 | 9.4° | 3.4° | 1.45 | 22.9° |

(`mpjpe` in mm.) **Unconfirmed** = the streamed SMPL names the clip, but the executed `q`
votes for a different clip (`idle_turn_270`, `jump_ff_360`).

**Checked 2026-10-01 (`sim2real/check_unconfirmed_1001.py`)**: sliding the streamed clip's
retargeted `dof` over the whole executed `q` of each run puts the best fit at **exactly the
frame the evaluator used** in all 4 episodes (offsets 2133 / 1618 / 2419 / 1454). So the
windows are time-aligned and the clip is the right one. They fail the robot-side identifier
only because tracking is loose: joint rms **16.6° / 16.5°** (dance) and **13.1° / 13.6°**
(high jump) vs 5.9–8.8° on confirmed runs (limit 12°), and a near-stationary clip
(`idle_turn_270`, 6.7–7.2°) then fits better than a badly tracked agile one. Their
`mpjpe_l` values are therefore valid measurements of **poor tracking on OOD clips**, not
pairing artefacts. They stay out of the headline locomotion mean because the two groups are
different clip types, not because they are unreliable.

| group | n | `mpjpe_l` | `mpjpe_pa` | joint err | cmd `mpjpe_l` | shaking | max tilt (mean / worst) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| confirmed (locomotion) | 6 | **30.80** | **25.52** | 5.33° | 83.5 | 0.459 | 9.8° / 13.5° |
| unconfirmed (dance, high jump) | 4 | 43.62 | 33.75 | 10.24° | 129.4 | 1.369 | 21.7° / 25.3° |

**Falls: 0 / 10** (35° threshold). The fall detector is **not valid for the jump clip**
(manifest §5.5), so "no fall" there is a weak statement; the tilt (23–25°) is nonetheless
the highest of any run here.

## 2. Clip-matched comparison against earlier policies

Earlier numbers from `0928_official_report.md` §2.2 (`mpjpe_l`, mm).

| clip | `low_latency` | `novr_050k` | `ll_062k` | **`llam_080k`** |
|---|:---:|:---:|:---:|:---:|
| `walk_180_R_003__A332_M` | 32.0 (n=7) | 25.9 (n=5) | 29.1 (n=2) | **30.1** (n=2; 37.5, 22.7) |
| `walk_sideway_045_stop_005__A042_M` | 49.3 (n=4) | 54.3 (n=2) | 24.9 (n=1) | **24.4** (n=1) |
| `jog_ff_start_180_R_002__A192_M` | 39.7 (n=3) | 32.2 (n=1) | 36.2 (n=1) | **37.3** (n=2; 38.3, 36.4) |
| `walk_backward_start_001__A030_M` | — | 46.6 (n=2) | 31.0 (n=5) | **25.5** (n=1) |
| **macro-mean, 3 shared clips** | 40.4 | 37.5 | 30.1 | **30.6** |
| macro-mean, 4 clips (≥2 policies) | — | — | 30.3 | **29.3** |

Reading:

1. `llam_080k` retains the `ll_062k` lateral-gait fix (sideway 24.4 mm, heading error 1.8°;
   `novr_050k` was 54 mm with 20–30° heading error).
2. Within `llam_080k`, same-clip reps differ by up to 15 mm (`walk_180`: 37.5 vs 22.7), which
   is larger than any between-policy difference with `ll_062k`. **No ranking between the two
   can be claimed.**
3. Jog is slightly worse than `ll_062k`/`novr_050k` (37.3 vs 36.2/32.2) and has the largest
   tilt among confirmed runs (13.5°) — consistent with jog being the highest-rate locomotion
   clip, but the gap is inside noise.
4. Both policies sit ~25 % below the `low_latency` baseline on matched clips.

## 3. Offline vs online

| checkpoint | offline rank | online rank (matched macro-mean) |
|---|:---:|:---:|
| `llam_080k` | 1 | 1–2 (30.6; tied with `ll_062k`) |
| `ll_062k` | 2 | 1–2 (30.1) |
| `novr_050k` | 3 | 3 (37.5) |
| `low_latency` | 4 | 4 (40.4) |

The online ordering is consistent with offline, with the top two **indistinguishable**. This
is the expected result if the offline gap between them is concentrated in OOD content: the
`eval_subset` gap was 0.007 success / ~1 mm L.

## 4. Out-of-distribution clips (indicative only)

- **Vogue dance** (2 reps): 43.5–44.0 mm, joint err 11.4°, worst joint `right_wrist_roll`
  (~40°), tilt 19–20°. Reps agree within 0.5 mm, so the result is repeatable even though it is
  unconfirmed.
- **High jump** (2 reps): 41–46 mm, tilt 23–25°, shaking 1.35–1.45 rad/s, worst joint
  `left_wrist_yaw` (~25°). Reps differ by 5 mm.
- No earlier policy has a confirmed score on these clips in `0928_official_report.md`; the
  manifest lists prior replays on 09-12/09-15 (dances ×3, jump ×5) under other policies, but
  those episodes were not scored there (policy attribution unchecked). **A direct OOD
  comparison with `ll_062k` is the natural next step** — it is where offline predicted
  `llam_080k` would win.

## 4b. Sim vs real, same checkpoint, same clips

The offline simulator evaluation of the same checkpoint already holds per-clip metrics:
`GR00T-WholeBodyControl/logs_eval/20260921_121122-EVAL_subset_LLAM080k/metrics_eval.json`
(`eval/all_metrics_dict`, 160 `eval_subset` clips keyed by `motion_keys`; none of the six clips
below terminated). The real values are the 1001 episodes above (mean over reps). Script:
`model_eval/sim_vs_real_llam080k.py`. Units: mpjpe in mm; `vel_dist` / `accel_dist` as defined in
`0928_official_report.md` §0.

### 4b.1 Per clip

| clip | n real (confirmed) | | mpjpe_l | mpjpe_pa | legs | upper | vel_dist | accel_dist |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `walk_180_R_003__A332_M` | 2 (2) | sim | 15.7 | 14.2 | 20.9 | 8.4 | 2.56 | 1.03 |
| | | real | 30.1 | 28.0 | 46.9 | 12.7 | 3.60 | 1.48 |
| | | **gap** | **+14.4** | +13.8 | **+26.0** | +4.2 | +1.04 | +0.44 |
| `walk_backward_start_001__A030_M` | 1 (1) | sim | 19.7 | 16.1 | 25.5 | 12.0 | 3.77 | 1.29 |
| | | real | 25.5 | 21.8 | 34.7 | 11.0 | 4.20 | 2.08 |
| | | **gap** | +5.7 | +5.7 | +9.2 | -1.1 | +0.43 | +0.79 |
| `jog_ff_start_180_R_002__A192_M` | 2 (2) | sim | 25.8 | 17.2 | 38.8 | 13.0 | 5.32 | 2.28 |
| | | real | 37.3 | 29.0 | 50.5 | 15.9 | 6.83 | 3.69 |
| | | **gap** | **+11.5** | +11.8 | +11.7 | +2.9 | +1.51 | +1.40 |
| `walk_sideway_045_stop_005__A042_M` | 1 (1) | sim | 19.3 | 16.7 | 30.7 | 9.6 | 2.80 | 0.80 |
| | | real | 24.4 | 17.3 | 35.7 | 9.1 | 2.80 | 1.17 |
| | | **gap** | +5.1 | +0.6 | +5.0 | -0.5 | 0.00 | +0.36 |
| `dance_vouge_shake_it_babe_360_R_002__A318_M` | 2 (0) | sim | 43.6 | 27.9 | 58.7 | 26.5 | 8.84 | 4.17 |
| | | real | 43.7 | 32.7 | 54.4 | 24.6 | 10.94 | 6.63 |
| | | **gap** | +0.2 | +4.8 | -4.3 | -1.9 | +2.10 | +2.46 |
| `high_jump_R_103__A389_M` | 2 (0) | sim | 37.6 | 26.6 | 36.5 | 32.6 | 7.63 | 3.33 |
| | | real | 43.5 | 34.8 | 43.6 | 26.7 | 10.60 | 6.22 |
| | | **gap** | +5.9 | +8.2 | +7.1 | -5.9 | +2.97 | +2.89 |

(gap = real - sim.)

### 4b.2 Macro-mean (each clip counts equally)

| metric | all 6 clips: sim | real | gap | real / sim | 4 confirmed locomotion clips: sim | real | gap | real / sim |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mpjpe_l | 26.95 | 34.10 | +7.15 | 1.27 | 20.14 | 29.34 | +9.19 | **1.46** |
| mpjpe_pa | 19.81 | 27.27 | +7.47 | 1.38 | 16.08 | 24.04 | +7.96 | 1.49 |
| legs | 35.18 | 44.32 | +9.14 | 1.26 | 28.96 | 41.96 | +13.00 | **1.45** |
| upper body | 17.03 | 16.66 | -0.36 | 0.98 | 10.77 | 12.17 | +1.40 | 1.13 |
| vel_dist | 5.15 | 6.50 | +1.34 | 1.26 | 3.61 | 4.36 | +0.75 | 1.21 |
| accel_dist | 2.15 | 3.54 | +1.39 | 1.65 | 1.35 | 2.10 | +0.75 | 1.55 |

### 4b.3 Reading

1. **The sim2real gap on tracking is not small**: whole-body `mpjpe_l` is 27 % higher on the robot
   than in the sim over all six clips, and **46 % higher on the four confirmed locomotion clips**
   (20.1 -> 29.3 mm).
2. **It sits in the legs.** Legs error is 1.45 x sim on locomotion (29.0 -> 42.0 mm, largest single
   gap: `walk_180` legs +26 mm). The upper body matches (ratio 0.98 over all clips, 1.13 on
   locomotion; gap 0.4–1.4 mm). This agrees with the one-step analysis
   (`sim2real/onestep_gap_report.md`), where ankle and leg joints are the poorly modelled ones.
3. **Dynamics are rougher on the robot**: `accel_dist` is 1.55–1.65 x and `vel_dist` 1.2–1.3 x sim.
4. **The agile clips show little `mpjpe_l` difference** (dance +0.2, high jump +5.9 mm), but they
   are hard in sim too (37–44 mm), so the headroom is already used; their velocity and
   acceleration gaps are the largest of any clip (`accel_dist` +2.5 / +2.9).
5. **Clip-level gaps are noisy.** Real n is 1–2 per clip and same-clip real reps differ by up to
   15 mm (`walk_180`: 37.5 vs 22.7), so a single clip's gap of ~5 mm is not significant; the
   macro-mean over clips is the number to use.

### 4b.4 Caveats specific to this comparison

- The sim figure is **one rollout per clip** of the training-side evaluator, with its own initial
  state, noise and termination rules. `mpjpe_l` there is the clip's own value (none terminated),
  but the real runs were driven through the deploy binary (ZMQ stream, 50 -> 30 fps resampling in
  the real evaluator), so part of the `vel_dist` / `accel_dist` gap may come from resampling and
  differences in how the two evaluators compute them. This was not isolated.
- The dance and high-jump real episodes are unconfirmed by the robot-side identifier (time
  alignment verified in §1); they are excluded from the 4-clip locomotion row.
- Sim is `llam_080k` at the same checkpoint, but evaluated on 2026-09-21; the real runs are
  2026-10-01 with `reference/example/` and the deploy-side observation pipeline.
## 5. Stress joints

Wrist joints dominate the worst-joint list in every run (`left/right_wrist_roll`,
`left_wrist_yaw`: 12–40°), as in earlier sessions. Mean joint error on confirmed runs is
5.3° (`ll_062k`: 5.35°), i.e. no change.

## 6. Caveats

1. **n = 6 confirmed episodes over 4 clips**; 1–2 reps per clip. Differences < ~10 mm are
   not resolvable (same-clip spread up to 15 mm here).
2. **Unconfirmed episodes (4/10)** are order-paired; the stream identity is certain, but
   whether the robot-side window is aligned is not verified. Not in the headline numbers.
3. `mpjpe_g` is degenerate online (root pinned at origin); ranked on L / PA.
4. Compared numbers for other policies are copied from the 09-28 report, with different
   session dates and hardware conditions; the 09-19 session was already flagged as confounded.
5. Robot logs were not passed through the `robot_log_data_quality` exclusion (PD-law/NaN)
   check in this report; the script's own parser was tolerant, no NaN runs were observed in
   the output.
6. Saturation (`~2 %`) and `cmd mpjpe_l` (84 mm on locomotion; `ll_062k` was 90.5) are in line
   with earlier sessions and are not informative about the ranking.

## 7. Recommendation

- `llam_080k` is **safe to deploy** and **no worse than `ll_062k`** on in-distribution
  locomotion; it can be the default hardware checkpoint given its much better OOD offline
  score.
- To actually test the OOD claim, run ≥3 reps each of the dances, `high_jump`, `reach_jump`
  and the kneel pair under both `llam_080k` and `ll_062k` back to back, and add reps of
  `walk_180`/`jog` (≥5) to resolve differences under ~10 mm.
- Keep the fall detector caveat in mind for jump/kneel clips (manifest §5.5).
