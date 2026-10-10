# Capability-Addition Fine-Tunes — Official Results, 2026-09-29

Consolidated results for the two **capability-addition (CA) fine-tunes** — teaching the G1
whole-body-control policy a skill it did not previously have, without losing the ones it
did:

| track | goal | new capability | status |
|---|---|---|---|
| **Sit** | chair sit-down / seated / stand-up cycle | ✅ **solved** — 100 % success, forgetting fully eliminated | **ship `MIXED_FT v2 004k`** |
| **Stair** | stair climbing | ✅ **solved** — 100 % success at `v3 028000` | **ship `v3 028000`** |

Both capability-addition tracks are now **solved**. Earlier stair conclusions (2k–6k
checkpoints, ≤ 25 % success) were simply **undertrained** — see §2.

**Source reports** (retained as the detailed record):
`GR00T-WholeBodyControl/model_eval/sit_ft_report.md`,
`GR00T-WholeBodyControl/model_eval/stair_ft_report.md`.
Base-checkpoint sweep: `checkpoint_comparison.md`.

---

## 0. Metric definitions

All numbers are `eval/success/*` from `metrics_eval.json`
(`gear_sonic/trl/callbacks/im_eval_callback.py`); the MPJPE variants are a port of
`smpl_sim.smpllib.smpl_eval.compute_metrics_lite`.

| metric | unit | dir | meaning |
|---|---|---|---|
| **Success rate** | fraction | ↑ | share of clips completed with **no termination** (`1 - mean(terminated)`); terminations = fall / anchor-pos / anchor-ori / end-effector / foot-position violation |
| **Progress rate** | fraction | ↑ | mean fraction of each clip completed before termination; successes count 1.0 — partial credit |
| **MPJPE-G** | mm | ↓ | **global** per-joint error, world frame, no alignment — includes root translation, so dominated by base drift |
| **MPJPE-L** | mm | ↓ | **local** (root-relative) — pelvis zeroed; pure pose/limb accuracy. **Primary quality number** |
| **MPJPE-PA** | mm | ↓ | **Procrustes-aligned** (rot+trans+scale per frame) — relative joint configuration only |
| **Accel / Vel dist** | mm | ↓ | acceleration / velocity error vs reference — motion smoothness |

Nesting is `PA ≤ L ≤ G`. A large G-vs-L gap means right pose, wrong place.

> **Read MPJPE together with success rate.** `eval/success/*` averages **only over clips
> that did not terminate**, so a policy that fails the hard clips is scored on the easy ones
> it survived. This matters acutely for stair (§2), where success is 0.00–0.25.

---

## 1. Sit fine-tune — **solved**

### 1.1 The problem: capability vs. forgetting

The tension in capability addition is that task-specific training degrades general tracking.
Joint training (sit clips + ~6000 general clips) removes it entirely:

| checkpoint | sit success | sit MPJPE-L | `eval_subset` success | `eval_subset` MPJPE-L |
|---|:-:|:-:|:-:|:-:|
| `REUBEN novr_050k` (base, no FT) | 0.850 | 63.62 | **0.981** | **24.60** |
| **`MIXED_FT v2 004k`** (joint) | **1.000** | **25.47** | **0.981** | 24.15 |

**`MIXED_FT v2 004k` matches the un-finetuned baseline on `eval_subset`** (0.981 / 24.15 vs
0.981 / 24.60) **while fully solving sit** (100 % success, MPJPE-L 63.62 → 25.47).

For contrast, sit-*only* finetuning cost ~15 points of general tracking at the same sit
performance (`SIT_FT 004k`: sit 1.000 / 21.49, but `eval_subset` 0.831 / 34.15, decaying
monotonically 0.875 → 0.831 → 0.825 across 2k/4k/6k). Full per-checkpoint tables are in
`sit_ft_report.md`.

### 1.2 `sit_evalset` (20 clips, full sit cycle, chair present)

| checkpoint | Success | Progress | MPJPE-G | MPJPE-L | MPJPE-PA | Accel | Vel |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `REUBEN novr_050k` (base) | 0.850 | 0.901 | 140.43 | 63.62 | 30.02 | 2.136 | 4.178 |
| `SIT_FT 002k` | **1.000** | **1.000** | 50.69 | 27.21 | 17.09 | 0.327 | 0.861 |
| `SIT_FT 004k` | **1.000** | **1.000** | **43.45** | **21.49** | **11.69** | 0.324 | 0.827 |
| `SIT_FT 006k` | **1.000** | **1.000** | 46.47 | 22.52 | 16.60 | **0.299** | **0.765** |
| `MIXED_FT v2 002k` | **1.000** | **1.000** | 54.00 | 28.38 | 17.41 | 0.402 | — |
| **`MIXED_FT v2 004k`** | **1.000** | **1.000** | 51.65 | 25.47 | 18.39 | 0.409 | — |
| `MIXED_FT v2 006k` | **1.000** | **1.000** | 51.05 | 25.48 | 17.57 | 0.369 | — |

20 clips = 8 `loop` (seated) + 6 `start` (stand→sit) + 6 `stop` (sit→stand). Every finetune
reaches 100 % success; the base reaches 0.850.

### 1.3 Retention on unseen general motions

`amass_evalset` (108) and `pico_evalset` (20) contain no chair, so they test retention on
distributions the finetune never saw:

| checkpoint | set | Success | Progress | MPJPE-G | MPJPE-L | MPJPE-PA |
|---|---|:-:|:-:|:-:|:-:|:-:|
| `MIXED_FT v2 002k` | amass | 0.750 | 0.845 | **236.7** | 32.8 | 24.1 |
| **`MIXED_FT v2 004k`** | amass | 0.741 | 0.841 | 288.1 | **32.7** | **24.0** |
| `MIXED_FT v2 006k` | amass | **0.759** | 0.842 | 319.3 | 33.1 | 24.8 |
| `MIXED_FT v2 002k` | pico | **0.750** | **0.864** | 346.4 | **29.7** | 24.1 |
| `MIXED_FT v2 004k` | pico | 0.650 | 0.835 | **291.3** | 32.9 | 25.4 |
| `MIXED_FT v2 006k` | pico | 0.700 | 0.828 | 352.4 | 30.3 | **23.7** |

Success is **flat** across 2k/4k/6k on both (amass spread 0.741–0.759 is under one clip in
108; pico's 0.650–0.750 is two clips in 20). MPJPE-L is unchanged, so articulation quality on
unseen motion is not degrading.

**One real trend: amass MPJPE-G climbs monotonically 236.7 → 288.1 → 319.3** while L/PA stay
flat — drift in *global root placement*, not articulation, the expected signature of a
task-specific finetune pulling the root prior toward its training distribution. It does not
appear on `eval_subset` because that set is drawn from the same corpus the mixed set samples
its general clips from.

### 1.4 Recommendation

**Ship `MIXED_FT v2 004k`** (`sonic_release_no_teleop_sit_mixed_v2-20260914_153300/model_step_004000.pt`).
If out-of-distribution global root accuracy matters more than peak sit precision, **002k** is
the conservative alternative at equal sit performance (100 % success, MPJPE-L 28.38 vs 25.47)
and the best amass MPJPE-G.

---

## 2. Stair fine-tune — **solved**

`stair_evalset`, 12 `stairs_climbing_*` motions. The **v3** run trains far longer than the
2k–6k range of the earlier attempts, and stair climbing is fully solved by ~22k steps.

### 2.1 Result

**Stair task** (`stair_evalset`, 12 clips):

| checkpoint | Success | Progress | MPJPE-G | MPJPE-L | MPJPE-PA | Accel | Vel |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `REUBEN novr_050k` (base, no FT) | 0.083 | 0.293 | **81.59** | 42.73 | 30.38 | 3.88 | 9.30 |
| **`v3 028000`** | **1.000** | **1.000** | 121.34 | **31.61** | **22.15** | **2.14** | **5.61** |

**General tracking retention** (`eval_subset`, 160 clips):

| checkpoint | Success | Progress | MPJPE-G | MPJPE-L | MPJPE-PA |
|---|:-:|:-:|:-:|:-:|:-:|
| `REUBEN novr_050k` (base, no FT) | **0.981** | **0.988** | **118.97** | **24.60** | **17.49** |
| **`v3 028000`** | **0.981** | **0.988** | 153.96 | 25.51 | 18.76 |

**Stair climbing goes from 1 of 12 motions to 12 of 12, with no forgetting.** MPJPE-L on the
stair task improves 42.73 → 31.61 mm and the motion is markedly smoother (accel 3.88 → 2.14,
vel 9.30 → 5.61) — while `eval_subset` success and progress are **identical to the base**
(0.981 / 0.988), at a cost of 0.9 mm MPJPE-L and 1.3 mm PA.

MPJPE-G rises on both sets. On the stair task the base figure is averaged over the **single**
clip it survived, so the two are not comparable (§0). On `eval_subset` the rise
(118.97 → 153.96) is real but is the same *global root placement* drift documented for the
sit track (§1.3) — L and PA stay flat, so articulation is unaffected.

### 2.2 Training trajectory

Success climbs steadily and saturates; it is not a plateau that more steps would have missed.
`eval_subset` retention holds flat across the whole range:

| step | 2k | 4k | 6k | 8k | 10k | 12k | 14k | 18k | 22k | 26k | **28k** | 30k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| stair Success | 0.000 | 0.250 | 0.250 | 0.417 | 0.750 | 0.833 | 0.917 | 0.917 | **1.000** | 0.917 | **1.000** | **1.000** |
| stair MPJPE-L | — | 39.18 | 37.71 | 33.97 | 32.42 | 33.64 | 33.05 | 30.16 | 28.54 | 29.61 | 31.61 | 30.75 |
| `eval_subset` Succ | — | — | — | — | — | — | 0.981 | 0.975 | 0.981 | 0.981 | **0.981** | 0.981 |
| `eval_subset` L | — | — | — | — | — | — | 25.78 | 25.26 | 25.12 | 24.71 | **25.51** | 25.99 |

First 100 % at **22k**; 28k and 30k also reach 100 % (26k dips to 0.917 — a single clip).
`028000` is the recommended checkpoint: it is inside a stable run of perfect stair success
**and** holds baseline-matching `eval_subset` retention.

### 2.3 Findings

1. **The earlier "stair is not learnable" conclusion was an artifact of undertraining.**
   At 2k–6k success was 0.000–0.250; the same recipe reaches 1.000 by 22k. Nothing about
   the task or the data was wrong.
2. **No forgetting.** `eval_subset` success/progress are **identical to the base**
   (0.981 / 0.988) at 028000, and hold flat across 14k–30k. The cost is 0.9 mm MPJPE-L —
   the same joint-training result the sit track established (§1.1), now confirmed on a
   second capability.
3. **Quality improves alongside success**, which is the important check — stair MPJPE-L
   falls 42.73 → 31.61 and PA 30.38 → 22.15, so the policy is not merely surviving the
   clips, it is tracking them well.
4. **Motion is substantially smoother**: accel 3.88 → 2.14, vel 9.30 → 5.61, i.e. roughly
   half the reference-tracking jerk of the base policy.
5. **The heightmap variant is superseded.** The earlier `hmap_add` vs no-heightmap
   comparison was run entirely inside the 2k–6k undertrained regime, where neither variant
   worked (peak 0.250, heightmap ≤ no-heightmap at every checkpoint). It says nothing about
   heightmap value at convergence and should not be cited; re-run it against `v3` if the
   question still matters.

### 2.4 Recommendation

**Ship `v3 028000`.** Stair fully solved (1.000 / 1.000) with baseline-matching general
tracking (0.981 / 0.988) — no outstanding blockers.

---

## 3. Cross-track comparison

| | Sit | Stair |
|---|---|---|
| base-policy success before FT | 0.850 | 0.083 |
| task success after FT | **1.000** (`MIXED_FT v2 004k`) | **1.000** (`v3 028000`) |
| progress | **1.000** | **1.000** |
| task MPJPE-L, base → FT | 63.62 → **25.47** | 42.73 → **31.61** |
| steps to solve | ~2k | ~22k |
| `eval_subset` success (base 0.981) | **0.981** | **0.981** |
| `eval_subset` MPJPE-L (base 24.60) | 24.15 | 25.51 |
| forgetting | **none** | **none** |
| verdict | **ship 004k** | **ship 028000** |

**Both capability additions succeed with zero forgetting.** Stair started from a much weaker
base (0.083 vs 0.850) and needed ~10× the training steps, but ends at the same 100 % success
and the same baseline-matching `eval_subset` retention.

Two lessons worth carrying forward:

1. **Joint training beats sequential finetuning** (§1.1), now confirmed on **two independent
   capabilities**. Sit-*only* training cost ~15 points of `eval_subset`; mixing general clips
   into the trainset recovered all of it at no cost to the new capability, and the stair
   track reproduces the result.
2. **Do not call a capability unlearnable from early checkpoints.** The stair track was
   written off at 2k–6k steps with ≤ 0.250 success; the identical recipe reaches 1.000 by
   22k. Success climbed steadily throughout (§2.2) — there was no plateau to justify
   stopping.

---

## 4. Known issues and caveats

### 4.1 Invalidated result: `SITLOOP_FT 002k`

Launched with the **3-encoder** `sonic_release` config but initialized from a **teleop-free**
checkpoint. The teleop encoder was therefore **randomly initialized**, then sampled for ~⅓ of
envs while two teleop aux losses at coefficient 1.0 pulled the shared FSQ token space toward a
random target. `strict=False` in the loader made it silent.

That — not sit-data narrowness — is the dominant cause of its `eval_subset` collapse
(0.981 → 0.244). The corrected rerun (`SIT_FT`) drops only to 0.875. **`SITLOOP_FT 002k` is
not evidence about catastrophic forgetting.**

### 4.2 Sit dataset was rebuilt 2026-09-11 — earlier numbers are not comparable

Three defects, all fixed: chair placed at **human** scale (robot penetrated the seat by ~35 mm
on every clip; 436/652 clips over tolerance), a legacy +31 mm lift on the 216 `loop`
trajectories, and a corrupted spawn frame on 208/210 train + 6/6 eval `stop` clips.

**Consequence:** on rebuilt data the un-finetuned baseline already scores **0.850**, versus
0.125 on the broken data. Most of what the original finetune appeared to "fix" was **chair
geometry, not a policy deficiency**. The obsolete `sitloop_evalset` table in the source report
is retained but marked superseded.

### 4.3 `MIXED_FT v1` diverged — adaptive-sampling collapse

Reward 23.8 → 6.7 at ~iteration 2000. Cause: `max_prob_per_bin`/`max_prob_per_motion` unset
(which disables *all* max-prob constraints via a legacy branch), so the sampler concentrated on
a few infeasible clips — `episodes_max_over_mean` 2.9 → **850**, `effective_num_bins`
5739 → 414. Amplified because a failing clip terminates in ~10–20 steps vs ~460, yielding ~30×
more episodes per unit env-time; with `use_failure_rate_decay: false` this was irreversible.

Fixed in v2 (now default): `max_prob_per_{motion,bin}=auto`,
`adp_samp_failure_rate_max_over_mean=10` (was 200), `uniform_sampling_rate=0.3` (was 0.1),
`use_failure_rate_decay=true`.

**Residual:** even v2 drifts — `effective_num_bins` 5739 → ~3600 and `episodes_max_over_mean`
16 → 26 by iteration 6000. This matches 006k scoring slightly worse than 004k on both eval
sets. For longer runs tighten `SAMP_MAX_OVER_MEAN` to 5 or raise `SAMP_UNIFORM_RATE`.

### 4.4 Evaluation-set limitations

- **`sit_evalset` is single-actor** (A248) at one seat height (~0.37 m), while training spans
  0.298–0.450 m across 37 actors. Height generalization is **untested**.
- **`stair_evalset` is 12 motions**, so one clip = 0.083 success. The early-checkpoint
  deltas are single clips; only the saturated 1.000 results are robust.
- **Stair retention is measured on `eval_subset` only.** No `amass_evalset` / `pico_evalset`
  run exists for any `v3` checkpoint, so out-of-distribution retention is untested — the sit
  track showed that is where the (mild) degradation appears (§1.3).
- **Termination-config caveat** (see `checkpoint_comparison.md`): `+manager_env/terminations=tracking/eval`
  *merges* rather than replaces, so 5 terms are always active. Applies uniformly to all runs
  here, so comparisons remain valid.
- **`pico_evalset` MPJPE-G is erratic** (346 → 291 → 352) and must not be read as a
  checkpoint-to-checkpoint signal; see the `transl = 0` issue in `gam/sim2real/0928_official_report.md` §0.

### 4.5 Checkpoint granularity

`save_interval=200` was configured but the `max_disk_usage: 14.8 GB` cap pruned everything
except 2k/4k/6k in the **sit** runs. Since sit is already solved at 2k, **an earlier
checkpoint (400–1000) may hold `eval_subset` ≥ 0.95 with sit fully solved** — worth
re-running with a higher disk cap. The stair `v3` run has 2k-granularity coverage to 30k and
is unaffected.

---

## 5. Next steps

1. **Run `amass_evalset` / `pico_evalset` on `v3 028000`** — `eval_subset` retention is
   confirmed, but out-of-distribution retention is untested, and that is where the sit track
   showed root-placement drift (§1.3, §4.4).
2. **Extend `sit_evalset` to multiple actors and seat heights** (training covers
   0.298–0.450 m; eval covers one).
3. **Re-run sit with a higher disk cap** to test checkpoints in the 400–1000 range.
4. **Re-evaluate the heightmap question against `v3`** if it still matters — the existing
   comparison is confined to the undertrained 2k–6k regime and is uninformative (§2.3).
5. **Deploy both checkpoints on hardware.** All results here are simulation; **no
   capability-addition checkpoint has any real-robot evaluation**. Compare against
   `gam/sim2real/0928_official_report.md`, which found the sim↔hardware ranking for the base
   checkpoints has only been partially verified.
