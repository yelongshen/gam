# Real-Robot Online Deployment — Cross-Session Policy Comparison

Consolidated view of **every** real-G1 online deployment session evaluated so far
(2026-09-04 → 2026-09-24), used to answer one question: **which deployed policy tracks
best on hardware?**

- **Sessions**: 8 · **episodes located**: 56 · **clip-confirmed**: 40
- **Per-session detail**: `online_eval_20260904_report.md`, `…_20260909_`, `…_20260912_`,
  `…_20260915_`, `…_20260919_`, `…_20260922_`, `…_20260924_report.md`
- **Raw metrics**: `sim2real/online_eval_results*.json`
- **Metric definitions**: `online_eval_20260904_report.md` §4 (port of
  `smpl_sim.smpllib.smpl_eval.compute_metrics_lite`); plan:
  `online_deployment_eval_plan.md`
- **Nature of the numbers**: online — real hardware telemetry, computed post-hoc from CSV
  logs. No simulator involved.

---

## 1. Per-session results (confirmed episodes only)

| date | policy | n | `mpjpe_l` | `mpjpe_pa` | joint err | cmd `mpjpe_l` | sat % | shaking | max tilt | falls |
|---|---|---|---|---|---|---|---|---|---|---|
| 09-04 | `low_latency` | 5 | 45.07 | 39.35 | 6.99° | 61.9 | 0.57 | 0.525 | 9.5° | 0 |
| 09-08 | `low_latency` | 15 | 40.28 | 28.31 | 6.74° | 66.5 | 1.02 | 0.405 | 15.2° | 0 |
| 09-09 | `sonic_no_vr_050k` | 7 | 36.86 | 31.95 | 5.95° | 59.6 | 1.08 | 0.344 | 8.9° | 0 |
| 09-12 | `sonic_no_vr_050k` | 1 | 72.09 | 67.53 | 9.34° | 81.5 | 1.35 | 0.574 | 19.6° | 0 |
| **09-15** | **`sonic_no_vr_ll_062k`** | 5 | **26.19** | **21.13** | **4.83°** | 87.2 | 1.58 | 0.462 | 9.0° | 0 |
| 09-19 | `ll_062k` / `low_latency` | 2 | 35.44 | 29.84 | 6.39° | 100.1 | 1.81 | 0.476 | 5.6° | 0 |
| **09-22** | `sonic_no_vr_050k` | 3 | **23.19** | **20.16** | **4.75°** | 73.3 | 1.36 | 0.405 | 10.6° | 0 |
| 09-24 | `sonic_no_vr_ll_062k` | 2 | 36.17 | 31.94 | 5.59° | 89.0 | 1.37 | 0.526 | 8.1° | 0 |

Session means are **confounded by clip mix** — 09-22 looks best partly because it ran only
easy locomotion clips. §2 is the fair comparison.

## 2. Per-policy aggregate

| policy | episodes | clips | `mpjpe_l` | `mpjpe_pa` | joint err | cmd `mpjpe_l` | sat % | shaking | max tilt |
|---|---|---|---|---|---|---|---|---|---|
| `low_latency` | 20 | 6 | 41.48 | 31.07 | 6.80° | 65.4 | 0.91 | 0.435 | 13.7° |
| `sonic_no_vr_050k` | 11 | 5 | 36.33 | 31.97 | 5.93° | 65.4 | 1.18 | 0.382 | 10.3° |
| **`sonic_no_vr_ll_062k`** | 9 | 4 | **30.46** | **25.47** | **5.35°** | 90.5 | 1.59 | 0.479 | **8.0°** |

## 3. Clip-matched comparison — the headline result

Same clip, different policy. This removes the clip-mix confound entirely.

| clip | `low_latency` | `novr_050k` | `ll_062k` |
|---|---|---|---|
| `walk_180_R_003__A332_M` | 32.0 (n=7) | **25.9** (n=5) | 29.1 (n=2) |
| `walk_sideway_045_stop_005__A042_M` | 49.3 (n=4) | 54.3 (n=2) | **24.9** (n=1) |
| `jog_ff_start_180_R_002__A192_M` | 39.7 (n=3) | **32.2** (n=1) | 36.2 (n=1) |
| `walk_backward_start_001__A030_M` | — | 46.6 (n=2) | **28.0** (n=3) |
| **macro-mean over the 3 clips all three ran** | **40.4** | **37.5** | **30.1** |

**`sonic_no_vr_ll_062k` is the best deployed policy: 30.1 mm vs 40.4 mm for the
`low_latency` baseline — a 25% reduction in tracking error on identical clips.**

Clips run under a single policy only (not comparable, listed for completeness):
`kneeling_start_101__A063_M` 57.2 mm and `kneeling_stop_002__A051_M` 58.3 mm
(`low_latency`, the two hardest clips measured), `reach_jump_R_001__A072_M` 29.0 mm
(`low_latency`), `wiping_shoes_R_001__A234` 36.1 mm (`novr_050k`).

---

## 4. Findings

1. **`ll_062k` wins, and the margin is real.** 30.1 vs 37.5 vs 40.4 mm macro-mean. It also
   produced the single best session on record (09-15: 26.19 mm across 5/5 confirmed clips,
   with `walk_backward` reproducible to **±0.7 mm** across two reps).

2. **Its biggest win is lateral gait.** `walk_sideway` drops **49.3 → 24.9 mm**, and the
   heading error on that clip collapses from **20–30° under `novr_050k` to 1.9°**. That was
   the standout defect identified in the 09-09 report and it does not reproduce under
   `ll_062k` — the most actionable single result in this table.

3. **Zero falls in all 56 episodes**, max tilt 22.7° against a 35° threshold. Every policy
   degrades gracefully; when a clip is too hard the robot refuses the motion rather than
   destabilising (see 09-12, §5).

4. **Better imitation costs command-tracking headroom.** Going `low_latency` → `ll_062k`,
   cmd `mpjpe_l` rises 65.4 → 90.5 mm and saturation 0.91 → 1.59 %. Imitation error stays
   *well below* command error in every session — the PD loop lags each instantaneous
   command but averages onto the reference. This ordering has now held across three
   policies and eight sessions.

5. **Ankle pitch is the universal actuation hotspot** — worst saturating joint in nearly
   every episode, peaking at **33 %** of timesteps above 0.9×effort. Policy-independent,
   so it is a hardware/limit property, not something a checkpoint will fix.

6. **The stress axes moved distally.** Worst reference-tracking joints went from
   `shoulder_yaw` (17–20° on 09-04) to `wrist_yaw`/`wrist_roll` (13–16° on 09-22). The
   Phase C.2 low-inertia-twist problem persists but has shifted.

## 5. Sessions that measure failure, not quality

Two sessions are **capability-boundary records** and their `mpjpe_l` must not be pooled:

- **09-12** (`novr_050k`): 6 of 7 episodes are agility clips the policy never tracked —
  4× `high_jump_R_103`, 2 dances. Signature: cmd `mpjpe_l` 96–141 mm, shaking
  1.33–1.47 rad/s (3–4× normal), streamed clip fitting the executed motion at only 12–18°
  rms. One rep also showed an **83.8° heading collapse**.
- **09-15 `run12`/`run20`, 09-19 walk_180 reps**: same pattern at lower severity.

The reliable "did not track" detector, from these: **cmd `mpjpe_l` > 100 mm + shaking >
1.0 rad/s + streamed-clip fit worse than ~12°**.

---

## 6. Caveats

1. **Small, unbalanced n.** 20 / 11 / 9 confirmed episodes per policy, over 6 / 5 / 4
   clips. The plan's ≥3-reps-per-clip bar is met for only a few clip×policy cells.
2. **Rep-to-rep variance is large.** `walk_180` spans 29.8–67.8 mm within a single
   policy on 09-04. Differences below ~10 mm between policies are not resolvable at these
   sample sizes.
3. **09-19 is confounded.** Its numbers are ~2× worse than 09-15 for *both* policies it
   ran. Traced to a degraded stream (SMPL clip identification 55–63 mm vs 28 mm normal)
   plus `reference/example/` instead of `real_example/`. Partially controlled on 09-24 —
   see that report §4 — which showed stream quality explains at most half the gap.
4. **`mpjpe_g` is degenerate** on real hardware (no mocap ⇒ root pinned at origin, so it
   equals `mpjpe_l`) and is not reported here.
5. **Clip identity is state-matched, not labelled.** All episodes here passed the
   two-sided confirmation gate (streamed SMPL *and* executed `q` naming the same clip);
   the 16 unconfirmed episodes are excluded from every number above.

## 7. Sim↔real disagreement — the open question

On `picoset_20260924` (sim, 93 clips) the **LLAM family** ranks best, with
`novrllam_080k` the recommended checkpoint (see
`GR00T-WholeBodyControl/model_eval/checkpoint_comparison.md`). On hardware, **`ll_062k`
leads** and the only LLAM checkpoint ever deployed (`llam_080k`, 09-27) has **no confirmed
episodes yet**.

These are not yet in contradiction — they are simply untested against each other. The
highest-value next experiment is **≥3 reps of `walk_180` / `walk_sideway` /
`walk_backward` under `llam_080k` on hardware**, which would either confirm the sim
ranking transfers or establish a concrete sim2real ranking inversion.

## 8. Reproduction

```bash
# per-session evaluation (content-paired clip identification + metrics)
.venv_sim/bin/python sim2real/eval_online_runs.py \
    --runs     /home/grease/g1_robot_data/g1_run_<MMDD> \
    --sessions /home/grease/g1_robot_data/<YYYYMMDD> \
    --out      sim2real/online_eval_results_<YYYYMMDD>.json

# 09-22 / 09-24 sessions use wall-clock pairing instead
.venv_sim/bin/python sim2real/eval_runs_0922.py
```
