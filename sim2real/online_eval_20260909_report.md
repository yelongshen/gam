# Real-Robot Online Deployment Evaluation — 2026-09-09 Session (report)

- **Data**: `g1_robot_data/g1_run_0909` (12 runs) + `g1_robot_data/20260909` (13 stream sessions)
- **Policies**: `policy/sonic_no_vr_050k` (runs 1–8), `policy/low_latency` (runs 10–14),
  both with `reference/real_example/`, 50 Hz, encoder on, fp16 off
- **Tooling**: `sim2real/eval_online_runs.py` (content pairing — see §1)
- **Raw output**: `sim2real/online_eval_results_20260909.json`
- **Result**: **11 episodes located, 7 CONFIRMED**, 0 falls
- Metric definitions: `sim2real/online_eval_20260904_report.md` §4; plan:
  `sim2real/online_deployment_eval_plan.md`

> **Cross-session comparison:** `sim2real/online_eval_policy_comparison.md` pools all
> 8 sessions and compares the deployed policies on matched clips.

---

## 1. Why content pairing

This session's run directories are named `g1_deploy_run_09082026_run<N>` and their only
absolute clock (`time_realtime_ms`) is the **robot's**, which is ~15 h away from the
streaming PC's `streamed_<HHMMSS>` names. The timestamp pairing used for 09-22 is therefore
unusable. Instead each side is identified independently against all 160 `eval_subset` clips
(SMPL stream → `eval_subset/smpl`, executed `q` → `eval_subset/robot`, yaw-invariant FFT
sliding SSD with motion-window and clip-length constraints), and:

- **agreement** — both sides name the same clip ⇒ `CONFIRMED`;
- **order** — the rest are zipped chronologically, evaluated, but flagged `confirmed=False`.

## 2. What was streamed (SMPL-side identification)

| session | clip | rms | margin | yaw |
|---|---|---|---|---|
| `073717` | `walk_180_R_003__A332_M` | 28.5 mm | ×2.16 | +86.6° |
| `074026` | `walk_180_R_003__A332_M` | 28.7 | ×2.15 | +86.6° |
| `074402` | `walk_180_R_003__A332_M` | 28.7 | ×2.19 | +86.9° |
| `074640` | `walk_180_R_003__A332_M` | 39.6 | ×1.53 | +87.1° |
| `075106` | `walk_sideway_045_stop_005__A042_M` | 27.4 | ×1.98 | +105.6° |
| `075325` | `walk_sideway_045_stop_005__A042_M` | 27.5 | ×1.97 | +105.6° |
| `075812` | `jog_ff_start_180_R_002__A192_M` | 38.8 | ×1.31 | −88.6° |
| `080124` | `jump_ff_180_R_001__A225` | 46.4 | ×1.02 | +88.3° |
| `081538` | `jog_ff_start_180_R_002__A192_M` | 38.3 | ×1.35 | −88.6° |
| `081839` | `jog_ff_start_180_R_002__A192_M` | 38.6 | ×1.34 | −88.7° |
| `082055` | `wiping_shoes_R_001__A234` | 45.4 | ×1.02 | +90.7° |
| `080809`, `082249` | `looking_around_up_002__A025_M` | **154–162** | ×1.03–1.12 | — → **rejected** (above the 90 mm bar) |

Note the **+105.6° yaw** on the two `walk_sideway` sessions vs +86–90° everywhere else: the
sender's frame offset is not a fixed constant across streams, which is exactly why the
matcher solves for yaw per candidate rather than assuming 90°.

## 3. Results

### 3.1 Per-clip aggregate (CONFIRMED only)

| clip | n | `mpjpe_l` | `mpjpe_pa` | joint err | non-fall |
|---|---|---|---|---|---|
| `walk_180_R_003__A332_M` | 3 | **27.0 ± 7.5 mm** | 24.2 ± 7.8 | 4.91 ± 0.42° | 100% |
| `walk_sideway_045_stop_005__A042_M` | 2 | **54.3 ± 2.8 mm** | 50.4 ± 3.2 | 8.04 ± 1.03° | 100% |
| `jog_ff_start_180_R_002__A192_M` | 1 | 32.2 | 25.5 | 5.56° | 100% |
| `wiping_shoes_R_001__A234` | 1 | 36.1 | 24.8 | 5.26° | 100% |

### 3.2 Per-episode

| run (policy) | session | clip | conf | `mpjpe_l` | pa | legs | foot | joint err (worst) | heading | cmd | sat / worst | shake | tilt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `run1` (050k) | `073717` | walk_180 | ✅ | **20.4** | 17.2 | 30.4 | 34.1 | 4.49° (`L_wrist_yaw` 12.4) | 2.2° | 63.6 | 1.24% / `L_ankle_pitch` 22.9% | 0.379 | 13.7° |
| `run2` (050k) | `074026` | walk_180 | ✅ | 37.5 | 35.2 | 59.2 | 87.6 | 5.48° (`L_wrist_yaw` 11.6) | 4.9° | 60.5 | 1.53% / 29.1% | 0.353 | 7.2° |
| `run3` (050k) | `074402` | walk_180 | ✅ | 23.2 | 20.3 | 35.1 | 42.9 | 4.76° (`L_wrist_yaw` 12.1) | 2.4° | 63.4 | 1.10% / 14.8% | 0.403 | 8.9° |
| `run4` (050k) | `082055` | wiping_shoes | ✅ | 36.1 | 24.8 | 36.7 | 38.7 | 5.26° (`R_shoulder_yaw` 13.6) | 1.3° | **28.6** | 0.00% | **0.017** | 1.8° |
| `run5` (050k) | `075106` | walk_sideway | ✅ | 51.5 | 47.2 | 88.3 | **127.4** | 7.01° (`R_hip_yaw` 12.9) | **20.3°** | 54.5 | 0.49% / 7.6% | 0.289 | 7.3° |
| `run5` (050k) | `075325` | walk_sideway | ✅ | 57.0 | 53.5 | 95.1 | **137.1** | 9.06° (`R_hip_yaw` 19.1) | **29.9°** | 56.9 | 0.56% / 9.0% | 0.269 | 8.1° |
| `run6` (050k) | `075812` | jog_ff_start | ✅ | 32.2 | 25.5 | 42.8 | 46.8 | 5.56° (`L_wrist_roll` 22.8) | 2.6° | 90.0 | 2.61% / 36.1% | 0.700 | 15.2° |
| `run7` (050k) | `074640` | walk_180 | ✗ order | 62.5 | 60.2 | 88.6 | 141.6 | 8.54° (`R_shoulder_yaw` 17.7) | 4.3° | 47.3 | 0.15% | 0.281 | 10.2° |
| `run8` (050k) | `080124` | jump_ff_180 | ✗ order | 42.1 | 39.8 | 54.8 | 71.8 | 8.43° (`L_knee` 26.0) | 6.8° | 43.1 | 0.29% | 0.096 | 6.3° |
| `run11` (**low_latency**) | `081538` | jog_ff_start | ✗ order | 43.1 | 33.8 | 53.9 | 72.4 | 7.12° (`L_wrist_roll` 22.2) | 5.2° | 92.3 | 2.15% / 30.6% | 0.727 | **20.8°** |
| `run12` (**low_latency**) | `081839` | jog_ff_start | ✗ order | **88.5** | 85.4 | 90.9 | 140.7 | 12.73° (`L_elbow` 54.4) | 8.2° | 78.4 | 0.82% | 0.563 | 15.9° |

(`sat` = fraction of timesteps above 0.9×effort; `shake` = high-pass `dq` RMS rad/s.)

## 4. Findings

1. **`walk_180` is solid at 27.0 ± 7.5 mm** under `sonic_no_vr_050k`, with `run2` (37.5 mm,
   foot 87.6) the only weak rep of three.
2. **`walk_sideway` fails at the *heading* level, not the joint level.** Both reps score
   51–57 mm with leg/foot errors 2.5–4× the `walk_180` reps, and the heading error is
   **20.3° / 29.9°** — the largest in any session evaluated so far. The robot performs the
   right joint motion but ends up facing the wrong way; lateral gait heading is the specific
   deficiency to chase.
3. **`jog_ff_start` is the actuation hotspot**: 2.2–2.6% mean saturation (`L_ankle_pitch`
   30–36% of timesteps), shaking 0.70–0.73 rad/s, tilt up to 20.8°. It is the most
   physically demanding clip in this session.
4. **The `low_latency` runs (11, 12) are visibly worse on the same clip.** `jog_ff_start`
   goes 32.2 mm (050k, run6) → 43.1 / 88.5 mm (low_latency, runs 11/12), with `L_elbow`
   drifting 54° in run12. Neither confirms on the robot side. This is consistent with the
   09-04-vs-09-22 cross-session gap and supports `sonic_no_vr_050k` over `low_latency`.
5. **`wiping_shoes` is a near-static clip** (shaking 0.017 rad/s, tilt 1.8°, zero
   saturation) — useful as a controller noise floor, useless as a tracking benchmark; its
   SMPL identification margin is only ×1.02 for the same reason.
6. **No falls** (max tilt 20.8°), consistent with every session to date.

## 5. Caveats

- Two `looking_around_up` sessions are rejected (SMPL rms 154–162 mm): the streamed content
  does not match any library clip well enough to anchor an episode.
- Four order-paired episodes (`run7`, `run8`, `run11`, `run12`) rest on chronological
  ordering, not evidence — treat their absolute numbers as indicative.
- Mixed policies within the day: only runs 1–8 are `sonic_no_vr_050k`. Do not pool the
  per-clip aggregate across the policy boundary.

## 6. Reproduction

```bash
.venv_sim/bin/python sim2real/eval_online_runs.py \
    --runs     /home/grease/g1_robot_data/g1_run_0909 \
    --sessions /home/grease/g1_robot_data/20260909 \
    --out      sim2real/online_eval_results_20260909.json
```
