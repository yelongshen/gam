# Real-Robot Online Deployment Evaluation — 2026-09-15 Session (report)

- **Data**: `g1_robot_data/g1_run_0915` (11 runs) + `g1_robot_data/20260915` (10 stream sessions)
- **Policy**: `policy/sonic_no_vr_ll_062k` for **all** runs (`reference/real_example/`, 50 Hz)
- **Tooling**: `sim2real/eval_online_runs.py` (content pairing)
- **Raw output**: `sim2real/online_eval_results_20260915.json`
- **Result**: **8 episodes located, 5 CONFIRMED**, 0 falls — **the cleanest locomotion
  session on record**
- Metric definitions: `sim2real/online_eval_20260904_report.md` §4

> **Cross-session comparison:** `sim2real/online_eval_policy_comparison.md` pools all
> 8 sessions and compares the deployed policies on matched clips.

---

## 1. Session structure

A textbook sweep: one clip per run, five locomotion clips first (runs 1–6, all confirmed),
then the agility stress block (runs 7–20) where the policy stops tracking. Both halves are
reported below; only the first is a tracking measurement.

## 2. What was streamed (SMPL-side identification)

| session | clip | rms | margin | outcome |
|---|---|---|---|---|
| `083556` | `walk_180_R_003__A332_M` | 28.2 mm | ×2.17 | ✅ confirmed (`run1`) |
| `083926` | `walk_sideway_045_stop_005__A042_M` | 27.4 | ×1.98 | ✅ confirmed (`run2`) |
| `084209` | `walk_backward_start_001__A030_M` | **18.4** | **×3.33** | ✅ confirmed (`run3`) |
| `084407` | `walk_backward_start_001__A030_M` | **18.7** | ×3.26 | ✅ confirmed (`run4`) |
| `084914` | `jog_ff_start_180_R_002__A192_M` | 38.2 | ×1.33 | ✅ confirmed (`run6`) |
| `085159` | `wiping_shoes_R_001__A234` | 45.5 | ×1.02 | order-paired (`run7`) |
| `091220` | `high_jump_R_103__A389_M` | 64.2 | ×1.62 | order-paired (`run12`) |
| `092520` | `dance_vouge_shake_it_babe_360_R_002__A318_M` | 78.7 | ×1.12 | order-paired (`run20`) |
| `085511` | `body_check_002__A122` | **234.2** | — | ❌ rejected (SMPL id far above the 90 mm bar) |
| `091502` | `jump_ff_270_001__A084` | **105.6** | ×1.04 | ❌ rejected |

## 3. Results

### 3.1 Per-clip aggregate (CONFIRMED only)

| clip | n | `mpjpe_l` | `mpjpe_pa` | `vel_dist` | joint err | non-fall |
|---|---|---|---|---|---|---|
| `walk_180_R_003__A332_M` | 1 | **20.6 mm** | 18.0 | 2.99 | 4.53° | 100% |
| `walk_backward_start_001__A030_M` | 2 | **24.6 ± 0.7 mm** | 20.8 ± 0.6 | 4.02 ± 0.07 | 4.53 ± 0.08° | 100% |
| `walk_sideway_045_stop_005__A042_M` | 1 | **24.9 mm** | 18.6 | 2.48 | 4.73° | 100% |
| `jog_ff_start_180_R_002__A192_M` | 1 | 36.2 | 27.5 | 6.72 | 5.83° | 100% |

### 3.2 Per-episode

| run | session | clip | conf | `mpjpe_l` | pa | legs | foot | vr_3pt | joint err (worst) | heading | cmd | sat / worst | shake | tilt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `run1` | `083556` | walk_180 | ✅ | **20.6** | 18.0 | 29.3 | 39.7 | 16.0 | 4.53° (`L_wrist_roll` 12.7) | 1.3° | 67.3 | 1.27% / `L_ankle_pitch` 18.4% | 0.337 | 9.1° |
| `run2` | `083926` | walk_sideway | ✅ | **24.9** | 18.6 | 37.4 | 54.6 | **8.6** | 4.73° (`L_wrist_roll` 16.8) | 1.9° | 74.0 | 0.90% / 14.8% | 0.252 | 7.2° |
| `run3` | `084209` | walk_backward | ✅ | 25.3 | 21.4 | 35.6 | 51.7 | 18.2 | 4.61° (`R_wrist_roll` 17.2) | 2.7° | 99.9 | 1.59% / 26.1% | 0.473 | 6.4° |
| `run4` | `084407` | walk_backward | ✅ | **23.9** | 20.2 | 33.1 | 47.9 | 18.3 | 4.45° (`R_wrist_roll` 17.1) | 2.9° | 100.2 | 1.51% / 20.1% | 0.457 | 6.6° |
| `run6` | `084914` | jog_ff_start | ✅ | 36.2 | 27.5 | 48.0 | 54.0 | 19.6 | 5.83° (`L_wrist_roll` 21.3) | 3.3° | 94.8 | 2.65% / 34.9% | 0.791 | 15.7° |
| `run7` | `085159` | wiping_shoes | ✗ order | 37.4 | 31.1 | 29.8 | 38.0 | 53.6 | 5.34° (`R_shoulder_yaw` 17.1) | 1.5° | **39.5** | 0.10% | **0.045** | 2.8° |
| `run12` | `091220` | high_jump | ✗ order | **130.5** | 97.2 | 91.2 | 43.4 | **177.5** | 18.82° (`R_knee` 38.6) | **90.6°** | 78.5 | 1.01% | 1.395 | **22.7°** |
| `run20` | `092520` | dance_vouge | ✗ order | 54.5 | 39.6 | 62.9 | 74.7 | 45.2 | 13.05° (`R_wrist_roll` 41.9) | 4.8° | **166.2** | **3.16%** / 27.0% | **1.831** | 22.4° |

## 4. Findings

1. **Best locomotion numbers measured so far.** All four locomotion clips land in
   **20.6–36.2 mm**, and `walk_backward` reproduces to **±0.7 mm across two reps** — an
   order of magnitude tighter than the 09-04 rep spread (σ=16 mm). Whatever changed between
   `sonic_no_vr_050k` and `sonic_no_vr_ll_062k`, rep-to-rep consistency improved markedly.
2. **`walk_sideway` is fixed here.** 24.9 mm with a **1.9° heading error**, versus 51–57 mm
   and **20–30° heading error** on 09-09 (`sonic_no_vr_050k`). The lateral-gait heading
   failure identified in the 09-09 report does not reproduce with this policy — the single
   most valuable finding of this session.
3. **Command error stayed high while imitation error dropped** (cmd `mpjpe_l` 67–100 mm vs
   imitation 21–36 mm). The PD loop still lags each instantaneous command and averages onto
   the reference; this ordering now holds across three policies.
4. **`jog_ff_start` remains the actuation hotspot** — 2.65% mean saturation, `L_ankle_pitch`
   saturating 34.9% of timesteps, shaking 0.79, tilt 15.7°, i.e. essentially unchanged from
   09-09 (32.2 mm / 2.61% / 0.70). The jog clip is limited by hardware headroom, not by the
   policy version.
5. **The agility block fails exactly as on 09-12.** `high_jump` (130.5 mm, `vr_3points`
   177.5, heading **90.6°**, tilt 22.7°) and `dance_vouge` (cmd **166.2 mm**, shaking
   **1.83**, saturation **3.16%** — the worst of any episode measured) confirm the capability
   boundary is a *policy/clip* property, not a `sonic_no_vr_050k` artifact.
6. **`wiping_shoes` again behaves as the noise floor** (shaking 0.045, tilt 2.8°, cmd
   39.5 mm) with a ×1.02 identification margin — near-static clips are unusable both as
   benchmarks and as anchors.
7. **No falls**, max tilt 22.7°.

## 5. Caveats

- Two sessions (`body_check_002`, `jump_ff_270_001`) are rejected at 234 mm / 106 mm SMPL
  identification error — the streamed content does not correspond to any library clip
  closely enough to anchor an episode.
- Three episodes are order-paired; their absolute values are indicative only.
- `run5`, `run8`, `run11` produce no usable episode (no plausible streamed clip left to pair).

## 6. Reproduction

```bash
.venv_sim/bin/python sim2real/eval_online_runs.py \
    --runs     /home/grease/g1_robot_data/g1_run_0915 \
    --sessions /home/grease/g1_robot_data/20260915 \
    --out      sim2real/online_eval_results_20260915.json
```
