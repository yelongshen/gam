# Real-Robot Online Deployment Evaluation — 2026-09-24 Session (report)

- **Data**: `g1_robot_data/g1_run_0924` (2 runs) + `g1_robot_data/20260924` (2 stream sessions)
- **Policy**: `policy/sonic_no_vr_ll_062k` for both runs, `reference/example/`, 50 Hz,
  encoder on, fp16 off — i.e. **the same configuration as the second half of 09-19**
- **Tooling**: `sim2real/eval_online_runs.py` (content pairing)
- **Raw output**: `sim2real/online_eval_results_20260924.json`
- **Result**: **2 episodes located, 2 CONFIRMED (100%)**, 0 falls
- Metric definitions: `sim2real/online_eval_20260904_report.md` §4

> **Cross-session comparison:** `sim2real/online_eval_policy_comparison.md` pools all
> 8 sessions and compares the deployed policies on matched clips.

---

## 1. Identification

Clean one-clip-per-run session; both runs confirm on both sides.

| run | session | SMPL vote | robot vote | verdict |
|---|---|---|---|---|
| `20260924_075444` span[4522:5692] | `streamed_075632` | `walk_180_R_003__A332_M` — 28.9 mm, ×2.13, +86.6° | same, 8.24° ×1.21 | ✅ |
| `20260924_075852` span[1664:3080] | `streamed_075948` | `walk_backward_start_001__A030_M` — 32.2 mm, ×1.78, −89.6° | same, 7.54° ×1.33 | ✅ |

**The stream itself is healthy this time**: `walk_180` identifies at **28.9 mm / ×2.13**,
matching the good sessions (09-09: 28.5–28.7 mm ×2.15; 09-15: 28.2 mm ×2.17) and unlike
09-19, where the same clip only reached 55.6–63.2 mm at ×1.02–1.22. This is the key control
for the 09-19 diagnosis — see §4.

## 2. Results

| run | clip | frames | `mpjpe_l` | pa | legs | foot | vr_3pt | upper | vel / accel | joint err (worst) | heading | cmd | sat / worst | shake | tilt | fall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `075444` | `walk_180_R_003__A332_M` | 560 (11.2 s) | **37.6** | 34.1 | 56.9 | 83.6 | 26.3 | 15.4 | 4.59 / 1.71 | 5.70° (`R_knee` 11.5) | 3.2° | 66.6 | 1.01% / `R_ankle_pitch` 17.7% | 0.332 | 8.3° | no |
| `075852` | `walk_backward_start_001__A030_M` | 199 (4.0 s) | **34.8** | 29.8 | 46.1 | 68.2 | 31.0 | 16.8 | 5.90 / 2.90 | 5.48° (`R_wrist_roll` 18.3) | 7.8° | **111.5** | 1.73% / `L_ankle_pitch` 30.7% | 0.720 | 7.9° | no |

Raw-SMPL diagnostics (not sim-comparable): robot-vs-SMPL 246.8–249.6 mm, retargeting alone
247.3–247.7 mm — i.e. the retargeting term again accounts for essentially all of it.

## 3. Cross-session placement of these two clips

| session | policy | reference dir | stream id quality (`walk_180`) | `walk_180` | `walk_backward` |
|---|---|---|---|---|---|
| 09-15 | `ll_062k` | `real_example/` | 28.2 mm ×2.17 | **20.6 mm** | **23.9 / 25.3 mm** |
| 09-19 | `ll_062k` | `example/` | 55.6–63.2 mm ×1.02–1.22 | 46.6 / 48.5 mm | 32.1 mm |
| **09-24** | `ll_062k` | `example/` | **28.9 mm ×2.13** | **37.6 mm** | **34.8 mm** |
| 09-22 | `sonic_no_vr_050k` | `example/` | 28.3–39.3 mm | 18.8 / 29.7 mm | 21.1 mm |

## 4. Findings

1. **The 09-19 "bad stream" hypothesis is only half the story.** 09-24 runs the *identical*
   policy + reference config as 09-19 but with a **healthy stream** (28.9 mm ×2.13), and the
   numbers only partially recover: `walk_180` 46.6–48.5 → **37.6 mm**, still **1.8× worse
   than 09-15's 20.6 mm**; `walk_backward` does not recover at all (32.1 → 34.8 mm vs
   23.9–25.3 mm on 09-15). Stream quality explains roughly half of the 09-19 gap; the
   remaining factor tracks the **`reference/example/` vs `reference/real_example/`**
   difference. That is now the single highest-value thing to test.
2. **`walk_backward` has a heavy control-loop cost here**: cmd `mpjpe_l` **111.5 mm** (the
   largest of any confirmed locomotion episode outside 09-15's 100 mm), shaking 0.72 rad/s,
   `L_ankle_pitch` saturating **30.7%** of timesteps on a 4-second walk. The imitation error
   (34.8 mm) is again far below the command error — the PD loop lags but averages onto the
   reference, consistent with every session since 09-04.
3. **Foot remains the worst FK subset** (83.6 mm on `walk_180`, 2.3× leg error) — the same
   phase/timing signature as 09-19, not a pose error.
4. **Heading is fine** (3.2° / 7.8°), so the lateral-gait heading failure of 09-09 does not
   reappear; `ll_062k` continues to look clean on that axis.
5. **No falls**, max tilt 8.3°, saturation ≤1.7% mean — mechanically a comfortable session.

## 5. Caveats

- n=1 per clip. Per the 09-04 finding on rep-to-rep variance (up to 2.3×), neither number
  should be used for a policy comparison on its own.
- Only two runs exist in this dataset, so there is no within-session control.

## 6. Reproduction

```bash
.venv_sim/bin/python sim2real/eval_online_runs.py \
    --runs     /home/grease/g1_robot_data/g1_run_0924 \
    --sessions /home/grease/g1_robot_data/20260924 \
    --out      sim2real/online_eval_results_20260924.json
```
