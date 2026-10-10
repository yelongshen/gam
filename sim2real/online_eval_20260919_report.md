# Real-Robot Online Deployment Evaluation — 2026-09-19 Session (report)

- **Data**: `g1_robot_data/g1_run_0919` (8 runs) + `g1_robot_data/20260919` (6 stream sessions)
- **Policies**: `policy/low_latency` (runs `073126`–`080352`) and
  `policy/sonic_no_vr_ll_062k` (runs `081337`, `081856`, `082250`), both with
  `reference/example/` — note this is the **generic** reference dir, not `real_example/`
  as on 09-09/09-12/09-15
- **Tooling**: `sim2real/eval_online_runs.py` (content pairing)
- **Raw output**: `sim2real/online_eval_results_20260919.json`
- **Result**: **5 episodes located, 2 CONFIRMED**, 0 falls
- Metric definitions: `sim2real/online_eval_20260904_report.md` §4

> **Cross-session comparison:** `sim2real/online_eval_policy_comparison.md` pools all
> 8 sessions and compares the deployed policies on matched clips.

---

## 1. Session character: weak stream identification throughout

Unlike 09-09/09-15, the SMPL identifications here are **uniformly poor**: the three
`walk_180` sessions score **55.6–63.2 mm** with margins of only **×1.02–1.22**, against
**28.2–28.7 mm at ×2.15–2.19** on the other days for the *same clip*. `walk_backward` lands
at 30.7–31.6 mm vs 18.4–19.1 mm elsewhere. Something about this day's streaming is
degraded — retiming, dropped chunks, or a different `--chunk-frames` setting — and it
propagates into every downstream number.

Run directories here *are* timestamped (`20260919_HHMMSS`), so `eval_runs_0922.py`'s
wall-clock pairing would also work; content pairing is used for consistency with the other
reports.

## 2. What was streamed (SMPL-side identification)

| session | clip | rms | margin | outcome |
|---|---|---|---|---|
| `074543` | `walk_180_R_003__A332_M` | 63.2 mm | ×1.02 | order-paired (`074421`, low_latency) |
| `075116` | `walk_backward_start_001__A030_M` | 31.6 | ×1.90 | ✅ confirmed (`074919`, low_latency) |
| `081621` | `walk_180_R_003__A332_M` | 55.6 | ×1.22 | order-paired (`081337`, ll_062k) |
| `082035` | `walk_180_R_003__A332_M` | 57.7 | ×1.05 | order-paired (`081856`, ll_062k) |
| `082350` | `walk_backward_start_001__A030_M` | 30.7 | ×1.98 | ✅ confirmed (`082250`, ll_062k) |
| `080453` | `neutral_dancecard_go_turn_walk_001__A536` | **174.1** | ×1.34 | ❌ rejected (above the 90 mm bar) |

## 3. Results

### 3.1 Per-clip aggregate (CONFIRMED only)

| clip | n | `mpjpe_l` | `mpjpe_pa` | `vel_dist` | joint err | non-fall |
|---|---|---|---|---|---|---|
| `walk_backward_start_001__A030_M` | 2 | **35.4 ± 3.4 mm** | 29.8 ± 1.7 | 5.06 ± 0.05 | 6.39 ± 1.14° | 100% |

### 3.2 Per-episode

| run (policy) | session | clip | conf | `mpjpe_l` | pa | legs | foot | vr_3pt | joint err (worst) | heading | cmd | sat / worst | shake | tilt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `074421` (low_latency) | `074543` | walk_180 | ✗ order | 51.9 | 48.1 | 74.9 | **118.9** | 36.9 | 7.05° (`L_wrist_roll` 16.1) | 7.2° | 70.1 | 0.81% / `R_ankle_pitch` 13.4% | 0.429 | 8.5° |
| `074919` (low_latency) | `075116` | walk_backward | ✅ | 38.8 | 31.6 | 44.2 | 65.0 | 33.5 | 7.53° (`R_elbow` 24.4) | 3.2° | 96.9 | 1.54% / 22.6% | 0.443 | 5.3° |
| `081337` (ll_062k) | `081621` | walk_180 | ✗ order | 48.5 | 46.3 | 74.6 | 116.5 | 37.4 | 6.57° (`R_knee` 14.1) | 4.1° | 62.6 | 0.63% / 11.1% | 0.341 | 7.2° |
| `081856` (ll_062k) | `082035` | walk_180 | ✗ order | 46.6 | 43.1 | 71.4 | 110.0 | 32.6 | 6.30° (`R_knee` 16.6) | 4.1° | 75.8 | 1.05% / 16.4% | 0.423 | 8.5° |
| `082250` (ll_062k) | `082350` | walk_backward | ✅ | **32.1** | 28.1 | 43.2 | 69.6 | 27.6 | 5.26° (`R_wrist_roll` 18.4) | 5.1° | 103.4 | 2.08% / `L_ankle_pitch` 33.2% | 0.510 | 6.0° |

## 4. Findings

1. **Everything is ~2× worse than 09-15 on the same clips.** `walk_180` sits at
   **46.6–51.9 mm** (vs 20.6 mm on 09-15) and `walk_backward` at **32.1–38.8 mm**
   (vs 23.9–25.3 mm). Crucially this holds for **both** policies in the session, including
   `sonic_no_vr_ll_062k`, which is the same policy that produced 09-15's best-ever numbers.
   The regression is therefore **not** attributable to the policy.
2. **The most likely cause is the stream itself.** The SMPL-side identification error for a
   *fixed* clip is a property of the streamed data alone (no robot involved), and it is
   2.2× worse than on 09-15 (63 vs 28 mm) with the margin collapsing to ×1.02. A degraded /
   retimed stream would both explain that and inflate every robot-side number downstream.
   The different `reference_motion_path` (`reference/example/` vs `real_example/`) is a
   second candidate and is cheap to check.
3. **Foot error is the dominant term** on `walk_180` (110–119 mm, 2.3–2.5× the leg error) —
   the same signature as the worst 09-04 reps, consistent with a timing/phase mismatch
   rather than a pose mismatch.
4. **`ll_062k` is still mildly better than `low_latency` within the session**: `walk_180`
   46.6/48.5 vs 51.9 mm, `walk_backward` 32.1 vs 38.8 mm. The ordering agrees with 09-09's
   `050k` > `low_latency` result.
5. **Ankle pitch saturation peaks at 33.2%** of timesteps (`082250`) — the highest single
   joint figure outside the agility sessions — despite this being a plain 4 s backward walk.
6. **No falls**, max tilt 8.5°, shaking 0.34–0.51 rad/s (normal).

## 5. Action items

> **Update (2026-09-24).** `sim2real/online_eval_20260924_report.md` runs the *identical*
> policy + `reference/example/` config with a **healthy stream** (`walk_180` identifies at
> 28.9 mm ×2.13, vs 55.6–63.2 mm ×1.02–1.22 here) and only partially recovers:
> `walk_180` 46.6–48.5 → **37.6 mm** (still 1.8× worse than 09-15's 20.6 mm) and
> `walk_backward` 32.1 → 34.8 mm (no recovery at all). **Stream quality explains at most
> half of this session's gap**; the residual tracks the `reference/example/` vs
> `reference/real_example/` difference, which is now the prime suspect (item 2 below).

1. Diff this session's streaming invocation against 09-15's (`--chunk-frames`, `--fps`,
   `--settle`, network path) — the stream-side identification gap is measurable and should be
   reproducible offline.
2. Re-run `walk_180` and `walk_backward` with `sonic_no_vr_ll_062k` and the `real_example/`
   reference to separate the reference-dir hypothesis from the stream-quality hypothesis.
   **Now the leading hypothesis** given the 09-24 control.
3. Until then, **do not pool 09-19 numbers with 09-15** when reporting `ll_062k` performance.

## 6. Reproduction

```bash
.venv_sim/bin/python sim2real/eval_online_runs.py \
    --runs     /home/grease/g1_robot_data/g1_run_0919 \
    --sessions /home/grease/g1_robot_data/20260919 \
    --out      sim2real/online_eval_results_20260919.json
```
