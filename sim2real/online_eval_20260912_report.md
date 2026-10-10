# Real-Robot Online Deployment Evaluation — 2026-09-12 Session (report)

- **Data**: `g1_robot_data/g1_run_0912` (6 runs) + `g1_robot_data/20260912` (7 stream sessions)
- **Policy**: `policy/sonic_no_vr_050k` for **all** runs (`reference/real_example/`, 50 Hz)
- **Tooling**: `sim2real/eval_online_runs.py` (content pairing)
- **Raw output**: `sim2real/online_eval_results_20260912.json`
- **Result**: **7 episodes located, 1 CONFIRMED** — this is an *agility stress* session and
  mostly a **tracking-failure** record, not a tracking-quality record
- Metric definitions: `sim2real/online_eval_20260904_report.md` §4

> **Cross-session comparison:** `sim2real/online_eval_policy_comparison.md` pools all
> 8 sessions and compares the deployed policies on matched clips.

---

## 1. Session character: the robot did not follow the clip

Five of the seven streamed clips are high-agility (`high_jump_R_103` ×4, two dance clips).
On those, the executed `q` stops resembling the reference, so the robot-side identification
**cannot** confirm the clip: it wanders to a generic `jump_ff_360` / `idle_turn` with a
margin of only ×1.02–1.17 (i.e. no discrimination at all), and the streamed clip itself only
fits the executed span at **12–18° rms** (vs 6.1–8.0° on a clip that is actually tracked).

This is why `eval_online_runs.py` evaluates order-paired episodes instead of dropping them —
discarding non-agreeing pairs would silently delete exactly the failures we want to measure.
Everything below except `walk_backward` is flagged `confirmed=False` by design.

## 2. What was streamed (SMPL-side identification)

| session | clip | rms | margin | robot-side fit of that clip |
|---|---|---|---|---|
| `082831` | `high_jump_R_103__A389_M` | 63.9 mm | ×1.64 | 15.10° |
| `083110` | `high_jump_R_103__A389_M` | 64.4 | ×1.62 | 15.18° |
| `083541` | `walk_backward_start_001__A030_M` | **19.1** | **×3.21** | **6.12°** ✅ |
| `084118` | `high_jump_R_103__A389_M` | 64.5 | ×1.62 | 15.94° |
| `084538` | `high_jump_R_103__A389_M` | 63.9 | ×1.63 | 15.02° |
| `084929` | `dance_hiphop_mike_tyson_R_fast_001__A319_M` | 51.1 | ×2.07 | 12.06° |
| `085741` | `dance_vouge_shake_it_babe_360_R_002__A318_M` | 78.0 | ×1.17 | 17.85° |

All seven sessions were paired; no orphans, no rejections.

## 3. Results

### 3.1 Per-episode

| run | session | clip | conf | `mpjpe_l` | pa | legs | foot | vr_3pt | joint err (worst) | heading | cmd | sat / worst | shake | tilt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `run1` | `082831` | high_jump | ✗ | 51.3 | 37.2 | 50.1 | 43.9 | 40.1 | 9.60° (`L_wrist_yaw` 27.0) | 2.4° | **116.2** | 1.49% / `waist_pitch` 21.5% | **1.422** | 16.1° |
| `run2` | `083110` | high_jump | ✗ | **126.1** | 101.3 | 85.9 | 59.4 | **185.4** | 18.90° (`L_shoulder_pitch` 41.5) | **83.8°** | 96.4 | 1.38% / 20.0% | 1.390 | 19.6° |
| `run2` | `083541` | **walk_backward** | ✅ | 72.1 | 67.5 | 95.5 | **156.0** | 72.1 | 9.34° (`L_knee` 22.1) | **82.9°** | 81.5 | 1.35% / 27.6% | 0.574 | 19.6° |
| `run3` | `084118` | high_jump | ✗ | 47.8 | 38.1 | 49.0 | 47.3 | 39.3 | 10.07° (`L_wrist_yaw` 27.5) | 4.2° | **124.9** | 2.68% / 22.3% | **1.466** | 18.3° |
| `run4` | `084538` | high_jump | ✗ | 41.2 | 31.2 | 41.3 | 33.8 | 33.3 | 9.22° (`L_wrist_yaw` 28.4) | 1.3° | 117.0 | 2.15% / `waist_pitch` 24.6% | 1.329 | 19.0° |
| `run5` | `084929` | dance_hiphop | ✗ | 45.1 | 34.4 | 50.8 | 55.6 | 42.4 | 9.18° (`L_wrist_yaw` 19.8) | 4.8° | **140.6** | 2.75% / 35.1% | 0.921 | 15.2° |
| `run6` | `085741` | dance_vouge | ✗ | 46.3 | 33.9 | 54.9 | 61.7 | 36.7 | 12.01° (`R_wrist_roll` 42.7) | 4.6° | **127.4** | 2.05% / 28.8% | 1.208 | 20.2° |

### 3.2 Only confirmed episode

`walk_backward_start_001__A030_M` / `run2` — **72.1 mm** `mpjpe_l`. For reference the same
clip scores **23.9–25.3 mm** on 09-15 and **32.1–38.8 mm** on 09-19, so this rep is the worst
of the seven `walk_backward` reps across all sessions.

## 4. Findings

1. **`high_jump_R_103` is not executable by this policy.** Four reps, none confirmed,
   cmd `mpjpe_l` 96–125 mm (vs 57–75 mm on tracked locomotion), shaking **1.33–1.47 rad/s**
   (3–4× normal), `waist_pitch` the worst saturating joint. The `mpjpe_l` numbers (41–51 mm)
   look deceptively ordinary — that is an artifact of scoring a *failed* execution against a
   short 130-frame reference; the honest signals are the cmd error, the shaking, and the
   12–18° clip-fit residual.
2. **`run2` is a distinct, more severe failure mode.** 126.1 mm `mpjpe_l`, `vr_3points`
   **185 mm**, `L_shoulder_pitch` off by 41.5°, heading error **83.8°** — the robot turned
   nearly sideways. Its `walk_backward` episode later in the same run also shows an 82.9°
   heading error, so the run had a persistent yaw problem rather than a per-clip glitch.
3. **The dance clips degrade the same way but less violently** — cmd error 127–141 mm,
   shaking 0.92–1.21, `R_wrist_roll` 42.7° on `dance_vouge`. Distal, low-inertia joints
   dominate, matching the Phase C.2 / `phaseE_waist_roll_chatter.md` picture.
4. **Ankle pitch saturation is elevated session-wide** (1.4–2.8% mean, up to 35% of timesteps
   on the hiphop clip) — the highest of any session except 09-15's `run20`.
5. **Still no falls** (max tilt 20.2°), i.e. the policy degrades gracefully: it refuses the
   motion rather than destabilising.

## 5. Recommended reading of this session

Do **not** quote this session's `mpjpe_l` as a tracking result. Use it as:

- a **capability boundary** record (jump/dance out of distribution for `sonic_no_vr_050k`), and
- a **failure-signature** reference: cmd `mpjpe_l` > 100 mm + shaking > 1.0 rad/s + streamed
  clip fitting worse than ~12° is a reliable "did not track" detector.

## 6. Reproduction

```bash
.venv_sim/bin/python sim2real/eval_online_runs.py \
    --runs     /home/grease/g1_robot_data/g1_run_0912 \
    --sessions /home/grease/g1_robot_data/20260912 \
    --out      sim2real/online_eval_results_20260912.json
```
