# Real-Robot Online Deployment Evaluation — 2026-09-22 Session (report)

Second execution record for **Step 1** of `sim2real/online_deployment_eval_plan.md`,
using the same metric suite as the 2026-09-04 report
(`sim2real/online_eval_20260904_report.md`, §4) but a **new session protocol**:

> **One motion clip replay per run.** Each policy run in `g1_run_0922/` replays exactly
> one `eval_subset` clip, and the matching `20260922/.../streamed_<HHMMSS>/` session holds
> the SMPL stream that was sent for that replay. The clip identity is therefore
> **cross-checked from both sides** (streamed SMPL *and* executed robot state) before any
> metric is computed.

- **Data**: `/home/grease/g1_robot_data/{g1_run_0922, 20260922/20260922}`
- **Reference**: `ego_dataset/eval_subset/{smpl,robot}/<clip>.pkl` (all **160** clips searched)
- **Policy under test**: `policy/sonic_no_vr_050k` (encoder on, fp16 off, 50 Hz, 29 dof) —
  **different policy from the 09-04 session**, which ran `policy/low_latency`
- **Tooling**: `sim2real/eval_runs_0922.py` (new), reusing `sim2real/online_eval_clips.py`
- **Raw output**: `sim2real/online_eval_results_20260922.json`
- **Nature of the numbers**: **online** (real hardware telemetry), computed post-hoc from
  CSV logs. No simulator involved. Same validity limits as the 09-04 report §7.
- **Related day reports**: `online_eval_20260904_report.md`, `online_eval_20260909_report.md`,
  `online_eval_20260912_report.md`, `online_eval_20260915_report.md`,
  `online_eval_20260919_report.md`, `online_eval_20260924_report.md` (the last five use
  `sim2real/eval_online_runs.py`, which pairs runs to stream sessions by content instead of
  by wall clock).

> **Cross-session comparison:** `sim2real/online_eval_policy_comparison.md` pools all
> 8 sessions and compares the deployed policies on matched clips.

---

## 1. Clip identification: two independent votes

Neither side carries a usable label (`motion_name.csv` only ever says `"streamed"`, the
stream logs have no name field), so the clip is recovered from **state** twice:

| side | signal | reference | search |
|---|---|---|---|
| **SMPL** (what was sent) | `streamed_*/smpl_joint.csv`, 24×3 @50 Hz, root-relative | `eval_subset/smpl/*.pkl` → `smpl_joints` | yaw-invariant FFT sliding SSD over all 160 clips |
| **Robot** (what was executed) | `q.csv` (29 dof @50 Hz) inside the `"streamed"` span | `eval_subset/robot/*.pkl` → `dof`, resampled 30→50 Hz | FFT sliding SSD over all 160 clips |

A clip is **CONFIRMED** only when both winners are the same name. Three non-obvious
details were required to make this reliable:

1. **The stream is yaw-rotated.** The streamed SMPL is emitted in the sender's world frame,
   which differs from the reference clip frame by a constant rotation about z (measured
   **+86.6…+90.4°**, and **−89.9°** for one session). Naïve matching therefore fails —
   the correct clip scored *worse* than unrelated ones. The search now solves for the
   optimal yaw in closed form at every offset: with
   `A = Σ(sx·rx + sy·ry)`, `B = Σ(sy·rx − sx·ry)`, `C = Σ(sz·rz)`, the best rotated
   cross-correlation is `√(A²+B²) + C` at `θ = atan2(B, A)` — one extra FFT pair, no grid
   search. The fitted yaw is then undone before the raw-SMPL diagnostics.
2. **Sessions are mostly idle.** Each ~20 s session contains ~4–11 s of motion inside a
   settle/hold. Unconstrained, a *static* clip wins by fitting the idle tail — the first
   pass "confirmed" `wiping_shoes_R_001__A234` on both sides for run `084036`, which was
   spurious. Candidate offsets are now restricted to cover the detected motion window.
3. **Length plausibility.** Since exactly one clip is streamed per session, candidates whose
   length is outside **0.6–1.6×** the motion-window length are dropped.

### 1.1 Match table

| run | session | stream motion window | SMPL vote (rms, margin over runner-up, yaw) | robot vote (rms) | verdict |
|---|---|---|---|---|---|
| `20260922_083033` | *(none)* | — | — | `wiping_shoes_R_001__A234` (8.11°) | ❌ **unconfirmed** |
| `20260922_083850` | *(none)* | — | — | — | ❌ **aborted** (0 frames logged) |
| `20260922_084036` | `streamed_084128` | 128–670 (10.8 s) | `walk_180_R_003__A332_M` — 39.3 mm, ×1.52, +86.9° | `walk_180_R_003__A332_M` (7.80°) | ✅ |
| `20260922_085326` | `streamed_085420` | 128–501 (7.5 s) | `walk_180_R_003__A332_M` — 28.3 mm, ×2.18, +86.6° | `walk_180_R_003__A332_M` (5.95°) | ✅ |
| `20260922_085555` | `streamed_085658` | 112–302 (3.8 s) | `walk_backward_start_001__A030_M` — 19.4 mm, ×3.17, −89.9° | `walk_backward_start_001__A030_M` (6.24°) | ✅ |

Run↔session pairing is by wall clock (session start inside the run's
`[start, start + duration]` window, from the directory timestamps + `time_ms`).

**Two runs produce no metrics, by design of the gate:**

- `20260922_083850` logged **0 frames** (aborted run).
- `20260922_083033` has **no paired stream session** — its robot-side best is only
  8.11° rms (vs 5.95–7.80° for the confirmed ones) with a ×1.10 margin, i.e. no credible
  clip. It is reported and skipped rather than guessed.
- `streamed_083353` is an **orphan session** (streamed with no usable robot log, most
  likely the stream for the aborted `083850` run). Its own SMPL-side best is weak
  (63 mm) — reported, not evaluated.

---

## 2. Results

### 2.1 Per-clip aggregate (PRIMARY imitation metric, mean ± std)

| clip | n | `mpjpe_l` | `mpjpe_pa` | `vel_dist` | joint err | non-fall |
|---|---|---|---|---|---|---|
| `walk_180_R_003__A332_M` | 2 | **24.2 ± 5.4 mm** | 22.0 ± 5.2 | 3.4 ± 0.7 | 4.86 ± 0.47° | 100% |
| `walk_backward_start_001__A030_M` | 1 | **21.1 mm** | 16.5 | 3.6 | 4.51° | 100% |

### 2.2 Per-episode

| # | clip / run | frames | `mpjpe_l` | `mpjpe_pa` | legs | foot | vr_3pt | upper | joint err (worst) | heading err | cmd `mpjpe_l` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | walk_180 / `084036` | 560 (11.2 s) | 29.7 | 27.2 | 46.2 | 63.4 | 20.1 | 12.4 | 5.33° (`L_wrist_yaw` 14.2) | 3.27° | 70.7 |
| 2 | walk_180 / `085326` | 560 (11.2 s) | **18.8** | 16.8 | 27.0 | 36.1 | 14.4 | 9.4 | 4.40° (`L_wrist_yaw` 13.3) | 1.77° | 57.2 |
| 3 | walk_backward / `085555` | 199 (4.0 s) | 21.1 | 16.5 | 25.3 | 35.4 | 18.9 | 9.9 | 4.51° (`R_wrist_roll` 16.2) | 2.72° | 92.0 |

### 2.3 Robustness

| # | saturation mean / worst | shaking | swing r/p | max tilt | non-fall |
|---|---|---|---|---|---|
| 1 | 1.42% / `R_ankle_pitch` **17.0%** | 0.485 rad/s | 3.08 / 3.81° | **18.8°** | ✅ |
| 2 | 0.84% / `L_ankle_pitch` 12.5% | 0.309 | 2.82 / 2.20° | 6.4° | ✅ |
| 3 | 1.82% / `R_ankle_pitch` **26.6%** | 0.421 | 2.14 / 2.56° | 6.6° | ✅ |

### 2.4 Diagnostics vs raw SMPL (morphology-contaminated, NOT sim-comparable)

| # | robot-vs-SMPL `mpjpe_l` (pa) | retargeting alone (pa) |
|---|---|---|
| 1 | 251.3 (105.7) | 246.7 (101.0) |
| 2 | 250.2 (100.5) | 247.3 (95.8) |
| 3 | 252.0 (95.9) | 251.5 (96.4) |

---

## 3. Findings

1. **Tracking improved ~1.9× vs the 09-04 session on the shared clip.** `walk_180_R_003`
   goes from **45.6 ± 16.2 mm** (n=3, `policy/low_latency`) to **24.2 ± 5.4 mm** (n=2,
   `policy/sonic_no_vr_050k`), with mean joint error down 6.46° → 4.86°. The n's are small
   and the policy *and* session protocol both changed, so this is a strong signal but not a
   controlled A/B.
2. **Rep-to-rep spread also shrank** — 29.7 vs 18.8 mm across the two `walk_180` reps
   (1.6×), against 2.3× (29.8→67.8 mm) on 09-04. The 09-04 finding that n=1 comparisons are
   invalid still holds, just with a smaller noise floor.
3. **Imitation error is again well below command error** (18.8–29.7 mm vs 57–92 mm cmd
   `mpjpe_l`): the PD loop lags each instantaneous command but averages onto the reference.
   This reproduces 09-04 finding #4 on a different policy.
4. **Ankle pitch is still the actuation hotspot** — worst saturation joint in all 3
   episodes, peaking at **26.6%** of timesteps above 0.9×effort on `walk_backward`, i.e.
   *worse* than the 09-04 peak (21.7%). Foot remains the worst FK subset (35–63 mm).
5. **The stress axes moved from shoulder-yaw to wrist.** Worst reference-tracking joints are
   now `left_wrist_yaw` (13.3–14.2°) and `right_wrist_roll` (16.2°); `shoulder_yaw` errors
   dropped to 4.7–7.2°. The low-inertia-twist problem (Phase C.2 / `phaseE_waist_roll_chatter.md`)
   is still present but has shifted distally.
6. **Episode 1 (`084036`) is the outlier across every axis** — 1.6× the positional error,
   1.6× the shaking, and **18.8° max tilt** vs 6.4° on the identical clip 13 minutes later.
   That tilt is the largest in either session and is within 2× of the 35° fall threshold:
   worth pulling the raw traces before treating the `walk_180` mean as representative.
7. **Retargeting still dominates the "human-fidelity" number.** `retarget_diag` is
   **247–252 mm** — nearly all of the 250 mm robot-vs-SMPL figure — up from 114–126 mm on
   09-04 because that session's diagnostics were not yaw-corrected in the same way; treat
   the absolute value as a property of the SMPL→G1 map, not of the controller.

---

## 4. Reproduction

```bash
.venv_sim/bin/python sim2real/eval_runs_0922.py \
    --runs      /home/grease/g1_robot_data/g1_run_0922 \
    --sessions  /home/grease/g1_robot_data/20260922/20260922 \
    --smpl-ref  /home/grease/ego_dataset/eval_subset/smpl \
    --robot-ref /home/grease/ego_dataset/eval_subset/robot \
    --out       sim2real/online_eval_results_20260922.json
```

The script prints, per run: the stream motion window, the top-3 SMPL-side and robot-side
clip candidates with their rms/offset/yaw, the confirm-or-skip verdict, and the full metric
block. Episode definitions (clip, offset, `t0_ms`/`t1_ms`, match quality) are persisted in
`episode_defs` in the JSON, so the evaluation is replayable without re-running the search.

---

## 5. Validity limits

All of §7 of the 09-04 report applies unchanged (`mpjpe_g` is degenerate without mocap; the
30→50 Hz reference resampling mildly low-passes `vel_dist`/`accel_dist`; `motor_torque` is
an estimate). Additional to this session:

- **n is small** (2 + 1 confirmed episodes from 5 runs). The plan's ≥3-reps-per-clip bar is
  not met for either clip.
- **Cross-session comparison is confounded**: `policy/sonic_no_vr_050k` here vs
  `policy/low_latency` on 09-04, plus a different streaming protocol. Attribute the
  improvement to "this configuration", not to the policy weights alone.
- **The clip-identity gate is conservative on purpose.** Margins of ×1.5–×3.2 on the SMPL
  side are comfortable, but ×1.52 (episode 1) is the weakest; it is corroborated by the
  robot side and by the motion-window length (542 streamed frames vs the clip's 560).

---

## 6. Next steps

1. Re-run `walk_180` for ≥3 more reps with this policy to pin down whether episode 1's
   18.8° tilt / 29.7 mm is a tail event or a bimodal failure mode.
2. Log the clip name on the streaming side (`stream_clip_mode2.py`) so the state-matching
   pipeline becomes a *verification* step rather than the only source of identity.
3. Extend the confirmed-episode set beyond the two locomotion clips before drawing any
   policy-level conclusion (the `eval_clip_manifest.md` set has 139 usable clips).
