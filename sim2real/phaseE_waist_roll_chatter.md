# Phase E — `waist_roll` velocity aliasing / chatter (sessions `g1_run_0905`,
`g1_run_0908`, `g1_run_0914`, cross-checked against `aug11`)

**Question:** the Phase B `Kp_eff`/`Kd_eff` refit, when re-run on new online sessions,
occasionally produces catastrophic `tau_est` vs. `tau_sim` mismatches on `waist_roll` —
far larger than the ~1.5–1.6× "2× 5020" gain anomaly documented in
`sim2real/phaseB_actuator.md` §3b. Is this a hardware fault, a bug in the
`sonic_no_vr_ll_062k` policy, or something else?

**Short answer:** it is **not a fault and not a policy-specific bug**. Commanding a
`q_target` beyond the joint's physical range turns out to be a **routine occurrence
under every policy checkpoint tested**, on `ankle_pitch`/`ankle_roll`/`waist_pitch` even
more often than on `waist_roll` (§12). What *is* unique to `waist_roll` is a
**velocity-aliasing/chatter signature** in how the joint's logged `dq` responds —
reproducible across every session tested (including one recorded under a completely
different policy checkpoint), consistent with a lightly-damped/backlash-prone
mechanical mode in the "2× 5020" paired-actuator design being intermittently excited.
Severity varies continuously from run to run rather than being on/off, and is not
confined to one robot, one session, or one checkpoint.

**Data:** `g1_robot_data/{g1_run_0905,g1_run_0908,g1_run_0914}` (deploy runs,
`policy/sonic_no_vr_ll_062k`) and `g1_robot_data/g1_real_deploy_logs` (`aug11` session,
`policy/low_latency`, used only as an independent control).

**Tooling:** `data_process/g1_params.py` (nominal `Kp`/`Kd`, joint order,
`action_to_q_target`), `data_process/g1_joint_ranges.py` (per-joint physical position
limits, from `g1_29dof.xml`), `model_eval/sim2real_phaseB_refit.py` (per-session
`Kp_eff`/`Kd_eff` free-fit, reused from `phaseB_actuator.md` §3b's recipe),
`model_eval/sim2real_phaseB_tau_compare.py` (`tau_est` vs. nominal `tau_sim`
MSE/RMSE/corr), `model_eval/sim2real_phaseE_dq_aliasing.py` (the `dq_logged` vs.
`dq_fd` diagnostic developed in this investigation),
`model_eval/sim2real_phaseE_qtarget_overshoot.py` (§12's out-of-range `q_target` check).

---

## 1. Starting point: the `Kp_eff`/`Kd_eff` refit reproduces on new data

Re-running the Phase B free-fit (`tau = Kp_eff·(q_target−q) − Kd_eff·dq`) on
`g1_run_0905` and `g1_run_0908` reproduces the original `aug11` "2× 5020" signature
almost exactly:

| joint | `Kp` nom | `Kd` nom | 0905 `Kp`/`Kd` ratio | 0908 `Kp`/`Kd` ratio | `g1_run_0914` ratio |
|---|---|---|---|---|---|
| `L_ankle_pitch` | 28.50 | 1.814 | 1.079 / 1.591 | 1.085 / 1.612 | 1.105 / 1.661 |
| `R_ankle_pitch` | 28.50 | 1.814 | 1.099 / 1.692 | 1.090 / 1.591 | 1.088 / 1.647 |
| `waist_pitch` | 28.50 | 1.814 | 1.083 / 1.611 | 1.067 / 1.553 | 1.027 / 1.375 |
| `L_ankle_roll` | 28.50 | 1.814 | 0.992 / 0.512 | 1.013 / 0.549 | 1.016 / 0.528 |
| `R_ankle_roll` | 28.50 | 1.814 | 1.025 / 0.554 | 1.024 / 0.552 | 1.004 / 0.546 |
| `waist_roll` | 28.50 | 1.814 | 1.023 / 0.536 | 1.000 / 0.525 | 0.961 / 0.461 |
| `L_hip_pitch` (control) | 99.10 | 6.309 | 0.992 / 0.982 | 0.992 / 0.971 | 0.970 / 0.926 |

Pitch joints (ankle ×2, waist) need `Kd_eff ≈ 1.4–1.7×` nominal; roll joints (ankle ×2,
waist) need `Kd_eff ≈ 0.46–0.55×` nominal; control joints match nominal to within ~5%.
Confirmed independently on three new sessions, spanning at least 10 days.

This finding **is not what this document is about** — it is the established Phase B
result, confirmed again. What follows is a second, larger, and unrelated anomaly that
surfaced while re-running it.

## 2. A much larger anomaly on `waist_roll` specifically

Comparing `tau_est` (real, logged) against the **nominal** PD law
`tau_sim = Kp_nom·(q_target−q) − Kd_nom·dq` (no correction), pooled per session:

| session | joint | MSE | RMSE (N·m) | corr |
|---|---|---|---|---|
| `g1_run_0905` (16,203 samples) | `waist_roll` | 0.305 | 0.553 | 0.972 |
| `g1_run_0908` (18,234 samples) | `waist_roll` | 0.307 | 0.554 | 0.985 |
| `g1_run_0914` (10,629 samples, **pooled**) | **`waist_roll`** | **9.429** | **3.071** | **0.810** |

`g1_run_0914`'s pooled `waist_roll` RMSE is **~5.5× higher** than the same joint in the
other two sessions, and its correlation is the worst of any joint in any session
checked. Every other joint (including the "2× 5020" siblings) stayed within the range
established by §1's gain fit. This is a distinct, larger phenomenon, isolated almost
entirely to `waist_roll`, and almost entirely inside `g1_run_0914`.

## 3. Which runs? Only 2 of 9 usable runs in `g1_run_0914`

Per-run breakdown (`model_eval/sim2real_phaseB_per_run.py`,
`sim2real_phaseB_tau_compare.py`):

| run | n | `waist_roll` RMSE | `waist_roll` corr | verdict |
|---|---|---|---|---|
| run1, run2, run3, run4, run6, run7 | 853–1151 | 0.29–0.66 | 0.988–0.998 | normal |
| run12 | 1045 | 0.94 | 0.986 | mildly elevated |
| **run8** | 2282 | **3.92** | **0.736** | **outlier** |
| **run20** | 1004 | **7.93** | **0.734** | **outlier** |
| run5, run11 | — | — | — | too few Mode-2 samples for the full `q`/`dq`/`action`/`motor_torque` fit |

`run8` (2,282 samples, the largest single run) and `run20` alone account for the
session's elevated pooled mean.

**Ruled out immediately:**
- `motor_error.csv`: **zero nonzero error rows** in either run, across all 29 motors.
- `motor_temperature.csv`: `waist_roll` sits at 35–61 °C (`run8`) and 34–45 °C
  (`run20`) — ordinary operating range, not even the hottest joint in either run, and
  `run20` (the *worse* torque-fit outlier) runs `waist_roll` **cooler** than `run8`.
- Torque saturation: neither `tau_est` nor the nominal-PD `tau_sim` approach the
  `waist_roll_joint` `actuatorfrcrange="-50 50"` limit in `g1_29dof.xml` during the
  worst windows (`tau_sim` peaks at 49.2/28.8 N·m in `run20`/`run8`); clipping `tau_sim`
  to ±50 changes the RMSE by **0.000**.
- CSV/analysis-side alignment bug: `q`/`dq`/`action`/`motor_torque` row counts match
  exactly within each run (only `encoder_mode` is off by one row, a harmless tail
  truncation).

## 4. What the raw trace shows: `q_target` and `dq` swinging far outside the physical joint

Raw rows around the worst-error sample in each run (`waist_roll` hard range is
`±0.52 rad` per `g1_29dof.xml`):

```
run20, rows 2470-2486 (raw, encoder_mode==2 throughout):
  q_target =  -1.27, +1.02, -1.31, +2.46, -0.92, +2.35, -1.41, +2.90, -1.38, +2.25 ...
  dq       = -11.1, +14.4, -19.2, +19.6, -16.4, +20.0, -24.2, +21.5, -25.3, +18.8 ...

run8, rows 7532-7542:
  dq       = +20.3, -21.3, +20.1, -18.1, +19.8, -15.6, +16.5, -17.2, +21.0, -16.0 ...
```

`q_target` reaches **2.9 rad — 5.6× the joint's physical range** in `run20`; `dq` swings
to **±25 rad/s (~1430°/s)**, alternating sign nearly every 20 ms control tick, in both
runs, for tens of consecutive frames. Real `q` itself only wobbles by ~0.1–0.3 rad over
the same window — it does **not** show a clean back-and-forth matching `dq`'s sign
pattern.

## 5. Not simple torque saturation

Within the "chatter" rows (`|e| > 0.6 rad`):

| | run20 | run8 |
|---|---|---|
| mean \|tau_est\| | 22.8 N·m | 22.5 N·m |
| mean \|tau_sim\| (nominal PD, unclipped) | 14.8 N·m | 6.2 N·m |
| sign(tau_est) == sign(tau_sim) | 80.0% | 75.5% |
| samples within 2 N·m of that run's own `\|tau_est\|` max | 1/180 | 2/106 |

`tau_est` is not pinned near a ceiling (ruling out saturation), is **1.5–3.6× larger**
than the nominal-PD prediction, and **disagrees in sign ~20–25% of the time** — a
linear PD law cannot itself produce a sign error, so the inputs (`e`, `dq`) evaluated at
these instants are not trustworthy snapshots of the real dynamics.

## 6. Root-cause test: does the logged `dq` agree with the position trace itself?

**Method** (`model_eval/sim2real_phaseE_dq_aliasing.py`): re-derive velocity purely from
the logged `q`, independent of however firmware computes `dq`:

```python
dq_fd[i] = (q[i+1] - q[i]) / dt[i]        # ground truth, from position alone
corr = np.corrcoef(dq_logged[mode2], dq_fd[mode2])[0, 1]
```

Two honest measurements of the same physical velocity must agree in **direction**
(correlation should be strongly positive), even if they differ in *magnitude* (e.g. due
to different filtering/bandwidth — a moderate positive std-ratio is expected and
benign). A correlation dropping toward zero, and especially **going negative**, means
the two disagree about which way the joint is even moving — the fingerprint of
**aliasing**: a real oscillation faster than the 50 Hz sample rate can flip apparent
sign on a coarse position trace, while a firmware velocity estimate derived from a
faster internal loop correctly tracks the true fast motion.

Session-wide (Mode-2 only), `waist_roll`:

| run20 | run8 |
|---|---|
| corr(dq_logged, dq_fd) = **−0.818** | corr = **−0.958** |
| std(dq_logged) = 8.78 rad/s | std(dq_logged) = 3.98 rad/s |
| std(dq_fd) = 2.46 rad/s | std(dq_fd) = 1.18 rad/s |
| ratio = 3.57× | ratio = 3.37× |

A negative correlation this strong, with 3.4–3.6× magnitude excess, is not measurement
noise — noise gives correlation near zero, not strongly negative.

## 7. Per-run severity is a gradient, not a fault switch

Per-run `corr(dq_logged, dq_fd)` for `waist_roll`, all 9 usable runs in `g1_run_0914`:

| run1 | run2 | run3 | run4 | run6 | run7 | run12 | **run8** | **run20** |
|---|---|---|---|---|---|---|---|---|
| +0.850 | +0.743 | +0.815 | +0.762 | +0.774 | −0.659 | +0.341 | **−0.958** | **−0.818** |
| ratio 1.13 | 1.12 | 1.10 | 1.13 | 1.04 | 1.81 | 1.43 | **3.37** | **3.57** |

`run7` and `run12` already show early degradation before `run8`/`run20`'s severe
episodes — a smooth continuum, not a binary fault. Same picture in the other sessions:

| session | # runs with corr < 0.5 (of usable runs) |
|---|---|
| `g1_run_0905` | 1/4 |
| `g1_run_0908` | 8/12 |
| `g1_run_0914` | 4/9 |

`g1_run_0908` alone has 8 of 12 runs already below the 0.5 threshold — this is not a
`g1_run_0914`-specific problem.

## 8. Decisive test 1 — does it appear under a completely different policy checkpoint?

`aug11` was recorded under **`policy/low_latency`**, weeks before `sonic_no_vr_ll_062k`
was trained/deployed. If this were a `062k`-specific control-policy bug, `aug11` should
be clean. Chunking its 24,455 Mode-2 samples into 24 windows of 1,000 and running the
same diagnostic on `waist_roll`:

| window 8 | window 10 | window 22 | window 23 |
|---|---|---|---|
| corr = 0.401 | corr = 0.224 | corr = 0.418 | **corr = −0.473** |

**4 of 24 windows degrade, including one fully negative — under a different checkpoint,
recorded before `062k` existed.** This rules out "bug specific to `sonic_no_vr_ll_062k`."

## 9. Decisive test 2 — does the same "2× 5020" family co-degrade, or is it isolated?

Per-run `corr(dq_logged, dq_fd)`, 25 usable runs across `g1_run_0905`+`0908`+`0914`:

| | `waist_roll` | `L_ankle_roll` | `R_ankle_roll` | `waist_pitch` | `L_ankle_pitch` | **`L_hip_pitch` (control)** |
|---|---|---|---|---|---|---|
| best run | 0.850 | 0.923 | 0.914 | 0.938 | 0.976 | **0.990** |
| worst run | **−0.958** | 0.444 | 0.331 | 0.106 | 0.645 | **0.835** |
| range | **1.81** | 0.48 | 0.58 | 0.83 | 0.33 | **0.16** |

`L_hip_pitch` (single-motor, high-`Kp` control joint) never drops below 0.83 across 25
runs in 3 sessions; `waist_roll` spans a 1.81-wide range including fully negative
values. There is a modest cross-joint correlation in *which* runs are worst
(`corr(waist_roll, L_hip_pitch) across runs = 0.55`) — some of this tracks how dynamic
the whole run was (low-excitation runs are noisier for every joint) — but that shared
confound cannot explain the differential *severity*: the same run-level factor barely
touches the well-conditioned control joint while repeatedly driving `waist_roll` to zero
or negative.

## 10. All 29 joints, every run — confirming scope

Full per-joint diagnostic (`model_eval/sim2real_phaseE_dq_aliasing.py`, no
`--split-bad-runs`/`--per-run`), min/median/max across 25 runs:

| joint | min corr | median | # runs < 0.5 (of 25) | # runs < 0 |
|---|---|---|---|---|
| **`waist_roll`** | **−0.958** | 0.450 | **13** | **3** |
| `R_sho_roll` | −0.793 | 0.865 | 2 | 1 |
| `L_sho_roll` | −0.746 | 0.904 | 2 | 1 |
| `R_hip_yaw` | −0.729 | 0.919 | 1 | 1 |
| `L_sho_pitch` | −0.283 | 0.949 | 1 | 1 |
| `R_hip_roll` | −0.242 | 0.927 | 1 | 1 |
| `R_ankle_roll` / `L_ankle_roll` | 0.331 / 0.444 | 0.78 / 0.76 | 2 each | 0 |
| remaining 21 joints | ≥ 0.645 | 0.91–0.98 | 0 | 0 |

`waist_roll` is degraded in **>50% of all runs tested** (13/25), with the only 3
fully-negative sessions of any joint. A handful of other joints (mostly single-motor
shoulder/hip joints) show a single isolated dip each — most plausibly a low-excitation
noise-floor artifact in one particular take, not a recurring pattern. **21 of 29
joints never drop below 0.65 in any of the 25 runs.**

### Pooled per-session table, all 29 joints (sorted worst-first)

| joint | `aug11` (low_latency) | `g1_run_0905` | `g1_run_0908` | `g1_run_0914` |
|---|---|---|---|---|
| `waist_roll` | 0.607 | 0.326 | 0.414 | **−0.790** |
| `R_sho_roll` | 0.857 | 0.794 | 0.852 | 0.517 |
| `L_sho_roll` | 0.879 | 0.820 | 0.930 | 0.591 |
| `L_ankle_roll` | 0.618 | 0.700 | 0.761 | 0.675 |
| `waist_yaw` | 0.893 | 0.901 | 0.923 | 0.651 |
| `R_ankle_roll` | 0.769 | 0.720 | 0.768 | 0.659 |
| `waist_pitch` | 0.840 | 0.773 | 0.810 | 0.703 |
| … (21 more joints) | 0.86–0.96 | 0.83–0.96 | 0.88–0.98 | 0.74–0.97 |
| `L_knee` / `R_knee` (most robust) | 0.955 / 0.960 | 0.963 / 0.955 | 0.972 / 0.975 | 0.968 / 0.964 |

`waist_roll` is the only joint whose *session-pooled* correlation goes negative, and only
in `g1_run_0914` — where the two severe episodes (`run8`, `run20`) dominate the pooled
statistic; in every other session it stays weakly positive (0.33–0.61) but is still the
single worst joint.

## 11. Splitting `g1_run_0914` into good vs. bad runs: the episodes are whole-body, not `waist_roll`-only

`model_eval/sim2real_phaseE_dq_aliasing.py --sessions g1_run_0914 --split-bad-runs run8 run20`:

| joint | good (9 runs) | **bad (run8+run20)** | drop |
|---|---|---|---|
| `waist_roll` | 0.309 | **−0.864** | catastrophic, sign-flipped |
| `R_sho_roll` | 0.910 | **0.197** | severe, stays positive |
| `waist_pitch` / `waist_yaw` | 0.88 | 0.51–0.55 | moderate |
| `L_hip_pitch` (control) | 0.965 | 0.837 | mild |
| `R_knee` (control) | 0.974 | 0.920 | mild |

`run8`/`run20` measurably degrade **almost every joint's** `dq_logged`/`dq_fd`
agreement, not just `waist_roll` — consistent with these being genuine whole-body
high-dynamics/disturbance episodes. But only the already-fragile joints (`waist_roll`
worst, then `sho_roll`, `ankle_roll`, `waist_pitch`/`waist_yaw`) cross from "somewhat
noisier" into "qualitatively wrong / anti-correlated." The robust single-motor joints
barely move.

## 12. ⚠️ CORRECTION — is the out-of-range `q_target` unique to `waist_roll`, or to this policy?

The original version of this section used a mis-derived index
(`np.where(ISAACLAB_TO_MUJOCO == JIDX)`, which solves the **inverse** mapping) to pull
`waist_roll`'s raw policy-action channel, and reported `run8`'s channel as "tame"
(`max|action| < 2`). **That was wrong** — the correct source column for hw joint `i` is
`ISAACLAB_TO_MUJOCO[i]` (i.e. index directly, not searched for). Re-reading the correct
channel (`ISAACLAB_TO_MUJOCO[13] = 5`) shows `run8`'s raw `waist_roll` action **also**
spikes to `max|action| = 6.98`, essentially identical to `run20`'s 6.62 — not tame at
all. The "run20 = policy spike, run8 = pure hardware chatter" distinction drawn in the
original §12 is **retracted**.

### Two better questions, answered directly with data

`model_eval/sim2real_phaseE_qtarget_overshoot.py` reconstructs `q_target` for every
Mode-2 sample and checks it against each joint's physical `range` in `g1_29dof.xml`,
across all four sessions/policies.

**Q1 — Is this specific to `sonic_no_vr_ll_062k`?** No. `aug11` (`policy/low_latency`,
recorded before `062k` existed) shows the **same phenomenon, and more severely** on
some joints:

| joint | `aug11` (low_latency) | `g1_run_0905` (062k) | `g1_run_0908` (062k) | `g1_run_0914` (062k) |
|---|---|---|---|---|
| `R_ankle_pitch` | **13.70%** of samples, worst 1.87× half-range | 7.51% | 5.48% | 4.56% |
| `L_ankle_pitch` | **10.26%**, worst 2.81× | 2.28% | 6.01%, worst 3.66× | 4.53% |
| `waist_pitch` | **12.06%**, worst 1.94× | 0.86% | 2.25% | 2.99% |
| `L_ankle_roll` | 2.55% | 1.16% | 1.91% | 2.43% |
| `R_ankle_roll` | 2.04% | 1.52%, worst 3.54× | 2.10%, worst 3.85× | 1.48% |
| **`waist_roll`** | 0.07% | 0.30%, worst 2.32× | 0.45%, worst 2.37× | **4.83%, worst 4.58×** |
| `L_knee` / `R_knee` | 0.31% / 0.23% | 0.49% / 0.39% | 0.14% / 0.07% | 0.41% / 0.29% |

**Commanding a `q_target` beyond the joint's physical range is a routine, expected
occurrence under every policy checkpoint tested** — the deploy binary evidently does
not clamp `q_target` to the joint's mechanical range before sending it (the physical
hardware's own hard stop, and the resulting saturated PD error, are presumably relied
on instead). This is consistent with an RL policy whose raw action head is unbounded
(no `tanh`/clip observed in the reconstructed values), clipped only by the deploy
binary's torque limit (`np.clip(tau, ±EFFORT_LIMIT)`), never by a position limit.

**Q2 — Is it unique to `waist_roll`?** No. In every session, **`ankle_pitch` (L and R)
and `waist_pitch` exceed their range far more often than `waist_roll`** — up to 13.7% of
samples in `aug11`, vs. `waist_roll`'s 0.07–4.83%. `ankle_roll` (L/R) sits in between.
Even `knee` and `elbow` occasionally exceed their range (worst overshoot up to 1.15 rad
on `R_knee` in `g1_run_0905`), though at far lower frequency (<0.5%) and the control
joints (hip_pitch, hip_yaw, shoulder, wrist) essentially never do.

### What this changes, and what it doesn't

This means the earlier framing — "`waist_roll` gets uniquely extreme commands" — was
**too narrow**. Out-of-range `q_target` is a **routine feature of the whole "2× 5020"
tier (and occasionally beyond it), under every policy checkpoint checked**, not a
`waist_roll`-specific or `062k`-specific pathology.

**What §6–§11's aliasing/chatter finding is still about, and remains unaffected by this
correction:** those sections never depended on the (wrong) action-index computation —
they compare the logged `q`/`dq` directly against a position-derived finite difference,
using only `q_target`'s implied *position error* (`e`), not the raw action value. The
real finding stands: `ankle_pitch`/`ankle_roll`/`waist_pitch` **also** occasionally
receive out-of-range commands (this section), but only `waist_roll` turns those (routine,
shared) out-of-range commands into the severe, sign-inverting `dq` aliasing signature
seen in §6–§11. **The distinguishing factor is not "does this joint get an extreme
command" (many do, routinely) — it's "what does this joint's mechanism do in response,"**
which is exactly the differential fragility documented in §9–§10 (`waist_roll` and its
"2× 5020" siblings vs. the robust single-motor control joints).

## 13. Conclusion

1. ✅ **Not a hardware fault.** Zero `motor_error.csv` flags, normal
   `motor_temperature.csv`, in every run checked.
2. ✅ **Not a `sonic_no_vr_ll_062k`-specific policy bug.** The identical aliasing
   signature (window corr dropping to 0.2–0.4, one window fully negative) appears in
   `aug11`, recorded under `policy/low_latency`, before `062k` existed (§8). Separately,
   §12 shows **out-of-range `q_target` commands themselves** are routine under every
   policy checkpoint tested, not a `062k`-specific defect.
3. ✅ **Not confined to one robot/session.** Present at graded severity in every
   session tested — `aug11`, `g1_run_0905`, `g1_run_0908`, `g1_run_0914` (§7, §10).
4. ✅ **Specific to how `waist_roll`'s mechanism responds, not to receiving extreme
   commands per se.** §12 shows `ankle_pitch`/`waist_pitch` receive out-of-range
   `q_target` *more often* than `waist_roll` does, under every policy tested — but only
   `waist_roll` (and, less severely, its "2× 5020" siblings) turns that routine
   overshoot into the severe, sign-inverting `dq` aliasing/chatter signature. The
   well-conditioned single-motor control joint (`L_hip_pitch`) stays robust (>0.83)
   across all 25+ runs regardless of run conditions (§9, §10).
5. ✅ **Consistent with, and independently corroborating, Phase B's own finding** that
   `waist_roll`/`ankle_roll` were 16–27× worse-conditioned than the pitch-axis siblings
   in the original `Kp_eff`/`Kd_eff` regression (`phaseB_actuator.md` §3c) — a
   completely separate analysis method flagging the same joint family as fragile.
6. ⚠️ **The specific trigger of `run8` and `run20` is not yet isolated.** Both runs show
   `waist_roll`'s raw policy-action channel spiking to similar magnitude (§12), so a
   policy-side spike is present in both — but that alone doesn't explain why these two
   runs (among many with presumably similar policy behavior) are the ones that tip
   `waist_roll` into severe, sign-inverted chatter while other runs with elevated
   action values (§10's single-dip joints) do not. The differential fragility of the
   joint's mechanism, not the presence of a large command, remains the best-supported
   explanation for *why waist_roll specifically fails*; *why these two runs specifically
   trigger it* is still open.

**Most likely physical mechanism** (consistent with, not separately proven beyond, the
evidence above): `waist_roll`'s two-motor "2× 5020" drive shares torque through a
mechanism with some backlash/compliance. Combined with its comparatively low commanded
stiffness (`Kp=28.5` vs. `99` for hip/knee), whenever the joint's direction of motion
reverses under load it can briefly chatter within the backlash gap before the
drivetrain re-engages. That chatter runs faster than the 50 Hz control/logging loop can
resolve, producing the aliased, sign-inverted `dq_logged` vs. `dq_fd` signature — worse
when the motion is more dynamic/reversal-heavy, occasionally severe, but not a discrete
fault.

### What would make this airtight (open, not yet done)

Log-only analysis, however thorough, cannot fully substitute for:
- a **physical backlash/play measurement** on a `waist_roll` joint, or
- a **higher-rate (>50 Hz) log** of the same joint, to directly observe the chatter
  frequency instead of inferring it from aliasing.

### Practical implication

If `waist_roll` tracking quality feeds into any eval framework (e.g.
`sim2real/eval_clip_manifest.md`), it should be treated as **inherently
motion-dependent noise**, not a fixed per-robot calibration constant — any per-clip
metric involving `waist_roll` needs multiple reps to separate "this clip excites the
chatter mode" from "this robot's `waist_roll` is miscalibrated."

---

## Reproducing this analysis

```bash
# Kp_eff/Kd_eff refit (§1), reused from phaseB_actuator.md's recipe:
.venv_sim/bin/python model_eval/sim2real_phaseB_refit.py g1_run_0905 g1_run_0908 g1_run_0914

# tau_est vs nominal tau_sim, all 29 joints, per session (§2):
.venv_sim/bin/python model_eval/sim2real_phaseB_tau_compare.py g1_run_0905 g1_run_0908 g1_run_0914

# Per-run breakdown, find outlier runs within a session (§3):
.venv_sim/bin/python model_eval/sim2real_phaseB_per_run.py g1_run_0914 --joint waist_roll

# dq_logged vs dq_fd aliasing diagnostic (§6-§11):
.venv_sim/bin/python model_eval/sim2real_phaseE_dq_aliasing.py \
    --sessions aug11 g1_run_0905 g1_run_0908 g1_run_0914          # per-session table, all 29 joints
.venv_sim/bin/python model_eval/sim2real_phaseE_dq_aliasing.py \
    --sessions g1_run_0914 --per-run                              # per-run breakdown, one joint
.venv_sim/bin/python model_eval/sim2real_phaseE_dq_aliasing.py \
    --sessions g1_run_0914 --split-bad-runs run8 run20             # good vs bad run split (§11)

# out-of-range q_target check, all 29 joints, all sessions/policies (§12):
.venv_sim/bin/python model_eval/sim2real_phaseE_qtarget_overshoot.py \
    --sessions aug11 g1_run_0905 g1_run_0908 g1_run_0914
```
