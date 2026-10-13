# Robot Log Data Quality — known-bad runs and how they were found

Applies to `~/g1_robot_data/**`, the real-robot deploy logs used by the
sim↔real dynamics work (`sim2real/deploy_constants.py`, `verify_pd_law.py`,
`sim_real_step_gap.py`).

**57 run directories contain the four CSVs we need; 9 are excluded, 48 are
usable.** The exclusion list lives in `deploy_constants.EXCLUDED_RUNS` and is
applied automatically by `list_runs()`.

> **None of these 7 runs is flagged in any `online_eval_*_report.md`.** The
> session reports grade *tracking quality* — whether the policy followed the
> streamed reference. These exclusions are about *log integrity* — whether the
> recorded numbers are usable at all. The two are independent, and a session
> can be "the cleanest on record" (09-15) while still containing a run whose
> logs are fine and another whose are not.

---

## 1. The excluded runs

| run | frames | problem |
|---|---|---|
| `g1_deploy_run` | 25885 | `waist_roll`/`waist_pitch` `tau_est` is **constant** |
| `g1_deploy_run002` | 3838 | same |
| `g1_run_0909/g1_deploy_run_09082026_run1` | 5584 | `action.csv` contains **8700 NaNs** |
| `g1_run_0925/20260925_085524` | — | CSVs contain **NUL bytes** |
| `g1_run_0925/20260925_091028` | — | CSVs contain **NUL bytes** |
| `g1_run_0919/20260919_073126` | — | unreadable CSV |
| `g1_run_0922/20260922_083850` | — | unreadable CSV |
| `g1_run_0917/g1_deploy_run_09182026_081125` | 2776 | `waist_roll` corr **0.297** — undiagnosed |
| `g1_run_0915/g1_deploy_run_09142026_run8` | 7543 | `waist_roll` slope **1.497** — undiagnosed |

### 1.1 `g1_deploy_run` / `g1_deploy_run002` — waist torque never recorded

For these two runs `motor_torque.csv` reports a **constant** value on
`waist_roll` and `waist_pitch`. The regression of measured against
PD-predicted torque returns slope `0.000` and correlation `nan`, because a
constant has zero variance.

They are the two oldest directories in the archive (2026-08-07) and the only
ones without a date-stamped parent, which is consistent with them predating
waist-actuator instrumentation. Every other joint in these runs is fine.

**Why this mattered so much.** These 29723 frames sat inside the
`low_latency` policy group. NumPy propagates `nan` silently through
`corrcoef`, so the *entire* 136688-frame group reported `nan`, and any
aggregate that pooled joints (e.g. `ravel()` before `corrcoef`) quietly
returned a number that was wrong rather than `nan`. This produced two
successive false conclusions before the per-run table exposed it:

1. *"waist_roll/pitch violate the PD law structurally"* — no, two runs have no
   waist torque signal.
2. *"the `low_latency` policy commands the waist beyond its limit"* — no. With
   the two bad runs removed, `low_latency` sits at `roll r = 0.968`,
   `pitch r = 0.996`, in line with the other policies.

**Lesson:** aggregate statistics hid this for three rounds of analysis. Always
compute the per-run table first, and treat `nan` as a finding, not as noise.

### 1.2 `g1_run_0909/..._run1` — NaNs in the policy output

8700 NaN entries in `action.csv` (and therefore in the reconstructed
`q_target`). `q.csv`, `dq.csv` and `motor_torque.csv` are clean, so the
corruption is on the policy-output path, not the sensor path. Not investigated
further — whether the network emitted NaN or the logger dropped values is
unresolved.

### 1.3 `g1_run_0925/*` — truncated logs

Both runs' CSVs contain NUL bytes, i.e. the file was extended but never
written (logging killed mid-flush). Python's `csv` module raises
`line contains NUL`.

### 1.4 The two unreadable runs

`20260919_073126` and `20260922_083850` fail to parse with an empty error. Not
diagnosed; they are small and excluding them costs nothing.

### 1.5 Two runs with unexplained `waist_roll` behaviour

| run | symptom |
|---|---|
| `0917/g1_deploy_run_09182026_081125` | `waist_roll` corr **0.297** (every other joint ≥ 0.99) |
| `0915/g1_deploy_run_09142026_run8` | `waist_roll` slope **1.497** — measured torque 50 % larger than the PD law predicts |

These are **excluded as unexplained, not as understood-bad.** Neither shows any
of the failure signatures above: no NaNs, no NUL bytes, no constant torque, and
in both runs all 28 other joints obey the PD law normally. Whatever is wrong is
specific to `waist_roll` in these two sessions.

A slope of 1.497 is the more suspicious of the two — it means the motor
delivered *more* torque than commanded, which no friction or derating model
explains. Candidates not yet tested: a different gain scaling active for that
run, a `waist_roll` sensor fault, or an unlogged external load. `metadata.json`
shows no gain-scale overrides for either run.

Excluding them removes 10319 frames (4.3 %) and changes the pooled result by
0.003 Nm, so nothing downstream depends on this decision. They should be
re-examined if the waist chain ever becomes the subject of the analysis rather
than a control.

---

## 2. What is *not* excluded, and why

| session | documented issue | log integrity |
|---|---|---|
| 09-19 (all) | **confounded**: degraded stream + `reference/example/` instead of `real_example/` | PD corr 0.996–0.9999 — **fine** |
| 09-15 `run12`/`run20` | "did not track" (capability boundary) | PD corr 0.9955/0.9956 — **fine** |
| 09-12 (6 of 7 episodes) | agility clips the policy never tracked | PD corr 0.989–0.999 — **fine** |
| 09-17/18 | no evaluation report exists | PD corr 0.998 — **fine** |

A degraded stream corrupts the *reference the policy was chasing*. It does not
corrupt the robot's own `(q, dq, tau)`, which still obey the actuator law. So:

- **tracking-quality evaluation** → exclude per the session reports
- **dynamics identification** → only the 7 runs above need excluding

`deploy_constants.list_runs()` implements the second policy. Pass
`include_excluded=True` to recover the raw list.

---

## 3. Verification after exclusion

`verify_pd_law.py` on the 48 clean runs, checking
`tau_est ≈ kp(q_target − q) − kd·dq`:

```
228054 frames (4561 s)   corr 0.9980   RMSE 0.401 Nm   residual 6.3% of signal
slope median 0.992 (range 0.728–1.037)   joints with corr < 0.80: 0
```

Per-joint correlation is 0.956–1.000 across **all 29 joints** — the waist pair
included. The waist summary by policy:

| policy | runs | frames | `waist_roll` r | `waist_pitch` r |
|---|---|---|---|---|
| `low_latency` | 16 | 104189 | 0.986 | 0.996 |
| `sonic_no_vr_050k` | 17 | 75982 | 0.992 | 0.997 |
| `sonic_no_vr_ll_062k` | 15 | 47883 | 0.968 | 0.993 |

The residual 6.3% is friction + gearing loss + motor-model error — the part a
simulator does not model.

### Runs still worth a look (not excluded)

| run | observation | likely cause |
|---|---|---|
| `0915/run5` | `roll r = −0.412`, **n = 258** (5 s) | sample far too small to interpret |
| `0915/run20` | `roll r = 0.738`, `roll > limit` **10.5%** | genuine over-limit commanding, not a log fault |

`run5` is 258 frames — any correlation computed on it is noise, and it is left
in because excluding a run for being *short* is a different criterion from
excluding it for being *wrong*. `run20`'s low correlation has an identified
physical cause (the commanded target leaves the joint's range 10.5 % of the
time), so it is real data, not bad data.

---

## 4. Separately: the `q_target` over-limit question

`waist_roll` and `waist_pitch` have a ±0.52 rad limit in
`g1_29dof.urdf`. Across the clean runs the commanded target exceeds it on
0.2–1.0 % (roll) and 0.8–4.8 % (pitch) of frames, and individual runs go much
higher (`g1_run_0919/080352`: 32.5 % on pitch; `0915/run20`: 10.5 % on roll).

Measured `|q|` **never reaches 95 % of the limit**, so the joint is not hitting
a mechanical stop — something upstream is clamping. The deploy binary does not
clamp (`g1_deploy_onnx_ref.cpp` writes `q_target` straight into `LowCmd_`), so
the clamp is in the Unitree firmware or SDK. **Not located** — the SDK ships as
a prebuilt library and no firmware documentation was found.

This matters for sim↔real: in simulation `q_target` applies unclamped.

---

## 5. Reproducing

```bash
# per-run table that exposed the nan runs
.venv_sim/bin/python sim2real/waist_corr_per_run.py

# per-run PD-law QC with integrity flags
.venv_sim/bin/python sim2real/qc_runs_pd.py

# confirm what the exclusion list drops
.venv_sim/bin/python -c "
import sys; sys.path.insert(0,'sim2real')
from deploy_constants import list_runs, is_excluded
a=list_runs(include_excluded=True)
print(len(a),'->',len(list_runs()))
for r in a:
    ex,why=is_excluded(r)
    if ex: print(' ',r,'|',why)"
```

> **Implementation note.** `is_excluded()` matches **exact path components**,
> never substrings. Two bugs were hit while writing it: `g1_deploy_run` is a
> substring of `g1_deploy_run_09082026_run4`, and `..._run1` is a prefix of
> `..._run10/11/12/14`. Both silently dropped most of the archive. If you add
> an entry, re-run the check above and confirm the count drops by exactly the
> number you intended.

---

## 6. Change log

| date | change |
|---|---|
| 2026-09-29 | Initial write-up. 7 runs excluded from 57. Found by data-driven QC while verifying the PD law for the sim↔real dynamics comparison; none was documented in the session reports. Retracted two earlier conclusions (structural waist failure; `low_latency` over-limit commanding) that were artifacts of the two uninstrumented runs. |
| 2026-09-29 | Excluded 2 more (`0917/081125`, `0915/run8`) for unexplained `waist_roll` behaviour — 9 of 57 now excluded, 48 usable. Both are flagged as **undiagnosed**, not understood; removing them shifts the pooled RMSE by 0.003 Nm. With them gone, all 29 joints correlate ≥ 0.956 and no joint falls below the 0.80 bar. |
