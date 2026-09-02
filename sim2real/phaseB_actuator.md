# Phase B — Actuator Identification (session `aug11`)

**Question:** does the real motor do what the simulated one does?

**Short answer:** at the **torque** level, yes — remarkably so (R² > 0.999). The gap is not in
torque delivery. It is in **what the MuJoCo model omits**: joint friction, rotor armature, the
ankle's parallel linkage, and a too-low ankle torque limit. Separately, the **damping** on the
six "2× 5020" joints is mis-specified (see §3b).

> **Revision note.** §3.3 originally attributed a +8–10 % ankle gain anomaly to the parallel
> linkage. Cross-checking against `dev_notes/g1_pd_gains_full_answer.md` and running the tests
> it implies **refuted** that: the anomaly tracks the assigned `Kp` group exactly (including
> waist roll/pitch, which are *not* parallel). See §3b.

**Data:** `g1_robot_data/g1_real_deploy_logs`, Mode-2 window, 21,977 steps / 439.6 s,
`policy/low_latency`.

---

## 0. What `motor_torque` actually is (checked first)

`g1_deploy_onnx_ref.cpp:2846` → `motor_torque[i] = unitree_joint_state[i].tau_est()`.

It is the **measured/estimated realised torque** from the motor, **not** the command. This
matters: Phase A's "PD law holds at corr = 1.000" is therefore a statement about **hardware**,
not a tautology.

Control architecture:

```
deploy binary  --(50 Hz, DDS)-->  q_target, dq_target, kp, kd, tau_ff
motor driver   --(internal, high rate)-->  tau = kp(q*-q) + kd(dq*-dq) + tau_ff
tau_est        <-- realised torque, logged at 50 Hz
```

The PD loop runs **inside the motor driver**, not in the policy binary.

---

## 1. B1 — Torque realisation is near-ideal

Fit per joint: `tau_est = gain · tau_cmd + offset`, where
`tau_cmd = Kp(q_target − q) − Kd·dq`.

| statistic | value |
|---|---|
| **R²** | median **0.99931**, min 0.96893 |
| **gain** | median **0.9909**, range **0.9708 – 1.0965** |
| residual RMS | 0.013 – 0.608 N·m |

So the real motor delivers the commanded PD torque to within ~1 % for most joints. **A learned
actuator network (Hwangbo et al.) is not needed here** — the classic actuator gap is largely
absent on this robot.

### But the deviation is structured, not noise

| joint | gain | Phase A tracking RMS |
|---|---|---|
| `R_ankle_pitch` | **1.0965** | **26.69°** |
| `L_ankle_pitch` | **1.0838** | **28.72°** |
| `waist_pitch` | **1.0821** | **22.72°** |
| `R_ankle_roll` | 1.0275 | — |
| `waist_roll` | 1.0199 | — |
| `L_ankle_roll` | 1.0159 | — |
| … | ~0.99 | — |
| `R_wri_roll` | **0.9708** | — |
| `L_wri_roll` | 0.9737 | — |

The joints with gain > 1 are **exactly** the worst-tracking joints from Phase A. Everything else
sits at ~0.99. This is a systematic effect, and §3 identifies the mechanism.

---

## 2. B2 — Per-joint rigid-body ID: **NOT identifiable from this data** ❌

Fitting `tau = I·q̈ + b·q̇ + f_c·sign(q̇) + c` per joint gives:

| symptom | value |
|---|---|
| median R² | **0.026** |
| inertia `I` | **negative on 6 joints** (physically impossible) |

**These numbers are reported only to be rejected.** The fit fails because:

1. The robot is a **coupled multibody** system — torque at a joint depends on the whole
   configuration; a per-joint lump cannot represent `M(q)q̈ + C(q,q̇)q̇ + G(q)`.
2. The data is **closed-loop teleop**, not designed excitation. Inputs are narrow-band and
   correlated across joints, so `I`, `b` and `f_c` are not separately identifiable.
3. `q̈` comes from double-differencing 50 Hz `dq` — poor SNR at the frequencies that matter.

**Proper identification needs dedicated open-loop excitation** (chirps / sinusoids per joint,
robot supported), which is a hardware experiment, not a log-analysis task.

---

## 3. The actual gap: three things the sim model omits

Inspecting `gear_sonic_deploy/g1/g1_29dof.xml`:

| feature | occurrences in sim model |
|---|---|
| `armature` | **0** |
| `frictionloss` | **0** |
| `damping` | **0** |
| `<equality>` / `<tendon>` | **0** |

A representative joint is bare:

```xml
<joint name="left_ankle_pitch_joint" pos="0 0 0" axis="0 1 0"
       range="-0.87267 0.5236" actuatorfrcrange="-50 50"/>
```

### 3.1 No rotor armature

`policy_parameters.hpp` defines armature per motor type
(`ARMATURE_5020 = 0.003610`, `7520_14 = 0.010178`, `7520_22 = 0.025102`, `4010 = 0.00425`) and
derives all `Kp`/`Kd` from them — but the **MuJoCo model carries no `armature` attribute**, so
the simulated joints have only link inertia. Reflected rotor inertia is missing.

### 3.2 No joint friction

`frictionloss = 0` everywhere: the simulated robot is **frictionless**. Tan et al. and Hwangbo
et al. both identify actuator friction as a dominant transfer error. B2 could not identify
friction reliably (§2), but its absence in sim is certain.

### 3.3 The ankle parallel linkage is not modelled ⭐

`robot_parameters.hpp:76` documents that the G1 ankle is a **coupled 2-DOF mechanism** with two
actuation modes:

```cpp
enum class Mode { PR = 0,  // series  Pitch/Roll
                  AB = 1 };// parallel A/B motors
```

and the joint enum gives the ankle joints **dual names** (`LeftAnklePitch == LeftAnkleB == 4`,
`LeftAnkleRoll == LeftAnkleA == 5`). The run used `mode_pr_ = Mode::PR`
(`g1_deploy_onnx_ref.cpp:2173`), so the driver maps joint-space pitch/roll commands onto the two
physical A/B motors **through a linkage Jacobian**.

The MuJoCo model has **no `<equality>` or `<tendon>` constraint** — ankle pitch and roll are two
independent revolute joints in series.

This is the most likely mechanism behind §1's anomaly: the linkage Jacobian rescales the
effective joint-space gain, which shows up as **+8–10 % on ankle pitch** and smaller offsets on
ankle roll — while direct-drive joints sit at ~0.99. `waist_pitch` (+8.2 %) is a candidate for
the same treatment and worth checking on the hardware.

> **Caveat.** This mechanism is *inferred* — consistent with the enum, the PR mode, and the fact
> that only ankle/waist deviate, but not proven from these logs. Confirming it requires either
> the linkage geometry or an A/B-mode hardware test.

---

## 3b. ⚠️ CORRECTION — the linkage hypothesis in §3.3 is REFUTED

After cross-checking against
`GR00T-WholeBodyControl/dev_notes/g1_pd_gains_full_answer.md`, which independently
confirms the gain table, the `dq_target = tau_ff = 0` assumption, the `tau_est()` source, and
the exact fitting recipe used here, the §3.3 explanation does **not** survive its own tests.

**Test 1 — saturation?** No. Only 0.55 % / 0.18 % / 0.09 % of ankle-pitch / ankle-pitch /
waist-pitch samples come within 5 % of the effort limit. (Though note `L_ankle_pitch` reaches
**72.2 N·m**, well above the ±50 `actuatorfrcrange` in `g1_29dof.xml` — the sim limit is too
low, a separate finding.)

**Test 2 — load-dependent?** No. `corr(gain, mean|tau|) = 0.53`, but the counterexample is
decisive: `L_hip_pitch` carries **12.41 N·m at gain 0.9956**, while `L_ankle_pitch` carries
**11.52 N·m at gain 1.0838**. Same load, opposite deviation.

**Test 3 — grouped by assigned `Kp`?** ✅ **Perfect separation.**

| assigned `Kp` | motor group | n | mean gain |
|---|---|---|---|
| 14.2506 | `5020` | 10 | 0.9841 |
| 16.7783 | `4010` | 4 | 0.9862 |
| **28.5012** | **`5020` × 2** | **6** | **1.0543** |
| 40.1792 | `7520_14` | 3 | 0.9887 |
| 99.0984 | `7520_22` | 6 | 0.9954 |

**All 6 joints assigned the "2× 5020" gain have gain > 1 (1.016–1.097); all 23 other joints
have gain < 1 (0.971–0.996).** No overlap.

That group is `{L/R ankle_pitch, L/R ankle_roll, waist_roll, waist_pitch}` — it includes
**waist roll/pitch, which are not a parallel A/B mechanism**. So the anomaly follows the
**gain assignment**, not the ankle linkage. §3.3 is withdrawn as the explanation.

### What the data implies instead

Re-fitting `tau = Kp_eff·(q_target − q) − Kd_eff·dq` with **both gains free**:

| joint | `Kp` nominal | `Kp` fitted | `Kd` nominal | `Kd` fitted | R² |
|---|---|---|---|---|---|
| `L_ankle_pitch` | 28.5012 | **30.12** | 1.8144 | **3.01** | 0.99722 |
| `R_ankle_pitch` | 28.5012 | **30.37** | 1.8144 | **2.91** | 0.99640 |
| `waist_pitch` | 28.5012 | **30.39** | 1.8144 | **2.79** | 0.99888 |
| `L_ankle_roll` | 28.5012 | 27.66 | 1.8144 | 0.90 | 0.99053 |
| `waist_roll` | 28.5012 | 27.96 | 1.8144 | 0.89 | 0.99037 |
| `L_hip_pitch` | 99.0984 | 98.39 | 6.3088 | 6.18 | 0.99962 |
| `L_sho_pitch` | 14.2506 | 14.12 | 0.9072 | 0.78 | 0.99732 |

Two distinct effects inside the "2×" group:

- **pitch joints** (ankle ×2, waist): `Kp` ≈ +6 %, and **`Kd` ≈ 1.6× nominal** (2.8–3.0 vs
  1.81). The stiffness error is modest; the *damping* error is large.
- **roll joints** (ankle ×2, waist): `Kp` ≈ −2 %, **`Kd` ≈ 0.5× nominal** (0.89–1.02 vs 1.81).

Directly-driven joints need no correction (`hip_pitch` 98.39 vs 99.10, `sho_pitch` 14.12 vs
14.25 — both within ~1 %), which is what makes the "2×" group stand out.

**Most likely cause:** the firmware's effective damping on the doubled-gain joints differs from
`Kd = 2ζ·armature·ω` with the 2× factor applied. Whether the 2× is applied to `Kp` only, or
differently to `Kd`, cannot be settled from these logs — it needs the firmware's gain handling
or a controlled hardware test. **The ankle parallel linkage remains a real sim/hardware
mismatch (§3.3) but is not the cause of the gain anomaly.**

## 4. Conclusions

1. ✅ **Torque delivery is not the gap.** R² > 0.999, gain ≈ 0.99. The real motor does what a
   commanded PD torque source should do. No actuator network required.
2. ⚠️ **The deviation follows the assigned gain, not the mechanism.** All 6 "2× 5020" joints
   (ankle pitch/roll, waist roll/pitch) have gain > 1; all 23 others < 1 — perfect separation
   (§3b). Free-fitting shows pitch joints need `Kd` ≈ **1.6×** nominal and roll joints ≈ **0.5×**,
   while `Kp` is within ~6 %. Direct-drive joints need no correction.
3. ❌ **Per-joint mechanical ID is not identifiable** from closed-loop teleop logs (R² = 0.026,
   negative inertias). Needs dedicated excitation.
4. ⭐ **The sim model omits armature, friction and the ankle parallel linkage** — all three are
   verified absent from `g1_29dof.xml`, and all three are the standard sim2real culprits.

### Recommended fixes, in value order

| # | Fix | Effort | Basis |
|---|---|---|---|
| 1 | Add `armature` per joint from `policy_parameters.hpp` | trivial (XML attr) | values already exist |
| 2 | Add `frictionloss` per joint | small | needs §5 experiment to set values |
| 3 | Fix `Kd` on the six "2× 5020" joints (pitch ≈ 1.6× nominal, roll ≈ 0.5×) | small | §3b |
| 4 | Raise ankle `actuatorfrcrange` — real `L_ankle_pitch` hits **72.2 N·m** vs ±50 in the XML | trivial | §3b |
| 5 | Model the ankle A/B linkage (`<equality>`) — a real mismatch, but NOT the gain cause | medium | §3.3, §3b |

### 5. To identify friction/inertia properly

Robot supported (feet off ground), per joint:
- slow triangle-wave sweeps at several amplitudes → Coulomb + viscous friction from the
  torque–velocity hysteresis loop;
- chirp 0.5–10 Hz → effective inertia from the torque/acceleration transfer function;
- log at ≥200 Hz if possible (50 Hz makes `q̈` noisy).

This is the Tan et al. system-identification step, and it cannot be replaced by post-hoc
analysis of teleop logs.
