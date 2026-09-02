# Phase B — Actuator Identification (session `aug11`)

**Question:** does the real motor do what the simulated one does?

**Short answer:** at the **torque** level, yes — remarkably so (R² > 0.999). The gap is not in
torque delivery. It is in **three things the MuJoCo model omits entirely**: joint friction,
rotor armature, and the ankle's parallel linkage.

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

## 4. Conclusions

1. ✅ **Torque delivery is not the gap.** R² > 0.999, gain ≈ 0.99. The real motor does what a
   commanded PD torque source should do. No actuator network required.
2. ⚠️ **Effective gains deviate systematically** on ankle pitch/roll and waist pitch (+2 to
   +10 %), matching the worst-tracking joints from Phase A.
3. ❌ **Per-joint mechanical ID is not identifiable** from closed-loop teleop logs (R² = 0.026,
   negative inertias). Needs dedicated excitation.
4. ⭐ **The sim model omits armature, friction and the ankle parallel linkage** — all three are
   verified absent from `g1_29dof.xml`, and all three are the standard sim2real culprits.

### Recommended fixes, in value order

| # | Fix | Effort | Basis |
|---|---|---|---|
| 1 | Add `armature` per joint from `policy_parameters.hpp` | trivial (XML attr) | values already exist |
| 2 | Add `frictionloss` per joint | small | needs §5 experiment to set values |
| 3 | Model the ankle A/B linkage (`<equality>`), or apply the measured +8–10 % gain correction | medium | §3.3 |

### 5. To identify friction/inertia properly

Robot supported (feet off ground), per joint:
- slow triangle-wave sweeps at several amplitudes → Coulomb + viscous friction from the
  torque–velocity hysteresis loop;
- chirp 0.5–10 Hz → effective inertia from the torque/acceleration transfer function;
- log at ≥200 Hz if possible (50 Hz makes `q̈` noisy).

This is the Tan et al. system-identification step, and it cannot be replaced by post-hoc
analysis of teleop logs.
