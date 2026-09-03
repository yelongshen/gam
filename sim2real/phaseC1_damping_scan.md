# Phase C.1 — Damping Coefficient Scan (session `aug11`)

**Question:** are the Phase-B-derived `dof_damping` calibration values (`{4: 1.19, 10: 1.09,
14: 0.98}`) actually the error-minimizing values for the Phase C.1 one-step-ahead prediction
test, or can a grid search find something better on this specific metric?

**Short answer:** a grid search finds numerically lower one-step RMS error at higher damping
values (`b=3.0` for both ankle-pitch joints), but this is an **empirical overfit to the one-step
metric**, not a better physical model. It also degrades several coupled joints and correlates
with increased MuJoCo numerical instability warnings. The Phase-B values remain the
physically-justified choice.

---

## 1. Where the Phase-B values (1.19 / 1.09 / 0.98) come from

They are **not** the result of a grid search — they come from `sim2real/phaseB_actuator.md` §3b,
a **free-fit regression** of real motor torque against joint velocity:

```
e   = q_target - q_real                      (position error)
tau_est = Kp_fit * e + Kd_fit * dq + offset   (fit per joint from real hardware logs)
```

| joint | `Kd` fitted (real hardware) | nominal `Kd` (commanded, `g1_params.KDS`) | extra damping = fitted − nominal |
|---|---|---|---|
| `L_ankle_pitch` | 3.01 | 1.8144 | **1.1956 ≈ 1.19** |
| `R_ankle_pitch` | 2.91 | 1.8144 | **1.0956 ≈ 1.09** |
| `waist_pitch`   | 2.79 | 1.8144 | **0.9756 ≈ 0.98** |

Since both sim and the real controller command the *same* nominal `Kd`, this residual is
interpreted as **extra physical viscous friction / back-EMF on real hardware** that neither
MuJoCo nor IsaacLab models (`frictionloss = 0` in both). It is applied in
`sim2real_phaseC1_onestep_prediction.py:build_model()` via:

```python
for j, b in extra_damping.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, MJ_JOINT_NAMES[j])
    m.dof_damping[m.jnt_dofadr[jid]] = b
```

Caveat from Phase B: the ankle-pitch fit is well-conditioned (`cond` 1.3–1.5, low
`corr(e, dq)`); `waist_pitch` is noted as "less clear" (`cond` 4.1) — consistent with the near-
zero q1step sensitivity to `waist_pitch` damping found below.

## 2. Grid-search tool

`model_eval/sim2real_phaseC1_damping_scan.py` sweeps `dof_damping` for one or more joints
(others held fixed) and reports the one-step-ahead RMS error (`q1step`, degrees) per candidate:

```
.venv_sim/bin/python model_eval/sim2real_phaseC1_damping_scan.py \
    --joints 4 10 14 --lo 0.5 --hi 3.0 --steps 6 --duration 439.6
```

### Result: no interior minimum found in [0, 3.0]

| joint | b=0.5 | b=1.0 | b=1.5 | b=2.0 | b=2.5 | b=3.0 |
|---|---|---|---|---|---|---|
| `L_ankle_pitch` q1step(deg) | 10.37 | 8.51 | 7.17 | 6.19 | 5.44 | **4.90** |
| `R_ankle_pitch` q1step(deg) | 7.60 | 6.37 | 5.53 | 4.90 | 4.44 | **4.09** |
| `waist_pitch` q1step(deg)   | 1.604 | 1.599 | 1.603 | 1.602 | 1.606 | 1.601 (flat, insensitive) |

Error decreases **monotonically** for both ankle-pitch joints across the entire tested range —
the search hit the upper bound (`b=3.0`) without finding a turnaround. `waist_pitch` shows no
meaningful sensitivity to damping at all (~0.3% spread, noise-level).

Every candidate ≥ 0.5 also triggered MuJoCo `QACC`/`QVEL`/`QPOS` "unstable" warnings at various
DOFs (not necessarily the joint being scanned) — a sign the model may be operating outside a
well-conditioned numerical regime at these values.

## 3. Full 29-joint comparison: baseline vs Phase-B vs best-scan

Config `best_scan = {4: 3.0, 10: 3.0, 14: 1.0}` (best found in the scan) vs.
`phaseB = {4: 1.19, 10: 1.09, 14: 0.98}` vs. `baseline = {}` (b=0 everywhere), full 439.6 s
session, all 29 joints, one-step-ahead teacher-forced RMS in degrees:

| joint | baseline | phaseB | best_scan | phaseB Δ% | best_scan Δ% |
|---|---|---|---|---|---|
| L_hip_pitch | 7.0572 | 6.9652 | 6.9643 | -1.3% | -1.3% |
| L_hip_roll | 0.5250 | 0.5579 | 0.5776 | +6.3% | +10.0% |
| L_hip_yaw | 3.6017 | 3.6406 | 3.6598 | +1.1% | +1.6% |
| L_knee | 6.6192 | 6.5398 | 6.4855 | -1.2% | -2.0% |
| **L_ankle_pitch** | 12.9891 | 7.9439 | **4.8891** | **-38.8%** | **-62.4%** |
| L_ankle_roll | 3.7328 | 4.0318 | 4.1669 | +8.0% | +11.6% |
| R_hip_pitch | 7.8019 | 7.7174 | 7.7209 | -1.1% | -1.0% |
| R_hip_roll | 0.7476 | 0.7653 | 0.7896 | +2.4% | +5.6% |
| R_hip_yaw | 1.6593 | 1.6732 | 1.6607 | +0.8% | +0.1% |
| R_knee | 6.7117 | 6.6450 | 6.6473 | -1.0% | -1.0% |
| **R_ankle_pitch** | 9.3737 | 6.1896 | **4.0814** | **-34.0%** | **-56.5%** |
| R_ankle_roll | 2.3196 | 2.5345 | 2.6796 | +9.3% | +15.5% |
| waist_yaw | 0.3840 | 0.3813 | 0.3820 | -0.7% | -0.5% |
| waist_roll | 0.2822 | 0.2795 | 0.2776 | -1.0% | -1.6% |
| waist_pitch | 1.6077 | 1.5992 | 1.5980 | -0.5% | -0.6% |
| L_sho_pitch | 5.4602 | 5.3827 | 5.3665 | -1.4% | -1.7% |
| L_sho_roll | 1.2496 | 1.2084 | 1.1708 | -3.3% | -6.3% |
| L_sho_yaw | 1.5482 | 1.4986 | 1.4852 | -3.2% | -4.1% |
| L_elbow | 2.8287 | 2.7494 | 2.8150 | -2.8% | -0.5% |
| L_wri_roll | 4.1584 | 4.0519 | 3.9192 | -2.6% | -5.8% |
| L_wri_pitch | 1.4395 | 1.3892 | 0.9779 | -3.5% | -32.1% |
| L_wri_yaw | 2.7490 | 2.6922 | 2.6913 | -2.1% | -2.1% |
| R_sho_pitch | 6.0281 | 5.9443 | 5.9350 | -1.4% | -1.5% |
| R_sho_roll | 0.7894 | 0.7101 | 0.7217 | -10.0% | -8.6% |
| R_sho_yaw | 4.5622 | 4.5093 | 4.4853 | -1.2% | -1.7% |
| R_elbow | 2.3658 | 2.3393 | 2.4004 | -1.1% | +1.5% |
| R_wri_roll | 1.9857 | 1.8777 | 1.8124 | -5.4% | -8.7% |
| R_wri_pitch | 1.1650 | 1.0798 | 1.2367 | -7.3% | +6.2% |
| R_wri_yaw | 1.1683 | 1.3296 | 1.2474 | +13.8% | +6.8% |
| **ALL-29 MEAN** | **3.5487** | **3.2492** | **3.0636** | **-8.4%** | **-13.7%** |

### Observations

1. **Directly targeted joints improve much more under `best_scan`**: ankle-pitch joints drop
   -62.4%/-56.5% vs -38.8%/-34.0% for Phase B. `waist_pitch` is unaffected either way (~-0.5%),
   confirming it is not a useful calibration target for this metric.
2. **Coupling side-effects get worse with more damping**: joints kinematically adjacent to the
   ankle (`ankle_roll`, `hip_roll`) regress further under `best_scan` (e.g. `R_ankle_roll`
   +9.3%→+15.5%, `L_hip_roll` +6.3%→+10.0%).
3. **Unrelated arm/wrist joints swing erratically and inconsistently** between configs
   (`L_wri_pitch` -3.5%→-32.1%; `R_wri_pitch` -7.3%→+6.2%; `R_elbow` -1.1%→+1.5%). These joints
   have no direct kinematic link to ankle/waist damping, so large, non-monotonic swings are
   most plausibly numerical/coupling noise rather than a genuine physical effect — this is
   consistent with the recurring MuJoCo instability warnings seen during the scan at these
   damping magnitudes.

## 4. Conclusion

- The **Phase-B calibration (`1.19 / 1.09 / 0.98`) is the physically-grounded choice** — derived
  from an independent regression against real torque/velocity data, not tuned against this
  particular one-step metric.
- The **best-scan config (`b=3.0` on both ankles)** achieves a lower one-step RMS error
  (-13.7% vs -8.4% overall) but:
  - has no interior minimum in the tested range (may not even be a local optimum — the true
    minimum, if any, lies beyond `b=3.0`),
  - amplifies regressions on kinematically-coupled roll joints,
  - produces erratic swings on unrelated arm joints,
  - correlates with more frequent MuJoCo numerical instability warnings,
  - is **not validated against a closed-loop / longer-horizon rollout**, unlike the
    teacher-forced one-step setup which cannot reveal instability from compounding drift.
- **Recommendation:** keep the Phase-B values as the primary calibration. Treat the scan result
  as evidence that the one-step metric alone is not sufficient to select damping values safely;
  any further increase in ankle damping should be validated against a stability/rollout test
  before being adopted.

## 5. Cross-check: torque-domain metric (`tau_sim_total` vs real `motor_torque`)

The comparisons above all use the **position-domain** metric (`q1step`, one-step-ahead RMS
error in degrees). A second, independent metric was added in
`model_eval/sim2real_phaseC1_tau_comparison.py`: compare a **torque-domain** quantity against
the real robot's measured `motor_torque` log.

### What is compared

MuJoCo cleanly separates two torque-like quantities and neither is a "measurement" — both are
exactly known from the model:

- `qfrc_actuator` = the actuator command (`ctrl`, clipped) — **always exactly equal to the PD
  law's output**, completely unaffected by `dof_damping`/`frictionloss` (verified empirically:
  identical to 4 decimal places across `b=0, 1.19, 5.0`).
- `qfrc_passive` = `-dof_damping * qvel` (+ any `frictionloss` term) — the passive force implied
  by the model's damping/friction settings at the current state.

`tau_sim_total = qfrc_actuator + qfrc_passive` is used as the sim-side analogue of the real
robot's `motor_torque` (`tau_est`), which is a genuine hardware measurement that already
conflates command + real friction + back-EMF etc. (there is no such command/measurement split
inside MuJoCo — both terms above are deterministic and fully known from the model parameters).

### Results (full 439.6 s session, calibrated joints only)

| joint | config | RMS (N·m) | corr | gain (`tau_real ≈ gain·tau_sim`) |
|---|---|---|---|---|
| `L_ankle_pitch` | baseline (no fix) | 1.1166 | 0.9977 | 1.0562 |
| | **Phase B fix** | **1.0716** (-4.0%) | **0.9990** | 1.0602 |
| | best-scan fix | 1.4368 (+28.7%) | 0.9945 | 1.0606 |
| `R_ankle_pitch` | baseline (no fix) | 1.0797 | 0.9974 | 1.0640 |
| | **Phase B fix** | **1.0094** (-6.5%) | **0.9990** | 1.0659 |
| | best-scan fix | 1.3959 (+29.3%) | 0.9913 | 1.0621 |
| `waist_pitch` | baseline (no fix) | 0.8125 | 0.9996 | 1.0651 |
| | Phase B fix | 0.8103 (~flat) | 0.9998 | 1.0666 |
| | best-scan fix | 0.8107 (~flat) | 0.9998 | 1.0666 |

### Interpretation

- **Phase B fix improves both RMS and correlation** on both ankle-pitch joints in the
  torque domain, consistent with (and independently corroborating) its position-domain result.
- **best-scan fix (`b=3.0`) makes the torque-domain fit substantially worse** (~29% higher RMS,
  correlation drops) despite improving the position-domain (`q1step`) metric it was tuned
  against — direct evidence of overfitting to `q1step` at the expense of physical plausibility.
- **The `gain` (≈1.06, i.e. real torque ~6% larger than sim) barely moves under either fix.**
  `qfrc_passive`'s magnitude (a few N·m from `dof_damping·dq` at typical velocities) is too
  small relative to the torques involved (tens of N·m) to meaningfully close a 6% multiplicative
  gap. This means the Phase B `Kd`-residual (linear-in-velocity, `dof_damping`) captures part of
  the real discrepancy (RMS/correlation improve) but **not the full gain>1 anomaly** documented
  in `sim2real/phaseB_actuator.md` §3b — a Coulomb (`frictionloss`) term, an additive bias, or
  the previously-considered Kp/linkage effects likely account for the remainder.

### Which metric should be trusted more: `q1step` (position) or torque RMSE?

**Torque-domain RMSE/correlation is the more trustworthy metric for selecting damping values**,
for two reasons:

1. **It cross-validates.** The Phase B values were *derived* from a torque-domain regression
   (§1 above) — showing they *also* improve an independently-computed torque-domain metric here
   is evidence of a real, reproducible physical effect, not a metric-specific artifact.
   The best-scan values were tuned *against* `q1step` and only look good on `q1step` — testing
   them against the untouched torque-domain metric reveals they are worse there, unmasking the
   overfit.
2. **`q1step` (position) is one integration step removed from the actual quantity being
   calibrated.** `dof_damping` acts directly as a force/torque term; testing it directly in the
   torque domain is a more direct measurement of the parameter's fit than propagating it through
   one step of the full rigid-body dynamics and inertia matrix to see the resulting position
   error, which is a noisier, more indirect readout (and, per §2 above, does not even have an
   interior minimum in a sane damping range — a bad sign for using it as the sole objective).

**Practical guidance:** use `q1step` as a *sanity check* / secondary confirmation (a good
damping value should not make position prediction worse), but use the **torque-domain
RMSE/correlation/gain** as the primary objective when selecting or scanning `dof_damping` (and
any future `frictionloss`) values — it is closer to the physical quantity being fit and is far
less prone to being gamed by numerically-unstable but position-metric-favorable damping.

## 6. Reproduction

```bash
# main C.1 baseline-vs-phaseB comparison (position domain, q1step)
.venv_sim/bin/python model_eval/sim2real_phaseC1_onestep_prediction.py --duration 439.6

# per-joint damping scan (position domain, q1step)
.venv_sim/bin/python model_eval/sim2real_phaseC1_damping_scan.py \
    --joints 4 10 14 --lo 0.0 --hi 3.0 --steps 11 --duration 439.6

# torque-domain cross-check (tau_sim_total vs real motor_torque)
.venv_sim/bin/python model_eval/sim2real_phaseC1_tau_comparison.py --duration 439.6

# frictionloss scan (Coulomb term) on top of a fixed dof_damping
.venv_sim/bin/python model_eval/sim2real_phaseC1_friction_scan.py \
    --joints 4 10 14 --lo 0.0 --hi 3.0 --steps 7 --duration 439.6
```

## 7. Combined optimal config (damping + friction)

See `sim2real/optimal_calibration.md` for the final least-squares-optimal
`dof_damping`/`dof_frictionloss` combination (derived from this torque-domain analysis), the
full 29-joint comparison against baseline/Phase B, and a discussion of how (and whether) to
integrate it into the MuJoCo deploy-test model vs. the IsaacLab training environment.
