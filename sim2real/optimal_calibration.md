# Optimal sim2real torque calibration (Phase C.1 derived)

This document records the final **optimal** `dof_damping` / `dof_frictionloss` calibration
derived from the Phase C.1 torque-domain analysis, and how to integrate it into the
MuJoCo deploy-test model and (optionally) the IsaacLab training environment.

## 1. The optimal config

```python
dof_damping      = {4: 0.806, 10: 0.870, 14: 0.537}   # L_ankle_pitch, R_ankle_pitch, waist_pitch
dof_frictionloss = {14: 0.25}                          # waist_pitch only
```

(indices are `JOINT_NAMES`/`MJ_JOINT_NAMES` order: 4=`left_ankle_pitch_joint`,
10=`right_ankle_pitch_joint`, 14=`waist_pitch_joint`.)

### Derivation

1. **`dof_damping`** — least-squares-optimal linear-viscous coefficient, fit directly against
   the torque residual `residual = tau_real - tau_cmd` (real `motor_torque` minus the PD-law
   command), via
   `b* = -Σ(residual·dq) / Σ(dq²)`
   computed per joint over the full 439.6 s `aug11` session. This is close to, but slightly
   lower than, the original Phase B `Kd`-residual estimate (`1.19/1.09/0.98`,
   `sim2real/phaseB_actuator.md` §3b), which was derived independently via a full
   `tau_est = Kp·e + Kd·dq + offset` regression (with an offset term this analysis omits).
2. **`dof_frictionloss`** — scanned (`model_eval/sim2real_phaseC1_friction_scan.py`) on top of
   the optimal damping. Ankle-pitch joints: **no frictionloss helps** (RMS increases
   monotonically from `f=0`). `waist_pitch`: small improvement at `f≈0.25` (see caveat below on
   how MuJoCo's frictionloss is solved on this DOF).

### Full 29-joint torque RMS comparison (N·m, full 439.6 s session)

| joint | (1) `tau_cmd` vs `tau_est` (real, no sim) | (2) sim-real gap, baseline (`b=f=0`) | (3) sim-real gap, Phase B | (4) sim-real gap, **optimal** |
|---|---|---|---|---|
| L_ankle_pitch | 1.1166 | 1.1166 | 1.0716 | **1.0580** |
| R_ankle_pitch | 1.0797 | 1.0797 | 1.0094 | **1.0044** |
| waist_pitch | 0.8125 | 0.8125 | 0.8103 | **0.7374** |
| (all other 26 joints) | unchanged (≤0.0005 solver noise) across all 4 configs | | | |
| **ALL-29 MEAN** | 0.3042 | 0.3042 | 0.3001 | **0.2970** |

- `waist_pitch` sees its first *meaningful* improvement (-9.2% vs Phase B's -0.3%) once
  frictionloss is added — this joint's residual was poorly explained by viscous damping alone
  (R² ≈ 1.7%, `sim2real/phaseC1_damping_scan.md` §5), so the small Coulomb term helps where pure
  damping could not.
- Ankle joints improve modestly beyond Phase B (extra ~1-2%) since 0.806/0.870 sit closer to the
  true least-squares optimum for *this specific RMS metric* than Phase B's 1.19/1.09.

### Important caveat — how much this actually explains

Per the R² analysis in `sim2real/phaseC1_damping_scan.md` §5: even at the true optimum, a
linear-damping term only explains **~10-15% of the ankle-pitch residual variance**
(`corr(residual, dq)` only -0.34 to -0.42) and **~2% for waist_pitch** even before adding
friction. **This is the ceiling of what `dof_damping`/`dof_frictionloss` alone can fix** — most
of the sim-real torque gap on these joints remains unexplained, and is more likely attributable
to the Kp/ankle-linkage effects flagged (but not fully resolved) in `phaseB_actuator.md` §3.3/3b,
or a genuine bias/offset term not modeled here.

## 2. Integration into MuJoCo deploy-test model

**Where:** `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml` — the same file
already updated to align `armature` with IsaacLab (see git history). Add per-joint
`damping`/`frictionloss` overrides on the specific `<joint>` elements (rather than the
`<default class="...">` blocks, since only 3 of ~15 joints per class need non-zero values):

```xml
<joint name="left_ankle_pitch_joint" ... class="ankle_motor" damping="0.806" />
<joint name="right_ankle_pitch_joint" ... class="ankle_motor" damping="0.870" />
<joint name="waist_pitch_joint" ... class="torso_motor" damping="0.537" frictionloss="0.25" />
```

This makes `build_model()`'s runtime patching (in `sim2real_phaseC1_onestep_prediction.py`)
unnecessary going forward for these 3 joints — the calibration becomes a permanent part of the
deploy-test model rather than an ad-hoc override applied at test time. Any *new* scan/tuning
work can continue to use the `extra_damping`/`extra_friction` override mechanism for
experimentation before "graduating" a new value into the XML.

**Scope of impact:** this MuJoCo model is used for deploy-side sim testing / evaluation
(`gear_sonic_deploy`, `model_eval/sim2real_phase*` scripts) — **not** for policy training.
Updating it improves the fidelity of any sim-based evaluation of the *already-trained* policy,
including future closed-loop rollout tests (Phase C.2, see `sim2real/phaseC2_discussion.md`).

## 3. Integration into IsaacLab training environment

**Where:** `g1.py` (IsaacLab robot config), via `ImplicitActuatorCfg`, alongside the
`armature=...` settings noted in `phaseB_actuator.md` §"But §3 was wrong about the sim has no
armature". IsaacLab's actuator config supports per-joint `damping`/`friction` fields directly.

**Should this be done?** This is a bigger decision than the MuJoCo deploy-model update, because
it changes the **training** distribution, not just an evaluation tool:

- **Pro:** if real hardware genuinely has this extra ankle/waist damping+friction, training with
  it modeled should make the learned policy more robust to it at deployment (closes part of the
  sim2real gap at the *training* stage, not just post-hoc evaluation).
- **Con / risk:** the calibration was fit from **closed-loop teleop replay data** on 3 joints
  only, explains a small fraction of the actual torque residual (§1 caveat above), and has not
  been validated in closed-loop rollout (Phase C.2 is not yet implemented) — retraining a policy
  against an under-validated physical parameter change risks a wasted training run or, worse, a
  policy that overfits to an incorrect friction model.
- **Recommended path:** do **not** retrain yet. First (a) implement the Phase C.2 closed-loop
  rollout test to check the optimal config doesn't destabilize longer-horizon prediction, and
  (b) if resources allow, use dedicated open-loop excitation (chirps/sinusoids, robot supported)
  per `phaseB_actuator.md` §2's recommendation, to get a properly identifiable friction model
  (current data is not sufficiently exciting/well-conditioned for confident per-joint ID) before
  committing to a training-time change.
- **If/when retraining**: standard practice is to **domain-randomize** damping/friction within a
  reasonable range around the fitted best-estimate (e.g. `damping ~ U(0.5, 1.2)` for ankle
  pitch) rather than hardcoding the single fitted point, to keep the trained policy robust to
  the (large) residual uncertainty this analysis has shown still exists.

## 4. Integration into real-robot deployment code

**No changes needed.** The real robot's PD gains (`Kp`, `Kd`, `policy_parameters.hpp`) are
already fixed hardware/firmware values — this calibration only concerns the **simulation
model's** representation of what the real robot's dynamics already are. It does not change
anything about how the real robot is controlled.

## 5. Reproduction

```bash
# derive optimal damping (least-squares closed form) + scan optimal frictionloss
.venv_sim/bin/python model_eval/sim2real_phaseC1_friction_scan.py \
    --joints 14 --lo 0.0 --hi 1.0 --steps 5 --duration 439.6 \
    --damping "4:0.806,10:0.870,14:0.537"

# full 29-joint torque comparison across baseline / Phase B / optimal
# (see sim2real/phaseC1_damping_scan.md §6 for the individual-metric scripts;
#  the 4-column comparison combining tau_estimate_vs_measured + tau_with_friction
#  was run ad-hoc in this session - see chat history for the exact snippet)
```
