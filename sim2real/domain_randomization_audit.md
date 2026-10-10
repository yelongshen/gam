# Domain randomization audit — measured against real-robot data

Compares the training-time domain randomization in
`GR00T-WholeBodyControl/gear_sonic/config/manager_env/events/tracking/level0_4.yaml`
against distributions measured on **47 clean real-robot runs** (228 k frames,
~4500 s) from `~/g1_robot_data`.

Measurement scripts: `gam/sim2real/verify_pd_law.py`,
`gam/model_eval/rma_feasibility_check.py`. Excluded runs are listed in
`gam/sim2real/robot_log_data_quality.md`.

---

## 0. Headline

**The single largest measured real-world variation — effective actuator gain —
is not randomized at all.** Everything currently randomized (mass, COM, contact
friction, pushes) is plausible but none of it was set from data, and one term
(`randomize_rigid_body_mass`) is silently overridden to a much wider range than
its own default.

---

## 1. What is randomized today

From `tracking/level0_4.yaml` plus the terms it composes:

| term | mode | range | set from data? |
|---|---|---|---|
| `physics_material` (contact friction) | startup | static 0.3–1.6, dynamic 0.3–1.2, restitution 0–0.5 | no |
| `add_joint_default_pos` | startup | ±0.01 rad | no |
| `base_com` (torso COM) | startup | x ±0.025, y ±0.05, z ±0.05 m | no |
| `randomize_rigid_body_mass` | **startup** (overridden) | **0.8–2.5× scale**, `wrist_yaw` + `torso_link` only | no |
| `push_robot` | interval 4–6 s | vel ±0.5 m/s, ang ±0.52–0.78 rad/s | no |

**NOT randomized:** actuator stiffness/damping, joint friction/armature,
effort limits, control latency, IMU noise.

> Note the override: `terms/randomize_rigid_body_mass.yaml` defines
> `mode: reset`, `body_names: ".*"`, `0.8–1.2`. The composition file replaces
> this with `mode: startup`, only wrists+torso, and **0.8–2.5**. A 2.5× wrist
> mass is a big claim; it is not obviously wrong (payload in the hands), but it
> should be stated deliberately rather than buried in an override.

---

## 2. What the real robot actually varies by

### 2.1 Effective actuator gain — the missing term

Defined as `tau_est / tau_PD` per joint per run, where
`tau_PD = kp(q_target − q) − kd·dq`. The PD law itself is verified at
corr 0.998, so this ratio isolates how much torque the motor really delivered
relative to what the commanded gains imply.

Pooled over all non-waist joints and 47 runs:

| percentile | gain |
|---|---|
| p1 | **0.472** |
| p5 | 0.731 |
| p25 | 0.937 |
| **p50** | **0.986** |
| p75 | 1.025 |
| p95 | 1.112 |
| p99 | 1.294 |

Per joint, the spread is very uneven:

| joint | p5 | p50 | p95 | comment |
|---|---|---|---|---|
| `left_hip_pitch` | 0.963 | 0.991 | 1.018 | tight |
| `left_knee` | 0.934 | 0.995 | 1.032 | tight |
| `right_ankle_roll` | **0.623** | 0.881 | 0.996 | **consistently weak** |
| `left_wrist_roll` | **0.362** | **0.663** | 1.216 | **very wide** |
| `left_shoulder_yaw` | 0.798 | 1.033 | **1.313** | very wide |
| `right_wrist_roll` | 0.502 | 0.970 | 1.321 | very wide |

Two distinct phenomena, which should not be conflated:

- **Hips and knees are tight** (±5 %). Big actuators, heavily loaded, the model
  is good.
- **Wrists, shoulder-yaw and ankle-roll are wide** (0.36–1.32). These are the
  low-inertia / low-torque joints where stiction dominates, and they are
  exactly the joints earlier reports flagged as the worst reference-trackers
  (`left_shoulder_yaw`, `left_wrist_roll`, 17–20° error).

### 2.2 Unmodelled torque (friction + gearing loss)

`|tau_est − tau_PD|`, median per run:

| group | p5 | p50 | p95 | as % of effort limit |
|---|---|---|---|---|
| legs | 0.182 | 0.290 | 0.568 Nm | 0.3 % |
| arms | 0.025 | 0.127 | 0.188 Nm | 0.7 % |

In absolute Nm this is small, but relative to the **small** actuators it is
not: `left_ankle_pitch` loses 0.512 Nm, **2.0 %** of its 25 Nm limit, and the
ankles are the joints already measured at 4–7 % torque saturation during
walking.

---

## 3. Recommended changes

### 3.1 ADD actuator gain randomization — highest priority

This is the measured gap. `isaaclab.envs.mdp.events.randomize_actuator_gains`
exists and is unused.

```yaml
# gear_sonic/config/manager_env/events/terms/randomize_actuator_gains.yaml
randomize_actuator_gains:
  _target_: isaaclab.managers.EventTermCfg
  func: isaaclab.envs.mdp:randomize_actuator_gains
  mode: "startup"
  params:
    asset_cfg:
      _target_: isaaclab.managers.SceneEntityCfg
      name: "robot"
      joint_names: [".*"]
    stiffness_distribution_params: [0.85, 1.15]   # measured p5-p95 = 0.73-1.11
    damping_distribution_params: [0.85, 1.15]
    operation: "scale"
    distribution: "uniform"
```

**Why 0.85–1.15 and not the measured 0.73–1.11?** The measured p5 is dragged
down by the wrists and ankle-roll; hips/knees sit inside ±5 %. A single
symmetric range that covers the well-behaved majority is the safe first step.
Widening to per-joint ranges (wrists 0.4–1.3) is the follow-up, and should be
done only after confirming the wide wrist values are physical rather than an
artefact of those joints' tiny torque signal (see §5).

### 3.2 ADD joint friction/armature randomization

`randomize_joint_parameters` (also unused) covers `friction` and `armature`.
The measured unmodelled torque is 0.13 Nm (arms) to 0.29 Nm (legs) median.

```yaml
randomize_joint_parameters:
  _target_: isaaclab.managers.EventTermCfg
  func: isaaclab.envs.mdp:randomize_joint_parameters
  mode: "startup"
  params:
    asset_cfg:
      _target_: isaaclab.managers.SceneEntityCfg
      name: "robot"
      joint_names: [".*"]
    friction_distribution_params: [0.0, 0.3]    # Nm, covers measured p95 0.19-0.57
    armature_distribution_params: [0.9, 1.3]
    operation: "scale_and_add"   # check the exact signature before use
```

> **Do not** set armature from motor datasheets. Replacing the model's uniform
> 0.01 with per-motor values (0.0036 / 0.0102 / 0.0251) made one-step
> prediction *worse* (arm `dq` RMSE 0.439 → 0.663). The 0.01 is a calibrated
> effective value that already absorbs transmission compliance. Randomize
> *around* it; do not replace it.

### 3.3 REVIEW the mass override

`0.8–2.5×` on wrists and torso is asymmetric (−20 % / +150 %) and was not
derived from anything measurable here. Either justify it (hand payload?) or
move it back toward the term's own `0.8–1.2` default. Worth a deliberate
decision, not an override.

### 3.4 CONSIDER control latency

Not currently randomized, and not measurable from the logs either — the
`q_target` reconstruction matched at zero lag, so the logger records command
and state at the same tick. Real end-to-end delay (policy → DDS → motor →
sensor → policy) is invisible in this data. If the training stack supports an
action-delay term, a 0–2 tick (0–40 ms) randomization is standard practice and
cheap insurance.

---

## 4. Expected effect, and how to check it

These changes should be validated against the **existing** sim↔real
disagreement, not assumed to help:

> `online_eval_policy_comparison.md` §7: on `picoset_20260924` (sim) the LLAM
> family ranks best, but on hardware `ll_062k` leads. The two rankings do not
> agree.

If actuator-gain randomization is the missing ingredient, a policy retrained
with §3.1 should shrink that disagreement. The cheapest measurement is the
one-step metric already built:

```bash
.venv_sim/bin/python model_eval/sim2real_phaseC1b_onestep_qdq.py --base float
# baseline today: arms ddq corr 0.458, legs 0.485
```

A policy is not required to move that number — it measures the *simulator*, not
the policy. What should move is real-robot success rate and `mpjpe_l` on the
frozen clip manifest.

---

## 5. Caveats

1. **The gain metric is a ratio of two small numbers on the light joints.**
   `left_wrist_roll` has ~0.26 Nm RMS torque; a 0.36–1.33 gain spread there may
   be measurement noise rather than real variation. Trust the hips/knees
   numbers (large signal, tight spread) more than the wrist numbers.
2. **All 47 runs are the same robot, same floor, same payload.** So this
   quantifies *within-condition* variation (wear, temperature, pose-dependent
   loading), **not** the across-condition variation domain randomization is
   ultimately meant to cover. The ranges above are therefore a **lower bound**
   on what training should span.
3. **Waist roll/pitch excluded** from the pooled statistics: commanded targets
   exceed the ±0.52 rad URDF limit on up to 24 % of frames in some runs, so
   their apparent gain reflects an upstream clamp, not the actuator.
4. **No contact-force or base-state data**, so nothing here constrains the
   contact-related randomization (`physics_material`) — which remains
   unvalidated against reality.
