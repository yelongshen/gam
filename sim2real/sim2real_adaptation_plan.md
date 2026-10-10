# Closing the sim2real gap: in-context adaptation and world modelling

A plan, grounded in what this repo's own measurements say — not a generic
literature summary. Every number quoted below was measured on
`~/g1_robot_data` with the scripts named alongside it.

---

## 0. TL;DR

The single most valuable action is **not** a new algorithm. It is
**logging the base state**. Three of the four analyses below are currently
impossible to run correctly, and the fourth (Phase C.1 one-step) has its leg
numbers contaminated, purely because base position / linear velocity / contact
force are never recorded. That is a ~50-line change to the deploy binary and it
unblocks everything else.

After that, the highest-leverage method is **explicit in-context adaptation
(RMA-style)** — and the architecture already has most of the pieces.

---

## 1. What we actually measured (the starting point)

| measurement | value | script |
|---|---|---|
| PD law `tau = kp(q*-q) - kd·dq` vs `tau_est` | corr **0.998**, residual **6.3 %** | `sim2real/verify_pd_law.py` |
| One-step `q` RMSE, arms | **0.246°** (real step 0.254°) | `model_eval/sim2real_phaseC1b_onestep_qdq.py --base float` |
| One-step `dq` RMSE, arms | **0.352 rad/s** (real step 0.183) | same |
| One-step `q` RMSE, legs | **1.403°** (real step 0.610°) | same |
| `ddq` corr sim vs real | **0.46 – 0.49** | same |
| Welded base vs floating+contact, leg `dq` RMSE | 6.73 → **1.52** (4.4×) | same |
| Known-bad runs excluded | **9 of 57** | `sim2real/robot_log_data_quality.md` |

Two conclusions follow directly:

1. **The actuator model is already right.** 6.3 % residual on the PD law means
   torque generation is not where the gap lives. Do not spend effort there.
2. **The gap is in the rigid-body + contact dynamics.** `ddq` correlation of
   ~0.47 means sim and real disagree about acceleration roughly half the time,
   even with teacher forcing and exact initial conditions.

A third, negative result worth remembering: replacing the XML's uniform
armature (0.01) with per-motor-type values **made one-step error worse**
(arm `dq` RMSE 0.439 → 0.663). The shipped 0.01 is a *calibrated effective*
value that already absorbs transmission compliance and friction. **Do not
"correct" calibrated sim parameters using datasheet numbers.**

---

## 2. Prerequisite: log the base state (blocking, ~1 day)

Currently logged: `base_quat`, `base_ang_vel`, `base_accel` (IMU), `q`, `dq`,
`action`, `motor_torque`.
**Not logged: base position, base linear velocity, contact/foot forces.**

`body_pos.csv` exists but is all-zero in 51/51 files — it is the *streamer's*
reference, not robot state.

Add to `state_logger.cpp`, from Unitree's `LowState`:

```cpp
// unitree_sdk2 LowState_ carries these already
ls->foot_force()            // 4x foot force sensors -> contact phase, GRF proxy
// plus whatever odometry/state-estimate the SDK exposes on this firmware
```

Why this is the top priority:

- **Legs cannot be validated without it.** Effective inertia at a stance ankle
  is 0.650 kg·m² (measured) vs 0.013 in a contact-free model — a **51×** gap
  that swamps every other effect.
- It converts the one-step test from "arms only" to "whole body".
- Foot force gives contact phase for free, which every method in §3–§5 wants.

If the firmware genuinely exposes no odometry, a **contact-based leg odometry**
estimator (standard for legged robots: when a foot is planted, integrate its
velocity backwards to get base velocity) is a few hundred lines and is a
well-understood, low-risk component.

---

## 3. In-context adaptation (RMA-style) — recommended first method

**Reference:** Kumar, Fu, Pathak, Malik, *RMA: Rapid Motor Adaptation for
Legged Robots*, RSS 2021, [arXiv:2107.04034](https://arxiv.org/abs/2107.04034).

RMA trains two pieces entirely in sim:

1. a **base policy** `a = π(s, z)` conditioned on a latent `z` that encodes the
   environment "extrinsics" (friction, payload, motor strength, terrain);
2. an **adaptation module** `z = φ(history of s, a)` that *infers* `z` from the
   recent state-action history — because on hardware the true extrinsics are
   unobservable.

At deploy time `φ` re-estimates `z` continuously, so the robot adapts within a
fraction of a second without any gradient step. That is in-context learning in
the precise sense: adaptation happens in the forward pass, from context.

### Why this fits here unusually well

The deployed observation is already the exact input RMA's adaptation module
wants (`policy/sonic_no_vr_llam_080k/observation_config.yaml`):

```yaml
his_body_joint_positions_10frame_step1     # q      x10   (290)
his_body_joint_velocities_10frame_step1    # dq     x10   (290)
his_last_actions_10frame_step1             # a      x10   (290)   <- autoregressive
his_base_angular_velocity_10frame_step1    # omega  x10
his_gravity_dir_10frame_step1              # g      x10
token_state                                # 64-dim encoder output
```

A **200 ms window of (q, dq, a)** — RMA uses ~0.5 s of the same three signals.
And there is already a 64-dim token channel feeding the decoder, already logged
per-step (`token_state.csv`, 64 columns).

So the change is **not architectural**. It is:

| step | work |
|---|---|
| 1 | Add domain randomisation over the parameters that matter *here*: joint friction, armature, link mass/COM, ground friction, latency, and **PD gain scale** (the deploy binary has `--motor-kp-scale`/`--motor-kd-scale`, so this axis is real). |
| 2 | Train the base policy with privileged `z` = the sampled randomisation vector. |
| 3 | Train `φ` by supervised regression from the 10-frame history to `z`. |
| 4 | Deploy: concatenate `φ`'s output alongside / into the existing token. |

### Validation gates (do not skip)

- `φ`'s inferred `z` must **correlate with a physically meaningful quantity on
  real data**. Test: run the 48 clean real runs through `φ` and check the
  inferred friction/gain terms are (a) stable within a run, (b) different
  across the three deployed policies' sessions if their gains differed.
  If `z` is noise, RMA reduces to a more expensive baseline.
- Improvement must show up in the **one-step `ddq` correlation** (currently
  0.46–0.49). That is the metric the gap actually lives in.

### Honest risk

RMA was demonstrated on a 12-DoF quadruped with a hand-designed reward. This is
a 29-DoF humanoid doing motion tracking from SMPL. The adaptation module has to
disambiguate extrinsics from a *much* richer action distribution. Expect `z`
identifiability to be the hard part, which is exactly why the first gate above
exists.

---

## 4. Residual dynamics learning — cheapest real win

Learn what the simulator gets wrong, as a correction term:

```
ddq_corrected = ddq_mujoco(q, dq, tau)  +  f_theta(q, dq, tau, contact)
```

`f_theta` is trained on real logs to minimise one-step prediction error — i.e.
directly on the `sim2real_phaseC1b_onestep_qdq.py` objective. Then train the
policy in the *corrected* simulator.

**Why it is attractive here:** the data already exists. 48 clean runs,
228 k frames, 4561 s, with `(q, dq, q_target, tau_est)` at 50 Hz and a verified
PD law. Nothing new needs collecting for the arms; legs need §2 first.

**Why it is not first:** a residual model learned on the current motion
distribution will not extrapolate to motions the policy has not performed yet,
and it can be gamed — the policy may learn to exploit the residual model's
errors. Use it as a *simulator improvement*, validated by the one-step metric,
not as a component of the control loop.

Concretely: if `f_theta` raises arm `ddq` correlation from 0.46 to >0.8 on
held-out runs, the corrected simulator is worth retraining in. If it does not,
the gap is not in a learnable static residual and §3/§5 are the answer.

---

## 5. World modelling / closed-loop rollout

`sim2real/phaseC2_discussion.md` already specifies this correctly, and
`model_eval/sim2real_phaseC2_closed_loop_rollout.py` exists. The distinction it
draws — teacher-forced one-step vs autoregressive rollout — is the right frame.

What to add, in order:

1. **Divergence horizon as the headline metric.** Not "RMSE at the end", but
   "how many milliseconds until `q_sim` and `q_real` differ by 5°". That number
   is interpretable, comparable across configs, and is what actually determines
   whether a learned model is usable for planning.
2. **Run it on the 48 clean runs**, not the single `aug11` session.
3. **Use it to test the damping candidates.** `phaseC1_damping_scan.md` found
   `{4: 3.0, 10: 3.0, 14: 1.0}` improves the *one-step* metric. One-step
   improvements from added damping are exactly the kind that can destabilise a
   rollout. This is the test that settles it.

A learned latent world model (Dreamer-style) is **premature**. The analytic
model's one-step `ddq` correlation is 0.47; a learned model has to beat that
before it is worth the complexity, and §4 is the cheaper way to find out
whether that is achievable.

---

## 6. Things that will *not* help (measured)

| idea | why not |
|---|---|
| Better actuator/PD model | already corr 0.998, residual 6.3 % |
| Per-motor armature from datasheets | measured **worse**: arm `dq` RMSE 0.439 → 0.663 |
| Friction compensation as the main fix | fitting `residual ~ a·sign(dq) + b·dq` gives **R² 0.02–0.05**; friction does not explain the residual |
| More real data of the same kind | 48 runs / 4561 s already exist; the blocker is *which signals*, not how many frames |

---

## 7. Recommended order

```
[1] Log base pos/vel + foot force                      ~1 day    BLOCKING
     └─ unblocks leg validation, contact phase, odometry

[2] Re-run one-step with real contact                  ~1 day
     └─ gives the first trustworthy whole-body ddq correlation
     └─ GATE: if leg ddq corr > 0.8, the model is fine and the gap is
              elsewhere (policy robustness, not dynamics)

[3] Residual dynamics f_theta on existing logs         ~1 week
     └─ GATE: held-out ddq corr > 0.8 ⇒ retrain policy in corrected sim

[4] RMA-style in-context adaptation                    ~3-4 weeks
     └─ GATE: inferred z must be stable per-run and vary across
              known-different conditions

[5] Closed-loop rollout / divergence horizon           ongoing
     └─ the acceptance test for [3] and [4], and for the damping candidates
```

Steps 2 and 3 are cheap and might make 4 unnecessary. Do them first.

---

## 8. Open questions

- **Is `z` identifiable from 200 ms?** RMA uses ~0.5 s on a 12-DoF robot. The
  existing observation is 10 frames at 50 Hz = 200 ms on a 29-DoF robot. May
  need lengthening, which changes the ONNX input shape and the deploy config.
- **Where is the `q_target` clamp?** Commanded waist targets exceed the ±0.52 rad
  URDF limit on up to 24 % of frames in some runs, yet measured `|q|` never
  reaches 95 % of the limit — something upstream clamps, and it is not in
  `g1_deploy_onnx_ref.cpp`. The SDK ships prebuilt. Until this is found, sim
  and real disagree about what a commanded target even *means*.
- **Does the sim robot stand at all under pure PD hold?** A static test
  collapsed from 0.752 m to 0.089 m in ~2 s
  (`model_eval/sim2real_standing_torque_check.py`). This is expected — the real
  system is closed-loop at 50 Hz and pure position hold is not a balance
  controller — but it means **any open-loop sim validation is invalid** and the
  policy must be in the loop.

---

## 9. References

- Kumar, Fu, Pathak, Malik. *RMA: Rapid Motor Adaptation for Legged Robots.*
  RSS 2021. [arXiv:2107.04034](https://arxiv.org/abs/2107.04034)
- Internal: `sim2real/phaseC2_discussion.md` (one-step vs world modelling),
  `sim2real/phaseC1_damping_scan.md` (damping calibration),
  `sim2real/phaseB_actuator.md` (actuator identification),
  `sim2real/robot_log_data_quality.md` (which runs are usable).
