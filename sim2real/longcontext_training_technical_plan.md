# Technical Plan: Training a Long-Context / Adaptive G1 Policy

This expands `sim2real/longcontext_adaptation_plan.md` §2 into a concrete technical
plan, grounded in the **actual deployed architecture** — inspected directly from
`gear_sonic_deploy/policy/sonic_pretrained/observation_config.yaml` (the real exported
checkpoint config, `model_step_150000.pt`), **not** the generic
`observation_config_example.yaml` template (an earlier pass at this doc incorrectly
used the template and concluded the policy was memoryless — corrected below).

## 0. What the current policy actually is (corrected architecture understanding)

The deployed policy is an **encoder → 64D `token_state` → decoder** split, with the
decoder already receiving **multi-frame proprioceptive history**, not a single frame:

```yaml
# decoder input (994D total)
observations:
  - name: "token_state"                              # 64D, from encoder
  - name: "his_base_angular_velocity_10frame_step1"   # 10 frames x 3D  = 30D
  - name: "his_body_joint_positions_10frame_step1"    # 10 frames x 29D = 290D
  - name: "his_body_joint_velocities_10frame_step1"   # 10 frames x 29D = 290D
  - name: "his_last_actions_10frame_step1"            # 10 frames x 29D = 290D
  - name: "his_gravity_dir_10frame_step1"             # 10 frames x 3D  = 30D
```

```yaml
# encoder input (1751D), mode-dependent (g1 / teleop / smpl), producing token_state
encoder_observations:
  - motion_joint_positions_10frame_step5 / motion_joint_velocities_10frame_step5
  - motion_anchor_orientation (+ _10frame_step5)
  - motion_joint_positions_lowerbody_10frame_step5 / _velocities_lowerbody_10frame_step5
  - vr_3point_local_target / vr_3point_local_orn_target   (teleop mode)
  - smpl_joints_10frame_step1 / smpl_anchor_orientation_10frame_step1  (smpl mode)
  - motion_joint_positions_wrists_10frame_step1
```

**Corrected conclusions:**

1. **The decoder is NOT memoryless.** It already consumes 10 frames (0.2s @ 50Hz) of
   its own proprioceptive history (joint positions, velocities, actions, base angular
   velocity, gravity direction) every step. My earlier plan's "Phase 1: add history"
   was based on the wrong config file and is largely already done.
2. **The `his_*_Nframe_stepM` naming convention already generalizes** to arbitrary
   window length and stride — widening from `10frame_step1` (0.2s) to e.g.
   `50frame_step1` (1s) or `20frame_step5` (2s, coarser sampling) is a **config/export
   change**, not a new architecture, assuming the underlying feature-extraction code
   (wherever `his_*_Nframe_stepM` observations are computed) supports arbitrary N/M.
3. **No explicit tracking-error signal.** The decoder gets raw `body_joint_positions`
   history but never `(motion_joint_positions − body_joint_positions)` directly — the
   network must infer mismatch implicitly by relating proprioceptive history to the
   encoder's `token_state` (which itself is derived from `motion_*` reference data, not
   robot state). A `his_tracking_error_Nframe_stepM` stream is still a legitimate,
   low-risk, additive observation.
4. **No cross-episode memory.** Both the history buffers and the encoder's
   `token_state` reset at episode boundaries (standard episodic RL) — **this remains
   the real, unaddressed gap relative to LocoFormer**, whose key architectural claim
   is context that spans episode boundaries (learning from a fall in one episode to
   adjust behavior in the next). This is now the primary target, not "add history from
   scratch."
5. **The encoder/decoder split already mirrors what a "Option B" (recurrent/attention
   temporal encoder) proposal would build** — the encoder already summarizes
   multi-frame reference data into a compact token consumed by a fast decoder. The
   remaining architectural work is less about building a new temporal module and more
   about **(a) widening its window, (b) feeding it error/surprise signals, and
   (c) making it persist across episodes.**

## 1. Revised phased plan

### Phase 1 — Widen existing history + add explicit tracking-error signal (low risk)

- Extend `his_body_joint_positions_10frame_step1` etc. to a longer window (e.g.
  `50frame_step1` = 1s, or a dual-resolution scheme: fine recent history +
  coarser/strided older history, similar to the encoder's existing `_10frame_step5`
  pattern for motion reference).
- Add a new `his_tracking_error_Nframe_stepM` observation
  (`motion_joint_positions − body_joint_positions`, historical) directly, rather than
  relying on the network to reconstruct it implicitly from separate `token_state` and
  `his_body_joint_positions` streams — a more direct "surprise" signal for adaptation.
- Consider **DOF-selective width**: full extended history only for the joints flagged
  in Phase C.2 (ankle pitch, waist pitch, wrist roll, shoulder yaw), current-window
  history for the rest, to control the size of the 994D→larger decoder input.
- Retrain/fine-tune with the **same** dynamics/reward as today — isolate the effect of
  window length + error signal alone.
- **This is the cheapest, lowest-risk experiment and should be run first.**

### Phase 2 — Cross-episode persistence (the actual LocoFormer-equivalent gap)

- Modify the training loop so `token_state` (or a dedicated recurrent hidden state) is
  **not reset at episode boundaries** — carried forward the way LocoFormer's context
  spans episodes, enabling "learn from a fall this episode, adjust the next."
- Requires IsaacLab-side changes to episode/reset handling (a nontrivial training
  infra change, distinct from the observation-config-only change in Phase 1).
- Directly targets the **disturbance-recovery** metric in
  `sim2real/online_deployment_eval_plan.md` §4 — the property Phase 1 alone likely
  cannot produce, since within-episode history has no mechanism to remember a fall
  from a previous rollout.

### Phase 3 — Procedural dynamics randomization (full LocoFormer recipe)

- Randomize per-episode armature/damping/friction/effort-limit (ranges in §2 below),
  plus optional link mass/inertia variation, so Phase 2's cross-episode adaptation has
  something genuinely variable to adapt *to*. Phase 2 without this has little to adapt
  across, since the simulator would otherwise be static.
- This is the compute- and infra-heavy phase; gate on Phase 1/2 showing a clear signal
  first.

## 2. Domain randomization ranges (needed for Phase 3, informed by this session)

| parameter | current fitted point (`optimal_calibration.md`) | suggested training range |
|---|---|---|
| `L/R_ankle_pitch` `dof_damping` | 0.806 / 0.870 | `U(0.3, 1.8)` — fit only explains ~10-15% of residual variance, wide margin needed |
| `waist_pitch` `dof_damping` | 0.537 | `U(0.2, 1.2)` |
| `waist_pitch` `dof_frictionloss` | 0.25 | `U(0.0, 0.6)` |
| `wrist_roll` / `shoulder_yaw` armature, damping | not separately fitted this session | randomize generously (~±50% of nominal) given the low-inertia sensitivity finding — never targeted by the Phase B/C.1 calibration effort |
| link mass/inertia (stretch) | n/a | small (~±5-10%) per-episode perturbation |

## 3. Risks and mitigations

| risk | mitigation |
|---|---|
| Widening `his_*_Nframe_stepM` increases decoder input size and inference latency, risking the <50ms real-time budget | Benchmark actual C++ deploy latency (`g1_deploy_onnx_ref`) before committing; consider DOF-selective width (Phase 1) or strided/coarser older history rather than naive linear scaling |
| Cross-episode persistence (Phase 2) destabilizes training (violates standard episodic-RL assumptions) | Start with short cross-episode carry-over (e.g. persist across 2-3 episodes) before attempting full LocoFormer-scale context; closely monitor training stability metrics |
| Retraining with domain randomization (Phase 3) degrades in-distribution (ID) performance while chasing OOD robustness | Track Test-Repetition (ID) metrics as a regression guard throughout, per `online_deployment_eval_plan.md` §5's stratified reporting |
| New encoder/decoder ONNX graph shapes break the existing C++ deploy pipeline | Validate the export pipeline early with a shape-only dummy change before investing in a real training run |
| Overfitting to this session's specific fitted damping/friction point instead of the true (still poorly identified) residual | Use wide randomization ranges (§2) explicitly wider than the point estimate's confidence, per `optimal_calibration.md`'s existing caution |

## 4. Summary of corrected phased recommendation

```
Phase 1 (low risk): widen his_*_Nframe_stepM windows (esp. Phase-C.2-flagged DOFs) +
                     add explicit his_tracking_error_Nframe_stepM. Retrain with SAME
                     dynamics as today. -> cheapest test of "does more/explicit
                     history help", since basic history already exists.

Phase 2 (moderate):  cross-episode persistence of token_state/recurrent hidden state
                     -> the actual unaddressed LocoFormer-equivalent gap.

Phase 3 (heavy):     add procedural dynamics randomization (per §2) so Phase 2's
                     cross-episode adaptation has real variation to adapt to
                     -> approaches the full LocoFormer recipe.

Gate every phase transition on sim2real/online_deployment_eval_plan.md's protocol,
especially the wrist-roll/shoulder-yaw stress subset and disturbance-recovery metrics.
```

## 5. Predicted-next-state residual as an explicit input feature

A concrete refinement of Phase 1/2: have the policy output **both** an action and a
predicted next state, then feed the **prediction residual**
(`state_t − predicted_state_hat_t`, computed from the *previous* step's prediction —
see causality note below) back in as an input feature for the current step. This is
not a novel idea in isolation — it has direct precedent in both classical adaptive
control and modern deep-RL/robotics (§5.3), and is lower-risk than full autoregressive
"world model" planning (§5.2 explains why).

### 5.1 Causality (must get this right)

At each step the policy has only *just* produced a prediction for the *next* state —
it cannot use that prediction as an input to itself in the same step. The residual
available at time `t` is necessarily computed against the prediction made at `t-1`:

```
at t-1: policy(obs_{t-1}, residual_{t-1}) -> action_{t-1}, predicted_state_hat_t
at t:   residual_t = state_t - predicted_state_hat_t     # now available
        policy(obs_t, residual_t) -> action_t, predicted_state_hat_{t+1}
```

### 5.2 Why this is safe (unlike planning/imagined rollouts)

This design uses the prediction **only as a single-step observation feature**, never
rolled forward autoregressively to *plan* multiple steps ahead (Dreamer/MBRL-style
imagined rollouts). This matters directly here: Phase C.2 (this session) showed that
even our best-calibrated, physically-grounded MuJoCo model diverges badly under
autoregressive (no-reset) rollout, especially on the low-inertia wrist-roll/
shoulder-yaw axes (60°+ RMS by 60s). A *learned* forward model used for multi-step
planning would very likely be worse, and any policy trusting imagined rollouts on
exactly the axes we've already shown to be most sensitive would be actively misled.
Using the residual as a single-step feature (this section) sidesteps that failure mode
entirely — it never compounds.

### 5.3 Theoretical/intuitive justification for why the residual should help

Three complementary framings, all pointing the same direction:

**(a) Control-theoretic — the residual is (to first order) linear in the unknown
dynamics parameter.** Model true dynamics as depending on a hidden physical parameter
`z` (e.g. the real `dof_damping`/`frictionloss`/armature — exactly what
`optimal_calibration.md` fit offline): `x_{t+1} = f(x_t, a_t; z_true) + noise`. The
predictor only knows a nominal model `f(x_t, a_t; z0)`. The residual is
`r_t = f(x_{t-1}, a_{t-1}; z_true) - f(x_{t-1}, a_{t-1}; z0)`, which to first order
(Taylor expansion around `z0`) is `r_t ≈ J_z(x_{t-1}, a_{t-1}) · (z_true - z0)` — i.e.
the residual is approximately a **linear readout of exactly the unobservable parameter
mismatch**, scaled by a known sensitivity `J_z = ∂f/∂z`. This is the same mechanism
classical adaptive control (MRAC) and Neural-Fly's composite adaptation law exploit.
It also connects directly to this session's Phase C.1 damping scan: the measured
`corr(residual, dq)` there was an empirical instance of exactly this `J_z` sensitivity,
for one specific `z` (viscous damping) we already characterized.

**(b) Statistical/filtering — the residual is the Kalman-filter "innovation."** In a
Kalman filter, the innovation (observed minus predicted) is *by construction* the only
part of a new observation carrying information not already captured by the current
belief. Feeding `r_t` back in asks the policy to perform an implicit Bayesian update
of its internal estimate of the current dynamics regime — a plain RNN fed only raw
history *could* in principle learn an equivalent computation, but handing it the
innovation directly is more sample-efficient than making it rediscover
subtraction+filtering from scratch (why RMA/Neural-Fly-style explicit designs tend to
be more sample-efficient than fully implicit long-context RL like LocoFormer).

**(c) Practical/inductive-bias — it removes a hard subtraction+alignment burden from a
small, fast network.** Without the explicit residual, the network must internally
learn to compute the same subtraction, correctly time-aligned, from raw stacked
history — nontrivial for a small, low-latency decoder (the actual constraint here,
per the 994D `sonic_pretrained` decoder). Precomputing `r_t` outside the network is a
form of feature engineering matched to the network's limited capacity/latency budget.

### 5.4 Design recommendation

- **Predict a compact/targeted subset of state, not the full 58D `(q, dq)`.** Per
  RMA's design choice, focus prediction+residual specifically on the joints flagged in
  Phase C.2 (ankle pitch, waist pitch, wrist roll, shoulder yaw) — most other joints
  already track well, and including them would just add uninformative noise to the
  residual signal.
- **Add an explicit auxiliary prediction loss**, not just a downstream-input-only
  design (à la ICM) — relying purely on task reward to shape the predictor
  indirectly is a weaker, slower training signal than a direct supervised loss on the
  prediction itself.
- **Run as an ablation against RMA's compact-latent-`ẑ` design** (predict a low-dim
  physical-parameter-like latent instead of raw next-state) — cheaper, more
  interpretable, and directly comparable against this session's already-fitted
  `optimal_calibration.md` point estimates as a sanity check on whether the learned
  latent converges near the manually-fitted values.

### 5.5 References for this design

- **RMA: Rapid Motor Adaptation for Legged Robots** (Kumar, Fu, Pathak, Malik, RSS
  2021, arXiv:2107.04034) — closest match: trains an adaptation module to regress a
  compact latent environment-extrinsics vector from history, conditioning the policy.
- **Neural-Fly** (O'Connell, Shi, Shi, Azizzadenesheli, Anandkumar, Yue, Chung,
  *Science Robotics* 2022) — online adaptive drone control using exactly a
  prediction-error-driven composite adaptation law, with a stability (Lyapunov)
  guarantee — direct formal precedent for the residual-feedback mechanism.
- **Intrinsic Curiosity Module (ICM)** (Pathak, Agrawal, Efros, Darrell, ICML 2017) —
  trains a forward model with an explicit prediction loss and uses the prediction
  error as a signal (there, intrinsic reward) — precedent for "prediction error is
  itself a useful learned signal," and for adding a direct auxiliary loss rather than
  relying only on downstream task reward.
- **RL²: Fast Reinforcement Learning via Slow Reinforcement Learning** (Duan et al.,
  2016) — the fully implicit alternative (no explicit prediction/residual, just a
  recurrent policy shaped by task reward alone) — useful as the "null hypothesis"
  baseline to compare the explicit-residual design against.

## References

- `sim2real/longcontext_adaptation_plan.md` — parent plan (steps 1-3).
- `sim2real/online_deployment_eval_plan.md` — the measurement protocol every phase
  gate uses.
- `sim2real/optimal_calibration.md` — current fitted damping/friction point estimates
  (used as randomization centers, §2, and as an ablation sanity-check target, §5.4).
- `gear_sonic_deploy/policy/sonic_pretrained/observation_config.yaml` — the actual
  deployed checkpoint's observation schema (994D decoder / 1751D encoder / 64D token),
  showing the existing `his_*_10frame_step1` history mechanism this plan proposes
  widening, and the mode-dependent (`g1`/`teleop`/`smpl`) encoder this plan proposes
  extending toward cross-episode persistence.
- LocoFormer: Generalist Locomotion via Long-context Adaptation, CoRL 2025,
  arXiv:2509.23745.
- RMA: Rapid Motor Adaptation for Legged Robots, RSS 2021, arXiv:2107.04034.
- Neural-Fly, O'Connell et al., Science Robotics 2022.
- Curiosity-driven Exploration by Self-supervised Prediction (ICM), Pathak et al.,
  ICML 2017.
- RL²: Fast Reinforcement Learning via Slow Reinforcement Learning, Duan et al., 2016.
