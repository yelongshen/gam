# Plan: Long-Context / World-Model Test-Time Adaptation for the G1 Policy

Motivated by the Phase C.2 finding that low-inertia "twist" axes (`wrist_roll`,
`shoulder_yaw`) are structurally most sensitive to sim2real dynamics mismatch, and by
**LocoFormer** (Liu, Pathak, Agarwal, CoRL 2025, arXiv:2509.23745) — which shows a
generalist policy can adapt to unseen/mismatched dynamics **at test time** via long
temporal context + massive domain randomization, rather than relying on a single
hand-calibrated simulator to be exactly right.

This plan covers: (1) how to measure current real-deployment robustness as a baseline,
(2) how to train a long-context/adaptive policy, (3) how to measure the new policy with
the same protocol for an apples-to-apples comparison.

---

## 1. Measure current policy performance/robustness on real deployment (baseline)

### 1.1 Reuse existing infrastructure — no new tooling needed for the baseline itself

This repo already has the building blocks; the gap is packaging them into one repeatable
**robustness battery** rather than one-off scripts:

| what | existing tool |
|---|---|
| Hardware tracking error (`q_target` vs `q`) | `sim2real_phaseA_latency.py` §5 methodology |
| Peak/mean torque, saturation | `motor_torque.csv` logs + `phaseA_latency.md` §5 "mean |τ|" |
| Sim-side one-step/torque fidelity (context, not robustness per se) | `sim2real_phaseC1_*` scripts |
| Simulation task success / non-fall rate, jerk, energy | `model_eval/EVAL_METRICS_DRAFT.md` §1B, `backup_sim_eval_metrics.py`, `backup_run_sim_eval.py` |
| Checkpoint-to-checkpoint comparison table (accel_dist etc.) | `model_eval/checkpoint_comparison.md` |
| Cross-model comparison notes | `notes_pico_evalset_3model_comparison.md`, `notes_amass_108clips_3model_comparison.md` |

### 1.2 Define the baseline robustness battery (what to measure, concretely)

See `sim2real/online_deployment_eval_plan.md` for the full systemic protocol
(sampled-SMPL-clip real-robot replay, two-flavor MPJPE, torque saturation rate,
task success/non-fall rate stratified by Test-Content/Test-Repetition, the
dedicated wrist-roll/shoulder-yaw stress subset, and shaking/swing/fall reward
terms). Summary of what it covers:

1. **Per-joint hardware tracking RMS** (`q_target - q`, degrees) across the full
   `aug11`-style Mode-2 session — reuse `phaseA_latency.md` methodology. This is the
   primary "is the robot doing what it's told" metric.
2. **Per-joint torque saturation rate** (`% of steps where |motor_torque| > 0.9 *
   actuatorfrcrange`) — flags joints working at their physical limit, a leading
   indicator of fragility under disturbance.
3. **Task success / non-fall rate** across a **diverse, fixed motion test set**
   (reuse the `EVAL_METRICS_DRAFT.md` Test-Content/Test-Repetition split: agility,
   dance/gesture, unstructured/crawl, basic locomotion) — run in sim first (cheap,
   safe), spot-check on hardware for the subset that's safe to run physically.
4. **Explicit stress cases targeting the low-inertia joints identified in Phase C.2**:
   design or select motion clips with fast wrist/forearm rotation and fast torso yaw,
   since these are exactly the DOFs shown to be most sensitive to dynamics mismatch.
   Track `L/R_wrist_roll`, `L/R_shoulder_yaw` tracking error specifically as a
   dedicated sub-metric, not just buried in an all-29-joint mean (a lesson directly
   from this session's Phase C.2 analysis — the all-joint mean masked large per-joint
   problems).
5. **Recovery behavior under induced disturbance** (if safe to test): light pushes,
   a held/dropped payload, or momentarily saturating one motor — record whether the
   policy recovers or falls, and how quickly tracking error returns to baseline. This
   is the most direct real-world analogue of "adaptation," and the metric the new
   policy is specifically trying to improve.

### 1.3 Output of step 1

A single **baseline report** (numbers + the stress-test videos/logs) with the same
structure as this plan's step 3 report, so the two are directly diffable. Suggested
location: `sim2real/baseline_robustness_report.md` (to be produced once the battery is
run — not yet executed in this session).

---

## 2. Plan: train a long-context / world-model-adaptive policy

**See `sim2real/longcontext_training_technical_plan.md` for the detailed, corrected
technical plan** — an earlier pass at this section incorrectly assumed the deployed
policy was a memoryless single-frame MLP (based on the generic
`observation_config_example.yaml` template). Inspecting the **actual deployed**
`sonic_pretrained/observation_config.yaml` shows the decoder already consumes 10
frames (0.2s @ 50Hz) of proprioceptive history (`his_body_joint_positions_10frame_step1`,
etc.) plus a 64D `token_state` from a multi-frame-aware encoder. The real gaps are:
(1) the history window is short and has no explicit tracking-error signal, and
(2) there is no cross-episode memory (the actual LocoFormer-equivalent gap). The
linked document lays out a corrected 3-phase plan: widen/add explicit history (low
risk) → cross-episode persistence (moderate) → procedural dynamics randomization
(heavy, full LocoFormer recipe).

### 2.1 Two candidate approaches (not mutually exclusive) — summary

**(a) Long-context in-context adaptation (LocoFormer-style)**
- Extend the policy's temporal context to **span multiple episodes** — the existing
  `his_*_10frame_step1` window and `token_state` both reset per episode today; this is
  the concrete, corrected gap (see `longcontext_training_technical_plan.md` §0.4).
- Train with **aggressive domain randomization** over exactly the parameters this
  session's analysis flagged as uncertain/impactful: `dof_damping` and
  `dof_frictionloss` on ankle-pitch/waist-pitch (randomize around the fitted optimal
  point, e.g. `damping ~ U(0.3, 1.8)` per `sim2real/optimal_calibration.md` §3), and
  **especially** armature/damping/friction on the low-inertia wrist-roll/shoulder-yaw
  axes identified in this session's Phase C.2 root-cause analysis, since those are
  structurally the most sensitive to mismatch and the least explained by our current
  calibration (R² ≈ 10-15% at best).
- The policy should see enough within-episode variation (procedurally varied dynamics,
  per LocoFormer's "procedurally generated robots") that it learns to **infer the
  current dynamics from recent tracking error/torque feedback**, rather than assuming
  a fixed model.

**(b) Explicit world-model / system-ID head (auxiliary prediction task)**
- Add an auxiliary head that predicts next-step `q`/`dq` (or torque) from current
  state+action, trained jointly with the policy (a lightweight learned residual on top
  of the analytic MuJoCo dynamics, rather than replacing it).
- Feed this head's own recent prediction error back into the policy's observation
  (a simple, cheap "am I surprised by the dynamics right now" signal) — this gives an
  explicit, interpretable adaptation signal rather than relying purely on an implicit
  long-context transformer to discover it.
- This is more directly inspired by classical system identification + adaptive control
  than LocoFormer's fully implicit approach, and may be cheaper to validate/debug given
  this project's existing Phase B/C.1 tooling (the auxiliary head's prediction error
  can be directly compared against `tau_sim_total` residuals already characterized).

**Recommendation:** start with (a) since it's the empirically validated recipe
(LocoFormer, CoRL 2025) and reuses existing large-scale RL infra (IsaacLab `g1.py`);
treat (b) as a stretch goal / ablation to understand *why* (a) works, using this
session's existing residual/R² analysis tooling as the diagnostic.

### 2.2 Concrete training-pipeline changes needed (see technical plan for full detail)

1. **Context length**: widen the existing `his_*_10frame_step1` observations (e.g. to
   `50frame_step1`) and add an explicit tracking-error history stream — a
   config/export-level change, not a new architecture, since the multi-frame
   observation mechanism already exists.
2. **Domain randomization ranges**: extend IsaacLab's `ImplicitActuatorCfg` (already
   holding the armature values aligned in this session) to randomize
   `damping`/`friction` per-episode within a *training* range informed by this
   session's fitted values + generous margin (since our own fit only explains a
   fraction of the real residual — the randomization range should be wider than just
   ± the fitted uncertainty).
3. **Procedural dynamics variation** (LocoFormer's "procedurally generated robots"):
   at minimum, randomize per-joint armature/damping/friction/effort-limit within
   plausible ranges every episode; more ambitiously, vary link masses/inertias
   slightly to simulate manufacturing variance and wear.
4. **Curriculum**: start training with less aggressive randomization then anneal to
   the full range, standard practice to avoid destabilizing early RL training.
5. **Episode-boundary-crossing context**: per LocoFormer's key architectural claim,
   make sure context is *not* reset at episode boundaries during training, so the
   policy can learn cross-episode adaptation (e.g. "I fell last episode because of X,
   adjust this episode").

### 2.3 What NOT to change yet

- **Do not** hand-bake the `sim2real/optimal_calibration.md` point-estimate values into
  a single fixed training config as the *only* dynamics — that reintroduces the exact
  single-simulator brittleness this plan is trying to move away from. Use them as the
  *center* of a randomization range, not a fixed target.
- **Do not** retrain against the currently best-known-imperfect friction/damping model
  without the domain randomization above — per `optimal_calibration.md` §3's existing
  recommendation, a naive retrain risks overfitting to a poorly-identified physical
  parameter.

---

## 3. Measure the new policy using the same protocol as step 1

1. **Re-run the exact same robustness battery from §1.2** on the new long-context
   policy — same motion test set, same hardware tracking/torque/success-rate metrics,
   same explicit wrist-roll/shoulder-yaw stress sub-metrics, same disturbance-recovery
   tests.
2. **Primary comparison metrics** (old vs new policy, sim first then hardware):
   - All-29-joint and per-joint hardware tracking RMS (should improve or at least not
     regress on the previously-worst joints: ankle pitch, waist pitch, wrist roll,
     shoulder yaw).
   - Task success / non-fall rate on the OOD (Test-Content) split — this is the
     headline metric LocoFormer reports gains on ("robust control even with large
     disturbances").
   - Disturbance-recovery time/success — the metric most directly targeted by
     long-context adaptation; expect the clearest improvement here specifically,
     since teacher-forced/one-step metrics (Phase C.1-style) cannot even measure this
     property (see `sim2real/phaseC2_discussion.md`).
3. **Regression checks before hardware deployment**: run the new policy through the
   same Phase C.1/C.2-style sim analysis used in this session (one-step prediction,
   closed-loop rollout) as a cheap pre-flight check, even though these metrics were
   originally built for *simulator* calibration, not policy evaluation — they can
   double as a sanity check that the new policy doesn't do something pathological in
   sim before ever touching hardware.
4. **Report format**: produce `sim2real/longcontext_policy_report.md` mirroring the
   baseline report's structure exactly, with a side-by-side diff table (same layout as
   the RMS comparison tables used throughout `phaseC1_damping_scan.md` /
   `optimal_calibration.md` in this session) so the improvement (or regression) is
   immediately legible per joint and per test category.

---

## Status

This is a **planning document only** — none of steps 1–3 have been executed yet. The
existing repo infrastructure (§1.1) is sufficient to begin step 1 immediately without
new tooling. Step 2 requires IsaacLab training-pipeline changes and a training budget
decision (out of scope for this session). Step 3 is a direct re-application of step 1's
protocol once a candidate policy exists.

## References

- LocoFormer: Generalist Locomotion via Long-context Adaptation. Liu, Pathak, Agarwal.
  CoRL 2025. arXiv:2509.23745. https://arxiv.org/abs/2509.23745
- `sim2real/optimal_calibration.md` — current best-fit damping/friction values and
  caveat on how much of the real residual they actually explain.
- `sim2real/phaseC2_discussion.md` — why teacher-forced (Phase C.1) metrics cannot
  measure adaptation/recovery behavior, motivating the closed-loop-style protocol used
  in step 3 here.
- `model_eval/EVAL_METRICS_DRAFT.md` — existing metrics framework this plan's
  robustness battery is built on top of.
