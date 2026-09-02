# Sim2Real Gap Study — Plan

Plan for measuring (and then closing) the sim-to-real gap for the G1 whole-body
tracking policy, using the **paired human↔robot recordings** already on disk.

**Status:** planning. No phase started yet.
**Prerequisite reading:** `PICO_SMPL_STREAMING_DATASETS_NOTE.md` (dataset inventory,
pairing evidence, alignment pitfalls).

---

## 1. Why this is worth doing here

Most groups closing a sim2real gap do **not** have paired logs: they have a policy that
works in sim, and a robot that behaves differently, with no synchronized record of the
same command driving both. We do:

- the **same human SMPL command stream** (PICO capture),
- the **robot's full response** to it (`state_logger` CSV bundle),
- the **same checkpoint** (`policy/low_latency/`) that was also benchmarked in sim
  (`notes_pico_evalset_3model_comparison.md`),
- and the **policy's own input tensor** (`token_state`, 64-d) logged on the real robot.

That last item is the unusual one — it lets us split "the policy saw different things"
from "the policy saw the same thing and the body responded differently".

---

## 2. What the literature says to do

| # | Approach | Key reference | Take-away |
|---|---|---|---|
| 1 | **System ID + actuator model + latency** | Tan et al. 2018, *Sim-to-Real: Learning Agile Locomotion for Quadruped Robots* (arXiv:1804.10332) | Narrowed the gap by "improving the physics simulator … using system identification, developing an accurate actuator model, and simulating latency". For position-controlled robots the dominant error is **the actuator and the latency**, not rigid-body dynamics. |
| 2 | **Learned actuator network** | Hwangbo et al. 2019, *Learning agile and dynamic motor skills for legged robots*, Science Robotics 4(26) (arXiv:1901.08652) | When the actuator is too complex to model analytically, train a small net mapping joint-state history → realized torque **from real hardware logs**, and embed it in sim. |
| 3 | **Domain randomization** | Zhao et al. 2020, *Sim-to-Real Transfer in Deep RL for Robotics: a Survey*, IEEE SSCI (arXiv:2009.13303) | Randomize dynamics so the policy is robust to any single mismatch. Cheap, but **hides** the gap rather than measuring it, and costs performance. |
| 4 | **High-frequency control + _targeted_ randomization** | Haarnoja et al. 2023, *Learning Agile Soccer Skills for a Bipedal Robot*, Science Robotics (arXiv:2304.13653) | Closest morphology (20-DOF humanoid, zero-shot). Transfer came from "sufficiently high-frequency control, **targeted** dynamics randomization, and perturbations" — targeted, i.e. informed by measurement. |

**Consensus: measure first, then fix the specific mismatch.** Blanket randomization is the
fallback for when you *cannot* measure. We can measure, so randomization is step 4, not step 1.

---

## 3. Data assets (verified)

Two complete pairs, each with exactly **one contiguous Mode-2 (live SMPL) window** — no
stitching required:

| | Aug 6 session | Aug 11 session |
|---|---|---|
| Human (PICO, 70 Hz) | `paired_smpl_g1_deploy` 33,596 fr / 505 s | `logs/smpl_raw_real_robot` 51,438 fr / 731 s |
| Robot (`state_logger`, 50 Hz) | `g1_robot_data/g1_deploy_run` 25,892 fr / 518 s | `g1_robot_data/g1_real_deploy_logs` 31,034 fr / 684 s |
| **Mode-2 window** | **26.5 → 517.8 s (491.3 s, 95%)** | **194.7 → 684.4 s (489.7 s, 72%)** |
| Checkpoint | `policy/low_latency/` | `policy/low_latency/` |

≈ **981 s (16.4 min)** of paired teleop total.

Robot-side signals (per frame, 50 Hz):

| Signal | Width | Role in this study |
|---|---|---|
| `q` | 29 | measured joint position — **the ground truth to match** |
| `action` | 29 | policy output = position target |
| `dq` | 29 | measured joint velocity |
| `motor_torque` | 29 | realized torque — **actuator ID target** |
| `token_state` | 64 | **policy input** (encoded SMPL) — separates obs gap from dynamics gap |
| `base_quat`, `base_ang_vel`, `base_accel` | 4/3/3 | root state, tilt/stability |
| `motor_temperature` | 58 | thermal / stall detection |
| `motion_name`, `motion_playing`, `encoder_mode` | 1 each | segment gating |

Excluded: `g1_deploy_run002` (reference-motion replay, `encoder_mode` 0 only — no human input).

---

## 4. Phases

Ordered by **value ÷ effort**. Phases A and B need **no simulator** and target exactly what
Tan et al. identify as the dominant error sources.

### Phase A — Latency identification *(no sim required)*

**Question:** how stale is the human command by the time it moves the robot?

Tan et al. treat latency as a primary transfer killer and simulate it explicitly. We can
measure it directly because both sides carry wall-clock timestamps (`timestamp_realtime`
on PICO, `time_realtime_ms` on the robot).

1. Cross-correlate PICO `smpl_joints` against robot `q` to recover the end-to-end delay.
2. Decompose where possible: PICO capture → ZMQ → encoder → policy → motor command.
3. Report the **distribution**, not just the mean (jitter matters as much as latency).

**Deliverable:** `sim2real/phaseA_latency.md` + a latency figure. Feeds a concrete delay
constant for the simulator.

### Phase B — Actuator identification *(no sim required)*

**Question:** does the real motor do what the simulated one does?

The deploy binary commands position targets with fixed gains
(`policy_parameters.hpp`: `Kp`/`Kd` derived from armature, `DAMPING_RATIO = 2`).

1. Fit the nominal PD law `τ̂ = Kp(action − q) − Kd·dq` and compare to logged `motor_torque`.
2. Analyse residuals per joint for **static friction, saturation, backlash, deadband**.
3. If residuals are structured and not analytically capturable → train a small
   **actuator network** (Hwangbo et al.): joint-state history → realized torque.

**Deliverable:** `sim2real/phaseB_actuator.md`, per-joint identified parameters, plus an
optional `actuator_net.onnx`. This is the highest-value phase: it consumes data we already
have and directly patches the simulator.

### Phase C — Sim replay and gap decomposition *(sim required)*

Replay the **same** SMPL segments through sim with the **same** checkpoint, then compare
layer by layer:

```
same human SMPL ─→ encoder ─→ tokens ─→ policy ─→ action ─→ SIM  dynamics ─→ q_sim
                                                          └→ REAL dynamics ─→ q_real
```

| Layer | Metric | If it diverges … |
|---|---|---|
| Observation | `token_state` MSE | encoder / streaming pipeline mismatch |
| Action | `action` MSE | policy is behaving differently |
| **Dynamics** | **`q_sim − q_real` per joint** | **the actual sim2real gap** |
| Stability | tilt (`base_quat`) divergence time | when trajectories separate |
| Actuation | `motor_torque` sim vs real | unmodeled friction / saturation |

Preferred route **C-a**: MuJoCo + `g1_deploy_onnx_ref` driven by
`data_process/stream_clip_mode2.py` — the *same binary* as the real run, so the comparison
isolates dynamics instead of confounding it with an eval-harness rewrite.
Alternative **C-b**: `eval_agent_trl.py` on `pico_evalset`, reusing existing numbers.

**Deliverable:** `sim2real/phaseC_gap.md` with a per-joint gap breakdown.
*Hypothesis to test (not assume): ankles/knees dominate (contact-rich), arms track well.*

### Phase D — Apply fixes and validate

1. Patch the simulator with Phase A latency + Phase B actuator model.
2. Re-run Phase C; the residual gap should shrink.
3. Only then add **targeted** randomization (Haarnoja et al.) over parameters Phase B
   showed to be genuinely uncertain.
4. A/B on the same clips, same checkpoint.

**Deliverable:** `sim2real/phaseD_validation.md` — before/after gap table.

---

## 5. Shared prerequisite: the aligned dataset

All phases need one resampled 50 Hz table per session (human SMPL + robot
`q`/`dq`/`action`/`motor_torque`/`token_state`/`base_quat`).

**Deliverable:** `data_process/build_sim2real_pairs.py` → `sim2real/<session>.npz`,
reporting the alignment residual as a quality gate.

### ⚠️ Alignment pitfalls (already measured — do not rediscover)

1. **Clock drift ≈ 24,000 ppm.** The optimal offset between two recorders slid
   166.18 s → 164.06 s over 300 s (−7.1 ms/s). A **single global offset accumulates ~7 s
   over 300 s**, inflating RMS to 0.043 m even though a locally-aligned match is ~3 mm.
   → Fit **offset + rate**, or re-align per window. Never assume a constant offset.
2. **Never assume a frame rate.** Assuming 50 Hz for a `streamed_*` bundle gives r = 0.25
   at the wrong offset; the true rate is 70 Hz (r = 0.94). Solve for the rate.
3. **Time-warped matching lies.** Resampling two clips to equal length and correlating
   gives a plausible-looking r = 0.74 at a *bogus* offset. Preserve the time axis.
4. **`joint_pos` in PICO `.npz` is NOT robot joint state.** Only indices 23–28 (hand DOFs)
   are populated; 23 of 29 columns are identically zero. Correlating it against `q.csv`
   scores ~0.05 even on the known-good pair. Use `base_quat` ↔ `body_quat_w`, or
   `token_state` ↔ `smpl_joints`.
5. The human stream **starts before** the robot in both sessions (+1.99 s Aug 6, +96 s
   Aug 11), so index 0 ≠ index 0.

---

## 6. Scope limits (state these in any result)

- **`transl` is all-zero** in the converted clips: the PICO capture stores no pelvis world
  position, so world locomotion is absent and **root-position gap is out of scope**. This
  study measures **joint-level tracking and actuation** — which is where Tan et al. and
  Hwangbo et al. both located the gap anyway.
- **No fall detection in the deploy binary.** `CheckSafety()` only checks LowState
  availability/staleness; nothing inspects posture. If the robot fell, the policy kept
  commanding a downed robot. → **Screen `base_quat` tilt in Phase A/B and exclude those
  windows**, otherwise the gap will be dominated by post-fall garbage.
- Two sessions, one operator, one checkpoint — conclusions are about *this* robot and
  *this* policy, not humanoids in general.
- Sim-side comparison inherits the `num_envs` sensitivity noted in
  `notes_pico_evalset_3model_comparison.md`; hold it fixed.

---

## 7. Suggested order

```
build_sim2real_pairs.py   (prerequisite, §5)
        │
        ├── Phase A  latency        ── no sim ── highest value/effort
        ├── Phase B  actuator ID    ── no sim ── highest value overall
        │
        └── Phase C  sim replay     ── needs sim
                 └── Phase D  fix + validate
```

Start with the prerequisite + Phases A and B: they require no simulator runs, surface data
problems early, and address the error sources the literature identifies as dominant.

---

## 8. References

- Tan, J., Zhang, T., Coumans, E., Iscen, A., Bai, Y., Hafner, D., Bohez, S., Vanhoucke, V.
  (2018). *Sim-to-Real: Learning Agile Locomotion For Quadruped Robots.* arXiv:1804.10332.
- Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., Hutter, M.
  (2019). *Learning agile and dynamic motor skills for legged robots.* Science Robotics 4(26).
  arXiv:1901.08652.
- Zhao, W., Peña Queralta, J., Westerlund, T. (2020). *Sim-to-Real Transfer in Deep
  Reinforcement Learning for Robotics: a Survey.* IEEE SSCI 2020, pp. 737–744.
  arXiv:2009.13303.
- Haarnoja, T., Moran, B., Lever, G., Huang, S. H., Tirumala, D., Humplik, J., Wulfmeier, M.,
  Tunyasuvunakool, S., et al. (2023). *Learning Agile Soccer Skills for a Bipedal Robot with
  Deep Reinforcement Learning.* Science Robotics. arXiv:2304.13653.
