# Phase C.2 (discussion) — One-step prediction vs. world modeling

This note captures the discussion clarifying what Phase C.1's `q_sim` / `q_real` comparison
means, and how a genuine "Phase C.2" closed-loop rollout test would differ from it. No C.2
script exists yet; this is a design discussion to guide building one.

## Does `q_sim` / `q_real` indicate the next-step state?

Yes. In `sim2real_phaseC1_onestep_prediction.py::one_step_predict()`:

```python
data.qpos[qadr] = q_real[t]      # teacher-forced: start from REAL state at t
data.qvel[dofadr] = dq_real[t]
...
mujoco.mj_step(model, data)      # ONE dt forward
q_sim_next[t] = data.qpos[qadr]  # = "q_sim(t+dt)", a prediction of q_real(t+1)
```

```
same human SMPL → encoder → tokens → policy → action(t)
                                                  │
                            ┌─────────────────────┴─────────────────────┐
                            ▼                                           ▼
                    SIM dynamics, ONE STEP                        REAL dynamics
                    q_sim(t+dt)                                   q_real(t+dt)
                    (both start from SAME q_real(t), dq_real(t))
```

`q_sim(t+dt)` = sim's prediction of the next joint position given the exact same current state
and action as the real robot. `q_real(t+dt)` = what actually happened on hardware. Comparing the
two, every step, gives the one-step RMS error (`q1step`) reported by Phase C.1.

## How this differs from "world modeling"

The key distinction is **teacher forcing (reset every step)** vs. **autoregressive rollout**.

| | Phase C.1 (this test) | World model / learned dynamics model |
|---|---|---|
| State input at each step | Always the real robot's measured state (reset every step) | The model's own previous output — state is whatever it predicted at t-1 |
| Error propagation | None — errors cannot compound; every step judged independently | Compounds/accumulates over the rollout; small per-step errors can snowball into large drift or divergence |
| What it measures | Local/instantaneous fidelity: "given the true state, is one step of integration accurate?" | Long-horizon predictive validity: "can the model simulate/imagine a whole trajectory autoregressively without external grounding?" |
| Model type | Known physics engine (MuJoCo, analytic `M(q)q̈ + C(q,q̇)q̇ + G(q)` + PD law); this test calibrates known-form parameters (damping/friction/armature) | Typically a learned function approximating dynamics; used for planning, imagination-based RL, or model-based control |
| Failure mode tested | "Is my analytic model's local derivative correct?" (small `dt`, drift impossible by construction) | "Does my model stay calibrated when I trust its own predictions for many steps?" — exactly what failed in the earlier `sim2real_phaseB2_compensation_test.py` open-loop replay (diverged, robot fell by t=30-50s) |

**Summary:** Phase C.1 tests the physics engine's *local* accuracy — a system-identification /
calibration question ("is the dynamics model, with these damping/armature parameters, correct
one step at a time?"). World modeling tests *global/autoregressive* predictive accuracy of a
model (usually learned) that must remain self-consistent over long horizons without ever being
reset to ground truth. Teacher-forcing in C.1 is deliberately designed to avoid the
world-modeling failure mode (compounding drift/instability) so it can isolate "is the dynamics
model itself right?" from "does accumulated error make the model unusable over time?".

## What a real Phase C.2 (closed-loop rollout) would need

To actually test the world-modeling question, C.2 would:
1. Initialize sim state once from the real robot's state at `t0` (no reset afterward).
2. Feed each `action(t)` (from the real log, or ideally re-run through the policy) and let sim
   integrate forward continuously — i.e. `q_sim(t+dt)` becomes the input to the next step,
   **not** overwritten by `q_real(t+dt)`.
3. Track how `q_sim(t)` diverges from `q_real(t)` over increasing horizon length, and whether the
   calibrated damping values (Phase B: `{4: 1.19, 10: 1.09, 14: 0.98}`, or the scanned
   alternative `{4: 3.0, 10: 3.0, 14: 1.0}` from `sim2real_phaseC1_damping_scan.py`) delay or
   prevent the instability seen in the earlier open-loop `phaseB2_compensation_test.py` attempt.
4. This is the appropriate test to validate whether the higher-damping "best-scan" config
   (which only improved the *one-step* metric) is actually safe/stable in closed loop, or
   whether it destabilizes the rollout the way the uncalibrated baseline did.

Not yet implemented — flagged here as the natural next step after the Phase C.1 damping scan.
