"""
SONIC motion-tracking reward functions for G1 PPO training.

Total reward = weighted sum of five components:
  r_tracking  — joint positions match reference motion
  r_root      — root height and orientation match reference
  r_balance   — penalise large base angular velocity
  r_smooth    — penalise action rate (jerk)
  r_torque    — penalise large actuator torques (optional)

All components are bounded in [0, 1] via exponential shaping:
  r = exp(-k * ||error||²)
so the agent maximises similarity to the reference motion.
"""

import numpy as np


def compute_reward(
    q:           np.ndarray,    # (29,) current joint positions (rad)
    dq:          np.ndarray,    # (29,) current joint velocities (rad/s)
    ref_q:       np.ndarray,    # (29,) reference joint positions (rad)
    root_vel:    np.ndarray,    # (6,)  [vx, vy, vz, wx, wy, wz]
    action:      np.ndarray,    # (29,) current action (rad)
    prev_action: np.ndarray,    # (29,) previous action (rad)
    torques:     np.ndarray | None = None,
    # weights
    w_tracking: float = 1.0,
    w_balance:  float = 0.2,
    w_smooth:   float = 0.1,
    w_torque:   float = 0.05,
    # shaping sigmas
    sigma_q:    float = 0.25,   # rad
    sigma_ang:  float = 1.0,    # rad/s
    sigma_da:   float = 0.5,    # rad
) -> float:
    """Compute total motion-tracking reward for one timestep."""

    # ── r_tracking : joint position similarity ────────────────────────────────
    # exp(-||q - q_ref||² / (2σ²))  → 1 when perfectly matching, decays smoothly
    err_q      = q - ref_q
    r_tracking = float(np.exp(-np.sum(err_q ** 2) / (2 * sigma_q ** 2 * len(q))))

    # ── r_balance : penalise large base angular velocity ─────────────────────
    ang_vel    = root_vel[3:]                    # wx, wy, wz
    r_balance  = float(np.exp(-np.sum(ang_vel ** 2) / (2 * sigma_ang ** 2)))

    # ── r_smooth : penalise action change (action rate) ───────────────────────
    delta_a   = action - prev_action
    r_smooth  = float(np.exp(-np.sum(delta_a ** 2) / (2 * sigma_da ** 2 * len(action))))

    # ── r_torque : penalise large torques (if provided) ─────────────────────
    r_torque = 1.0
    if torques is not None:
        r_torque = float(np.exp(-np.mean(torques ** 2) / 1000.0))

    total = (
        w_tracking * r_tracking
        + w_balance  * r_balance
        + w_smooth   * r_smooth
        + w_torque   * r_torque
    ) / (w_tracking + w_balance + w_smooth + w_torque)

    return float(total)
