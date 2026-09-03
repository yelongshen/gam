"""Per-joint Kp/Kd correction factors to make the MuJoCo deploy-test sim
reproduce the *realized* PD behaviour measured on the real G1, instead of the
idealized nominal PD law.

Background (`sim2real/phaseB_actuator.md`, session `aug11`):

The deploy binary commands the SAME nominal (Kp, Kd) to both the real robot
and this MuJoCo sim (they are read from the same `WeakMotorJointIndex` order
and travel over the same DDS `LowCmd` message -- see
`DefaultEnv.compute_body_torques` in `base_sim.py`). Fitting
`tau_est = Kp_eff(q_target-q) - Kd_eff*dq` against the REAL robot's logged
torque shows that 6 joints -- both ankles and waist roll/pitch, the ones
sharing the "2x 5020" nominal gain group -- have a systematically different
*effective* gain than the value that was actually commanded:

  pitch subgroup (L/R ankle_pitch, waist_pitch):  Kp ~= +6%,  Kd ~= 1.6x nominal
  roll  subgroup (L/R ankle_roll,  waist_roll):   Kp ~= -2%,  Kd ~= 0.5x nominal

All other (directly-driven) joints matched nominal to within ~1% and get no
correction. This is a real-hardware effect (firmware/gearbox), not a sim
config bug, so it cannot be "fixed" on the robot from log analysis alone.
Applying the multipliers here on the SIM side lets the MuJoCo deploy-test
environment reproduce what the real robot actually does, closing the
kp/kd gap identified in Phase B.

Enable via config: `SIM2REAL_GAIN_CORRECTION: True` in the WBC yaml.
Disabled by default so nothing changes unless explicitly opted in.

See also `sim2real/phaseD_apply_fixes.md` for how to validate this.
"""

from typing import Dict, Tuple

import numpy as np

# joint_name -> (kp_scale, kd_scale), applied on top of whatever kp/kd the
# deploy binary commands for that motor index. Only the joints identified in
# Phase B as diverging get a non-unity entry.
GAIN_CORRECTION: Dict[str, Tuple[float, float]] = {
    "left_ankle_pitch_joint": (1.057, 1.66),
    "right_ankle_pitch_joint": (1.065, 1.60),
    "waist_pitch_joint": (1.066, 1.54),
    "left_ankle_roll_joint": (0.970, 0.50),
    "right_ankle_roll_joint": (0.970, 0.50),  # not directly fitted; assumed
    # symmetric to left_ankle_roll (§3b of phaseB_actuator.md only reports L).
    "waist_roll_joint": (0.981, 0.49),
}


def build_scale_arrays(weak_motor_joint_index: Dict[str, int], num_motors: int):
    """Build (kp_scale, kd_scale) arrays of length `num_motors`, indexed the
    same way as the motor command arrays used in `compute_body_torques`
    (i.e. by `WeakMotorJointIndex[joint_name]`). Unlisted joints get scale 1.0.
    """
    kp_scale = np.ones(num_motors, dtype=np.float64)
    kd_scale = np.ones(num_motors, dtype=np.float64)
    for joint_name, (kp_s, kd_s) in GAIN_CORRECTION.items():
        idx = weak_motor_joint_index.get(joint_name)
        if idx is None or idx >= num_motors:
            continue
        kp_scale[idx] = kp_s
        kd_scale[idx] = kd_s
    return kp_scale, kd_scale
