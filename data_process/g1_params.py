"""
G1 policy/actuator constants, extracted verbatim from the deploy binary's own
header so analysis code cannot drift from what the robot actually ran:

    gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp

Values were produced by COMPILING against that header and printing the arrays
(not transcribed by hand). Regenerate with the snippet at the bottom of this file.

WHY THIS MATTERS
----------------
The `state_logger` CSVs do NOT store q and action in the same space:

  q.csv       -> hardware / MuJoCo joint order, `default_angles` ALREADY added back
  action.csv  -> raw policy output, IsaacLab joint order, UNSCALED

so the actual commanded position target must be reconstructed:

    q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] * g1_action_scale[i]

Comparing `action` directly against `q` gives a median error of 0.667 rad (38 deg)
and a zero-lag correlation of ~-0.015, i.e. meaningless.

Note `g1_action_scale` spans 0.0745 (wrist roll/pitch) to 0.548 (hip/waist yaw) --
a ~7x range -- so this is not a small correction.
"""
import numpy as np

# action_scale = 0.25 * effort_limit / stiffness
G1_ACTION_SCALE = np.array([
    0.3506614664, 0.3506614664, 0.5475464652, 0.3506614664, 0.4385773139,
    0.4385773139, 0.3506614664, 0.3506614664, 0.5475464652, 0.3506614664,
    0.4385773139, 0.4385773139, 0.5475464652, 0.4385773139, 0.4385773139,
    0.4385773139, 0.4385773139, 0.4385773139, 0.4385773139, 0.4385773139,
    0.07450087033, 0.07450087033, 0.4385773139, 0.4385773139, 0.4385773139,
    0.4385773139, 0.4385773139, 0.07450087033, 0.07450087033,
])

DEFAULT_ANGLES = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0, 0, 0, 0,
    0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0,
])

ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23,
    25, 27, 12, 16, 20, 22, 24, 26, 28,
])

MUJOCO_TO_ISAACLAB = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
    24, 18, 25, 19, 26, 20, 27, 21, 28,
])

# stiffness = armature * (2*pi*10)^2 ; damping = 2 * 2.0 * armature * (2*pi*10)
KPS = np.array([
    99.09842682, 99.09842682, 40.17923737, 99.09842682, 28.5012455, 28.5012455,
    99.09842682, 99.09842682, 40.17923737, 99.09842682, 28.5012455, 28.5012455,
    40.17923737, 28.5012455, 28.5012455, 14.25062275, 14.25062275, 14.25062275,
    14.25062275, 14.25062275, 16.77832794, 16.77832794, 14.25062275, 14.25062275,
    14.25062275, 14.25062275, 14.25062275, 16.77832794, 16.77832794,
])

KDS = np.array([
    6.308801651, 6.308801651, 2.5578897, 6.308801651, 1.814445734, 1.814445734,
    6.308801651, 6.308801651, 2.5578897, 6.308801651, 1.814445734, 1.814445734,
    2.5578897, 1.814445734, 1.814445734, 0.907222867, 0.907222867, 0.907222867,
    0.907222867, 0.907222867, 1.06814146, 1.06814146, 0.907222867, 0.907222867,
    0.907222867, 0.907222867, 0.907222867, 1.06814146, 1.06814146,
])

# hardware / MuJoCo order (the order q.csv, motor_torque.csv etc. are written in)
JOINT_NAMES = [
    'L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ankle_pitch', 'L_ankle_roll',
    'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ankle_pitch', 'R_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'L_sho_pitch', 'L_sho_roll', 'L_sho_yaw', 'L_elbow', 'L_wri_roll', 'L_wri_pitch', 'L_wri_yaw',
    'R_sho_pitch', 'R_sho_roll', 'R_sho_yaw', 'R_elbow', 'R_wri_roll', 'R_wri_pitch', 'R_wri_yaw',
]

GROUPS = {
    'legs': list(range(0, 12)),
    'waist': list(range(12, 15)),
    'left_arm': list(range(15, 22)),
    'right_arm': list(range(22, 29)),
}


def action_to_q_target(action):
    """Raw logged `action` (T,29, IsaacLab order) -> commanded joint position
    target (T,29, hardware order), matching what the deploy binary sends:

        motor_command.q_target[i] = default_angles[i]
                                  + action[isaaclab_to_mujoco[i]] * g1_action_scale[i]
    """
    action = np.asarray(action)
    return DEFAULT_ANGLES[None, :] + action[:, ISAACLAB_TO_MUJOCO] * G1_ACTION_SCALE[None, :]


# Regenerate constants:
#   cat > /tmp/dump_params.cpp <<'EOF'
#   #include <vector>
#   #include "policy_parameters.hpp"
#   #include <cstdio>
#   int main(){ for(int i=0;i<29;i++) printf("%.10g, ", g1_action_scale[i]); }
#   EOF
#   g++ -I gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include -o /tmp/dp /tmp/dump_params.cpp && /tmp/dp
