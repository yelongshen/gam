"""
Sim2Real "补偿测试" (compensation validation test) for Phase B viscous-damping findings.

WHAT THIS TESTS
---------------
Phase B fit an EXTRA damping term `b` per joint from real-robot logs by solving
    tau_est = Kp*(q_target - q) - (Kd_nom + b)*dq
This script asks: if we inject that same extra damping as MuJoCo joint
`damping` (a passive term, NOT part of the PD control law -- see the
armature/damping/friction note), does OPEN-LOOP replay of the real robot's
own `q_target` sequence reproduce the real robot's OWN measured `q` trajectory
and tracking-error RMS better than the (already IsaacLab-aligned) baseline
with zero extra damping?

This is a much cheaper test than the full ZMQ/policy/streamer pipeline:
MuJoCo's `<motor>` actuators are direct torque actuators driven by
`mj_data.ctrl`, and `compute_body_torques()` in base_sim.py computes exactly
`tau = Kp*(q_target-q) + Kd*(dq_target-dq)` in Python before writing it to
`ctrl`. So we can replay the REAL `q_target(t)` sequence directly against a
bare MuJoCo model with the SAME Kp/Kd/dt as the real controller, with no
policy network, no ZMQ, no sim_loop process -- just mj_step in a loop.

METHOD
------
1. Load the aligned aug11 session (human/robot already resampled to 50 Hz).
2. Take the real `q_target(t)` (from `action_to_q_target`) for the 5 test
   joints (3 hypothesis joints + 2 controls).
3. For each candidate extra damping `b` (0 = current post-armature-alignment
   baseline, plus the Phase B analytic optima), set `dof_damping[joint] += b`
   in the (already IsaacLab-armature-aligned) MuJoCo model, then step the sim
   open-loop with q_target(t) as the position target, holding all OTHER
   joints at their real q_target trajectory too (so gravity/coupling loads are
   realistic) but only scoring the 5 test joints.
4. Compare simulated RMS(q_sim - q_target) against the REAL RMS(q_real -
   q_target) already measured in Phase A (sim2real/phaseA_latency.md §5):
       L_ankle_pitch  28.72 deg
       R_ankle_pitch  26.69 deg
       waist_pitch    22.72 deg
       L_hip_pitch    (control, small)
       L_sho_pitch    (control, small)
   A good `b` should bring sim RMS close to the real RMS on the 3 hypothesis
   joints while leaving the 2 control joints roughly unchanged.

USAGE
-----
  .venv_sim/bin/python model_eval/sim2real_phaseB2_compensation_test.py \\
      --duration 60 --candidates 0,0.5,0.8,1.0,1.09,1.19,0.98,1.3,1.5,2.0
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_process'))
from load_sim2real_session import load_session          # noqa: E402
from g1_params import action_to_q_target, KPS, KDS, JOINT_NAMES  # noqa: E402

XML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'gear_sonic/data/robot_model/model_data/g1')
XML = os.path.join(XML_DIR, 'scene_43dof.xml')
BODY_XML = os.path.join(XML_DIR, 'g1_29dof_with_hand.xml')

MJ_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
    'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]

# The 43-DOF model interleaves 22 hand-finger actuators between the left and
# right arm body actuators (motor #21 left_wrist_yaw is immediately followed
# by finger actuators, THEN right_shoulder_pitch resumes at a much higher
# index) -- so the 29 body actuators are NOT contiguous in ctrl/qpos and must
# be addressed by NAME, never by a fixed slice.
MJ_ACTUATOR_NAMES = [
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee',
    'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee',
    'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
    'left_elbow', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
    'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
]

# (name, joint index in the 29-DOF hardware order, Phase A real tracking RMS in deg)
TEST_JOINTS = [
    ('L_ankle_pitch', 4, 28.72),
    ('R_ankle_pitch', 10, 26.69),
    ('waist_pitch', 14, 22.72),
    ('L_hip_pitch', 0, None),   # control: no b expected
    ('L_sho_pitch', 15, None),  # control: no b expected
]

FS = 50.0


def build_model(extra_damping, weld_base=True):
    """Load the IsaacLab-armature-aligned MuJoCo model and add `extra_damping`
    (dict: joint_index -> extra Nm*s/rad) on top of the existing dof_damping
    (which is 0 after the armature-alignment patch).

    If `weld_base`, the pelvis free-joint is removed by writing a patched copy
    of the body XML (and a small scene wrapper) alongside the originals, so
    relative `meshdir` paths still resolve via `from_xml_path` (MuJoCo has no
    runtime "remove joint" API). This fixes the robot in space so open-loop
    replay tests joint-level dynamics without whole-body balance -- see
    replay_open_loop() docstring for why this is necessary.
    """
    if weld_base:
        import re
        body = open(BODY_XML).read()
        patched, n = re.subn(r'<joint name="floating_base_joint"[^/]*/>', '', body, count=1)
        assert n == 1, "expected exactly one floating_base_joint (pelvis) to remove"
        patched_body_path = os.path.join(XML_DIR, '_tmp_welded_g1_29dof_with_hand.xml')
        open(patched_body_path, 'w').write(patched)

        scene = open(XML).read().replace(
            os.path.basename(BODY_XML), os.path.basename(patched_body_path))
        patched_scene_path = os.path.join(XML_DIR, '_tmp_welded_scene_43dof.xml')
        open(patched_scene_path, 'w').write(scene)

        m = mujoco.MjModel.from_xml_path(patched_scene_path)
    else:
        m = mujoco.MjModel.from_xml_path(XML)
    for j, b in extra_damping.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, MJ_JOINT_NAMES[j])
        dof = m.jnt_dofadr[jid]
        m.dof_damping[dof] = b
    return m


def replay_isolated(model, q_target_seq, dt, test_joints):
    """Drive ONLY `test_joints` with real PD dynamics (Kp/Kd + candidate extra
    damping). All OTHER 24 joints are kinematically PINNED to the real
    q_target trajectory every step (qpos/qvel forced, bypassing the actuator
    entirely) -- they still exert correct gravity/Coriolis coupling on the
    test joints via the real, correctly-posed multibody configuration, but
    their own (non-)tracking cannot contaminate the test joints' error the
    way full open-loop replay did.

    This replaces the earlier "replay ALL 29 joints open-loop" approach, which
    had two problems: (1) with a free floating base it toppled/diverged by
    t=30s, and (2) even after welding the base, uncontrolled drift on the 24
    non-test joints fed back into gravity/Coriolis terms and produced
    inconsistent, sometimes-worse-with-more-damping results on the 3
    hypothesis joints. Pinning removes that confound; only test_joints'
    dynamics can differ across candidates now.

    `test_joints`: list of hardware-order joint indices (0-28) to simulate.
    Returns q_sim (T, 29) -- pinned joints are simply copies of q_target.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for j in range(29):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MJ_JOINT_NAMES[j])
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = q_target_seq[0, j]
    mujoco.mj_forward(model, data)

    n_sub = max(1, int(round(dt / model.opt.timestep)))
    T = len(q_target_seq)
    q_sim = np.zeros((T, 29))
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in MJ_JOINT_NAMES])
    dofadr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                        for n in MJ_JOINT_NAMES])
    actadr = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in MJ_ACTUATOR_NAMES])

    test_mask = np.zeros(29, dtype=bool)
    test_mask[test_joints] = True
    pinned = ~test_mask

    prev_target = q_target_seq[0].copy()
    for t in range(T):
        target = q_target_seq[t]
        # kinematically pin the non-test joints: exact position + finite-diff velocity
        data.qpos[qadr[pinned]] = target[pinned]
        data.qvel[dofadr[pinned]] = (target[pinned] - prev_target[pinned]) / dt

        q = data.qpos[qadr]
        dq = data.qvel[dofadr]
        tau = np.zeros(29)
        tau[test_mask] = (KPS[test_mask] * (target[test_mask] - q[test_mask])
                          - KDS[test_mask] * dq[test_mask])
        tau = np.clip(tau, -139, 139)
        data.ctrl[actadr] = tau

        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        # re-pin immediately after stepping so pinned joints never carry
        # integrated drift into the next iteration's coupling terms
        data.qpos[qadr[pinned]] = target[pinned]
        data.qvel[dofadr[pinned]] = 0.0
        mujoco.mj_forward(model, data)

        q_sim[t] = data.qpos[qadr]
        prev_target = target
    return q_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='aug11')
    ap.add_argument('--duration', type=float, default=60.0,
                     help='seconds of the usable window to replay (keep small: this is python mj_step, not real-time)')
    ap.add_argument('--start', type=float, default=0.0,
                     help='offset (s) into the usable window to start replay from')
    args = ap.parse_args()

    S = load_session(args.session)
    r = S['robot']
    lo, hi = S['overlap']
    grid = np.arange(lo + args.start, min(hi, lo + args.start + args.duration), 1 / FS)
    print(f"=== Compensation test - session {args.session} - replaying {len(grid)/FS:.1f}s "
          f"({len(grid)} steps) ===\n")

    def resample(t, v):
        return np.stack([np.interp(grid, t, v[:, j]) for j in range(v.shape[1])], 1)

    q_real = resample(r['t'], r['q'])
    qt_real = resample(r['t'], action_to_q_target(r['action']))

    # candidates: baseline 0, plus the Phase B analytic optima per joint.
    # test_joints = the 3 hypothesis joints + 2 controls, ALL simulated
    # simultaneously (their mutual coupling IS part of what we want to keep
    # realistic -- e.g. waist_pitch loading affects hip_pitch); everything
    # else (24 joints) is kinematically pinned per replay_isolated().
    test_joints = [j for _, j, _ in TEST_JOINTS]
    cand_sets = {
        'baseline (b=0, IsaacLab-aligned)': {},
        'calibrated (Phase B analytic optima)': {4: 1.19, 10: 1.09, 14: 0.98},
    }

    print(f"{'joint':16s} {'real RMS(deg)':>14s} " +
          " ".join(f"{name[:22]:>24s}" for name in cand_sets))
    results = {}
    for name, extra in cand_sets.items():
        model = build_model(extra)
        q_sim = replay_isolated(model, qt_real, 1 / FS, test_joints)
        rms = np.degrees(np.sqrt(((q_sim - qt_real) ** 2).mean(0)))
        results[name] = rms

    for jname, j, real_rms in TEST_JOINTS:
        real_str = f"{real_rms:.2f}" if real_rms is not None else "(control)"
        row = f"{jname:16s} {real_str:>14s} "
        for name in cand_sets:
            row += f"{results[name][j]:24.2f} "
        print(row)

    print("\nInterpretation: for the 3 hypothesis joints (ankle pitch x2, waist_pitch), the")
    print("'calibrated' column should move CLOSER to the real RMS than 'baseline'. For the 2")
    print("control joints (hip_pitch, sho_pitch) both columns should stay small/similar --")
    print("if they also move a lot, the calibration is leaking into unaffected joints.")


if __name__ == '__main__':
    main()
