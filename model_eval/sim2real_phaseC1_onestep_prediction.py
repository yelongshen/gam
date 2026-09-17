"""
Phase C.1 -- single-step dynamics prediction test (no policy re-inference).

WHAT THIS TESTS
---------------
"load the real robot state-action into simulation, and calculate tau_est and
q_sim (next state)":

    same human SMPL --> encoder --> tokens --> policy --> action(t)
                                                    |
                            +-----------------------+-----------------------+
                            v                                               v
                    [SIM dynamics, ONE STEP]                      [REAL dynamics]
                    q_sim(t+dt), tau_sim(t)                       q_real(t+dt), tau_est(t)
                    (both start from the SAME q_real(t), dq_real(t))

We deliberately do NOT re-run the policy network. `action(t)` is REUSED
verbatim from the real robot log (as the user requested) -- this only
validates the SIM DYNAMICS MODEL, not the policy/observation pipeline (that
split was already established in Phase A: stage 3, encoder->policy->motor
command, was shown to add no gap).

WHY TEACHER FORCING (this is NOT the same as the earlier open-loop replay)
---------------------------------------------------------------------------
Earlier attempts (sim2real_phaseB2_compensation_test.py) rolled the sim
forward continuously: once sim state drifts from real state (even slightly,
from imperfect armature/friction), the REUSED action sequence is no longer
appropriate for where the sim actually is -- errors compound and the robot
eventually falls (observed: instability at t=30-50s).

Here, EVERY step resets sim's q/dq to the REAL robot's measured q_real(t),
dq_real(t) before applying action(t) and stepping ONE dt forward. This is a
"one-step-ahead prediction" / teacher-forcing test: it asks "if sim starts
from the exact same state real robot was in, and applies the exact same
command, does ONE STEP of sim dynamics land close to where the exact same
command took the real robot?" -- with no possibility of multi-step drift,
so it is stable for the ENTIRE 439.6 s Mode-2 window, on all 29 joints at
once (full multibody coupling from the real pose, no isolation-topology
issues like the earlier pinned-joint experiment had).

WHAT tau_sim(t) MEANS HERE
---------------------------
MuJoCo's <motor> actuators are direct torque actuators: `qfrc_actuator` after
clipping to ctrlrange equals the commanded torque, i.e. this is essentially
`tau_cmd = Kp*e - Kd*dq` again (already validated against real tau_est at
R^2=0.999 in Phase B, so it is NOT new information). The genuinely new
signal from this test is q_sim(t+dt), which depends on the sim's inertia/
armature/damping/friction model integrating that torque -- exactly the
piece Phase B could not validate (B2 failed: no designed excitation).

USAGE
-----
  .venv_sim/bin/python model_eval/sim2real_phaseC1_onestep_prediction.py \\
      --duration 439.6
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

TEST_JOINTS = [(JOINT_NAMES[j], j, None) for j in range(29)]
FS = 50.0
EFFORT_LIMIT = np.array([139, 139, 88, 139, 50, 50] * 2 + [88, 50, 50] + [25, 25, 25, 25, 25, 5, 5] * 2)


def build_model(extra_damping):
    """Load the IsaacLab-armature-aligned MuJoCo model (base already welded
    in the XML source used here has a free joint; for this single-step test
    the free joint doesn't matter since we overwrite qpos/qvel every step
    anyway and only integrate ONE dt -- but we weld it regardless for
    consistency/safety, same as the earlier tests)."""
    import re
    body = open(BODY_XML).read()
    patched, n = re.subn(r'<joint name="floating_base_joint"[^/]*/>', '', body, count=1)
    assert n == 1
    patched_body_path = os.path.join(XML_DIR, '_tmp_welded_g1_29dof_with_hand.xml')
    open(patched_body_path, 'w').write(patched)
    scene = open(XML).read().replace(os.path.basename(BODY_XML), os.path.basename(patched_body_path))
    patched_scene_path = os.path.join(XML_DIR, '_tmp_welded_scene_43dof.xml')
    open(patched_scene_path, 'w').write(scene)

    m = mujoco.MjModel.from_xml_path(patched_scene_path)
    for j, b in extra_damping.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, MJ_JOINT_NAMES[j])
        m.dof_damping[m.jnt_dofadr[jid]] = b
    return m


def one_step_predict(model, q_real, dq_real, q_target, dt, kp=None, kd=None):
    """For every t, RESET sim state to (q_real[t], dq_real[t]) [teacher
    forcing], apply the real command via the real PD law, step ONE dt, and
    return (q_sim_next, tau_sim) both of shape (T,29). q_sim_next[t] is the
    prediction for q_real[t+1]; tau_sim[t] is the sim-side commanded/realised
    torque this step (compare to motor_torque[t]).

    `kp`/`kd`: optional (29,) arrays overriding the module-level KPS/KDS for
    this call only (used to test e.g. `--motor-kp-scale`/`--motor-kd-scale`
    style multiplicative gain changes -- see cand_sets in main()). Defaults
    to the nominal KPS/KDS if not given."""
    kp = KPS if kp is None else kp
    kd = KDS if kd is None else kd
    data = mujoco.MjData(model)
    n_sub = max(1, int(round(dt / model.opt.timestep)))
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in MJ_JOINT_NAMES])
    dofadr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                        for n in MJ_JOINT_NAMES])
    actadr = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in MJ_ACTUATOR_NAMES])

    T = len(q_real)
    q_sim_next = np.zeros((T, 29))
    tau_sim = np.zeros((T, 29))

    for t in range(T):
        # teacher forcing: overwrite sim state with the REAL state at time t
        data.qpos[qadr] = q_real[t]
        data.qvel[dofadr] = dq_real[t]
        mujoco.mj_forward(model, data)

        tau = kp * (q_target[t] - q_real[t]) - kd * dq_real[t]
        tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)
        data.ctrl[actadr] = tau
        tau_sim[t] = tau

        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        q_sim_next[t] = data.qpos[qadr]

    return q_sim_next, tau_sim


def scaled_gains(kp_scale=None, kd_scale=None):
    """Build (kp, kd) arrays = KPS/KDS with per-joint-index multipliers
    applied. kp_scale/kd_scale: dict {joint_index: multiplier}."""
    kp = KPS.copy()
    kd = KDS.copy()
    for j, s in (kp_scale or {}).items():
        kp[j] *= s
    for j, s in (kd_scale or {}).items():
        kd[j] *= s
    return kp, kd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='aug11')
    ap.add_argument('--duration', type=float, default=439.6,
                     help='seconds of the usable window to test (default = full Mode-2 window)')
    ap.add_argument('--start', type=float, default=0.0)
    args = ap.parse_args()

    S = load_session(args.session)
    r = S['robot']
    lo, hi = S['overlap']
    grid = np.arange(lo + args.start, min(hi, lo + args.start + args.duration), 1 / FS)
    print(f"=== Phase C.1 - one-step prediction - session {args.session} ===")
    print(f"testing {len(grid)/FS:.1f}s ({len(grid)} steps), teacher-forced every step\n")

    def resample(t, v):
        return np.stack([np.interp(grid, t, v[:, j]) for j in range(v.shape[1])], 1)

    q_real = resample(r['t'], r['q'])
    dq_real = resample(r['t'], r['dq'])
    tau_real = resample(r['t'], r['motor_torque'])
    qt_real = resample(r['t'], action_to_q_target(r['action']))

    # drop the last sample (no q_real[t+1] to compare against)
    q_real, dq_real, tau_real, qt_real = q_real[:-1], dq_real[:-1], tau_real[:-1], qt_real[:-1]
    q_real_next = resample(r['t'], r['q'])[1:len(q_real) + 1]

    # Each candidate: extra_damping (additive dof_damping, applied in build_model)
    # and/or kp_scale/kd_scale (multiplicative, applied to the PD law itself via
    # scaled_gains() -- this is what --motor-kp-scale/--motor-kd-scale do on the
    # real deploy binary). See sim2real/phaseE_waist_roll_chatter.md and the
    # discussion of upstream commit 087f9ac for why these specific candidates
    # were chosen: joints 4/10 = L/R_ankle_pitch.
    #
    # FRAMING (corrected): `baseline (nominal)` -- nominal gains, UNCALIBRATED
    # model -- is the GROUND TRUTH, because that is exactly the training-time
    # dynamics the policy was optimized against (IsaacLab uses the same nominal
    # Kp/Kd, see phaseB_actuator.md sec 3c). We never ran real hardware with
    # scaled gains, so `q_real_next*` (what the real robot WOULD do under
    # scaled gains) cannot be read from logs. Instead we PREDICT it using the
    # calibrated model (dof_damping = Phase B fit, our best available proxy for
    # real hardware's extra friction/damping, assumed independent of what gain
    # is commanded) driven by the SCALED PD law. The question this answers:
    # does that predicted real-with-scaled-gains state land closer to the
    # training ground truth than the ACTUAL observed (unscaled) real state did?
    cand_sets = {
        'baseline (nominal, = training ground truth)':
            dict(extra_damping={}, kp_scale=None, kd_scale=None),
        'Phase B calibrated (proxy for REAL, unscaled)':
            dict(extra_damping={4: 1.19, 10: 1.09, 14: 0.98}, kp_scale=None, kd_scale=None),
        'calibrated + scaled 1.5/1.5 (predicted REAL*, upstream)':
            dict(extra_damping={4: 1.19, 10: 1.09, 14: 0.98},
                 kp_scale={4: 1.5, 10: 1.5}, kd_scale={4: 1.5, 10: 1.5}),
        'calibrated + scaled Kp=1.0/Kd=1.5 (predicted REAL*, measured)':
            dict(extra_damping={4: 1.19, 10: 1.09, 14: 0.98},
                 kp_scale=None, kd_scale={4: 1.5, 10: 1.5}),
    }
    highlight_joints = {4, 10}

    raw_q_sim_next = {}
    for name, cfg in cand_sets.items():
        model = build_model(cfg['extra_damping'])
        kp, kd = scaled_gains(cfg['kp_scale'], cfg['kd_scale'])
        q_sim_next, _tau_sim = one_step_predict(model, q_real, dq_real, qt_real, 1 / FS, kp=kp, kd=kd)
        raw_q_sim_next[name] = q_sim_next
        print(f"  done: {name}")

    names = list(cand_sets)
    ground_truth = raw_q_sim_next[names[0]]          # nominal-gain, uncalibrated: what training saw

    def rms_deg(a, b):
        return np.degrees(np.sqrt(((a - b) ** 2).mean(0)))

    # error_before: the ACTUAL observed real robot (unscaled gains, real logged
    # q_real_next) vs the training ground truth. This is the existing,
    # already-measured sim2real gap -- no model needed, q_real_next is data.
    error_before = rms_deg(q_real_next, ground_truth)

    # error_after[name]: the PREDICTED real-robot-under-scaled-gains state
    # (calibrated model + scaled PD law) vs the SAME training ground truth.
    # Note this is compared against `ground_truth`, NOT against q_real_next --
    # we have no real data for the scaled-gain case, so q_real_next cannot
    # appear in this half of the comparison.
    error_after = {name: rms_deg(raw_q_sim_next[name], ground_truth) for name in names[1:]}

    print("\nAll values: q1step-style RMS (deg) vs the training ground truth "
          "(nominal-gain, uncalibrated sim).")
    print(f"{'joint':16s} {'q_real_next (actual, unscaled)':>32s} " +
          " ".join(f"{n:>32s}" for n in names[1:]))
    for jname, j, _ in TEST_JOINTS:
        if j not in highlight_joints:
            continue
        row = f"{jname:16s} {error_before[j]:32.4f} "
        row += " ".join(f"{error_after[n][j]:32.4f}" for n in names[1:])
        print(row)
    print(f"{'ALL-29 MEAN':16s} {error_before.mean():32.4f} " +
          " ".join(f"{error_after[n].mean():32.4f}" for n in names[1:]))

    print("\nDoes scaling move the PREDICTED real-under-scaled-gains state CLOSER to "
          "the training ground truth than the ACTUAL (unscaled) real state was?")
    for jname, j, _ in TEST_JOINTS:
        if j not in highlight_joints:
            continue
        for n in names[1:]:
            delta = error_after[n][j] - error_before[j]
            pct = 100 * delta / error_before[j]
            verdict = "CLOSER to training (helps)" if delta < 0 else "FARTHER from training (hurts)"
            print(f"  {jname:16s} | {n:55s} | {pct:+7.1f}%  -> {verdict}")
    for n in names[1:]:
        delta = error_after[n].mean() - error_before.mean()
        pct = 100 * delta / error_before.mean()
        verdict = "CLOSER to training (helps)" if delta < 0 else "FARTHER from training (hurts)"
        print(f"  {'ALL-29 MEAN':16s} | {n:55s} | {pct:+7.1f}%  -> {verdict}")

    print("\nInterpretation: q_sim_next[baseline] = ground truth = what the policy was")
    print("trained against (nominal Kp/Kd, no friction -- matches IsaacLab, phaseB_actuator.md")
    print("sec 3c). q_real_next = the ACTUAL logged real robot state under unscaled gains --")
    print("this is the observed sim2real gap (error_before). We have NO real hardware log")
    print("with scaled gains, so q_real_next* (what real WOULD do under scaled gains) is")
    print("PREDICTED using the Phase-B-calibrated model (best available proxy for real")
    print("dynamics/friction) driven by the scaled PD law (error_after). If error_after <")
    print("error_before, that supports scaling as bringing real deployment BACK toward the")
    print("training distribution; if error_after > error_before, scaling would predictably")
    print("push real deployment FURTHER from what the policy was trained to expect.")


if __name__ == '__main__':
    main()
