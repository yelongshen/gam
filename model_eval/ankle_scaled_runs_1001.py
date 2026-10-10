#!/usr/bin/env python3
"""Per-run one-step MAE on 1001 runs: runs with kp,kd x1.5 on ankle pitch (joints 4,10) vs matched
unscaled runs of the same clip. Teacher-forced, floating base + contact (see onestep_gap_per_joint.py).

For every run it simulates twice: with NOMINAL gains and with kp,kd x1.5 on joints 4 and 10, and also
regresses the measured motor torque on the nominal PD torque (slope ~1.5 => the robot really ran 1.5x).
"""
import os, sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model_eval"))
sys.path.insert(0, os.path.join(ROOT, "sim2real"))
import sim2real_phaseC1b_onestep_qdq as M  # noqa: E402
from deploy_constants import CONTROL_DT, KP, KD, load_run, load_csv  # noqa: E402

J = [4, 10]
RUNS = "/home/grease/g1_robot_data/g1_run_1001"
PAIRS = [  # (clip, unscaled run, scaled run)
    ("walk_180", "083356", "084808"),
    ("jog_ff_start_180", "083950", "085817"),
    ("dance_vouge", "085157", "085443"),
    ("high_jump", "090040", "090302"),
]
model = M.build_model(True, weld_base=False)
data = mujoco.MjData(model)
n_sub = int(round(CONTROL_DT / model.opt.timestep))
KP0, KD0 = KP.copy(), KD.copy()


def set_gains(scale):
    kp, kd = KP0.copy(), KD0.copy()
    kp[J] *= scale; kd[J] *= scale
    M.KP, M.KD = kp, kd


def run_one(rid, scale):
    set_gains(scale)
    r = os.path.join(RUNS, "20261001_" + rid)
    d = load_run(r)
    q, dq, qt, tau = d["q"], d["dq"], d["q_target"], d["tau"]
    mv = np.abs(dq).max(axis=1) > 0.05
    lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
    q, dq, qt, tau = q[lo:hi], dq[lo:hi], qt[lo:hi], tau[lo:hi]
    bq, _ = load_csv(os.path.join(r, "base_quat.csv"), "base_q")
    bw, _ = load_csv(os.path.join(r, "base_ang_vel.csv"), "base_w")
    bq, bw = bq[lo:hi], bw[lo:hi]
    QE, DE = [], []
    for t in range(len(q) - 1):
        mujoco.mj_resetData(model, data)
        qn, dqn, _ = M.one_step(model, data, q[t], dq[t], qt[t], n_sub, weld_base=False,
                                base_quat=bq[t], base_ang_vel=bw[t])
        QE.append(qn - q[t + 1]); DE.append(dqn - dq[t + 1])
    QE, DE = np.abs(np.degrees(np.asarray(QE))), np.abs(np.asarray(DE))
    # measured torque vs nominal PD torque on joints 4, 10
    pd = KP0 * (qt - q) - KD0 * dq
    slope = [float((pd[:, j] * tau[:, j]).sum() / (pd[:, j] ** 2).sum()) for j in J]
    return dict(n=len(QE), q_j=QE[:, J].mean(0), dq_j=DE[:, J].mean(0), q_all=QE.mean(), dq_all=DE.mean(),
                q_rest=np.delete(QE, J, 1).mean(), dq_rest=np.delete(DE, J, 1).mean(), slope=slope)


rows = []
for clip, a, b in PAIRS:
    for tag, rid in (("unscaled", a), ("SCALED  ", b)):
        n0 = run_one(rid, 1.0)
        n15 = run_one(rid, 1.5)
        rows.append((clip, tag, rid, n0, n15))
        print(f"{clip:18s} {tag} {rid} steps={n0['n']}  torque slope vs nominal PD (L,R ankle pitch) = "
              f"{n0['slope'][0]:.2f}, {n0['slope'][1]:.2f}", flush=True)
        for nm, r_ in (("sim nominal kp,kd", n0), ("sim kp,kd x1.5 ", n15)):
            print(f"    {nm}: ankle-pitch q MAE L/R {r_['q_j'][0]:.3f}/{r_['q_j'][1]:.3f} deg  "
                  f"dq MAE L/R {r_['dq_j'][0]:.3f}/{r_['dq_j'][1]:.3f} rad/s | other joints q {r_['q_rest']:.3f} dq {r_['dq_rest']:.3f}",
                  flush=True)
