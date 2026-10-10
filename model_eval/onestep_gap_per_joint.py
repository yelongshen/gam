#!/usr/bin/env python3
"""Teacher-forced one-step sim-vs-real gap per joint, CLEAN runs only.

For every step t of every clean real run (deploy_constants.list_runs(), 9 bad runs
excluded): reset the sim to the measured (q, dq)[t] (+IMU base orientation/ang-vel when the
base floats), apply the real q_target[t] through the deploy PD law, step one control dt,
compare with the measured (q, dq)[t+1].

Reports, per joint: bias (mean err), RMSE, normalised RMSE (vs the real one-step change),
and a persistence baseline (predict q[t+1]=q[t], dq[t+1]=dq[t]) so "better than doing
nothing" is explicit. Also split by moving/stationary steps.

Usage:
  .venv_sim/bin/python model_eval/onestep_gap_per_joint.py --base float --real-armature \
      --out model_eval/onestep_gap_float_realarm.json
"""
import argparse, json, os, sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model_eval"))
sys.path.insert(0, os.path.join(ROOT, "sim2real"))
from sim2real_phaseC1b_onestep_qdq import build_model, one_step  # noqa: E402
from deploy_constants import CONTROL_DT, JOINT_NAMES, list_runs, load_run, load_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["weld", "float"], default="float")
    ap.add_argument("--real-armature", action="store_true")
    ap.add_argument("--max-runs", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=1500, help="per run (evenly spaced)")
    ap.add_argument("--after", type=int, default=None, help="keep runs dated > YYYYMMDD")
    ap.add_argument("--upto", type=int, default=None, help="keep runs dated <= YYYYMMDD")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    weld = a.base == "weld"
    model = build_model(a.real_armature, weld_base=weld)
    data = mujoco.MjData(model)
    n_sub = int(round(CONTROL_DT / model.opt.timestep))
    def run_date(path):
        top = path.split("g1_robot_data/")[1].split("/")[0]
        dg = "".join(c for c in top if c.isdigit())[-4:]
        return int("2026" + dg) if len(dg) == 4 else 0

    runs = list_runs()
    if a.after is not None:
        runs = [r for r in runs if run_date(r) > a.after]
    if a.upto is not None:
        runs = [r for r in runs if run_date(r) <= a.upto]
    runs = runs[: a.max_runs]
    QE, DE, QP, DP, MV, RID, RQ, RD = [], [], [], [], [], [], [], []
    used = 0
    for ri, r in enumerate(runs):
        d = load_run(r)
        q, dq, qt = d["q"], d["dq"], d["q_target"]
        if any(np.isnan(v).any() for v in (q, dq, qt)):
            continue
        mv = np.abs(dq).max(axis=1) > 0.05
        if mv.sum() < 200:
            continue
        lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
        q, dq, qt = q[lo:hi], dq[lo:hi], qt[lo:hi]
        bq = bw = None
        if not weld:
            try:
                bq, _ = load_csv(os.path.join(r, "base_quat.csv"), "base_q")
                bw, _ = load_csv(os.path.join(r, "base_ang_vel.csv"), "base_w")
                bq, bw = bq[lo:hi], bw[lo:hi]
            except Exception:  # noqa: BLE001
                continue
        T = len(q) - 1
        idx = np.arange(T) if T <= a.max_steps else np.linspace(0, T - 1, a.max_steps).astype(int)
        for t in idx:
            mujoco.mj_resetData(model, data)
            qn, dqn, _ = one_step(model, data, q[t], dq[t], qt[t], n_sub, weld_base=weld,
                                  base_quat=None if weld else bq[t],
                                  base_ang_vel=None if weld else bw[t])
            QE.append(qn - q[t + 1]); DE.append(dqn - dq[t + 1])
            QP.append(q[t] - q[t + 1]); DP.append(dq[t] - dq[t + 1])
            MV.append(np.abs(dq[t]).max() > 0.3); RID.append(ri)
            RQ.append(q[t + 1]); RD.append(dq[t + 1])
        used += 1
    QE, DE, QP, DP, MV, RQ, RD = map(np.asarray, (QE, DE, QP, DP, MV, RQ, RD))
    print(f"base={a.base} armature={'real' if a.real_armature else 'xml'} "
          f"runs used={used}/{len(runs)} steps={len(QE)} (moving {MV.mean():.0%})")

    def rm(x):
        return np.sqrt((x ** 2).mean(0))

    rows = []
    hdr = f"{'joint':24s}{'q bias':>8}{'q RMSE':>8}{'persist':>8}{'ratio':>7} | {'dq bias':>8}{'dq RMSE':>8}{'persist':>8}{'ratio':>7}"
    print("\nq in deg, dq in rad/s.  ratio = sim RMSE / persistence RMSE (<1: sim beats 'nothing changes')")
    print(hdr); print("-" * len(hdr))
    for j in range(29):
        r = dict(joint=JOINT_NAMES[j],
                 q_bias=float(np.degrees(QE[:, j].mean())), q_rmse=float(np.degrees(rm(QE)[j])),
                 q_persist=float(np.degrees(rm(QP)[j])),
                 dq_bias=float(DE[:, j].mean()), dq_rmse=float(rm(DE)[j]), dq_persist=float(rm(DP)[j]),
                 q_rmse_moving=float(np.degrees(rm(QE[MV])[j])), dq_rmse_moving=float(rm(DE[MV])[j]),
                 dq_rmse_still=float(rm(DE[~MV])[j]))
        # relative to the REAL value (pointwise (sim-real)/real is undefined near 0, so use
        # aggregates): rel_rmse = RMSE/RMS(real); rel_bias = mean(sim-real)/mean|real|;
        # rel_mae = sum|sim-real| / sum|real|
        for nm, E_, X_ in (("q", QE, RQ), ("dq", DE, RD)):
            r[nm + "_real_rms"] = float(np.sqrt((X_[:, j] ** 2).mean()))
            r[nm + "_real_absmean"] = float(np.abs(X_[:, j]).mean())
            r[nm + "_rel_rmse"] = float(rm(E_)[j] / max(r[nm + "_real_rms"], 1e-9))
            r[nm + "_rel_bias"] = float(E_[:, j].mean() / max(r[nm + "_real_absmean"], 1e-9))
            r[nm + "_rel_mae"] = float(np.abs(E_[:, j]).sum() / max(np.abs(X_[:, j]).sum(), 1e-9))
        r["q_ratio"] = r["q_rmse"] / max(r["q_persist"], 1e-9)
        r["dq_ratio"] = r["dq_rmse"] / max(r["dq_persist"], 1e-9)
        rows.append(r)
        print(f"{r['joint']:24s}{r['q_bias']:8.3f}{r['q_rmse']:8.3f}{r['q_persist']:8.3f}{r['q_ratio']:7.2f} | "
              f"{r['dq_bias']:8.3f}{r['dq_rmse']:8.3f}{r['dq_persist']:8.3f}{r['dq_ratio']:7.2f}")
    groups = {"legs": range(0, 12), "waist": range(12, 15), "arms": range(15, 29), "all": range(29)}
    print()
    for g, js in groups.items():
        js = list(js)
        print(f"{g:6s} q RMSE {np.degrees(np.sqrt((QE[:, js]**2).mean())):.3f} deg "
              f"(persist {np.degrees(np.sqrt((QP[:, js]**2).mean())):.3f})   "
              f"dq RMSE {np.sqrt((DE[:, js]**2).mean()):.3f} rad/s "
              f"(persist {np.sqrt((DP[:, js]**2).mean()):.3f})")
    if a.out:
        json.dump(dict(base=a.base, real_armature=a.real_armature, steps=int(len(QE)),
                       runs=used, joints=rows), open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
