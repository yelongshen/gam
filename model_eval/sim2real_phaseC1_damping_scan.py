"""
Phase C.1 -- damping coefficient scan for the one-step prediction test.

Sweeps a range of `dof_damping` values (Nm*s/rad) for one or more target
joints (independently, holding all others at their current best/zero value)
and reports the one-step-ahead position RMS error (q1step, degrees) for each
candidate value, so the minimum can be read off directly instead of guessed.

USAGE
-----
  .venv_sim/bin/python model_eval/sim2real_phaseC1_damping_scan.py \\
      --joints 4 10 14 --lo 0.0 --hi 2.0 --steps 11 --duration 439.6

  # scan only the ankle pitch joints over a finer grid:
  .venv_sim/bin/python model_eval/sim2real_phaseC1_damping_scan.py \\
      --joints 4 10 --lo 0.8 --hi 1.6 --steps 17
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_process'))
from load_sim2real_session import load_session          # noqa: E402
from g1_params import action_to_q_target, KPS, KDS, JOINT_NAMES  # noqa: E402

# reuse the exact same helpers as the main C.1 script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2real_phaseC1_onestep_prediction import (   # noqa: E402
    build_model, one_step_predict, FS, MJ_JOINT_NAMES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='aug11')
    ap.add_argument('--duration', type=float, default=439.6)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--joints', type=int, nargs='+', default=[4, 10, 14],
                     help='joint indices (into JOINT_NAMES / MJ_JOINT_NAMES) to scan, one at a time')
    ap.add_argument('--lo', type=float, default=0.0, help='lowest damping value to try')
    ap.add_argument('--hi', type=float, default=2.0, help='highest damping value to try')
    ap.add_argument('--steps', type=int, default=11, help='number of grid points between lo and hi')
    ap.add_argument('--fixed', type=str, default='',
                     help="comma-separated j:b pairs to hold fixed on OTHER joints while scanning, "
                          "e.g. '4:1.19,10:1.09,14:0.98' (the joint currently being scanned is "
                          "excluded from this even if listed)")
    args = ap.parse_args()

    fixed = {}
    if args.fixed:
        for pair in args.fixed.split(','):
            j, b = pair.split(':')
            fixed[int(j)] = float(b)

    S = load_session(args.session)
    r = S['robot']
    lo, hi = S['overlap']
    grid = np.arange(lo + args.start, min(hi, lo + args.start + args.duration), 1 / FS)
    print(f"=== Phase C.1 damping scan - session {args.session} ===")
    print(f"testing {len(grid)/FS:.1f}s ({len(grid)} steps) per candidate\n")

    def resample(t, v):
        return np.stack([np.interp(grid, t, v[:, j]) for j in range(v.shape[1])], 1)

    q_real = resample(r['t'], r['q'])
    dq_real = resample(r['t'], r['dq'])
    qt_real = resample(r['t'], action_to_q_target(r['action']))

    q_real, dq_real, qt_real = q_real[:-1], dq_real[:-1], qt_real[:-1]
    q_real_next = resample(r['t'], r['q'])[1:len(q_real) + 1]

    b_values = np.linspace(args.lo, args.hi, args.steps)

    for j in args.joints:
        jname = JOINT_NAMES[j]
        others = {k: v for k, v in fixed.items() if k != j}
        print(f"--- scanning joint {j} ({jname}), fixed others = {others or 'none'} ---")
        print(f"{'b (Nm*s/rad)':>14s} {jname+' q1step(deg)':>26s}  {'all-29 mean(deg)':>18s}")
        best_b, best_err = None, np.inf
        for b in b_values:
            extra = dict(others)
            extra[j] = float(b)
            model = build_model(extra)
            q_sim_next, _ = one_step_predict(model, q_real, dq_real, qt_real, 1 / FS)
            state_rms = np.degrees(np.sqrt(((q_sim_next - q_real_next) ** 2).mean(0)))
            err_j = state_rms[j]
            mean29 = state_rms.mean()
            flag = ''
            if err_j < best_err:
                best_err, best_b = err_j, b
                flag = '  <-- best so far'
            print(f"{b:14.3f} {err_j:26.4f}  {mean29:18.4f}{flag}")
        print(f"==> best b for {jname}: {best_b:.3f} (q1step={best_err:.4f} deg)\n")


if __name__ == '__main__':
    main()
