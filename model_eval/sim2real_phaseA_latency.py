"""
Phase A of the sim2real gap study: LATENCY IDENTIFICATION.

Tan et al. 2018 (arXiv:1804.10332) identify latency as a primary cause of
sim-to-real failure for position-controlled robots, and simulate it explicitly.
This measures it directly from the paired logs:

  A1  ACTUATOR / TRACKING lag    q_target[j] -> q[j]      (robot data only)
  A2  END-TO-END lag             human SMPL -> robot motion

METHOD NOTES (each prevents a specific observed failure)
--------------------------------------------------------
1. `action` is NOT comparable to `q` as logged: it is the raw policy output in
   IsaacLab order, while `q` is hardware order with `default_angles` added back.
   g1_params.action_to_q_target() moves median|q_target-q| 0.667 -> 0.056 rad
   and corr -0.015 -> 0.912.
2. Signals are HIGH-PASS filtered (first difference) before cross-correlation;
   raw joint angles are dominated by slow drift, giving ambiguous rail-pinned peaks.
3. Lag search bounded to +/-400 ms; peaks ON the rail are flagged unreliable.
4. Joints with low command excitation or weak peaks are excluded from aggregates.

Usage:
  .venv_sim/bin/python model_eval/sim2real_phaseA_latency.py --session aug11
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_process'))
from load_sim2real_session import load_session
from g1_params import action_to_q_target, JOINT_NAMES, GROUPS

FS = 50.0
MAX_LAG_S = 0.40


def resample(t, v, grid):
    v = np.asarray(v)
    if v.ndim == 1:
        v = v[:, None]
    return np.stack([np.interp(grid, t, v[:, j]) for j in range(v.shape[1])], 1)


def xcorr(a, b, fs=FS, max_lag_s=MAX_LAG_S, highpass=True):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if highpass:
        a, b = np.diff(a), np.diff(b)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = int(max_lag_s * fs)
    lags = np.arange(-n, n + 1)
    cs = np.empty(len(lags))
    for i, L in enumerate(lags):
        if L >= 0:
            x, y = a[L:], (b[:len(b) - L] if L else b)
        else:
            x, y = a[:L], b[-L:]
        m = min(len(x), len(y))
        cs[i] = float(np.dot(x[:m], y[:m]) / m) if m else -1.0
    k = int(np.argmax(cs))
    lag = lags[k] / fs
    if 0 < k < len(cs) - 1:
        d = cs[k - 1] - 2 * cs[k] + cs[k + 1]
        if abs(d) > 1e-12:
            lag = (lags[k] + 0.5 * (cs[k - 1] - cs[k + 1]) / d) / fs
    return lag, cs[k], (k == 0 or k == len(cs) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='aug11')
    ap.add_argument('--chunk', type=float, default=20.0)
    ap.add_argument('--min_corr', type=float, default=0.30)
    args = ap.parse_args()

    S = load_session(args.session)
    r, h = S['robot'], S['human']
    lo, hi = S['overlap']
    grid = np.arange(lo, hi, 1 / FS)

    print("=== Phase A - latency - session %s ===" % args.session)
    print("usable window (Mode-2 AND human present): %.1f s, %d samples @ %.0f Hz"
          % (hi - lo, len(grid), FS))

    q = resample(r['t'], r['q'], grid)
    qt = resample(r['t'], action_to_q_target(r['action']), grid)

    cz = np.array([np.corrcoef(qt[:, j], q[:, j])[0, 1] for j in range(29)])
    print("reconstruction check: median corr(q_target,q) = %.3f, median|q_target-q| = %.4f rad\n"
          % (np.nanmedian(cz), np.median(np.abs(qt - q))))

    print('--- A1  actuator lag:  q_target[j] -> q[j] ---')
    print("%-16s %9s %7s %9s  %s" % ('joint', 'lag(ms)', 'corr', 'cmd_std', 'note'))
    rows = []
    for j in range(29):
        cmd_std = qt[:, j].std()
        lag, c, rail = xcorr(qt[:, j], q[:, j])
        if cmd_std < 0.01:
            note = 'low-excitation'
        elif rail:
            note = 'ON RAIL (unreliable)'
        elif c < args.min_corr:
            note = 'weak corr'
        else:
            note = ''
            rows.append((j, lag, c))
        print("%-16s %9.1f %7.3f %9.4f  %s" % (JOINT_NAMES[j], 1000 * lag, c, cmd_std, note))

    if rows:
        L = np.array([x[1] for x in rows]) * 1000
        print("\nreliable joints: %d/29" % len(rows))
        print("%-12s %3s %15s %12s" % ('group', 'n', 'median lag(ms)', 'median corr'))
        for g, idx in GROUPS.items():
            sel = [x for x in rows if x[0] in idx]
            if sel:
                print("%-12s %3d %15.1f %12.3f"
                      % (g, len(sel), 1000 * np.median([x[1] for x in sel]),
                         np.median([x[2] for x in sel])))
        print("\nACTUATOR LAG  median = %.1f ms   IQR = [%.1f, %.1f] ms"
              % (np.median(L), np.percentile(L, 25), np.percentile(L, 75)))
        print("              = %.2f control steps @ %.0f Hz" % (np.median(L) / (1000 / FS), FS))

    print('\n--- A2  end-to-end lag:  human SMPL motion -> robot motion ---')
    hj = resample(h['t'], h['smpl_joints'].reshape(len(h['t']), -1), grid)
    hj = hj.reshape(len(grid), 24, 3)
    hum = np.linalg.norm(np.diff(hj[:, [15, 18, 19, 20, 21], :], axis=0), axis=2).sum(1)
    rob = np.abs(np.diff(q[:, 15:29], axis=0)).sum(1)
    lag, c, rail = xcorr(hum, rob, max_lag_s=1.0, highpass=False)
    print("whole window: lag = %7.1f ms   corr = %.3f%s"
          % (1000 * lag, c, '   ON RAIL (unreliable)' if rail else ''))

    n = int(args.chunk * FS)
    est = []
    for s in range(0, len(hum) - n, n):
        a, b = hum[s:s + n], rob[s:s + n]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        L_, c_, rail_ = xcorr(a, b, max_lag_s=1.0, highpass=False)
        if c_ >= args.min_corr and not rail_:
            est.append((L_ * 1000, c_))
    if est:
        L = np.array([e[0] for e in est])
        C = np.array([e[1] for e in est])
        print("\nper-%.0fs chunks: %d usable (corr>=%.2f, off-rail)" % (args.chunk, len(est), args.min_corr))
        print("  median %7.1f ms | mean %7.1f | std %6.1f" % (np.median(L), L.mean(), L.std()))
        print("  p10    %7.1f ms | p90  %7.1f" % (np.percentile(L, 10), np.percentile(L, 90)))
        print("  median chunk corr = %.3f" % np.median(C))
        print("\n  JITTER (std) = %.1f ms - matters as much as the mean" % L.std())
    else:
        print('\n  no usable chunks - end-to-end lag NOT established')


if __name__ == '__main__':
    main()
