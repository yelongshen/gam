#!/usr/bin/env python3
"""FFT of dq during a quiet (steady-state) window, to check whether idle
jitter is a coherent oscillation (limit cycle) or broadband noise."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.g1_params import JOINT_NAMES  # noqa: E402
from model_eval.sim2real_single_run_quicklook import load  # noqa: E402


def main():
    run_dir = sys.argv[1]
    t0 = float(sys.argv[2])
    t1 = float(sys.argv[3])

    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    dt = meta["logging"]["dt"]
    fs = 1.0 / dt

    dq, t = load(run_dir, "dq")
    q, _ = load(run_dir, "q")
    n = min(len(dq), len(q))
    dq, q, t = dq[:n], q[:n], t[:n]
    t_s = (t - t[0]) / 1000.0

    m = (t_s >= t0) & (t_s < t1)
    watch = ["R_knee", "L_knee", "waist_roll", "waist_pitch", "L_hip_pitch"]
    print(f"window [{t0},{t1})s, n={m.sum()} samples, fs={fs:.1f} Hz")
    for nm in watch:
        j = JOINT_NAMES.index(nm)
        sig = dq[m, j] - dq[m, j].mean()
        if len(sig) < 8:
            continue
        freqs = np.fft.rfftfreq(len(sig), d=dt)
        mag = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        # ignore DC/near-DC
        k = freqs > 0.3
        top = np.argsort(-mag[k])[:3]
        peak_freqs = freqs[k][top]
        peak_mags = mag[k][top] / mag[k].sum() * 100
        rms = np.sqrt(np.mean(sig ** 2))
        peaks_str = ", ".join(f"{f:.2f}Hz({p:.0f}%)" for f, p in zip(peak_freqs, peak_mags))
        print(f"  {nm:12s} rms(dq)={rms:6.3f} rad/s  top peaks: {peaks_str}")


if __name__ == "__main__":
    raise SystemExit(main())
