#!/usr/bin/env python3
"""Explore a consolidated sitting session: pelvis height, root speed, sit episodes.

Prints a timeline of low-pelvis (seated) intervals so the chunker thresholds can be chosen from data.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_pico_motion import quat_apply_wxyz, yup_to_zup  # noqa: E402

PELVIS, L_FOOT, R_FOOT = 0, 10, 11


def world(path):
    d = np.load(path, allow_pickle=True)
    loc = d["smpl_joints"].astype(np.float64)
    q = d["body_quat_w"].astype(np.float64)
    r = yup_to_zup(d["body_pos_w"].astype(np.float64))
    W = quat_apply_wxyz(q, loc) + r[:, None, :]
    W[..., 2] -= np.percentile(W[:, [L_FOOT, R_FOOT], 2].min(axis=1), 5)
    return W, d["_t"].astype(np.float64)


def smooth(x, n):
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


if __name__ == "__main__":
    W, t = world(sys.argv[1])
    T = len(t)
    fps = T / t[-1]
    pz = W[:, PELVIS, 2]
    pxy = W[:, PELVIS, :2]
    print(f"frames {T}  duration {t[-1]:.1f}s ({t[-1] / 60:.1f} min)  fps {fps:.1f}")
    print(f"pelvis z percentiles 1/5/25/50/75/95/99: {np.percentile(pz, [1, 5, 25, 50, 75, 95, 99]).round(3)}")
    sp = np.linalg.norm(np.diff(smooth(pxy[:, 0], 15)[:, None].repeat(1, 1), axis=0), axis=1)
    px, py = smooth(pxy[:, 0], 25), smooth(pxy[:, 1], 25)
    v = np.hypot(np.diff(px), np.diff(py)) / np.maximum(np.diff(t), 1e-3)
    v = np.concatenate([[0], v])
    print(f"root speed (smoothed) percentiles 50/75/90/99: {np.percentile(v, [50, 75, 90, 99]).round(3)}")
    # histogram of pelvis height
    h, e = np.histogram(pz, bins=np.arange(0, 1.3, 0.05))
    print("pelvis z histogram (0.05 m bins):")
    for c, lo in zip(h, e[:-1]):
        print(f"  {lo:4.2f}-{lo + 0.05:4.2f} {c:7d} {100 * c / T:5.1f}% " + "#" * int(60 * c / h.max()))
    # timeline at 1 s resolution: pelvis z (median) + speed
    step = int(round(fps))
    print("\n sec   pel_z  speed   (every 2 s)")
    for i in range(0, T - step, 2 * step):
        print(f" {t[i]:6.1f}  {np.median(pz[i:i + step]):.2f}  {np.median(v[i:i + step]):.2f}")
