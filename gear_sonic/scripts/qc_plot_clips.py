"""Per-clip QC plots: stature, foot height (floor contact) and frame jumps.

Reveals reconstruction problems that the activity-based chunker cannot see:
  * stature collapse   -> body-tracking lost / joints degenerate
  * foot float         -> single global floor offset is wrong for this clip
  * large frame jumps  -> tracking teleport

Usage:
    python gear_sonic/scripts/qc_plot_clips.py <clips_dir> --clips 006 018 003
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_pico_motion import quat_apply_wxyz, yup_to_zup


def clip_signals(path: str):
    d = np.load(path)
    loc = d["smpl_joints"].astype(np.float64)
    q = d["body_quat_w"].astype(np.float64)
    r = yup_to_zup(d["body_pos_w"].astype(np.float64))
    t = d["timestamp_monotonic"].reshape(-1).astype(np.float64)
    t = t - t[0]

    W = quat_apply_wxyz(q, loc) + r[:, None, :]
    W[..., 2] -= W[..., 2].min()

    foot = W[:, [10, 11], 2].min(axis=1)
    stature = W[:, 15, 2] - foot
    jump = np.concatenate([[0.0], np.linalg.norm(np.diff(W, axis=0), axis=2).max(axis=1)])
    return t, stature, foot, jump, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir")
    ap.add_argument("--clips", nargs="+", required=True, help="clip ids, e.g. 006 018")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(args.clips_dir, "qc_plots.png")
    n = len(args.clips)
    fig, axes = plt.subplots(n, 3, figsize=(15, 2.8 * n), squeeze=False)

    for row, cid in enumerate(args.clips):
        path = os.path.join(args.clips_dir, f"clip_{cid}.npz")
        t, stature, foot, jump, W = clip_signals(path)

        ax = axes[row][0]
        ax.plot(t, stature, lw=1.0, color="tab:blue")
        ax.axhline(1.5, color="g", ls="--", lw=1, label="expected ~1.5 m")
        ax.set_ylim(0, 2.0)
        ax.set_ylabel(f"clip_{cid}\nstature (m)")
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title("stature (head - lowest foot)")
            ax.legend(fontsize=7)

        ax = axes[row][1]
        ax.plot(t, foot, lw=1.0, color="tab:orange")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.axhline(0.15, color="r", ls=":", lw=1, label="float threshold")
        ax.set_ylabel("foot z (m)")
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title("lowest foot height (floor contact)")
            ax.legend(fontsize=7)

        ax = axes[row][2]
        ax.plot(t, jump, lw=0.8, color="tab:red")
        ax.set_ylabel("max joint jump (m)")
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title("per-frame max joint displacement")
        if row == n - 1:
            for a in axes[row]:
                a.set_xlabel("t (s)")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
