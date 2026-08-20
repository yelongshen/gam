"""
Visualize random QC-FAILED (low-quality) motion sequences as 3D skeleton MP4s.

Reads motion_qc.csv, samples N failed clips (qc_pass==0), runs the same FK as
the classifier, and renders each with its failure reason(s) in the title so you
can eyeball WHY it was filtered (foot_skate / static / floating / jitter / ...).

Usage:
    .venv_sim/bin/python visualize_low_quality.py                # 10 random
    .venv_sim/bin/python visualize_low_quality.py --n 10 --reason foot_skate
"""
import os
import csv
import argparse
import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_process"))
import classify_motions as C  # reuse FK + loader (returns meters, Z-up)

OUT_DIR = "/home/grease/gam/data/evaluation_visualization_set/low_quality"
os.makedirs(OUT_DIR, exist_ok=True)

SMPL_LINKS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]


def render(joints, title, out_file):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    T = joints.shape[0]
    stride = max(1, T // 200)
    frames = range(0, min(T, 600), stride)

    # fixed floor for context
    floor = np.percentile(joints[:, C.FOOT_JOINTS, 2], 5)

    def update(fi):
        ax.clear()
        p = joints[fi]
        m = ~np.all(p == 0, axis=1)
        ax.scatter(p[m, 0], p[m, 1], p[m, 2], c='red', s=6)
        for a, b in SMPL_LINKS:
            pa, pb = p[a], p[b]
            if not (np.all(pa == 0) or np.all(pb == 0)):
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                        c='darkred', linewidth=1.2)
        # draw floor plane reference
        c = p[0]
        r = 1.0
        xx, yy = np.meshgrid(np.linspace(c[0]-r, c[0]+r, 2),
                             np.linspace(c[1]-r, c[1]+r, 2))
        ax.plot_surface(xx, yy, np.full_like(xx, floor), alpha=0.12, color='gray')
        ax.set_xlim(c[0]-r, c[0]+r)
        ax.set_ylim(c[1]-r, c[1]+r)
        ax.set_zlim(floor-0.1, floor+2.0)
        ax.set_axis_off()
        ax.set_title(title, fontsize=8)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=40)
    fps = max(5, 30 // stride)
    ani.save(out_file, writer='ffmpeg', fps=fps, dpi=100)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qc_csv', default='data_analysis/motion_qc/motion_qc.csv')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--reason', default='', help='only clips failing this reason')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.qc_csv)))
    failed = [r for r in rows if int(r['qc_pass']) == 0 and r['fail_reasons']]
    if args.reason:
        failed = [r for r in failed
                  if args.reason in r['fail_reasons'].split(';')]
    print(f"{len(failed)} failed clips available"
          + (f" for reason '{args.reason}'" if args.reason else "") + ".")

    random.seed(args.seed)
    picks = random.sample(failed, min(args.n, len(failed)))

    for i, r in enumerate(picks):
        path = r['path']
        abspath = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        name = os.path.basename(path)
        reasons = r['fail_reasons']
        print(f"[{i+1}/{len(picks)}] {name}  FAIL={reasons}")
        try:
            j = C.load_joints(abspath)
            if j is None:
                print("   (could not load, skipping)")
                continue
            metrics = (f"skate={float(r['foot_skate']):.2f} "
                       f"float={float(r['float_gap']):.2f} "
                       f"jerk={float(r['max_jerk']):.0f} "
                       f"spd={float(r['max_speed']):.1f} "
                       f"mspd={float(r['mean_speed']):.3f}")
            title = f"FAIL[{reasons}]\n{r['category']} | {name}\n{metrics}"
            safe = name.replace('.npz', '').replace('.bvh', '')
            out = os.path.join(OUT_DIR, f"LOWQ_{reasons.replace(';','_')}_{safe}.mp4")
            render(j, title, out)
            print(f"   saved {os.path.basename(out)}")
        except Exception as e:
            print(f"   FAILED to render: {e}")

    print(f"\nDone. Videos in: {OUT_DIR}")


if __name__ == "__main__":
    main()
