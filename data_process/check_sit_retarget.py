import os, sys, glob
os.environ["VIS_SMPL_DIR"] = "/home/grease/ego_dataset/picoset_20261002_sit/smpl"
os.environ["VIS_ROBOT_DIR"] = "/home/grease/GMR/picoset_20261002_sit_retargeted_g1_fps30"
sys.path.insert(0, "/home/grease/gam/gear_sonic/scripts")
import numpy as np
from visualize_smpl_vs_robot import load_smpl, load_robot

names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.environ["VIS_SMPL_DIR"] + "/*.pkl"))
rows = []
for n in names:
    S, sf = load_smpl(n)
    R, nm, rf = load_robot(n)
    k = int(min(len(S) / sf, len(R) / rf) * 30)
    si = np.minimum((np.arange(k) * sf / 30).astype(int), len(S) - 1)
    ri = np.minimum((np.arange(k) * rf / 30).astype(int), len(R) - 1)
    S = S[si].copy(); R = R[ri].copy()
    S[..., 2] -= np.percentile(S[:, [10, 11], 2].min(1), 5)
    toe = [nm.index("left_toe_link"), nm.index("right_toe_link")]
    R[..., 2] -= np.percentile(R[:, toe, 2].min(1), 5)
    sp = S[:, 0, 2]; rp = R[:, nm.index("pelvis"), 2]
    seated = sp < 0.87
    sxy = np.linalg.norm(np.diff(S[:, 0, :2], axis=0), axis=1).sum(); rxy = np.linalg.norm(np.diff(R[:, nm.index("pelvis"), :2], axis=0), axis=1).sum()
    # feet height while seated (robot): feet should stay on the floor
    ft = R[:, toe, 2].min(1)
    rows.append((n, k / 30, float(np.median(sp[seated])) if seated.any() else np.nan, float(np.median(rp[seated])) if seated.any() else np.nan,
                 float(np.median(sp[~seated])), float(np.median(rp[~seated])), rxy / max(sxy, 1e-9),
                 float(np.corrcoef(S[:, 0, 0], R[:, nm.index("pelvis"), 0])[0, 1]), float(np.percentile(ft, 95))))
a = np.array([r[1:] for r in rows], dtype=float)
print(f"{len(rows)} clips")
print("seated pelvis z  smpl median %.3f   robot median %.3f   diff (robot-smpl) mean %.3f  max|diff| %.3f" % (np.nanmedian(a[:, 1]), np.nanmedian(a[:, 2]), np.nanmean(a[:, 2] - a[:, 1]), np.nanmax(np.abs(a[:, 2] - a[:, 1]))))
print("standing pelvis z smpl median %.3f   robot median %.3f   diff mean %.3f" % (np.median(a[:, 3]), np.median(a[:, 4]), np.mean(a[:, 4] - a[:, 3])))
print("root xy path ratio robot/smpl: median %.2f  min %.2f max %.2f" % (np.median(a[:, 5]), a[:, 5].min(), a[:, 5].max()))
print("x corr smpl vs robot: median %.3f  min %.3f" % (np.median(a[:, 6]), a[:, 6].min()))
print("robot toe height p95 (m): median %.3f max %.3f" % (np.median(a[:, 7]), a[:, 7].max()))
bad = [(rows[i][0], round(a[i, 2] - a[i, 1], 3), round(a[i, 5], 2), round(a[i, 6], 2)) for i in range(len(rows))
       if abs(a[i, 2] - a[i, 1]) > 0.1 or a[i, 5] < 0.8 or a[i, 5] > 1.25 or a[i, 6] < 0.9]
print("outliers (seated z diff>0.1 | path ratio off | x corr<0.9):", bad[:15], "... total", len(bad))
