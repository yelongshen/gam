import os

import joblib
import numpy as np

R = os.path.expanduser("~/ego_dataset")


def desc(v):
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape} {v.dtype}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]"
    if isinstance(v, dict):
        return f"dict[{len(v)}]"
    return f"{type(v).__name__}={v if not isinstance(v, str) or len(v) < 40 else v[:40]}"


def show(path, label):
    d = joblib.load(path)
    top = d
    key = None
    if isinstance(d, dict) and "dof" not in d and "pose_aa" not in d:
        key = next(iter(d))
        d = d[key]
    print(f"--- {label}: {os.path.basename(path)}  (outer key: {key})")
    for k, v in d.items():
        extra = ""
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.size:
            extra = f"  range [{v.min():.3f}, {v.max():.3f}]"
        print(f"   {k:22s} {desc(v)}{extra}")
    return d


ref_s = show(f"{R}/eval_subset/smpl/walk_180_R_003__A332_M.pkl", "eval_subset SMPL")
ref_r = show(f"{R}/eval_subset/robot/walk_180_R_003__A332_M.pkl", "eval_subset ROBOT")
la_s = show(f"{R}/lafan1_evalset/smpl/walk1_subject1.pkl", "lafan eval SMPL")
la_r = show(f"{R}/lafan1_evalset/robot_fixed/walk1_subject1.pkl", "lafan eval ROBOT (fixed)")
la_o = show(f"{R}/lafan1_evalset/robot/walk1_subject1.pkl", "lafan eval ROBOT (old)")
tr_s = show(f"{R}/lafan1_trainset/smpl/run1_subject2.pkl", "lafan train SMPL")
tr_r = show(f"{R}/lafan1_trainset/robot_fixed/run1_subject2.pkl", "lafan train ROBOT (fixed)")

print("\n=== first-frame / convention comparison (frame 0) ===")
for lbl, s, r in [("eval_subset", ref_s, ref_r), ("lafan eval", la_s, la_r), ("lafan train", tr_s, tr_r)]:
    sj = np.asarray(s["smpl_joints"])
    print(f"{lbl:12s} smpl_joints[0] pelvis={np.round(sj[0, 0], 3)}  foot z={np.round(sj[0, [10, 11], 2], 3)}"
          f"  |  transl[0]={np.round(np.asarray(s['transl'])[0], 3)}"
          f"  | robot root[0]={np.round(np.asarray(r['root_trans_offset'])[0], 3)} quat[0]={np.round(np.asarray(r['root_rot'])[0], 3)}")
print("\nSMPL fps:", ref_s.get("fps"), la_s.get("fps"), tr_s.get("fps"), "| robot fps:", ref_r.get("fps"), la_r.get("fps"), tr_r.get("fps"))
print("pose_aa shapes SMPL:", np.asarray(ref_s["pose_aa"]).shape[1:], np.asarray(la_s["pose_aa"]).shape[1:])
print("pose_aa shapes ROBOT:", np.asarray(ref_r["pose_aa"]).shape[1:], np.asarray(la_r["pose_aa"]).shape[1:])
print("smpl pose_aa nonroot absmax: eval_subset %.3f  lafan %.3f" % (
    np.abs(np.asarray(ref_s["pose_aa"]).reshape(len(ref_s["pose_aa"]), -1)[:, 3:]).max(),
    np.abs(np.asarray(la_s["pose_aa"]).reshape(len(la_s["pose_aa"]), -1)[:, 3:]).max()))
