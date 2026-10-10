import json, sys, collections
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg())
names = list(fk.body_names)
contact = [names.index(b) for b in E.SEAT_CONTACT_BODIES if b in names]
cnt_thru, cnt_in = collections.Counter(), collections.Counter()
depth_thru, depth_in = [], []
for c in sorted(p.stem for p in (root / "robot").glob("*.pkl")):
    rv = joblib.load(root / "robot" / f"{c}.pkl"); rv = rv[next(iter(rv))]
    pos = E.fk_positions(fk, rv)
    env = json.loads((root / "envs" / f"{c}.json").read_text())
    cx, cy, top = env["seat_center"]; yaw = env["seat_yaw"]
    hx, hy = env["seat_size"][0] / 2, env["seat_size"][1] / 2; th = env["seat_size"][2]
    pel = pos[:, names.index("pelvis"), 2]
    seated = pel < pel.min() + E.SEATED_BAND
    dx, dy = pos[..., 0] - cx, pos[..., 1] - cy
    lx = np.cos(yaw) * dx + np.sin(yaw) * dy; ly = -np.sin(yaw) * dx + np.cos(yaw) * dy
    inxy = (np.abs(lx) < hx - 0.01) & (np.abs(ly) < hy - 0.01)
    inz = (pos[..., 2] < top - 0.01) & (pos[..., 2] > top - th + 0.005)
    inside = inxy & inz
    for t in np.where(~seated)[0]:
        for b in np.where(inside[t])[0]:
            cnt_thru[names[b]] += 1; depth_thru.append(top - pos[t, b, 2])
    nc = [i for i in range(len(names)) if i not in contact]
    for t in np.where(seated)[0]:
        for b in nc:
            if inside[t, b]:
                cnt_in[names[b]] += 1; depth_in.append(top - pos[t, b, 2])
print("outside seated segment: links inside slab (body: frames) ->", dict(cnt_thru.most_common(8)), " depth below top median %.3f max %.3f" % (np.median(depth_thru), np.max(depth_thru)))
print("seated, non-contact links inside slab ->", dict(cnt_in.most_common(8)), " depth median %.3f max %.3f" % (np.median(depth_in), np.max(depth_in)))
