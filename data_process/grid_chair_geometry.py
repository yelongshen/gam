"""Grid-search the seat slab size / back offset so that the robot's links never pass through the seat
(knees during sit-down / stand-up) while the pelvis and hip links stay supported while seated.

Run with env_isaaclab python.   Writes nothing; prints the table.
"""
import csv, json, sys
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names)
contact = [names.index(b) for b in E.SEAT_CONTACT_BODIES if b in names]
support = [names.index(b) for b in ("pelvis", "left_hip_roll_link", "right_hip_roll_link") if b in names]
nc = [i for i in range(len(names)) if i not in contact]
params = {r["clip"]: r for r in csv.DictReader(open(root / "envs" / "chair_params.csv"))}
data = []
for c in sorted(p.stem for p in (root / "robot").glob("*.pkl")):
    rv = joblib.load(root / "robot" / f"{c}.pkl"); rv = rv[next(iter(rv))]
    pos = E.fk_positions(fk, rv)
    env = json.loads((root / "envs" / f"{c}.json").read_text())
    pel = pos[:, names.index("pelvis"), 2]
    seated = pel < pel.min() + E.SEATED_BAND
    data.append((c, pos, seated, env["seat_center"][2], float(params[c]["yaw"]), float(params[c]["seat_x"]), float(params[c]["seat_y"])))


def evaluate(w, b, th=0.05, wy=0.45):
    n_thru = n_in = 0
    sup = []
    for c, pos, seated, top, yaw, sx0, sy0 in data:
        cx = sx0 - b * np.cos(yaw); cy = sy0 - b * np.sin(yaw)
        dx, dy = pos[..., 0] - cx, pos[..., 1] - cy
        lx = np.cos(yaw) * dx + np.sin(yaw) * dy; ly = -np.sin(yaw) * dx + np.cos(yaw) * dy
        h = w / 2; hyw = wy / 2
        inxy = (np.abs(lx) < h - 0.01) & (np.abs(ly) < hyw - 0.01)
        inz = (pos[..., 2] < top - 0.01) & (pos[..., 2] > top - th + 0.005)
        inside = inxy & inz
        n_thru += int(inside[~seated].any(axis=1).sum() > 3)
        n_in += int(inside[seated][:, nc].any(axis=1).sum() > 3)
        # support: pelvis + hip links over the slab (xy margin 2 cm) for the seated frames
        sup_xy = (np.abs(lx[:, support]) < h - 0.02) & (np.abs(ly[:, support]) < hyw - 0.02)
        sup.append(float(sup_xy[seated].all(axis=1).mean()))
    sup = np.array(sup)
    return n_thru, n_in, float(np.median(sup)), float(sup.min()), int((sup < 0.95).sum())


print("rectangular slab: depth (along facing) x width 0.45 m ; thickness th")
print(f"{'depth':>6s}{'back':>6s}{'th':>6s} | {'THROUGH':>8s}{'IN-seated':>10s}{'support med':>12s}{'support min':>12s}{'<95%':>6s}")
for th in (0.05, 0.03):
    for w in (0.45, 0.40, 0.35, 0.30):
        for b in (0.0, 0.05, 0.10):
            t, i, m, mn, nb = evaluate(w, b, th=th)
            print(f"{w:6.2f}{b:6.2f}{th:6.2f} | {t:8d}{i:10d}{m:12.3f}{mn:12.3f}{nb:6d}")
