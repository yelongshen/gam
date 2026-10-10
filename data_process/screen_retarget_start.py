"""Static screen of a retargeted dataset for the 'bowed-forward IK' signature found in picoset_20261002_sit:
torso lean 28-62 deg in the first 2 s, lowest ankle 0.10-0.17 m above the floor at frame 0, first-frame reference joint speed 9-25 rad/s.

Run with env_isaaclab python:
  python data_process/screen_retarget_start.py <name> [<name> ...]     e.g. picoset_20260928 picoset_20260929 picoset_20260930
"""
import json, sys
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
out = {}
for ds in sys.argv[1:]:
    rows = []
    for p in sorted((Path("/home/grease/ego_dataset") / ds / "robot").glob("*.pkl")):
        rv = joblib.load(p); rv = rv[next(iter(rv))]
        fps = float(rv["fps"]); n = int(2.0 * fps)
        if len(rv["dof"]) < 12:
            continue
        pos = E.fk_positions(fk, rv)
        v = pos[:n, names.index("torso_link")] - pos[:n, names.index("pelvis")]
        lean = float(np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))).mean())
        low = pos[:, fidx, 2].min(1)
        spd = np.abs(np.diff(rv["dof"], axis=0)).max(1) * fps
        dof = rv["dof"][:n]
        rows.append(dict(clip=p.stem, T=len(low), lean=lean, f0=float(low[0]), spd0=float(spd[:3].max()),
                         waist_pitch=float(np.degrees(dof[:, 14].mean())), rhip=float(np.degrees(dof[:, 6].mean())),
                         pelvis0=float(pos[0, names.index("pelvis"), 2])))
    out[ds] = rows
    a = lambda k: np.array([r[k] for r in rows])
    bad = [r for r in rows if r["lean"] > 25 or r["spd0"] > 6 or r["f0"] > 0.10]
    print(f"== {ds}: {len(rows)} clips | lean first 2 s: median {np.median(a('lean')):.1f} p90 {np.percentile(a('lean'), 90):.1f} max {a('lean').max():.1f} deg"
          f" | frame-0 foot z: median {np.median(a('f0')):.3f} p90 {np.percentile(a('f0'), 90):.3f} max {a('f0').max():.3f}"
          f" | first-frame speed: median {np.median(a('spd0')):.1f} p90 {np.percentile(a('spd0'), 90):.1f} max {a('spd0').max():.1f} rad/s")
    print(f"   clips with lean > 25 deg: {sum(r['lean'] > 25 for r in rows)}   f0 foot > 0.10 m: {sum(r['f0'] > 0.10 for r in rows)}   first-frame speed > 6 rad/s: {sum(r['spd0'] > 6 for r in rows)}   any: {len(bad)}")
    print("   worst:", [(r["clip"][-14:], round(r["lean"]), round(r["f0"], 3), round(r["spd0"], 1)) for r in sorted(rows, key=lambda r: -r["lean"])[:6]])
json.dump(out, open("/tmp/screen_retarget_start.json", "w"))
