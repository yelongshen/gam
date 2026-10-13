"""Static whole-clip quality screen (independent of sim success): human posture,
human-vs-robot lean mismatch, floating/penetrating feet, pelvis height.
env_isaaclab python:  python data_process/screen_picoset_20261003_quality.py [set]"""
import sys, json
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
ds = sys.argv[1] if len(sys.argv) > 1 else "picoset_20261003"
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
root = Path("/home/grease/ego_dataset") / ds
def lean(v): return np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1)))
rows = []
for p in sorted((root / "robot").glob("*.pkl")):
    rv = joblib.load(p); rv = rv[next(iter(rv))]
    sm = joblib.load(root / "smpl" / p.name); sm = sm[next(iter(sm))] if "smpl_joints" not in sm else sm
    J = np.asarray(sm["smpl_joints"], float)
    hl = lean(J[:, 9] - J[:, 0])
    pos = E.fk_positions(fk, rv)
    rl = lean(pos[:, names.index("torso_link")] - pos[:, names.index("pelvis")])
    low = pos[:, fidx, 2].min(1); pel = pos[:, names.index("pelvis"), 2]
    spd = np.abs(np.diff(rv["dof"], axis=0)).max(1) * float(rv["fps"])
    rows.append(dict(clip=p.stem[7:], T=len(low), hl_med=float(np.median(hl)), hl_max=float(hl.max()),
        rl_med=float(np.median(rl)), lean_gap=float(np.median(rl - np.interp(np.linspace(0, 1, len(rl)), np.linspace(0, 1, len(hl)), hl))),
        foot_min=float(low.min()), foot_p5=float(np.percentile(low, 5)), pel_min=float(pel.min()), pel_med=float(np.median(pel)),
        spd_max=float(spd.max()), spd_p99=float(np.percentile(spd, 99))))
flags = {
 "human lean>60 (lying/crawl)": lambda r: r["hl_med"] > 60,
 "robot-human lean gap>20": lambda r: abs(r["lean_gap"]) > 20,
 "feet floating (p5 > 0.10m)": lambda r: r["foot_p5"] > 0.10,
 "feet penetrating (min < -0.03m)": lambda r: r["foot_min"] < -0.03,
 "pelvis too low (<0.35m)": lambda r: r["pel_min"] < 0.35,
 "joint speed spike (>25 rad/s)": lambda r: r["spd_max"] > 25,
 "very short (<2s @30)": lambda r: r["T"] < 60,
}
print(f"{ds}: {len(rows)} clips")
for k, f in flags.items():
    print(f"  {k}: {[ (r['clip'], ) [0] for r in rows if f(r)]}")
json.dump(rows, open("/tmp/quality_1003.json", "w"))
