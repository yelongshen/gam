import sys, json, csv
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names); fidx = S.foot_idx(names)
PELV_OFF = np.array([0.003, -0.351, 0.012])
meta = {r["clip"][:-4]: r for r in csv.DictReader(open("/home/grease/g1_robot_data/pico_raw/20261002_151150_sit_clips/clips.csv"))}
early = ["clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")]
rows = {}
for p in sorted((root / "robot").glob("*.pkl")):
    cid = p.stem.replace("pico1002sit_", "")
    rv = joblib.load(p); rv = rv[next(iter(rv))]
    pos = E.fk_positions(fk, rv)
    low = pos[:, fidx, 2].min(1); pel = pos[:, names.index("pelvis"), 2]
    dof = rv["dof"]; spd = np.abs(np.diff(dof, axis=0)).max(1) * float(rv["fps"])
    # human side
    s = joblib.load(root / "smpl" / p.name); s = s[next(iter(s))] if "smpl_joints" not in s else s
    J = np.asarray(s["smpl_joints"], float) - PELV_OFF + np.asarray(s["transl"], float)[:, None, :]
    hlow = J[:, [10, 11], 2].min(1); hpel = J[:, 0, 2]
    # first settled frame: clearance < 0.06 and speed < 3 rad/s and pelvis < 0.82
    ok = (low[1:] < 0.06) & (spd < 3.0) & (pel[1:] < 0.82)
    t0 = int(np.argmax(ok)) + 1 if ok.any() else -1
    rows[cid] = dict(low0=float(low[0]), pel0=float(pel[0]), spd0=float(spd[:3].max()), hlow0=float(hlow[0]), hpel0=float(hpel[0]), t0=t0, fps=float(rv["fps"]),
                     walk_in=float(meta[cid]["walk_in_s"]), start_s=float(meta[cid]["start_s"]), dur=float(meta[cid]["dur_s"]),
                     low_curve=[round(float(x), 3) for x in low[:60:6]], pel_curve=[round(float(x), 3) for x in pel[:60:6]])
print("clip  session_t  dur  | robot f0: clearance pelvis  speed(rad/s) | human f0: foot pelvis | first settled frame (30fps)")
for c in early:
    r = rows[c]
    print(f"{c}  {r['start_s']:7.1f}s {r['dur']:5.1f} |        {r['low0']:.3f}   {r['pel0']:.3f}  {r['spd0']:6.1f}      |      {r['hlow0']:.3f}  {r['hpel0']:.3f}     | {r['t0']}  ({r['t0'] / r['fps']:.2f} s)")
oth = [v for k, v in rows.items() if k not in early]
print("others median: robot f0 clearance %.3f pelvis %.3f speed %.1f | human f0 foot %.3f pelvis %.3f | first settled frame median %d (p90 %d)" % (
    np.median([v["low0"] for v in oth]), np.median([v["pel0"] for v in oth]), np.median([v["spd0"] for v in oth]),
    np.median([v["hlow0"] for v in oth]), np.median([v["hpel0"] for v in oth]), np.median([v["t0"] for v in oth]), np.percentile([v["t0"] for v in oth], 90)))
for c in ("clip_024", "clip_001"):
    print(c, "robot lowest foot z, frames 0..54 step 6:", rows[c]["low_curve"])
    print(c, "robot pelvis z,        frames 0..54 step 6:", rows[c]["pel_curve"])
# which other clips (not in 'early') are also unsettled at the start?
late = sorted([(k, v["t0"]) for k, v in rows.items() if k not in early and (v["t0"] < 0 or v["t0"] > 10)], key=lambda x: -x[1])
print("non-early clips with first settled frame > 10:", late[:15])
json.dump(rows, open("/tmp/spawn_rows.json", "w"))
