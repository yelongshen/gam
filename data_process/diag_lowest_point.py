import sys, csv
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, "/home/grease/gam/data_process")
import snap_robot_floor_check_chairs as S
E = S.E
root = Path("/home/grease/ego_dataset/picoset_20261002_sit")
fk = E.Humanoid_Batch(E._Cfg()); names = list(fk.body_names)
early = {"clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")}
meta = {r["clip"][:-4]: r for r in csv.DictReader(open("/home/grease/g1_robot_data/pico_raw/20261002_151150_sit_clips/clips.csv"))}
print("clip   seat_top  | where is the lowest body point of the clip? (phase) which body | min z over STAND frames | min z over SEATED frames")
rows = []
for p in sorted((root / "robot").glob("*.pkl")):
    cid = p.stem.replace("pico1002sit_", "")
    rv = joblib.load(p); rv = rv[next(iter(rv))]
    pos = E.fk_positions(fk, rv)                      # (T, B, 3) world
    zmin_t = pos[..., 2].min(1); pel = pos[:, names.index("pelvis"), 2]
    seated = pel < pel.min() + 0.06; stand = pel > 0.74
    t = int(np.argmin(zmin_t)); b = names[int(np.argmin(pos[t, :, 2]))]
    ph = "SEATED" if seated[t] else ("standing" if stand[t] else "transition")
    rows.append((cid, ph, b, float(zmin_t[stand].min()) if stand.any() else np.nan, float(zmin_t[seated].min())))
for cid, ph, b, zs, zq in rows:
    if cid in early:
        print(f"{cid}  {'EARLY':5s} | {ph:10s} {b:22s} | {zs:.3f} | {zq:.3f}")
oth = [r for r in rows if r[0] not in early]
import collections
print("others: phase of the lowest point ->", dict(collections.Counter(r[1] for r in oth)), "| median min z over stand frames %.3f, seated %.3f" % (np.nanmedian([r[3] for r in oth]), np.median([r[4] for r in oth])))
print("early : phase of the lowest point ->", dict(collections.Counter(r[1] for r in rows if r[0] in early)))
