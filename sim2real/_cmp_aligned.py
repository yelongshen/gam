import os

import joblib

R = os.path.expanduser("~/ego_dataset")
print("datasets:", sorted(d for d in os.listdir(R) if "ICL" in d))
for n in ["dance1_subject1__w100s", "CMU__CMU__141__141_15_stageii", "flip_090_003__A304_M"]:
    for ds in ["ICL_hardset", "ICL_hardset_aligned", "ICL_hardset_rep3"]:
        p = f"{R}/{ds}/smpl/{n}.pkl"
        if not os.path.exists(p):
            print(f"{ds:20s} {n[:30]:30s} (missing)")
            continue
        s = joblib.load(p)
        r = joblib.load(f"{R}/{ds}/robot/{n}.pkl")
        r = r[next(iter(r))]
        print(f"{ds:20s} {n[:30]:30s} smpl frames {len(s['transl'])} fps {s.get('fps')} | robot frames {len(r['dof'])} fps {r.get('fps')} | smpl keys {list(s)[:6]}")
