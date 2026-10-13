import os, sys
import numpy as np
sys.path.insert(0, "/home/grease/gam/model_eval")
import still_step_diagnosis as S
import world_model_qdq as W
import sim2real_phaseC1b_onestep_qdq as M
from deploy_constants import JOINT_NAMES

mujoco = M.mujoco
model = M.build_model(True, weld_base=False)
data = mujoco.MjData(model)
J = [0, 1, 4, 6, 10, 14, 16, 23, 17, 24]
paths = []
tr, va, vb = W.split_runs(0)
for tag, ps in (("train", tr), ("valA", va), ("valB", vb)):
    paths += [(tag, p) for p in ps]
print("run                tag    n   " + " ".join(f"{JOINT_NAMES[j][:12]:>12s}" for j in J))
rows = []
for tag, p in paths:
    s = S.still_samples(p, model, data, 0.10, 300)
    if s is None:
        continue
    name = f"{W.run_date(p)}/{W.run_id(p)}"
    rows.append((name, tag, len(s["r"]), [s["r"][:, j].mean() for j in J], [s["r"][:, j].std() for j in J]))
    print(f"{name:18s} {tag:5s} {len(s['r']):4d} " + " ".join(f"{m:12.2f}" for m in rows[-1][3]))
# between-run vs within-run spread of the mean r
print("\nbetween-run std of mean r  vs  mean within-run std  (Nm), per joint")
for k, j in enumerate(J):
    means = np.array([r[3][k] for r in rows]); wst = np.array([r[4][k] for r in rows])
    print(f"  {JOINT_NAMES[j]:22s} between-run {means.std():6.2f}   within-run {wst.mean():6.2f}   overall mean {means.mean():+6.2f}")
