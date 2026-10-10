import glob, sys
import numpy as np
import joblib
PELVIS_OFFSET = np.array([0.003, -0.351, 0.012])
for name, d in (("0930 (walk etc.)", "/home/grease/ego_dataset/picoset_20260930/smpl"), ("1002 sit", "/home/grease/ego_dataset/picoset_20261002_sit/smpl")):
    fs = sorted(glob.glob(d + "/*.pkl"))[:70]
    st, se, ps, pse, sp = [], [], [], [], []
    for f in fs:
        x = joblib.load(f); x = x[next(iter(x))] if "smpl_joints" not in x else x
        J = np.asarray(x["smpl_joints"], float) - PELVIS_OFFSET + np.asarray(x["transl"], float)[:, None, :]
        fmin = J[:, [10, 11], 2].min(1)
        J[..., 2] -= np.percentile(fmin, 5)
        fmin = J[:, [10, 11], 2].min(1); z = J[:, 0, 2]
        stand = z > 0.9
        seat = z < 0.85
        if stand.any(): st.append(np.median(fmin[stand])); ps.append(np.median(z[stand]))
        if seat.any(): se.append(np.median(fmin[seat])); pse.append(np.median(z[seat]))
    print(f"{name:18s} lowest foot joint z (after p5 floor): standing median {np.median(st):.3f}  [{np.min(st):.3f},{np.max(st):.3f}]"
          f" | seated median {np.median(se) if se else float('nan'):.3f}   standing pelvis {np.median(ps):.3f}  seated pelvis {np.median(pse) if pse else float('nan'):.3f}")
