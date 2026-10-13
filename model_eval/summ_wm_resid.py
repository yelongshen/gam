import json
d = json.load(open("/home/grease/gam/model_eval/world_model_residual_metrics.json"))
for v, e in d.items():
    print(f"== {v} ({e['steps']} steps)   [q MAE, q RMSE (deg) | dq MAE, dq RMSE (rad/s)]")
    for g in ("legs", "waist", "arms", "ankle_pitch", "all"):
        row = []
        for t, s in e["groups"].items():
            x = s[g]; row.append(f"{t.split()[1]:9s} {x[0]:.3f} {x[1]:.3f} | {x[2]:.3f} {x[3]:.3f}")
        print(f"  {g:12s}" + "   ".join(row))
    # share of joints where sim+model beats persist / pure_sim
    pj = e["per_joint"]; ks = list(pj)
    for nm, key in (("q MAE", "q_mae_deg"), ("dq MAE", "dq_mae")):
        b_p = sum(pj[ks[2]][key][j] < pj[ks[1]][key][j] for j in range(29))
        b_s = sum(pj[ks[2]][key][j] < pj[ks[0]][key][j] for j in range(29))
        print(f"  joints where sim+model beats persist / pure_sim on {nm}: {b_p}/29 , {b_s}/29")
    for j in (4, 10):
        print(f"  joint {j}: " + " | ".join(f"{t.split()[1]} q {pj[t]['q_mae_deg'][j]:.3f} dq {pj[t]['dq_mae'][j]:.3f}" for t in ks))
