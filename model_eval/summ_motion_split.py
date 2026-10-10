import json, sys
for tag, f in (("ALL-dates training", "world_model_residual_motion_all.json"),
               ("AFTER-0919 training", "world_model_residual_motion_after0919.json")):
    d = json.load(open("/home/grease/gam/model_eval/" + f))
    print("#" * 6, tag)
    for v, e in d.items():
        print(f"== {v}  total {e['steps']} steps")
        for sname in ("still", "moving"):
            if sname not in e.get("subsets", {}):
                continue
            s = e["subsets"][sname]
            print(f"  [{sname}] {s['steps']} steps ({100 * s['steps'] / e['steps']:.0f}%)    q MAE/RMSE deg | dq MAE/RMSE rad/s")
            for g in ("legs", "waist", "arms", "ankle_pitch", "all"):
                row = "   ".join(f"{t.split()[1]:9s}{x[0]:.3f}/{x[1]:.3f} | {x[2]:.3f}/{x[3]:.3f}"
                                 for t, x in ((t, s["groups"][t][g]) for t in s["groups"]))
                print(f"     {g:12s}{row}")
            pj = s["per_joint"]; ks = list(pj)
            bq = sum(pj[ks[2]]["q_mae_deg"][j] < pj[ks[1]]["q_mae_deg"][j] for j in range(29))
            bd = sum(pj[ks[2]]["dq_mae"][j] < pj[ks[1]]["dq_mae"][j] for j in range(29))
            sq = sum(pj[ks[0]]["q_mae_deg"][j] < pj[ks[1]]["q_mae_deg"][j] for j in range(29))
            sd = sum(pj[ks[0]]["dq_mae"][j] < pj[ks[1]]["dq_mae"][j] for j in range(29))
            print(f"     joints beating persist: sim+model q {bq}/29 dq {bd}/29 ; pure_sim q {sq}/29 dq {sd}/29")
