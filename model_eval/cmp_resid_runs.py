import json
A = json.load(open("/home/grease/gam/model_eval/world_model_residual_after0919_metrics.json"))
B = json.load(open("/home/grease/gam/model_eval/world_model_residual_metrics.json"))
for v in A:
    print(f"== {v} ({A[v]['steps']} steps)")
    for nm, d in (("after0919", A), ("all dates", B)):
        e = d[v]["groups"]
        for g in ("legs", "waist", "arms", "ankle_pitch", "all"):
            r = "  ".join(f"{t.split()[1]:9s}{e[t][g][0]:.3f} {e[t][g][1]:.3f}|{e[t][g][2]:.3f} {e[t][g][3]:.3f}" for t in e)
            print(f"  {nm:10s}{g:12s}{r}")
        pj = d[v]["per_joint"]; ks = list(pj)
        bp = sum(pj[ks[2]]["q_mae_deg"][j] < pj[ks[1]]["q_mae_deg"][j] for j in range(29))
        bd = sum(pj[ks[2]]["dq_mae"][j] < pj[ks[1]]["dq_mae"][j] for j in range(29))
        bs = sum(pj[ks[2]]["q_mae_deg"][j] < pj[ks[0]]["q_mae_deg"][j] and pj[ks[2]]["dq_mae"][j] < pj[ks[0]]["dq_mae"][j] for j in range(29))
        print(f"  {nm:10s}beats persist: q {bp}/29  dq {bd}/29 ; beats pure_sim on both q,dq: {bs}/29")
        for j in (4, 10):
            print(f"  {nm:10s}joint {j}: " + " | ".join(f"{t.split()[1]} q {pj[t]['q_mae_deg'][j]:.3f} dq {pj[t]['dq_mae'][j]:.3f}" for t in ks))
