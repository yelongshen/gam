import json
a = json.load(open("/home/grease/gam/model_eval/world_model_qdq_metrics.json"))
b = json.load(open("/home/grease/gam/model_eval/world_model_qdq_all_metrics.json"))
for v in a:
    print("==", v)
    for name, d in (("after0919-only", a), ("ALL dates", b)):
        o = d[v]["one_step"]["mlp"]["all"]; p = d[v]["one_step"]["persist"]["all"]
        print(f"  {name:15s} one-step all joints  q MAE {o[0]:.3f} RMSE {o[1]:.3f} | dq MAE {o[2]:.3f} RMSE {o[3]:.3f}   (persist q {p[0]:.3f}/{p[1]:.3f}  dq {p[2]:.3f}/{p[3]:.3f})")
        for h in ("5", "25"):
            r = d[v]["rollout"]["mlp"][h]; pr = d[v]["rollout"]["persist"][h]
            print(f"      rollout {h:>2s} steps q RMSE  " + "  ".join(f"{g} {r[g][1]:.2f} (persist {pr[g][1]:.2f})" for g in ("legs", "waist", "arms", "ankle_pitch")))
            print(f"                      dq RMSE " + "  ".join(f"{g} {r[g][3]:.2f}" for g in ("legs", "waist", "arms", "ankle_pitch")))
        g0 = d[v]["one_step"]["mlp"]
        print("      one-step by group  q RMSE / dq RMSE: " + "  ".join(f"{g} {g0[g][1]:.3f}/{g0[g][3]:.3f}" for g in ("legs", "waist", "arms", "ankle_pitch")))
