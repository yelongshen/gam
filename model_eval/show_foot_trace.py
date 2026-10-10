import json, sys
import numpy as np
fixed = {"clip_%s" % x for x in ("001", "008", "009", "024", "027", "032", "044", "045", "046")}
pick = json.load(open("/tmp/short_clips.json"))
keys = sorted("pico1002sit_" + c for c in pick)
recs = [json.loads(l) for l in open("/tmp/terms_sitmixv2.jsonl.trace")]
print("foot tracking error (max of L,R ankle) per env at control steps 0..6 ; z error = reference minus robot ankle height [m]")
print(f"{'clip':9s}{'set':6s}" + "".join(f"  step{r['step']}" for r in recs))
for i, k in enumerate(keys):
    c = k.replace("pico1002sit_", "")
    e = [max(r["err"][i]) for r in recs]; dz = [r["dz"][i] for r in recs]
    print(f"{c:9s}{'FIXED' if c in fixed else 'other':6s}" + "".join(f"  {x:6.3f}" for x in e) + "   | dz L/R step0: %+.3f %+.3f  step3: %+.3f %+.3f" % (dz[0][0], dz[0][1], dz[min(3, len(dz) - 1)][0], dz[min(3, len(dz) - 1)][1]))
