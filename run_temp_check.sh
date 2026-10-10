#!/usr/bin/env bash
cd /home/grease/gam
for d in /home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/2026*/; do
  .venv_sim/bin/python - "$d" <<'PYEOF'
import sys
sys.path.insert(0, "/home/grease/gam")
import numpy as np
from data_process.g1_params import JOINT_NAMES
from model_eval.sim2real_single_run_quicklook import load
d = sys.argv[1]
temp, _ = load(d, "motor_temperature")
if len(temp) == 0:
    print(f"{d}: EMPTY"); sys.exit()
jmax = int(np.nanargmax(np.nanmax(temp, axis=0)))
print(f"{d}: max={np.nanmax(temp):.0f}C on {JOINT_NAMES[jmax]}  "
      f"(final={temp[-1, jmax]:.0f}C)")
PYEOF
done
