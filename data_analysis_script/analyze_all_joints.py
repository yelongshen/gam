import pandas as pd

log_dir = "/home/grease/g1_deploy_run"

print(f"Loading Real Robot Logs from {log_dir}...")
df_q = pd.read_csv(f"{log_dir}/q.csv")

q_cols = [c for c in df_q.columns if c.startswith('q_')]

print(f"\n--- Checking all {len(q_cols)} joints for strictly zero activity ---")
all_zero_joints = []
active_joints = []

for j in q_cols:
    if (df_q[j] == 0).all():
        all_zero_joints.append(j)
    else:
        active_joints.append(j)

print(f"\n{len(all_zero_joints)} Joints that are strictly all zeros (0.0000 across entire run):")
print(", ".join(all_zero_joints) if all_zero_joints else "None")

print(f"\n{len(active_joints)} Joints that show activity:")
print(", ".join(active_joints) if active_joints else "None")
