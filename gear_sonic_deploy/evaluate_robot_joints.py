import numpy as np
import pandas as pd

log_dir = "/home/grease/g1_deploy_run"

# We check Q (positions), DQ (velocities), Torque (effort), and Temperature.
# Abnormalities typically surface as exploding velocities, maxed out torques, 
# thermal warnings, or extreme error margins between Q and Action (Target Q).

print("Loading Real Robot Logs from deploy_run...")
df_q = pd.read_csv(f"{log_dir}/q.csv")
df_dq = pd.read_csv(f"{log_dir}/dq.csv")
df_tau = pd.read_csv(f"{log_dir}/motor_torque.csv")
df_err = pd.read_csv(f"{log_dir}/motor_error.csv")
df_temp = pd.read_csv(f"{log_dir}/motor_temperature.csv")

q_cols = [c for c in df_q.columns if c.startswith('q_')]
dq_cols = [c for c in df_dq.columns if c.startswith('dq_')]
tau_cols = [c for c in df_tau.columns if c.startswith('tau_')]
err_cols = [c for c in df_err.columns if c.startswith('err_')]
temp_cols = [c for c in df_temp.columns if c.startswith('m_temp_')]

print(f"\n--- Checking {len(q_cols)} joints over {len(df_q)} frames (~{len(df_q)/50.0:.1f}s) ---")

# 1. Check for extreme velocities (usually indicative of jitter/spikes)
max_dq = df_dq[dq_cols].abs().max()
spiky_joints = max_dq[max_dq > 10.0]  # Very high radian/sec
if len(spiky_joints) > 0:
    print("\n[WARNING] Found extremely high peak velocities (Jitter/Snapping risk):")
    for j, val in spiky_joints.items():
        print(f"  {j}: {val:.2f} rad/s")
else:
    print("\n[OK] Peak joint velocities are completely within safe bounds under 10 rad/s.")

# 2. Check Torques (identifies struggling motors or collisions)
max_tau = df_tau[tau_cols].abs().max()
high_tau_joints = max_tau[max_tau > 25.0] # 25 Nm is very high for many G1 upper arm / limb joints
mean_tau = df_tau[tau_cols].abs().mean()

if len(high_tau_joints) > 0:
    print("\n[WARNING] Found joints hitting extreme absolute torque spikes (>25 Nm):")
    for j, val in high_tau_joints.items():
        print(f"  {j}: {val:.2f} Nm (Mean: {mean_tau[j]:.2f} Nm)")
else:
    print("\n[OK] Torques are completely stable with no severe spikes (>25 Nm).")

# 3. Check Thermal Warnings
max_temp = df_temp[temp_cols].max()
hot_joints = max_temp[max_temp > 50.0] # 50C is starting to get warm, 60+ is hot
if len(hot_joints) > 0:
    print("\n[WARNING] Found motors running hot (>50 C):")
    for j, val in hot_joints.items():
        mean_val = df_temp[j].mean()
        print(f"  {j}: Peak {val:.1f} C (Mean working temp {mean_val:.1f}C)")
else:
    print("\n[OK] All motors remained comfortably cool under 50 C.")

# 4. Check Tracking Errors (Difference between Target Q and Actual Q) 
# The log gives us motor_error directly, or we can look at it mathematically.
max_err = df_err[err_cols].abs().max()
mean_err = df_err[err_cols].abs().mean()
struggling_joints = mean_err[mean_err > 0.15] # If average tracking error is > 0.15 radians (~8.5 degrees), the PID is really struggling or lagging
if len(struggling_joints) > 0:
    print("\n[WARNING] These joints have very high average tracking errors (Poor tracking / lag):")
    for j, val in struggling_joints.items():
        print(f"  {j}: Mean Err {val:.3f} rad, Peak Err {max_err[j]:.3f} rad")
else:
    print("\n[OK] Overall mean trajectory tracking error is tight (under 0.15 rad).")
    
# Find the single spikiest tracker
worst_err_j = mean_err.idxmax()
print(f"  -> Worst tracked joint overall was {worst_err_j} (Mean Error: {mean_err[worst_err_j]:.3f} rad)")

print("\nEvaluation Complete.")
