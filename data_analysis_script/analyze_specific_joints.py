import pandas as pd

log_dir = "/home/grease/g1_deploy_run"

print(f"Loading Real Robot Logs from {log_dir}...")
df_q = pd.read_csv(f"{log_dir}/q.csv")
df_dq = pd.read_csv(f"{log_dir}/dq.csv")
df_tau = pd.read_csv(f"{log_dir}/motor_torque.csv")

def format_stats(series):
    v_min = series.min()
    v_max = series.max()
    v_mean = series.mean()
    v_std = series.std()
    
    unique_vals = series.nunique()
    all_zeros = (series == 0).all()
    
    return f"Min: {v_min:8.4f} | Max: {v_max:8.4f} | Mean: {v_mean:8.4f} | Std: {v_std:8.4f} | Unique: {unique_vals} | All Zeros: {all_zeros}"

print("\n--- Joint 13 Analysis ---")
print("Position (q_13): ", format_stats(df_q['q_13']))
print("Velocity (dq_13):", format_stats(df_dq['dq_13']))
print("Torque (tau_13): ", format_stats(df_tau['tau_13']))

print("\n--- Joint 14 Analysis ---")
print("Position (q_14): ", format_stats(df_q['q_14']))
print("Velocity (dq_14):", format_stats(df_dq['dq_14']))
print("Torque (tau_14): ", format_stats(df_tau['tau_14']))

