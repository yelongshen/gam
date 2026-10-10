import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gam/data_process"))
import gmr_lafan_posonly as G  # noqa: E402

cfg = "/tmp/gmr_test/_ik_config.json"
G._init(cfg)
GMR, P = G._G
s = joblib.load(os.path.expanduser("~/ego_dataset/lafan1_smpl_filtered_FPS30/dance2_subject3.pkl"))
sj = np.asarray(s["smpl_joints"], dtype=np.float64)[3600:4500]
tr = np.asarray(s["transl"], dtype=np.float64)[3600:4500]
frames = G.human_frames(sj, tr)
rt = GMR(src_human="lafan_posonly", tgt_robot="unitree_g1", actual_human_height=1.75, verbose=False)
for _ in range(30):
    rt.retarget(frames[0])
for t in (0, 150, 300, 450, 600):
    for _ in range(3 if t else 0):
        pass
    q = None
    for k in range(max(0, t - 5), t + 1):
        q = rt.retarget(frames[k])
    print(f"\nframe {t}")
    for body, task in rt.human_body_to_task2.items():
        e = task.compute_error(rt.configuration)
        pe = np.linalg.norm(e[:3])
        re = np.linalg.norm(e[3:])
        print(f"  {body:14s} pos err {pe * 100:6.1f} cm   rot err {re:5.2f}")
    # arm geometry: robot arm lengths
    import mujoco as mj
    m, d = rt.model, rt.configuration.data
    bid = {n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in ("left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link")}
    p = {n: d.xpos[i] for n, i in bid.items()}
    print("  G1 upper arm %.3f forearm %.3f" % (np.linalg.norm(p["left_elbow_link"] - p["left_shoulder_pitch_link"]),
                                                 np.linalg.norm(p["left_wrist_yaw_link"] - p["left_elbow_link"])))
    h = rt.scaled_human_data
    print("  scaled human upper arm %.3f forearm %.3f" % (np.linalg.norm(h["LeftForeArm"][0] - h["LeftArm"][0]),
                                                        np.linalg.norm(h["LeftHand"][0] - h["LeftForeArm"][0])))
