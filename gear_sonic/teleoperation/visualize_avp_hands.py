#!/usr/bin/env python3
"""
visualize_avp_hands.py
======================
Real-time 3D hand skeleton visualization of AVP hand streaming data.

Listens on UDP port 9870 and renders a 2x2 view:
  Top    row: raw AVP 27-joint skeleton (left / right)
  Bottom row: retargeted Dex3 3-finger FK skeleton (left / right)

Usage
-----
  python gear_sonic/teleoperation/visualize_avp_hands.py [--port 9870]
"""

import argparse
import json
import socket
import threading
import time

import numpy as np
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# AVP skeleton connectivity
# ---------------------------------------------------------------------------

BONES = [
    (26, 25), (25, 0),
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
    (5, 10), (10, 15), (15, 20),
]

_FINGER_RANGES = [
    (range(1, 5),  "#FF6B6B"),
    (range(5, 10), "#4ECDC4"),
    (range(10, 15), "#45B7D1"),
    (range(15, 20), "#96CEB4"),
    (range(20, 25), "#FFEAA7"),
]

def _avp_bone_color(a, b):
    for rng, col in _FINGER_RANGES:
        if a in rng or b in rng:
            return col
    if a in (25, 26) or b in (25, 26):
        return "#DDA0DD"
    return "#888888"

def _draw_avp(ax, joints, active, side):
    pts = np.array(joints, dtype=np.float32)
    a = 1.0 if active else 0.25
    for (i, j) in BONES:
        ax.plot([pts[i,0],pts[j,0]], [pts[i,1],pts[j,1]], [pts[i,2],pts[j,2]],
                color=_avp_bone_color(i,j), linewidth=2, alpha=a)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
               c=["#FFFFFF"]*len(pts), s=18, alpha=a, depthshade=False)
    w = pts[0]
    ax.text(w[0], w[1], w[2], f" {side}", fontsize=7, color="white", alpha=a)

# ---------------------------------------------------------------------------
# Retargeting — Geometric (vector) algorithm via dex-retargeting
# ---------------------------------------------------------------------------

MOTOR_NUM = 7
MOTOR_NAMES = ["Thumb Abd","Thumb MCP","Thumb IP","Idx MCP","Idx PIP","Mid MCP","Mid PIP"]

import yaml as _yaml
from dex_retargeting.retargeting_config import RetargetingConfig as _RetargetingConfig
from pathlib import Path as _Path

_ASSETS = _Path(__file__).parent / "assets"
_RetargetingConfig.set_default_urdf_dir(str(_ASSETS))
_cfg_yaml = _yaml.safe_load((_ASSETS / "unitree_hand/unitree_dex3.yml").read_text())

def _make_vector_cfg(d):
    """Build a clean 'vector' (Geometric) retargeting config dict."""
    return {
        "type": "vector",
        "urdf_path": d["urdf_path"],
        "target_joint_names": d["target_joint_names"],
        "target_origin_link_names": d["target_origin_link_names"],
        "target_task_link_names": d["target_task_link_names"],
        "target_link_human_indices": np.array(d["target_link_human_indices_vector"]),
        "scaling_factor": d.get("scaling_factor", 1.0),
        "low_pass_alpha": d.get("low_pass_alpha", 0.2),
    }

_left_retarget  = _RetargetingConfig.from_dict(_make_vector_cfg(_cfg_yaml["left"])).build()
_right_retarget = _RetargetingConfig.from_dict(_make_vector_cfg(_cfg_yaml["right"])).build()

_LEFT_HW  = ["left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint",
             "left_hand_middle_0_joint","left_hand_middle_1_joint",
             "left_hand_index_0_joint","left_hand_index_1_joint"]
_RIGHT_HW = ["right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint",
             "right_hand_index_0_joint","right_hand_index_1_joint",
             "right_hand_middle_0_joint","right_hand_middle_1_joint"]
_left_to_hw  = [_left_retarget.joint_names.index(n)  for n in _LEFT_HW]
_right_to_hw = [_right_retarget.joint_names.index(n) for n in _RIGHT_HW]
_left_indices  = _left_retarget.optimizer.target_link_human_indices   # shape (2,3)
_right_indices = _right_retarget.optimizer.target_link_human_indices

# Coordinate transform: AVP visionOS (OpenXR) world frame → Unitree URDF hand frame.
# Official pipeline from unitreerobotics/televuer tv_wrapper.py:
#   Step 1: change basis OpenXR → Robot:  R_ROBOT_OPENXR = [[0,0,-1],[-1,0,0],[0,1,0]]
#   Step 2: wrist-relative vectors (pts[tip]-pts[wrist] cancels translation)
#   Step 3: change initial pose convention:  R_TO_UNITREE_HAND = [[0,0,1],[-1,0,0],[0,-1,0]]
# Combined (step1 then step3): R = R_TO_UNITREE_HAND @ R_ROBOT_OPENXR = [[0,1,0],[0,0,1],[1,0,0]]
# In OpenXR/visionOS:  +Y=up, +X=right, -Z=toward robot (finger extension direction)
# In Unitree URDF:     +X=dorsal, -Y=distal (finger extension), ±Z=lateral
# Mapping: (-Z in OpenXR) → (-Y in URDF)  ✓  (both hands use same transform)
_R_AVP2ROBOT_RIGHT = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
_R_AVP2ROBOT_LEFT  = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])  # same as right per official code


def _thumb_angles_geometric(pts: np.ndarray, is_right: bool) -> np.ndarray:
    """Compute Dex3 thumb joint angles directly from AVP joint positions.

    Uses the opening angle between thumb and fingers for abduction, and the
    thumb-tip ↔ index-tip distance for flexion.

    Returns array [thumb_0_abd, thumb_1_mcp, thumb_2_ip] in radians.
    """
    p_w   = pts[0]   # wrist
    p_th  = pts[4]   # thumb tip
    p_idx = pts[9]   # index tip
    p_mid = pts[14]  # middle tip

    # Abduction: opening angle between (thumb_tip - wrist) and (index_tip - wrist)
    v_th  = p_th  - p_w
    v_idx = p_idx - p_w
    cos_sep = np.clip(np.dot(v_th / (np.linalg.norm(v_th)+1e-8),
                             v_idx / (np.linalg.norm(v_idx)+1e-8)), -1., 1.)
    sep_angle = np.arccos(cos_sep)   # 0..π
    # Typical human max thumb–index angle ≈ 1.2 rad; robot thumb_0 max = 1.745
    thumb_abd = float(np.clip(sep_angle * (1.745 / 1.2), 0., 1.745))

    # Flexion: thumb-tip ↔ index-tip distance, normalised by wrist-to-middle-tip
    # (middle-tip is a stable reference that doesn't change with thumb/pinch motion).
    # Open:   ti_dist/wm_dist ≈ 0.65 → flex_t = 0
    # Pinch:  ti_dist/wm_dist ≈ 0.05 → flex_t = 1
    # Fist:   ti_dist/wm_dist ≈ 0.25 → flex_t = 0.6
    p_mid     = pts[14]  # middle tip
    ti_dist   = np.linalg.norm(p_th - p_idx)
    wm_dist   = np.linalg.norm(p_mid - p_w) + 1e-8
    ratio     = ti_dist / wm_dist
    # Map: 0.65 (open) → 0,  0.0 (pinched) → 1
    flex_t    = float(np.clip(1.0 - ratio / 0.65, 0., 1.))
    thumb_mcp = flex_t * 1.2   # [0, 1.571]
    thumb_ip  = flex_t * 1.4   # [0, 1.745]

    return np.array([thumb_abd, thumb_mcp, thumb_ip])


def retarget(joints, is_right):
    """Geometric (vector) retargeting: AVP 27-joint -> Dex3 7-DOF (hardware order).

    Index/middle: vector optimizer minimises wrist→fingertip direction error
    (with AVP→robot coordinate transform applied).
    Thumb: direct geometric mapping from separation angle and pinch distance.
    """
    if len(joints) < 15:
        return np.zeros(MOTOR_NUM)
    pts        = np.array(joints, dtype=np.float64)
    retargeter = _right_retarget if is_right else _left_retarget
    indices    = _right_indices  if is_right else _left_indices
    to_hw      = _right_to_hw   if is_right else _left_to_hw
    R          = _R_AVP2ROBOT_RIGHT if is_right else _R_AVP2ROBOT_LEFT

    # Build index/middle vectors and rotate into robot frame
    raw_ref = pts[indices[1]] - pts[indices[0]]   # shape (3,3)
    ref     = (R @ raw_ref.T).T                   # shape (3,3) in robot frame
    q       = retargeter.retarget(ref)[to_hw]     # hardware order

    # Override thumb joints with geometric angles (vector solver gives poor thumb)
    q[0], q[1], q[2] = _thumb_angles_geometric(pts, is_right)
    return q

# ---------------------------------------------------------------------------
# Dex3 forward kinematics
# ---------------------------------------------------------------------------
# Frame: X=lateral(toward index), Y=distal(finger extension), Z=dorsal

def _rx(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def _rz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def _rodrigues(vec, axis, angle):
    c,s=np.cos(angle),np.sin(angle)
    return vec*c + np.cross(axis,vec)*s + axis*np.dot(axis,vec)*(1-c)

def dex3_fk(angles, is_right):
    q=angles.copy()
    sign=1.0 if is_right else -1.0
    pts={}
    pts["palm"]=np.zeros(3)

    # Index  (q[3] MCP in [-1.57,0],  q[4] PIP in [-1.75,0]  — negative = flex toward palm)
    # _rx(q) with negative angle rotates +Y toward -Z (palm), correct for flexion.
    idx_meta=np.array([-sign*0.015, 0.052, 0.0])
    pts["idx_meta"]=idx_meta
    R_im=_rx(q[3])
    idx_pip=idx_meta + R_im@np.array([0.0,0.038,0.0])
    pts["idx_pip"]=idx_pip
    idx_tip=idx_pip + R_im@_rx(q[4])@np.array([0.0,0.030,0.0])
    pts["idx_tip"]=idx_tip

    # Middle  (q[5] MCP in [-1.57,0],  q[6] PIP in [-1.75,0])
    mid_meta=np.array([sign*0.012, 0.055, 0.0])
    pts["mid_meta"]=mid_meta
    R_mm=_rx(q[5])
    mid_pip=mid_meta + R_mm@np.array([0.0,0.042,0.0])
    pts["mid_pip"]=mid_pip
    mid_tip=mid_pip + R_mm@_rx(q[6])@np.array([0.0,0.032,0.0])
    pts["mid_tip"]=mid_tip

    # Thumb  (q[0] abd in [-0.5,0.5],  q[1] MCP in [-1.5,0],  q[2] IP in [0,1.7])
    thumb_cmc=np.array([sign*0.042, 0.010, 0.002])
    pts["thumb_cmc"]=thumb_cmc
    thumb_rest=np.array([sign*np.sin(np.pi/4), np.cos(np.pi/4), 0.0])
    R_abd=_rz(sign*q[0])
    abd_dir=R_abd@thumb_rest
    thumb_mcp=thumb_cmc + abd_dir*0.038
    pts["thumb_mcp"]=thumb_mcp
    # flex_mcp > 0 when thumb is flexed (q[1] < 0)
    flex_mcp=-q[1]
    z_hat=np.array([0.0,0.0,1.0])
    # cross(z_hat, abd_dir) → positive flex_mcp rotates thumb tip toward -Z (palm). 
    # cross(abd_dir, z_hat) was the bug — it rotated toward +Z (dorsal).
    flex_axis=np.cross(z_hat, abd_dir)
    fn=np.linalg.norm(flex_axis)
    flex_axis=flex_axis/fn if fn>1e-6 else np.array([1.0,0.0,0.0])
    mcp_dir=_rodrigues(abd_dir, flex_axis, flex_mcp)
    thumb_ip=thumb_mcp + mcp_dir*0.030
    pts["thumb_ip"]=thumb_ip
    # q[2] > 0 = IP flex; same axis keeps curling in same direction
    ip_dir=_rodrigues(mcp_dir/(np.linalg.norm(mcp_dir)+1e-8), flex_axis, q[2])
    pts["thumb_tip"]=thumb_ip + ip_dir*0.025

    return pts

DEX3_BONES = [
    ("palm","thumb_cmc","#FF6B6B"),
    ("thumb_cmc","thumb_mcp","#FF6B6B"),
    ("thumb_mcp","thumb_ip","#FF6B6B"),
    ("thumb_ip","thumb_tip","#FF6B6B"),
    ("palm","idx_meta","#888888"),
    ("idx_meta","idx_pip","#4ECDC4"),
    ("idx_pip","idx_tip","#4ECDC4"),
    ("palm","mid_meta","#888888"),
    ("mid_meta","mid_pip","#45B7D1"),
    ("mid_pip","mid_tip","#45B7D1"),
    ("thumb_cmc","idx_meta","#888888"),
    ("idx_meta","mid_meta","#888888"),
]

def _draw_dex3(ax, angles, active, is_right):
    pts=dex3_fk(angles, is_right)
    a=1.0 if active else 0.25
    for (j1,j2,col) in DEX3_BONES:
        p1,p2=pts[j1],pts[j2]
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color=col,linewidth=2.5,alpha=a)
    coords=np.array(list(pts.values()))
    ax.scatter(coords[:,0],coords[:,1],coords[:,2],
               c=["#FFFFFF"]*len(coords),s=22,alpha=a,depthshade=False)
    label="\n".join(f"{MOTOR_NAMES[i]:>9s}: {angles[i]:+.2f}" for i in range(MOTOR_NUM))
    ax.text2D(0.02,0.98,label,transform=ax.transAxes,fontsize=5.5,color="#AAAAAA",
              verticalalignment="top",fontfamily="monospace",alpha=a)

# ---------------------------------------------------------------------------
# UDP receiver
# ---------------------------------------------------------------------------

class HandReceiver:
    def __init__(self, port):
        self._port=port
        self._lock=threading.Lock()
        self._left=None; self._right=None
        self._left_ts=0.0; self._right_ts=0.0
        self._running=False
        self._hz=0.0; self._frame_count=0; self._hz_time=time.time()

    def start(self):
        self._running=True
        threading.Thread(target=self._loop,daemon=True,name="AVP_UDP").start()

    def stop(self):
        self._running=False

    def _loop(self):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(("0.0.0.0",self._port))
        sock.settimeout(0.5)
        while self._running:
            try:
                data,_=sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                pkt=json.loads(data.decode())
                hand=pkt["hand"]; joints=pkt["joints"]; ts=pkt.get("t",time.time())
                with self._lock:
                    if hand=="left": self._left=joints; self._left_ts=ts
                    else:            self._right=joints; self._right_ts=ts
                    self._frame_count+=1
            except Exception:
                pass
            now=time.time(); elapsed=now-self._hz_time
            if elapsed>=1.0:
                with self._lock:
                    self._hz=self._frame_count/elapsed
                    self._frame_count=0; self._hz_time=now

    def get(self):
        now=time.time()
        with self._lock:
            l=self._left; r=self._right
            la=l is not None and (now-self._left_ts)<0.5
            ra=r is not None and (now-self._right_ts)<0.5
            hz=self._hz
        return l,r,la,ra,hz

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="gray",labelsize=5)
    ax.xaxis.pane.fill=False; ax.yaxis.pane.fill=False; ax.zaxis.pane.fill=False
    ax.grid(True,color="#333355",linewidth=0.4)
    ax.set_xlabel("X",color="gray",fontsize=6)
    ax.set_ylabel("Y",color="gray",fontsize=6)
    ax.set_zlabel("Z",color="gray",fontsize=6)

def _equal_3d(ax, pts, margin=0.12):
    if len(pts)==0: return
    mn,mx=pts.min(0),pts.max(0)
    ctr=(mn+mx)/2; half=max((mx-mn).max()/2, margin)
    ax.set_xlim(ctr[0]-half,ctr[0]+half)
    ax.set_ylim(ctr[1]-half,ctr[1]+half)
    ax.set_zlim(ctr[2]-half,ctr[2]+half)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=9870)
    args=ap.parse_args()

    try:
        _s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8",80)); local_ip=_s.getsockname()[0]; _s.close()
    except Exception:
        local_ip="unknown"

    print(f"[Visualizer] Listening on UDP :{args.port}")
    print(f"[Visualizer] Set AVP Host IP to: {local_ip}")

    receiver=HandReceiver(port=args.port)
    receiver.start()

    from matplotlib.animation import FuncAnimation

    plt.style.use("dark_background")
    fig=plt.figure(figsize=(14,10),facecolor="#1a1a2e")
    fig.suptitle("AVP Hand Teleoperation  —  Real-time Visualization",
                 color="white",fontsize=13,y=0.99)

    ax_avp_l=fig.add_subplot(2,2,1,projection="3d")
    ax_avp_r=fig.add_subplot(2,2,2,projection="3d")
    ax_dex_l=fig.add_subplot(2,2,3,projection="3d")
    ax_dex_r=fig.add_subplot(2,2,4,projection="3d")

    fig.text(0.01,0.73,"AVP\nRaw",color="#888888",fontsize=8,va="center",rotation=90)
    fig.text(0.01,0.27,"Dex3\nFK", color="#888888",fontsize=8,va="center",rotation=90)

    status_text=fig.text(0.5,0.005,"Waiting for AVP data...",
                         ha="center",color="#888888",fontsize=8)

    plt.tight_layout(rect=[0.03,0.02,1.0,0.97])

    def update(_frame):
        try:
            l,r,la,ra,hz=receiver.get()

            for ax in (ax_avp_l,ax_avp_r,ax_dex_l,ax_dex_r):
                ax.cla(); _style_ax(ax)

            ax_avp_l.set_title("AVP Raw — Left"  +(" [TRACKING]" if la else " [waiting]"),color="white",fontsize=9,pad=3)
            ax_avp_r.set_title("AVP Raw — Right" +(" [TRACKING]" if ra else " [waiting]"),color="white",fontsize=9,pad=3)

            avp_pts=[]
            if l is not None: _draw_avp(ax_avp_l,l,la,"L"); avp_pts.extend(l)
            if r is not None: _draw_avp(ax_avp_r,r,ra,"R"); avp_pts.extend(r)
            if avp_pts:
                pts=np.array(avp_pts)
                _equal_3d(ax_avp_l,pts); _equal_3d(ax_avp_r,pts)

            q_left  = retarget(l,is_right=False) if l is not None else np.zeros(MOTOR_NUM)
            q_right = retarget(r,is_right=True)  if r is not None else np.zeros(MOTOR_NUM)

            ax_dex_l.set_title("Dex3 FK — Left"  +(" [ACTIVE]" if la else " [zero pose]"),color="white",fontsize=9,pad=3)
            ax_dex_r.set_title("Dex3 FK — Right" +(" [ACTIVE]" if ra else " [zero pose]"),color="white",fontsize=9,pad=3)

            _draw_dex3(ax_dex_l,q_left, active=la,is_right=False)
            _draw_dex3(ax_dex_r,q_right,active=ra,is_right=True)

            for ax in (ax_dex_l,ax_dex_r):
                ax.set_xlim(-0.10,0.10)
                ax.set_ylim(-0.02,0.16)
                ax.set_zlim(-0.08,0.08)
                ax.view_init(elev=20,azim=-60)

            if l is None and r is None:
                status="Waiting for AVP data..."
            else:
                active=[s for s,f in [("L",la),("R",ra)] if f]
                ql=" ".join(f"{v:+.2f}" for v in q_left)
                qr=" ".join(f"{v:+.2f}" for v in q_right)
                status=f"Hz: {hz:.0f}  |  Tracking: {','.join(active) if active else 'none'}  |  L:[{ql}]  R:[{qr}]"
            status_text.set_text(status)
        except Exception as e:
            print(f"[update error] {e}", flush=True)

    # Keep a strong reference so FuncAnimation is not garbage-collected
    _ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)

    # On macOS, plt.show(block=False) can destroy the figure before get_fignums()
    # runs. Use fignum_exists() and let plt.pause() implicitly show the window.
    try:
        while plt.fignum_exists(fig.number):
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        print("\n[Visualizer] Stopped.")

if __name__=="__main__":
    main()
