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
# AVP skeleton connectivity  (WebXR 25-joint layout, matches official xr_teleoperate)
# ---------------------------------------------------------------------------
# Joint order (0-24) — same as WebXR XRHand spec and TeleVuer/vuer:
#  0=wrist  1-4=thumb(metacarpal→tip)  5-9=index  10-14=middle
#  15-19=ring  20-24=little
# (AVP HandSkeleton.allCases adds joints 25=forearmWrist, 26=forearmArm;
#  we ignore those extra two so our indexing is identical to the official code.)

BONES = [
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
    return "#888888"

def _draw_avp(ax, joints, active, side):
    pts = np.array(joints[:25], dtype=np.float32)  # use only first 25 joints
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
# Official hardware order from Unitree hand_retargeting.py (both hands):
#   [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
MOTOR_NAMES = ["Thumb Abd","Thumb MCP","Thumb IP","Mid MCP","Mid PIP","Idx MCP","Idx PIP"]

import yaml as _yaml
from dex_retargeting.retargeting_config import RetargetingConfig as _RetargetingConfig
from pathlib import Path as _Path

_ASSETS = _Path(__file__).parent / "assets"
_RetargetingConfig.set_default_urdf_dir(str(_ASSETS))
_cfg_yaml = _yaml.safe_load((_ASSETS / "unitree_hand/unitree_dex3.yml").read_text())

def _make_dexpilot_cfg(d):
    """Build a 'dexpilot' retargeting config dict (matches official xr_teleoperate)."""
    return {
        "type": "dexpilot",
        "urdf_path": d["urdf_path"],
        "target_joint_names": d["target_joint_names"],
        "wrist_link_name": d["wrist_link_name"],
        "finger_tip_link_names": d["finger_tip_link_names"],
        "target_link_human_indices": np.array(d["target_link_human_indices_dexpilot"]),
        "low_pass_alpha": d.get("low_pass_alpha", 0.2),
    }

_left_retarget  = _RetargetingConfig.from_dict(_make_dexpilot_cfg(_cfg_yaml["left"])).build()
_right_retarget = _RetargetingConfig.from_dict(_make_dexpilot_cfg(_cfg_yaml["right"])).build()

# Hardware joint order from official Unitree hand_retargeting.py (both hands):
#   [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
# Note: for the right hand the YAML target_joint_names has index before middle,
# but the official dex3_api_joint_names (used for DDS mapping) has middle before
# index. We follow the official DDS mapping here.
_LEFT_HW  = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
             "left_hand_middle_0_joint", "left_hand_middle_1_joint",
             "left_hand_index_0_joint",  "left_hand_index_1_joint"]
_RIGHT_HW = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
             "right_hand_middle_0_joint", "right_hand_middle_1_joint",
             "right_hand_index_0_joint",  "right_hand_index_1_joint"]
_left_to_hw  = [_left_retarget.joint_names.index(n)  for n in _LEFT_HW]
_right_to_hw = [_right_retarget.joint_names.index(n) for n in _RIGHT_HW]
_left_indices  = _left_retarget.optimizer.target_link_human_indices   # shape (2,6) for DexPilot
_right_indices = _right_retarget.optimizer.target_link_human_indices

# Coordinate transform: AVP visionOS (OpenXR) world frame → Unitree URDF hand frame.
# Official pipeline from unitreerobotics/televuer tv_wrapper.py:
#   Step 1: change basis OpenXR → Robot:  R_ROBOT_OPENXR = [[0,0,-1],[-1,0,0],[0,1,0]]
#   Step 2: wrist-relative vectors (pts[tip]-pts[wrist] cancels translation)
#   Step 3: change initial pose convention:  T_TO_UNITREE_HAND_rot = [[0,0,1],[-1,0,0],[0,-1,0]]
# Combined: R = T_TO_UNITREE_HAND_rot @ R_ROBOT_OPENXR = [[0,1,0],[0,0,1],[1,0,0]]
# In OpenXR/visionOS:  +Y=up, +X=right, -Z=toward robot (finger extension direction)
# In Unitree URDF:     +X=dorsal, -Y=distal (finger extension), ±Z=lateral
# Mapping: (-Z in OpenXR) → (-Y in URDF)  ✓  both hands use same transform (official)
_R_AVP2ROBOT_RIGHT = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
_R_AVP2ROBOT_LEFT  = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])


def retarget(joints, is_right):
    """Pure DexPilot retargeting: AVP joint positions -> Dex3 7-DOF (hardware order).

    Exactly matches the official unitreerobotics/xr_teleoperate algorithm:
      robot_hand_unitree.py Dex3_1_Controller.control_process()

      hand_data = np.array(hand_array[:]).reshape(25, 3)   # 25 joints
      ref_value = hand_data[indices[1,:]] - hand_data[indices[0,:]]
      q_target  = retargeting.retarget(ref_value)[dex_retargeting_to_hardware]

    Uses only the first 25 joints (WebXR XRHand layout). AVP sends 27;
    joints 25-26 (forearmWrist, forearmArm) are ignored.

    DexPilot indices (from unitree_dex3.yml):
      origins : [index_tip(9), middle_tip(14), middle_tip(14), wrist(0), wrist(0),  wrist(0) ]
      targets : [thumb_tip(4), thumb_tip(4),   index_tip(9),  thumb(4), index(9), middle(14)]

    Output hardware order (both hands): [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
    """
    if len(joints) < 25:
        return np.zeros(MOTOR_NUM)
    pts        = np.array(joints[:25], dtype=np.float64)   # reshape(25,3) — matches official
    retargeter = _right_retarget if is_right else _left_retarget
    indices    = _right_indices  if is_right else _left_indices
    to_hw      = _right_to_hw   if is_right else _left_to_hw
    R          = _R_AVP2ROBOT_RIGHT if is_right else _R_AVP2ROBOT_LEFT

    # Compute 6 DexPilot inter-finger vectors and rotate into Unitree URDF frame.
    # Official code applies no explicit rotation because tv_wrapper already
    # outputs positions in the URDF arm frame; we replicate with R matrix.
    raw_ref = pts[indices[1]] - pts[indices[0]]   # shape (6, 3) in OpenXR frame
    ref     = (R @ raw_ref.T).T                   # shape (6, 3) in URDF frame
    return retargeter.retarget(ref)[to_hw]         # hardware order

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
    """Forward kinematics for Dex3-1 visualization.

    Input ordering matches official hardware order (both hands):
      q[0]=thumb_0(abd)  q[1]=thumb_1(MCP)  q[2]=thumb_2(IP)
      q[3]=middle_0(MCP) q[4]=middle_1(PIP)
      q[5]=index_0(MCP)  q[6]=index_1(PIP)

    DexPilot sign conventions (from URDF NLopt bounds):
      index/middle: positive = flex  (range approx [0, 1.57/1.75])
      thumb_2:      negative = flex  (range approx [-1.75, 0])
      thumb_1:      negative = MCP flex (range approx [-0.92, 0.73])
    """
    q=angles.copy()
    sign=1.0 if is_right else -1.0
    pts={}
    pts["palm"]=np.zeros(3)

    # Middle  q[3]=MCP, q[4]=PIP  — DexPilot: positive = flex
    # _rx(-q) rotates +Y toward -Z (palm) for positive q = flexion.
    mid_meta=np.array([sign*0.012, 0.055, 0.0])
    pts["mid_meta"]=mid_meta
    R_mm=_rx(-q[3])
    mid_pip=mid_meta + R_mm@np.array([0.0,0.042,0.0])
    pts["mid_pip"]=mid_pip
    mid_tip=mid_pip + R_mm@_rx(-q[4])@np.array([0.0,0.032,0.0])
    pts["mid_tip"]=mid_tip

    # Index  q[5]=MCP, q[6]=PIP  — DexPilot: positive = flex
    idx_meta=np.array([-sign*0.015, 0.052, 0.0])
    pts["idx_meta"]=idx_meta
    R_im=_rx(-q[5])
    idx_pip=idx_meta + R_im@np.array([0.0,0.038,0.0])
    pts["idx_pip"]=idx_pip
    idx_tip=idx_pip + R_im@_rx(-q[6])@np.array([0.0,0.030,0.0])
    pts["idx_tip"]=idx_tip

    # Thumb  q[0]=abd, q[1]=MCP, q[2]=IP
    # DexPilot: thumb_1 negative = MCP flex; thumb_2 negative = IP flex
    thumb_cmc=np.array([sign*0.042, 0.010, 0.002])
    pts["thumb_cmc"]=thumb_cmc
    thumb_rest=np.array([sign*np.sin(np.pi/4), np.cos(np.pi/4), 0.0])
    R_abd=_rz(sign*q[0])
    abd_dir=R_abd@thumb_rest
    thumb_mcp=thumb_cmc + abd_dir*0.038
    pts["thumb_mcp"]=thumb_mcp
    # thumb_1 < 0 means MCP flex: -q[1] > 0 = positive flex amount
    flex_mcp=-q[1]
    z_hat=np.array([0.0,0.0,1.0])
    flex_axis=np.cross(z_hat, abd_dir)
    fn=np.linalg.norm(flex_axis)
    flex_axis=flex_axis/fn if fn>1e-6 else np.array([1.0,0.0,0.0])
    mcp_dir=_rodrigues(abd_dir, flex_axis, flex_mcp)
    thumb_ip=thumb_mcp + mcp_dir*0.030
    pts["thumb_ip"]=thumb_ip
    # thumb_2 < 0 means IP flex: -q[2] > 0 = positive flex amount
    ip_dir=_rodrigues(mcp_dir/(np.linalg.norm(mcp_dir)+1e-8), flex_axis, -q[2])
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

    plt.style.use("dark_background")
    plt.ion()   # interactive mode — draw() calls update the window immediately

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
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    try:
        while plt.fignum_exists(fig.number):
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

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        print("\n[Visualizer] Stopped.")

if __name__=="__main__":
    main()
