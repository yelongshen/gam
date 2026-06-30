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

# Official approach: pass YAML dict directly, matching hand_retargeting.py exactly:
#   RetargetingConfig.from_dict(self.cfg['left'])
# from_dict() handles np.array conversion for target_link_human_indices internally.
_left_retarget  = _RetargetingConfig.from_dict(_cfg_yaml["left"]).build()
_right_retarget = _RetargetingConfig.from_dict(_cfg_yaml["right"]).build()

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

# Coordinate transform: wrist-local geometric frame → Unitree URDF hand frame.
# Derived as R_old @ R_wrist_local_to_world, where R_old=[[0,1,0],[0,0,1],[1,0,0]]
# was the original OpenXR-world → URDF rotation, and the wrist-local frame has:
#   x_local = OpenXR +X (ulnar)   y_local = OpenXR -Z (distal)   z_local = OpenXR +Y (dorsal)
# Mapping:
#   x_local (ulnar)  → URDF +Z      y_local (distal) → URDF -Y      z_local (dorsal) → URDF +X
# Verification: URDF zero-pose fingertip at (0, -0.174, ±0.029) → both -Y (distal ✓) and ±Z (lateral ✓)
_R_WRIST_TO_URDF = np.array([[0., 0., 1.],
                              [0.,-1., 0.],
                              [1., 0., 0.]])


def _pts_to_wrist_frame(pts: np.ndarray, is_right: bool) -> np.ndarray:
    """Transform 25 joint positions into the estimated wrist-local frame.

    Replicates tv_wrapper.py fast_mat_inv(wrist_pose) using hand geometry.
    Axes are defined consistently for both hands so _R_WRIST_TO_URDF applies:
      origin : wrist (joint 0)
      y-axis : wrist → middle metacarpal (joint 10) — distal
      x-axis : ulnar direction (toward pinky), orthogonalized against y
               For right hand: pts[20]-pts[5]; for left hand negated so dorsal
               z = x×y always points dorsally (right-handed frame).
      z-axis : x × y — dorsal
    """
    origin = pts[0]
    y = pts[10] - origin
    yn = np.linalg.norm(y)
    if yn < 1e-9:
        return pts - origin
    y /= yn
    # Ulnar direction: pinky_meta - index_meta for right hand; negate for left so
    # that z = x × y always points dorsally (ensures right-handed coordinate system).
    x = pts[20] - pts[5]
    if not is_right:
        x = -x
    x -= np.dot(x, y) * y
    xn = np.linalg.norm(x)
    if xn < 1e-9:
        return pts - origin
    x /= xn
    z = np.cross(x, y)
    x = np.cross(y, z)          # re-orthogonalize x
    R_wrist = np.stack([x, y, z], axis=1)   # (3,3): columns are local axes in world
    return (pts - origin) @ R_wrist          # (25,3) in wrist-local frame


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
    pts        = np.array(joints[:25], dtype=np.float64)
    retargeter = _right_retarget if is_right else _left_retarget
    indices    = _right_indices  if is_right else _left_indices
    to_hw      = _right_to_hw   if is_right else _left_to_hw

    # Step 1: wrist-local frame (invariant to wrist rotation in world space).
    pts_local = _pts_to_wrist_frame(pts, is_right)
    # Step 2: 6 DexPilot inter-finger vectors in wrist-local frame.
    raw_ref = pts_local[indices[1]] - pts_local[indices[0]]  # (6, 3)
    # Step 3: rotate wrist-local frame → Unitree URDF frame for DexPilot.
    ref = (raw_ref @ _R_WRIST_TO_URDF.T)                    # (6, 3)
    return retargeter.retarget(ref)[to_hw]

# ---------------------------------------------------------------------------
# Dex3 forward kinematics — exact kinematics via pinocchio
# ---------------------------------------------------------------------------
# Display coordinate frame (URDF → display via _R_URDF_TO_DISPLAY):
#   FK_x = URDF_z  (lateral: right-hand index at −X, middle at +X; left mirrored)
#   FK_y = −URDF_y (distal: fingers extend in +Y)
#   FK_z = URDF_x  (palmar: thumb extends toward −Z at URDF zero; +Z = dorsal)
#
# Verified at URDF zero (pinocchio):
#   right index_tip URDF=(0.0017,−0.1735,−0.0285) → FK=(−0.029, 0.174, 0.002) ✓
#   right middle_tip URDF=(0.0018,−0.1735,+0.0285) → FK=(+0.029, 0.174, 0.002) ✓
#   right thumb_tip URDF=(−0.115,−0.023,0.000) → FK=(0.000, 0.023, −0.115) (palmar)

import pinocchio as _pin

_pin_models: dict = {}
_pin_datas:  dict = {}
for _side in ('left', 'right'):
    _m = _pin.buildModelFromUrdf(
        str(_ASSETS / f'unitree_hand/unitree_dex3_{_side}.urdf'))
    _pin_models[_side] = _m
    _pin_datas[_side]  = _m.createData()

# Hardware order [th0,th1,th2,mid0,mid1,idx0,idx1] → pinocchio q-vector index
# (same mapping for both hands per inspecting model joint order)
_HW2PIN_IDX = [4, 5, 6, 2, 3, 0, 1]

_R_URDF_TO_DISPLAY = np.array([[0., 0., 1.],
                                [0.,-1., 0.],
                                [1., 0., 0.]])


def dex3_fk(angles, is_right):
    """Exact Dex3-1 FK via pinocchio URDF.

    Input: angles in hardware order [th0, th1, th2, mid0, mid1, idx0, idx1]
    Output: dict of display-frame 3D positions for bone rendering.
    """
    side  = 'right' if is_right else 'left'
    model = _pin_models[side]
    data  = _pin_datas[side]
    pf    = 'right_hand_' if is_right else 'left_hand_'

    q = _pin.neutral(model)
    for i, qi in enumerate(_HW2PIN_IDX):
        q[qi] = float(angles[i])

    _pin.forwardKinematics(model, data, q)
    _pin.updateFramePlacements(model, data)

    R = _R_URDF_TO_DISPLAY

    def jp(name):   # joint origin position in display frame
        return R @ data.oMi[model.getJointId(name)].translation.copy()

    def fp(name):   # body/frame position in display frame
        return R @ data.oMf[model.getFrameId(name)].translation.copy()

    return {
        'palm':      np.zeros(3),
        'thumb_cmc': jp(f'{pf}thumb_0_joint'),
        'thumb_mcp': jp(f'{pf}thumb_1_joint'),
        'thumb_ip':  jp(f'{pf}thumb_2_joint'),
        'thumb_tip': fp('thumb_tip'),
        'idx_meta':  jp(f'{pf}index_0_joint'),
        'idx_pip':   jp(f'{pf}index_1_joint'),
        'idx_tip':   fp('index_tip'),
        'mid_meta':  jp(f'{pf}middle_0_joint'),
        'mid_pip':   jp(f'{pf}middle_1_joint'),
        'mid_tip':   fp('middle_tip'),
    }

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
# FK test mode  (--test flag, no UDP needed)
# ---------------------------------------------------------------------------

def _run_fk_test():
    """Static FK pose grid for visual debugging.

    Shows 4 reference poses × 2 hands (left top row, right bottom row).
    Angles use URDF joint limits directly so each column is physically meaningful.
    Joint order: [thumb_abd, thumb_mcp, thumb_ip, mid_mcp, mid_pip, idx_mcp, idx_pip]

    URDF limits (for reference):
      Right: index/middle  [0, +1.57/1.75]  positive = flex
             thumb_1       [-0.920, +0.720]  -0.920 = abducted
             thumb_2       [-1.750, 0]       -1.750 = flexed
      Left:  index/middle  [-1.57/-1.75, 0]  negative = flex
             thumb_1       [-0.724, +0.920]  +0.920 = abducted
             thumb_2       [0, +1.750]        +1.750 = flexed
    """
    # Each pose: (label, q_right, q_left)
    # Joint hardware order: [thumb_abd, thumb_mcp, thumb_ip, mid_mcp, mid_pip, idx_mcp, idx_pip]
    # Right URDF limits: thumb_0=[-1.047,+1.047]  thumb_1=[-0.920,+0.720]  thumb_2=[-1.745,0]
    #                    mid/idx_0=[0,+1.571]  mid/idx_1=[0,+1.745]
    # Left  URDF limits: thumb_0=[-1.047,+1.047]  thumb_1=[-0.724,+0.920]  thumb_2=[0,+1.745]
    #                    mid/idx_0=[-1.571,0]  mid/idx_1=[-1.745,0]
    POSES = [
        (
            "zero\n(URDF q=0 rest)",
            [ 0.0,  0.0,   0.0,    0.0,  0.0,   0.0,  0.0],   # right
            [ 0.0,  0.0,   0.0,    0.0,  0.0,   0.0,  0.0],   # left
        ),
        (
            "open\n(thumb spread,\nfingers straight)",
            [-1.047, -0.920,  0.0,    0.0,  0.0,   0.0,  0.0],  # right: th0/th1 at lower limit = spread
            [-1.047, +0.920,  0.0,    0.0,  0.0,   0.0,  0.0],  # left:  th1 upper = spread
        ),
        (
            "fist\n(thumb spread,\nfingers full flex)",
            [-1.047, -0.920, -1.745,  1.571, 1.745,  1.571, 1.745],  # right
            [-1.047, +0.920, +1.745, -1.571,-1.745, -1.571,-1.745],  # left
        ),
        (
            "pinch\n(tip-to-tip,\nindex flex)",
            [-1.000,  0.400, -1.745,  0.0,  0.0,  1.571, 0.800],  # right: tip dist ~8 mm
            [-1.000, -0.400, +1.745,  0.0,  0.0, -1.571,-0.800],  # left mirror
        ),
    ]

    plt.style.use("dark_background")
    n = len(POSES)
    fig = plt.figure(figsize=(4*n, 10), facecolor="#1a1a2e")
    fig.suptitle("Dex3-1 FK Reference Poses  [--test mode]\n"
                 "Top row = RIGHT hand   |   Bottom row = LEFT hand",
                 color="white", fontsize=11, y=0.99)

    for col, (pose_name, q_right, q_left) in enumerate(POSES):
        for row, (q, is_right, side_label) in enumerate([
            (q_right, True,  "RIGHT"),
            (q_left,  False, "LEFT"),
        ]):
            ax = fig.add_subplot(2, n, row * n + col + 1, projection="3d")
            ax.set_title(f"{side_label}: {pose_name}", color="white", fontsize=7, pad=2)
            _style_ax(ax)
            _draw_dex3(ax, np.array(q, dtype=float), active=True, is_right=is_right)
            ax.set_xlim(-0.14, 0.14)
            ax.set_ylim(-0.04, 0.22)
            ax.set_zlim(-0.18, 0.08)
            ax.view_init(elev=20, azim=-45)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=9870)
    ap.add_argument("--test", action="store_true",
                    help="Show static FK reference poses and exit (no AVP device needed)")
    args=ap.parse_args()

    if args.test:
        _run_fk_test()
        return

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
                ax.set_xlim(-0.12, 0.12)
                ax.set_ylim(-0.03, 0.20)
                ax.set_zlim(-0.16, 0.08)
                ax.view_init(elev=20, azim=-60)

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
            plt.pause(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        print("\n[Visualizer] Stopped.")

if __name__=="__main__":
    main()
