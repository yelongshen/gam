#!/usr/bin/env python3
"""
avp_g1_inspire_teleop.py
========================
Apple Vision Pro -> Unitree G1 with Inspire 5-finger hands (6 DOF per hand) teleoperation.

  AVP (AVPHandStreamer app)  --UDP :9870-->  this script
       25 hand joints / hand                    1. wrist-local frame (same preprocessing as avp_g1_dex3_teleop.py)
                                                2. DexPilot retargeting on the Inspire URDF (dex_retargeting)
                                                3. radians -> Inspire [0,1] (1 = open, 0 = closed)
                                                4. DDS rt/inspire/cmd (MotorCmds_, DFX version)

Hardware order per hand (Inspire DFX API): [pinky, ring, middle, index, thumb_bend(pitch), thumb_rotation(yaw)]
DDS motor ids: right hand 0-5, left hand 6-11.

Assets: gear_sonic/teleoperation/assets/inspire_hand/{inspire_hand.yml, inspire_hand_{left,right}.urdf}
(from unitreerobotics/xr_teleoperate, Apache-2.0). The yml uses the key names of Unitree's dex-retargeting fork;
they are mapped to the pip `dex_retargeting` 0.5.0 keys at load time (see load_retargeter).

NOT implemented: the Inspire FTP variant (topics rt/inspire_hand/ctrl/{l,r}, needs the `inspire_dds` package).

Usage
-----
  # dry run on fake hands, no headset, no robot
  python gear_sonic/teleoperation/avp_g1_inspire_teleop.py --sim --print-only
  # real headset, print only (no robot)
  python gear_sonic/teleoperation/avp_g1_inspire_teleop.py --print-only
  # drive the hands over DDS (NIC wired to the G1)
  python gear_sonic/teleoperation/avp_g1_inspire_teleop.py --net enp4s0

Requirements: numpy, pyyaml, dex_retargeting (+pinocchio, nlopt); unitree_sdk2_python for DDS.
"""
import argparse
import contextlib
import copy
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "external_dependencies" / "unitree_sdk2_python"))

from visualize_avp_stream import Receiver, local_ip, sim_sender  # noqa: E402

ASSETS = HERE / "assets"
YML = ASSETS / "inspire_hand" / "inspire_hand.yml"
NUM_MOTORS = 6
TOPIC_CMD = "rt/inspire/cmd"
STALE_S = 0.5

# Hardware order of the Inspire DFX API (per hand), as joint names of the URDF
HW_JOINTS = {
    "left": ["L_pinky_proximal_joint", "L_ring_proximal_joint", "L_middle_proximal_joint",
             "L_index_proximal_joint", "L_thumb_proximal_pitch_joint", "L_thumb_proximal_yaw_joint"],
    "right": ["R_pinky_proximal_joint", "R_ring_proximal_joint", "R_middle_proximal_joint",
              "R_index_proximal_joint", "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint"],
}
# Joint ranges (rad) from the URDF; the API value is (max - q) / (max - min): 1 = open, 0 = closed
RANGES = [(0.0, 1.7), (0.0, 1.7), (0.0, 1.7), (0.0, 1.7), (0.0, 0.5), (-0.1, 1.3)]
DDS_ID = {"right": list(range(0, 6)), "left": list(range(6, 12))}

# wrist-local geometric frame -> hand URDF frame (identical to avp_g1_dex3_teleop.py; the Inspire and Dex3 URDF
# hand frames share the same axes: fingers along -Y, thumb on -X, pinky side +Z for the right hand)
R_WRIST_TO_URDF = np.array([[0., 0., 1.],
                            [0., -1., 0.],
                            [1., 0., 0.]])


@contextlib.contextmanager
def quiet():
    """Silence pinocchio's 'Unable to resolve filename' mesh warnings while building (meshes are not needed)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved + [null]:
            os.close(fd)


def pts_to_wrist_frame(pts, is_right):
    """Wrist-local frame from joint geometry (y: wrist->middle metacarpal, x: ulnar, z: dorsal)."""
    origin = pts[0]
    y = pts[10] - origin
    yn = np.linalg.norm(y)
    if yn < 1e-9:
        return pts - origin
    y = y / yn
    x = pts[20] - pts[5]
    if not is_right:
        x = -x
    x = x - np.dot(x, y) * y
    xn = np.linalg.norm(x)
    if xn < 1e-9:
        return pts - origin
    x = x / xn
    z = np.cross(x, y)
    x = np.cross(y, z)
    return (pts - origin) @ np.stack([x, y, z], axis=1)


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_COLOR = {"thumb": "tab:red", "index": "tab:orange", "middle": "tab:green", "ring": "tab:blue",
                "pinky": "tab:purple", None: "0.45"}


def urdf_edges(side):
    """[(parent_link, child_link, finger_or_None)] of the Inspire URDF; finger = finger whose tip is below the edge."""
    import xml.etree.ElementTree as ET
    root = ET.parse(ASSETS / "inspire_hand" / f"inspire_hand_{side}.urdf").getroot()
    edges, kids = [], {}
    for j in root.iter("joint"):
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        edges.append((p, c))
        kids.setdefault(p, []).append(c)

    def finger_below(link):
        for f in FINGERS:
            if link.endswith(f"_{f}_tip"):
                return f
        for k in kids.get(link, []):
            f = finger_below(k)
            if f:
                return f
        return None

    return [(p, c, finger_below(c)) for p, c in edges]


class InspireRetargeter:
    def __init__(self, rtype=None):
        from dex_retargeting.retargeting_config import RetargetingConfig
        RetargetingConfig.set_default_urdf_dir(str(ASSETS))
        cfg = yaml.safe_load(YML.read_text())
        self.rt, self.idx, self.to_hw = {}, {}, {}
        self.last_full, self.last_local = {}, {}   # for visualisation
        self.edges = {s: urdf_edges(s) for s in ("left", "right")}
        with quiet():
            for side in ("left", "right"):
                c = copy.deepcopy(cfg[side])
                if rtype:
                    c["type"] = rtype
                dex = c.pop("target_link_human_indices_dexpilot")
                vec = c.pop("target_link_human_indices_vector")
                # map the fork's key names to pip dex_retargeting 0.5.0
                c["target_link_human_indices"] = dex if c["type"].lower() == "dexpilot" else vec
                r = RetargetingConfig.from_dict(c).build()
                self.rt[side] = r
                self.idx[side] = np.asarray(r.optimizer.target_link_human_indices)
                self.to_hw[side] = [r.joint_names.index(n) for n in HW_JOINTS[side]]

    def __call__(self, joints, side):
        """25 AVP joints (WebXR layout) -> (radians [6], api [6]) in Inspire hardware order."""
        pts = np.asarray(joints, dtype=np.float64)[:25]
        local = pts_to_wrist_frame(pts, side == "right")
        idx = self.idx[side]
        ref = (local[idx[1]] - local[idx[0]]) @ R_WRIST_TO_URDF.T
        full = np.asarray(self.rt[side].retarget(ref))
        q = full[self.to_hw[side]]
        api = np.array([np.clip((hi - q[i]) / (hi - lo), 0.0, 1.0) for i, (lo, hi) in enumerate(RANGES)])
        self.last_full[side] = full
        self.last_local[side] = local @ R_WRIST_TO_URDF.T   # human hand in the URDF hand frame
        return q, api

    def link_positions(self, side):
        """FK of the retargeted Inspire hand (URDF hand frame) from the last retarget() call."""
        rb = self.rt[side].optimizer.robot
        rb.compute_forward_kinematics(self.last_full[side])
        names = {n for e in self.edges[side] for n in e[:2]}
        return {n: np.array(rb.get_link_pose(rb.get_link_index(n))[:3, 3]) for n in names
                if rb.get_link_index(n) is not None}


class HandViz:
    """Simulation view: retargeted Inspire hand (colored, from URDF FK) over the human AVP hand (gray)."""

    def __init__(self, retarget):
        import matplotlib
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        self.plt, self.rtg = plt, retarget
        self.fig = plt.figure("Inspire retargeting (simulation)", figsize=(12, 6))
        self.ax = {"left": self.fig.add_subplot(1, 2, 1, projection="3d"),
                   "right": self.fig.add_subplot(1, 2, 2, projection="3d")}
        self.title = self.fig.suptitle("", fontsize=9, family="monospace")
        plt.ion()
        plt.show(block=False)

    @staticmethod
    def disp(p):   # URDF hand frame (fingers -Y, thumb -X) -> plot (lateral, depth, up)
        p = np.atleast_2d(p)
        return np.stack([p[:, 2], p[:, 0], -p[:, 1]], axis=1)

    def alive(self):
        return self.plt.fignum_exists(self.fig.number)

    def update(self, live, cmd, status):
        for side, ax in self.ax.items():
            ax.cla()
            ax.set_title(f"{side} hand" + ("" if live[side] else "  (no data)"), fontsize=9)
            if live[side] and side in self.rtg.last_full:
                h = self.disp(self.rtg.last_local[side])
                for ch in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8, 9), (0, 10, 11, 12, 13, 14), (0, 15, 16, 17, 18, 19),
                           (0, 20, 21, 22, 23, 24)):
                    ax.plot(h[list(ch), 0], h[list(ch), 1], h[list(ch), 2], "-", color="0.7", lw=1.2)
                pos = self.rtg.link_positions(side)
                for p, c, f in self.rtg.edges[side]:
                    if p in pos and c in pos:
                        a, b = self.disp(pos[p])[0], self.disp(pos[c])[0]
                        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "-", color=FINGER_COLOR[f], lw=2.2)
            ax.set_xlim(-0.12, 0.12)
            ax.set_ylim(-0.12, 0.12)
            ax.set_zlim(-0.04, 0.24)
            ax.set_xlabel("lateral", fontsize=7)
            ax.set_ylabel("depth", fontsize=7)
            ax.set_zlabel("up (fingers)", fontsize=7)
            ax.tick_params(labelsize=6)
            names = ["pinky", "ring", "mid", "index", "t.bend", "t.rot"]
            ax.text2D(0.02, 0.02, "  ".join(f"{n}:{v:.2f}" for n, v in zip(names, cmd[side])),
                      transform=ax.transAxes, fontsize=7, family="monospace")
        self.title.set_text(status + "   gray = human (AVP)   colored = Inspire (retargeted)   API 1=open 0=closed")
        self.plt.pause(0.001)


class InspireDDS:
    """Publishes both hands on rt/inspire/cmd (DFX). q in [0,1]: 1 = open."""

    def __init__(self, net):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
        ChannelFactoryInitialize(0, net)
        self.pub = ChannelPublisher(TOPIC_CMD, MotorCmds_)
        self.pub.Init()
        self.msg = MotorCmds_()
        self.msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
        self.send({"left": np.ones(6), "right": np.ones(6)})  # start open

    def send(self, api):
        for side in ("left", "right"):
            for k, mid in enumerate(DDS_ID[side]):
                self.msg.cmds[mid].q = float(api[side][k])
        self.pub.Write(self.msg)


def main():
    ap = argparse.ArgumentParser(description="AVP -> Inspire hand teleoperation")
    ap.add_argument("--net", default="enp4s0", help="NIC wired to the G1 (DDS)")
    ap.add_argument("--port", type=int, default=9870, help="UDP port of the AVPHandStreamer app")
    ap.add_argument("--hz", type=float, default=50.0, help="control rate")
    ap.add_argument("--type", choices=["dexpilot", "vector"], default=None, help="override the yml retargeting type")
    ap.add_argument("--ramp", type=float, default=1.0, help="seconds to ramp from open to the target when a hand appears")
    ap.add_argument("--print-only", action="store_true", help="retarget and print; do not touch DDS / the robot")
    ap.add_argument("--sim", action="store_true", help="feed fake hands (no headset)")
    ap.add_argument("--viz", action="store_true",
                    help="simulation window: retargeted Inspire hand (URDF FK) over the human hand; use with --print-only")
    ap.add_argument("--viz-hz", type=float, default=15.0, help="window refresh rate")
    ap.add_argument("--seconds", type=float, default=0.0, help="exit after N seconds (0 = until Ctrl+C)")
    a = ap.parse_args()

    print("[Inspire] building retargeters (DexPilot on the Inspire URDF) ...")
    retarget = InspireRetargeter(a.type)
    rx = Receiver(a.port)
    rx.start()
    stop = threading.Event()
    if a.sim:
        threading.Thread(target=sim_sender, args=(a.port, stop), daemon=True).start()
    print(f"[AVP] PC IP {local_ip()}  UDP :{a.port}  (enter this in the headset app and tap Start Streaming)")
    dds = None if a.print_only else InspireDDS(a.net)
    viz = HandViz(retarget) if a.viz else None
    t_viz = 0.0
    print("[Inspire] mode:", "print-only" if a.print_only else f"DDS {TOPIC_CMD} on {a.net}",
          "| order [pinky ring middle index thumb_bend thumb_rot], API 1=open 0=closed")

    cmd = {"left": np.ones(6), "right": np.ones(6)}
    ramp_t0 = {"left": 0.0, "right": 0.0}
    was = {"left": False, "right": False}
    dt, t_start, t_print = 1.0 / a.hz, time.time(), 0.0
    try:
        while a.seconds <= 0 or time.time() - t_start < a.seconds:
            t0 = time.time()
            joints, last, _, _, _, _ = rx.snapshot()
            rad = {}
            for side in ("left", "right"):
                live = joints[side] is not None and t0 - last[side] < STALE_S
                if live:
                    q, api = retarget(joints[side], side)
                    rad[side] = q
                    if not was[side]:
                        ramp_t0[side] = t0
                        print(f"[Inspire] {side} hand ON")
                    w = 1.0 if a.ramp <= 0 else min(1.0, (t0 - ramp_t0[side]) / a.ramp)
                    cmd[side] = (1 - w) * np.ones(6) + w * api
                elif was[side]:
                    print(f"[Inspire] {side} hand LOST -> holding last command")
                was[side] = live
            if dds is not None:
                dds.send(cmd)
            if viz is not None and t0 - t_viz > 1.0 / a.viz_hz:
                t_viz = t0
                if not viz.alive():
                    break
                viz.update(dict(was), cmd, f"PC {local_ip()}:{a.port}   L {'ON' if was['left'] else 'off'}  R {'ON' if was['right'] else 'off'}")
            if t0 - t_print > 0.5 and viz is None:
                t_print = t0
                f = lambda v: " ".join(f"{x:5.2f}" for x in v)  # noqa: E731
                print(f"L{'*' if was['left'] else ' '} api [{f(cmd['left'])}]   R{'*' if was['right'] else ' '} api [{f(cmd['right'])}]"
                      + (f"   rad L [{f(rad['left'])}] R [{f(rad['right'])}]" if len(rad) == 2 else ""))
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        rx.stop()
        if dds is not None:
            dds.send({"left": np.ones(6), "right": np.ones(6)})  # open hands on exit
            print("[Inspire] hands commanded open, bye")


if __name__ == "__main__":
    main()
