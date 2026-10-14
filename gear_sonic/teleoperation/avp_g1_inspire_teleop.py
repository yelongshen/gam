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

Robot side: Unitree's DFX_inspire_service (https://github.com/unitreerobotics/DFX_inspire_service) bridges
rt/inspire/cmd <-> the hands' serial ports (RS485 frame, address 0x05CE, raw = q * 1000, clipped to 0..1000; right hand
/dev/ttyUSB1, left /dev/ttyUSB2 -- hard-coded in inspire_g1.cpp). Run it on the machine the hands are wired to:
    sudo ./inspire_g1 --network <nic>        # build: mkdir build && cd build && cmake .. && make -j6
Check it is reachable from here:  python avp_g1_inspire_teleop.py --net <nic> --check

Usage
-----
  # dry run on fake hands, no headset, no robot
  python gear_sonic/teleoperation/avp_g1_inspire_teleop.py --sim --print-only
  # real headset, print only (no robot)
  python gear_sonic/teleoperation/avp_g1_inspire_teleop.py --print-only
  # drive the hands over DDS (NIC wired to the G1)
  python gear_sonic/teleoperation/avp_g1_inspire_teleop.py --net enp4s0
  # retarget here, send over UDP to inspire_relay.py running on the robot PC (which publishes DDS on its eth0)
  python gear_sonic/teleoperation/avp_g1_inspire_teleop.py --relay 192.168.8.227

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
TOPIC_STATE = "rt/inspire/state"
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
    def __init__(self, rtype=None, alpha=None, project_dist=None, escape_dist=None):
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
                if alpha is not None:
                    c["low_pass_alpha"] = alpha   # 1.0 = no smoothing; the yml value (0.2) smooths a lot
                if project_dist is not None:
                    c["project_dist"] = project_dist   # DexPilot: thumb-to-fingertip distance below which it assumes a pinch
                if escape_dist is not None:
                    c["escape_dist"] = escape_dist
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
    """Publishes both hands on rt/inspire/cmd (DFX) and tracks the actual hand pose from rt/inspire/state.

    q in [0,1]: 1 = open. The service applies BOTH hands from every message, so a hand we do not drive must be
    re-sent at its own current value (a default 0 would close it).
    """

    def __init__(self, net):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        ChannelFactoryInitialize(0, net)
        self.pub = ChannelPublisher(TOPIC_CMD, MotorCmds_)
        self.pub.Init()
        self.msg = MotorCmds_()
        self.msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
        self._state = None
        self.sub = ChannelSubscriber(TOPIC_STATE, MotorStates_)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg):
        self._state = np.array([float(msg.states[i].q) for i in range(12)])

    def wait_state(self, timeout=2.0):
        """Actual hand pose {'right': [6], 'left': [6]} in API units (0..1), or None if no state arrives."""
        t0 = time.time()
        while self._state is None and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.latest()

    def latest(self):
        return None if self._state is None else {"right": self._state[:6].copy(), "left": self._state[6:].copy()}

    def send(self, api):
        for side in ("left", "right"):
            for k, mid in enumerate(DDS_ID[side]):
                self.msg.cmds[mid].q = float(api[side][k])
        self.pub.Write(self.msg)


def smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class InspireRelay:
    """Same interface as InspireDDS, but talks UDP to inspire_relay.py running on the robot PC (which owns the DDS side).

    send(): {"type":"cmd","seq":ms,"q":[12]}  (also the keepalive).  State comes back as {"type":"state","q":[12]}.
    """

    def __init__(self, host, port=9871):
        import json as _json
        import socket as _socket
        self._json = _json
        self.addr = (host, port)
        self.sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        self.sock.settimeout(0.2)
        self._state = None
        self._state_t = 0.0
        self._running = True
        threading.Thread(target=self._recv, daemon=True).start()

    def _recv(self):
        while self._running:
            try:
                data, _ = self.sock.recvfrom(4096)
                m = self._json.loads(data.decode())
                if m.get("type") == "state" and len(m.get("q", [])) == 12:
                    self._state, self._state_t = np.array(m["q"], dtype=float), time.time()
            except Exception:
                continue

    def _tx(self, obj):
        try:
            self.sock.sendto(self._json.dumps(obj).encode(), self.addr)
        except OSError:
            pass

    def wait_state(self, timeout=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._tx({"type": "hello"})
            if self._state is not None:
                break
            time.sleep(0.2)
        return self.latest()

    def latest(self):
        return None if self._state is None else {"right": self._state[:6].copy(), "left": self._state[6:].copy()}

    def state_age(self):
        return time.time() - self._state_t if self._state is not None else float("inf")

    def send(self, api):
        q = [float(api["right"][k]) for k in range(6)] + [float(api["left"][k]) for k in range(6)]
        self._tx({"type": "cmd", "seq": int(time.time() * 1000), "q": q})

    def close(self):
        self._running = False
        self.sock.close()


def check_state(net, secs=3.0, relay=None):
    """Listen to the hand state (rt/inspire/state via DDS, or via inspire_relay.py if relay=(host,port)) and print it."""
    if relay is not None:
        r = InspireRelay(*relay)
        t0, got = time.time(), 0
        while time.time() - t0 < secs:
            st = r.wait_state(0.5)
            if st is None:
                continue
            got += 1
            print("[check] right q", " ".join(f"{v:.2f}" for v in st["right"]), "| left q",
                  " ".join(f"{v:.2f}" for v in st["left"]), "(via relay)")
            time.sleep(0.5)
        if got == 0:
            print(f"[check] NO state from the relay at {relay[0]}:{relay[1]}: is inspire_relay.py running there (and --allow "
                  "this PC's IP)?")
        r.close()
        return got > 0
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
    ChannelFactoryInitialize(0, net)
    sub = ChannelSubscriber("rt/inspire/state", MotorStates_)
    sub.Init()
    t0, got = time.time(), 0
    while time.time() - t0 < secs:
        msg = sub.Read(200)
        if msg is None:
            continue
        got += 1
        q = [float(msg.states[i].q) for i in range(12)]
        lost = [int(msg.states[i].lost) for i in (0, 6)]
        print("[check] right q", " ".join(f"{v:.2f}" for v in q[:6]), "| left q", " ".join(f"{v:.2f}" for v in q[6:]),
              f"| lost R/L {lost}")
        time.sleep(0.5)
    if got == 0:
        print(f"[check] NO rt/inspire/state on '{net}' within {secs:.0f}s: is inspire_g1 running and on the same DDS network?")
    return got > 0


def main():
    ap = argparse.ArgumentParser(description="AVP -> Inspire hand teleoperation")
    ap.add_argument("--net", default="enp4s0", help="NIC on the same network as the G1 / inspire service (DDS)")
    ap.add_argument("--check", action="store_true", help="only listen to rt/inspire/state for a few seconds and exit")
    ap.add_argument("--relay", default=None, metavar="HOST[:PORT]",
                    help="send the 12 hand values over UDP to inspire_relay.py on the robot PC (default port 9871) "
                         "instead of publishing DDS from this machine; the real hand pose comes back the same way")
    ap.add_argument("--port", type=int, default=9870, help="UDP port of the AVPHandStreamer app")
    ap.add_argument("--hz", type=float, default=50.0, help="control rate")
    ap.add_argument("--type", choices=["dexpilot", "vector"], default=None, help="override the yml retargeting type")
    ap.add_argument("--project-dist", type=float, default=None,
                    help="DexPilot only: if the thumb tip is closer than this (m, default 0.03) to a fingertip it assumes a "
                         "pinch and curls that finger, which can bend fingers on an open hand; lower it (e.g. 0.01) or use "
                         "--type vector")
    ap.add_argument("--escape-dist", type=float, default=None, help="DexPilot only: distance (m, default 0.05) that ends a pinch")
    ap.add_argument("--alpha", type=float, default=None,
                    help="retargeting low-pass filter (0..1, 1 = off). The yml uses 0.2, which smooths a lot; try 0.6-1.0 for a "
                         "snappier hand")
    ap.add_argument("--settle", "--ramp", dest="settle", type=float, default=3.0,
                    help="seconds to move from the hand's CURRENT pose to your tracked pose when a hand first appears "
                         "(smoothstep; the target keeps tracking your hand during the settle)")
    ap.add_argument("--assume-open", action="store_true",
                    help="if rt/inspire/state is not received, start the settle from 'open' instead of refusing to run")
    ap.add_argument("--lost", choices=["hold", "open"], default="open",
                    help="what a hand does once it is OFF (not tracked for --lost-grace seconds): hold the last pose, or "
                         "ramp to open over --off-ramp seconds. A hand that was never tracked is left where it is.")
    ap.add_argument("--lost-grace", type=float, default=1.0,
                    help="seconds without tracking before a hand counts as OFF; shorter dropouts just hold the last "
                         "command and resume without a new settle")
    ap.add_argument("--off-ramp", type=float, default=2.0,
                    help="seconds for the OFF ramp to open (also used to open the hands when the script exits); 0 = immediately")
    ap.add_argument("--print-only", action="store_true", help="retarget and print; do not touch DDS / the robot")
    ap.add_argument("--sim", action="store_true", help="feed fake hands (no headset)")
    ap.add_argument("--viz", action="store_true",
                    help="simulation window: retargeted Inspire hand (URDF FK) over the human hand; use with --print-only")
    ap.add_argument("--viz-hz", type=float, default=15.0, help="window refresh rate")
    ap.add_argument("--seconds", type=float, default=0.0, help="exit after N seconds (0 = until Ctrl+C)")
    a = ap.parse_args()

    relay_addr = None
    if a.relay:
        host, _, p = a.relay.partition(":")
        relay_addr = (host, int(p) if p else 9871)

    if a.check:
        sys.exit(0 if check_state(a.net, relay=relay_addr) else 1)

    print("[Inspire] building retargeters (DexPilot on the Inspire URDF) ...")
    retarget = InspireRetargeter(a.type, a.alpha, a.project_dist, a.escape_dist)
    rx = Receiver(a.port)
    rx.start()
    stop = threading.Event()
    if a.sim:
        threading.Thread(target=sim_sender, args=(a.port, stop), daemon=True).start()
    print(f"[AVP] PC IP {local_ip()}  UDP :{a.port}  (enter this in the headset app and tap Start Streaming)")
    dds = None if a.print_only else (InspireRelay(*relay_addr) if relay_addr else InspireDDS(a.net))
    viz = HandViz(retarget) if a.viz else None
    t_viz = 0.0
    print("[Inspire] mode:", "print-only" if a.print_only else
          (f"UDP relay -> {relay_addr[0]}:{relay_addr[1]}" if relay_addr else f"DDS {TOPIC_CMD} on {a.net}"),
          "| order [pinky ring middle index thumb_bend thumb_rot], API 1=open 0=closed")

    cmd = {"left": np.ones(6), "right": np.ones(6)}   # what we send; before a hand is tracked it stays at the actual pose
    if dds is not None:
        state = dds.wait_state(2.0)
        if state is None:
            if not a.assume_open:
                sys.exit(f"[Inspire] no hand state from {'the relay' if relay_addr else repr(a.net)}: cannot know where the hands "
                         "are. Is inspire_g1 (and inspire_relay.py, if --relay) running? (--assume-open starts the settle from "
                         "'open' anyway)")
            print("[Inspire] WARNING: no hand state; assuming the hands are OPEN at start")
        else:
            cmd = {s: np.clip(state[s], 0.0, 1.0) for s in ("left", "right")}
            f0 = lambda v: " ".join(f"{x:.2f}" for x in v)  # noqa: E731
            print(f"[Inspire] hands now: right [{f0(cmd['right'])}]  left [{f0(cmd['left'])}]  -> settle {a.settle:.1f}s to the tracked pose")
    start = {"left": cmd["left"].copy(), "right": cmd["right"].copy()}
    ramp_t0 = {"left": 0.0, "right": 0.0}
    was = {"left": False, "right": False}          # ACTIVE: tracked, or lost for less than --lost-grace
    ever = {"left": False, "right": False}         # tracked at least once
    last_live = {"left": 0.0, "right": 0.0}
    off_t0 = {"left": 0.0, "right": 0.0}
    off_from = {"left": cmd["left"].copy(), "right": cmd["right"].copy()}
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
                    last_live[side] = t0
                    if not was[side]:   # ON: first appearance, or back after being OFF
                        ramp_t0[side] = t0
                        start[side] = cmd[side].copy()   # settle starts from what we last commanded (= where the hand is)
                        was[side] = ever[side] = True
                        print(f"[Inspire] {side} hand ON -> settling over {a.settle:.1f}s")
                    w = 1.0 if a.settle <= 0 else smoothstep((t0 - ramp_t0[side]) / a.settle)
                    cmd[side] = (1 - w) * start[side] + w * api
                elif was[side] and t0 - last_live[side] > a.lost_grace:   # OFF
                    was[side] = False
                    off_t0[side], off_from[side] = t0, cmd[side].copy()
                    print(f"[Inspire] {side} hand OFF (no tracking for {a.lost_grace:.1f}s) -> "
                          + (f"opening over {a.off_ramp:.1f}s" if a.lost == "open" else "holding last pose"))
                if not live and not was[side] and ever[side] and a.lost == "open":
                    w = 1.0 if a.off_ramp <= 0 else smoothstep((t0 - off_t0[side]) / a.off_ramp)
                    cmd[side] = (1 - w) * off_from[side] + w * np.ones(6)
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
                act = dds.latest() if dds is not None else None
                lag = ("   lag |cmd-actual| R %.2f L %.2f" % tuple(float(np.max(np.abs(cmd[s] - act[s]))) for s in ("right", "left"))
                       if act is not None else "")
                print(f"L{'*' if was['left'] else ' '} api [{f(cmd['left'])}]   R{'*' if was['right'] else ' '} api [{f(cmd['right'])}]"
                      + lag + (f"   rad L [{f(rad['left'])}] R [{f(rad['right'])}]" if len(rad) == 2 else ""))
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        rx.stop()
        if dds is not None:
            # release: ramp both hands to open over --off-ramp (Ctrl+C again = open immediately)
            frm = {s: cmd[s].copy() for s in ("left", "right")}
            n = int(a.off_ramp * a.hz)
            try:
                for i in range(1, n + 1):
                    w = smoothstep(i / n)
                    dds.send({s: (1 - w) * frm[s] + w * np.ones(6) for s in frm})
                    time.sleep(dt)
            except KeyboardInterrupt:
                pass
            dds.send({"left": np.ones(6), "right": np.ones(6)})
            print("[Inspire] hands commanded open, bye")


if __name__ == "__main__":
    main()
