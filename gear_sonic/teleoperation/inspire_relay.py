#!/usr/bin/env python3
"""
inspire_relay.py  --  runs on the ROBOT PC (the machine whose DDS network contains the DFX_inspire_service).

Why: the retargeting (avp_g1_inspire_teleop.py, needs torch/pinocchio/dex_retargeting) runs on another PC. This relay
has NO heavy dependencies (python3 stdlib + unitree_sdk2py). It

  * receives the 12 hand values over UDP from the teleop PC       (JSON, port 9871)
  * publishes them as MotorCmds_ on rt/inspire/cmd                  (DDS, --iface, e.g. eth0)   -> inspire_g1 service -> hands
  * reads rt/inspire/state and sends the REAL hand pose back over UDP to the teleop PC (so its settle starts from
    where the hands actually are)

UDP packets (JSON, one per datagram):
  PC -> relay : {"type":"hello"}                                  announce / keepalive (relay then streams state back)
                {"type":"cmd","seq":n,"q":[12 floats 0..1]}       seq = ms clock (monotonic across sender restarts);
                                                                  q order = DDS motor ids: right 0-5, left 6-11,
                                                                  each [pinky ring middle index thumb_bend thumb_rot], 1 = open
  relay -> PC : {"type":"state","t":unix,"q":[12 floats]}

Safety: values are clamped to [0,1], non-finite/short packets are dropped, out-of-order seq is dropped, and if no cmd
arrives for --timeout seconds the relay STOPS publishing (the service then stops applying commands, the hands hold their
last position). --allow restricts which source IPs may command the hands.

    python3 gear_sonic/teleoperation/inspire_relay.py --iface eth0 --allow 192.168.8.192
    python3 gear_sonic/teleoperation/inspire_relay.py --dry-run          # no DDS: fake hand that follows the commands
"""
import argparse
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOPIC_CMD, TOPIC_STATE = "rt/inspire/cmd", "rt/inspire/state"


class DDSSide:
    def __init__(self, iface):
        sys.path.insert(0, str(REPO / "external_dependencies" / "unitree_sdk2_python"))
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        ChannelFactoryInitialize(0, iface)
        self.pub = ChannelPublisher(TOPIC_CMD, MotorCmds_)
        self.pub.Init()
        self.msg = MotorCmds_()
        self.msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
        self.state = None
        self.sub = ChannelSubscriber(TOPIC_STATE, MotorStates_)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg):
        self.state = [float(msg.states[i].q) for i in range(12)]

    def publish(self, q):
        for i in range(12):
            self.msg.cmds[i].q = float(q[i])
        self.pub.Write(self.msg)


class DryRunSide:
    """No DDS: a fake pair of hands (starting half closed) that moves toward whatever is commanded."""

    def __init__(self):
        self.state = [0.5] * 12
        self._last = time.time()

    def publish(self, q):
        now = time.time()
        step = 1.5 * (now - self._last)       # fake hand speed: 1.5 units/s
        self._last = now
        self.state = [s + max(-step, min(step, c - s)) for s, c in zip(self.state, q)]


def main():
    ap = argparse.ArgumentParser(description="UDP -> DDS relay for the Inspire hands (run on the robot PC)")
    ap.add_argument("--iface", default="eth0", help="DDS network interface that reaches the DFX_inspire_service")
    ap.add_argument("--port", type=int, default=9871, help="UDP port to listen on")
    ap.add_argument("--rate", type=float, default=50.0, help="DDS publish rate while commands are fresh (Hz)")
    ap.add_argument("--state-hz", type=float, default=25.0, help="rate of state packets sent back (Hz)")
    ap.add_argument("--timeout", type=float, default=0.5, help="stop publishing if no cmd for this long (s)")
    ap.add_argument("--allow", nargs="*", default=[], help="only accept packets from these source IPs (recommended)")
    ap.add_argument("--dry-run", action="store_true", help="do not touch DDS; simulate the hands")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)   # python >= 3.7
    except AttributeError:
        pass

    side = DryRunSide() if a.dry_run else DDSSide(a.iface)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", a.port))
    sock.settimeout(0.1)
    print(f"[relay] UDP :{a.port}  ->  DDS {TOPIC_CMD} on {'(dry-run)' if a.dry_run else a.iface}; "
          f"allow={a.allow or 'any'}  cmd timeout {a.timeout}s")

    lock = threading.Lock()
    S = {"q": None, "cmd_t": 0.0, "seq": -1.0, "peer": None, "peer_t": 0.0, "n_cmd": 0, "dropped": 0}
    running = True

    def recv_loop():
        while running:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if a.allow and addr[0] not in a.allow:
                S["dropped"] += 1
                continue
            try:
                m = json.loads(data.decode())
                typ = m.get("type")
            except Exception:
                S["dropped"] += 1
                continue
            now = time.time()
            with lock:
                if typ in ("hello", "cmd"):
                    if S["peer"] != addr:
                        print(f"[relay] peer {addr[0]}:{addr[1]}")
                    S["peer"], S["peer_t"] = addr, now
                if typ == "cmd":
                    q, seq = m.get("q"), m.get("seq")
                    if (not isinstance(q, list) or len(q) != 12 or not isinstance(seq, (int, float))
                            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in q)):
                        S["dropped"] += 1
                        continue
                    if seq <= S["seq"]:          # out of order / duplicate (the sender uses a ms clock as seq)
                        continue
                    S["q"] = [min(1.0, max(0.0, float(v))) for v in q]
                    S["seq"], S["cmd_t"] = seq, now
                    S["n_cmd"] += 1

    threading.Thread(target=recv_loop, daemon=True).start()

    dt_pub, dt_state = 1.0 / a.rate, 1.0 / a.state_hz
    t_state = t_print = 0.0
    publishing = False
    try:
        while True:
            t0 = time.time()
            with lock:
                q, fresh = S["q"], (S["q"] is not None and t0 - S["cmd_t"] < a.timeout)
                peer, peer_ok = S["peer"], (S["peer"] is not None and t0 - S["peer_t"] < 3.0)
            if fresh:
                side.publish(q)
                if not publishing:
                    print("[relay] commands flowing -> publishing")
                publishing = True
            elif publishing:
                print(f"[relay] no command for {a.timeout}s -> stopped publishing (hands hold their last position)")
                publishing = False
            if peer_ok and side.state is not None and t0 - t_state >= dt_state:
                t_state = t0
                sock.sendto(json.dumps({"type": "state", "t": t0, "q": side.state}).encode(), peer)
            if t0 - t_print > 3.0:
                t_print = t0
                st = "none" if side.state is None else " ".join(f"{v:.2f}" for v in side.state)
                print(f"[relay] peer={peer[0] if peer_ok else '-'} cmds={S['n_cmd']} dropped={S['dropped']} "
                      f"publishing={publishing} state[{st}]")
            time.sleep(max(0.0, dt_pub - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        sock.close()
        print("[relay] bye (hands left where they are)")


if __name__ == "__main__":
    main()
