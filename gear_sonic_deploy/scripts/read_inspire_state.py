#!/usr/bin/env python3
"""Read Inspire dexterous-hand joint state on the G1 over DDS, and detect which
Inspire variant is publishing (DFX vs FTP).

RUNS ON THE ROBOT (PC2 / Jetson). Read-only: this script never publishes a
command, so it cannot move the hands.

Two Inspire variants exist and they use different Unitree-side services/IDLs
(topic names and field names below are taken from Unitree's own
xr_teleoperate `robot_hand_inspire.py`):

  DFX (RH56DFX)  service: unitreerobotics/DFX_inspire_service (`sudo ./inspire_g1`)
                 topic:   rt/inspire/state         type: unitree_go MotorStates_
                 layout:  states[0..5]  = RIGHT hand, states[6..11] = LEFT hand
                          each joint's value is `states[i].q`
  FTP (RH56DFTP) service: NaCl-1374/inspire_hand_ws (`Headless_driver_double.py`)
                 topics:  rt/inspire_hand/state/l  and  rt/inspire_hand/state/r
                 type:    inspire_sdkpy.inspire_dds.inspire_hand_state
                 layout:  `angle_act` = 6 ints, 0..1000 (0 = closed, 1000 = open)

Joint order in both: pinky, ring, middle, index, thumb_bend, thumb_rotation.

If neither service is running, nothing is published on these topics and this
script will report "none" -- start the matching service first (see README
notes in the chat, or the repos above).

Usage:
    python3 gear_sonic_deploy/scripts/read_inspire_state.py --iface eth0
    python3 gear_sonic_deploy/scripts/read_inspire_state.py --iface eth0 --variant ftp --seconds 30
"""
import argparse
import sys
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

JOINTS = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]

DFX_TOPIC = "rt/inspire/state"
FTP_TOPICS = {"left": "rt/inspire_hand/state/l", "right": "rt/inspire_hand/state/r"}


class Latest:
    """Thread-safe store of the latest values + message counts per stream."""

    def __init__(self):
        self._lock = threading.Lock()
        self._values = {}
        self._counts = {}

    def put(self, key, values):
        with self._lock:
            self._values[key] = list(values)
            self._counts[key] = self._counts.get(key, 0) + 1

    def snapshot(self):
        with self._lock:
            return dict(self._values), dict(self._counts)


def make_dfx_handler(store):
    def handler(msg):
        try:
            states = msg.states
            if len(states) < 12:
                return
            store.put("dfx/right", [states[i].q for i in range(0, 6)])
            store.put("dfx/left", [states[i].q for i in range(6, 12)])
        except Exception as e:  # never let a bad sample kill the reader thread
            print("[dfx] handler error: %r" % (e,), file=sys.stderr)
    return handler


def make_ftp_handler(store, side):
    def handler(msg):
        try:
            angle = list(msg.angle_act)
            if len(angle) != 6:
                return
            store.put("ftp/" + side, angle)
        except Exception as e:
            print("[ftp/%s] handler error: %r" % (side, e), file=sys.stderr)
    return handler


def fmt(values):
    return " ".join("%s=%g" % (n, v) for n, v in zip(JOINTS, values))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="eth0", help="DDS network interface (default eth0)")
    ap.add_argument("--variant", choices=["auto", "dfx", "ftp"], default="auto",
                    help="which Inspire variant to listen for (default: auto = both)")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to listen")
    ap.add_argument("--hz", type=float, default=2.0, help="print rate")
    args = ap.parse_args()

    ChannelFactoryInitialize(0, args.iface)
    store = Latest()
    subs = []  # keep references so subscribers are not garbage-collected

    if args.variant in ("auto", "dfx"):
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        s = ChannelSubscriber(DFX_TOPIC, MotorStates_)
        s.Init(make_dfx_handler(store), 1)
        subs.append(s)
        print("listening (DFX): %s" % DFX_TOPIC)

    if args.variant in ("auto", "ftp"):
        try:
            from inspire_sdkpy import inspire_dds
        except ImportError:
            print("FTP: python package `inspire_sdkpy` not importable -> skipping FTP "
                  "(install NaCl-1374/inspire_hand_ws first)")
            if args.variant == "ftp":
                return 2
        else:
            for side, topic in FTP_TOPICS.items():
                s = ChannelSubscriber(topic, inspire_dds.inspire_hand_state)
                s.Init(make_ftp_handler(store, side), 1)
                subs.append(s)
                print("listening (FTP): %s" % topic)

    if not subs:
        print("nothing to listen to")
        return 2

    t_end = time.time() + args.seconds
    period = 1.0 / max(args.hz, 0.1)
    try:
        while time.time() < t_end:
            time.sleep(period)
            values, counts = store.snapshot()
            if not values:
                print("[%s] no hand state received yet..." % time.strftime("%H:%M:%S"))
                continue
            for key in sorted(values):
                print("[%s] %-10s n=%-5d %s" % (time.strftime("%H:%M:%S"), key,
                                                counts[key], fmt(values[key])))
    except KeyboardInterrupt:
        pass

    _, counts = store.snapshot()
    print("\n=== summary ===")
    if any(k.startswith("dfx/") for k in counts):
        print("DETECTED: Inspire DFX (rt/inspire/state is publishing)")
    if any(k.startswith("ftp/") for k in counts):
        print("DETECTED: Inspire FTP (rt/inspire_hand/state/* is publishing)")
    if not counts:
        print("DETECTED: none -- no Inspire state was published on the topics above.")
        print("The hand service is probably not running (or the hand is not powered/connected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
