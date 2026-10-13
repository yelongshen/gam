#!/usr/bin/env python3
"""Move ONE Inspire (DFX) hand to a half-way pose, gently, then return it.

NOT YET RUN ON HARDWARE. Read the notes below before using `--go`.

Talks to Unitree's DFX_inspire_service (`sudo ./inspire_g1`) over DDS:
    rt/inspire/cmd    <- unitree_go MotorCmds_    (12 entries; only `q` is used)
    rt/inspire/state  -> unitree_go MotorStates_  (12 entries)
Layout (from Unitree's xr_teleoperate / DFX_inspire_service README):
    index 0..5  = RIGHT hand,  index 6..11 = LEFT hand
    joint order = pinky, ring, middle, index, thumb_bend, thumb_rotation
    q is in [0, 1];  0 = closed, 1 = open (Unitree's example comment; the
    hand's owner states the same convention).

What this script does
  1. Reads the current pose of both hands (aborts if the service is silent).
  2. Plans: chosen hand -> `--target` (default 0.5) on the chosen joints.
  3. DRY RUN by default: prints the plan and sends NOTHING.
  4. With `--go`: ramps linearly over `--ramp-seconds`, holds, then ramps back
     to the starting pose (unless `--no-return`).

Things you must know
  * The command message always has 12 entries. A default entry is q=0, which
    would CLOSE the other hand. So the other hand is commanded to its own
    MEASURED starting pose for the whole run. This assumes the hand holds a
    position equal to its reported state; I have not verified that on hardware.
  * The interface is position-only. There is no force or speed limit here;
    the ramp time is the only speed control. Keep the hand clear of objects.
  * Ctrl-C re-sends the last commanded pose once and exits. What the hand does
    after this script exits (hold vs. relax) is not known to me.
  * If state messages stop arriving mid-run, the script stops sending.
  * The 0=closed / 1=open direction matches Unitree's code comment. A single
    `--joints` with the default `--target 0.5` is still the safest first test.

Usage
    # dry run (sends nothing):
    python3 gear_sonic_deploy/scripts/inspire_half_move.py --hand left

    # actually move, one finger first:
    python3 gear_sonic_deploy/scripts/inspire_half_move.py --hand left --joints index --go

    # whole hand to half-way, slower:
    python3 gear_sonic_deploy/scripts/inspire_half_move.py --hand left --ramp-seconds 5 --go
"""
import argparse
import sys
import threading
import time

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
                                         ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

CMD_TOPIC = "rt/inspire/cmd"
STATE_TOPIC = "rt/inspire/state"
JOINTS = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
HAND_OFFSET = {"right": 0, "left": 6}


class StateStore:
    """Latest 12 joint positions + when they last arrived (thread-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._q = None
        self._t = 0.0

    def update(self, msg):
        try:
            states = msg.states
            if len(states) < 12:
                return
            q = [float(states[i].q) for i in range(12)]
        except Exception:
            return
        with self._lock:
            self._q = q
            self._t = time.time()

    def get(self):
        with self._lock:
            return (list(self._q) if self._q is not None else None), self._t


def clamp01(v):
    return max(0.0, min(1.0, float(v)))


def fmt(vals):
    return " ".join("%s=%.3f" % (n, v) for n, v in zip(JOINTS, vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", choices=["left", "right"], required=True)
    ap.add_argument("--iface", default="eth0", help="DDS network interface (default eth0)")
    ap.add_argument("--target", type=float, default=0.5,
                    help="target q in [0,1] for the chosen joints (default 0.5)")
    ap.add_argument("--joints", nargs="+", choices=JOINTS, default=JOINTS,
                    help="which joints to move (default: all six)")
    ap.add_argument("--ramp-seconds", type=float, default=3.0,
                    help="time to move start->target (and back). default 3.0")
    ap.add_argument("--hold-seconds", type=float, default=2.0,
                    help="time to hold at the target. default 2.0")
    ap.add_argument("--no-return", action="store_true",
                    help="do not move back to the starting pose at the end")
    ap.add_argument("--rate", type=float, default=20.0, help="command rate in Hz (default 20)")
    ap.add_argument("--go", action="store_true",
                    help="actually send commands. Without this it is a dry run.")
    args = ap.parse_args()

    if not 0.0 <= args.target <= 1.0:
        print("--target must be in [0, 1]")
        return 2
    if args.ramp_seconds <= 0 or args.rate <= 0 or args.hold_seconds < 0:
        print("--ramp-seconds and --rate must be > 0, --hold-seconds >= 0")
        return 2

    ChannelFactoryInitialize(0, args.iface)
    store = StateStore()
    sub = ChannelSubscriber(STATE_TOPIC, MotorStates_)
    sub.Init(store.update, 1)

    t_end = time.time() + 5.0
    start12, _ = store.get()
    while start12 is None and time.time() < t_end:
        time.sleep(0.05)
        start12, _ = store.get()
    if start12 is None:
        print("No state on %s within 5 s. Is `sudo ./inspire_g1` running? Nothing was sent."
              % STATE_TOPIC)
        return 3

    start12 = [clamp01(v) for v in start12]
    off = HAND_OFFSET[args.hand]
    idx = [JOINTS.index(j) for j in args.joints]

    target12 = list(start12)
    for i in idx:
        target12[off + i] = args.target

    print("hand            : %s (message indices %d..%d)" % (args.hand, off, off + 5))
    print("start (measured): %s" % fmt(start12[off:off + 6]))
    print("target          : %s" % fmt(target12[off:off + 6]))
    print("moving joints   : %s" % ", ".join(args.joints))
    other = "right" if args.hand == "left" else "left"
    oo = HAND_OFFSET[other]
    print("other hand (%s) : will be commanded to its MEASURED pose and held: %s"
          % (other, fmt(start12[oo:oo + 6])))
    print("timing          : ramp %.1fs, hold %.1fs, %s, %.0f Hz"
          % (args.ramp_seconds, args.hold_seconds,
             "no return" if args.no_return else "then return to start", args.rate))

    if not args.go:
        print("\nDRY RUN: nothing was sent. Re-run with --go to move the hand.")
        return 0

    pub = ChannelPublisher(CMD_TOPIC, MotorCmds_)
    pub.Init()
    time.sleep(0.5)  # let DDS discovery finish so early messages are not lost

    last_cmd = list(start12)

    def send(pose12):
        nonlocal last_cmd
        msg = MotorCmds_(cmds=[unitree_go_msg_dds__MotorCmd_() for _ in range(12)])
        for k in range(12):
            msg.cmds[k].q = clamp01(pose12[k])
        pub.Write(msg)
        last_cmd = list(pose12)

    period = 1.0 / args.rate

    def state_is_fresh():
        _, t = store.get()
        return (time.time() - t) < 1.0

    def ramp(a12, b12):
        n = max(1, int(args.ramp_seconds * args.rate))
        for k in range(1, n + 1):
            f = k / float(n)
            send([a + (b - a) * f for a, b in zip(a12, b12)])
            if not state_is_fresh():
                print("State stopped arriving; no further commands will be sent.")
                return False
            time.sleep(period)
        return True

    def hold(pose12, seconds):
        t_stop = time.time() + seconds
        while time.time() < t_stop:
            send(pose12)
            if not state_is_fresh():
                print("State stopped arriving; no further commands will be sent.")
                return False
            time.sleep(period)
        return True

    try:
        print("\nramping to target ...")
        ok = ramp(start12, target12) and hold(target12, args.hold_seconds)
        if ok and not args.no_return:
            print("returning to start ...")
            ok = ramp(target12, start12)
        if not ok:
            return 4
    except KeyboardInterrupt:
        print("\nCtrl-C: re-sending the last commanded pose once, then exiting.")
        try:
            send(last_cmd)
        except Exception:
            pass
        return 130

    time.sleep(0.3)
    now12, _ = store.get()
    if now12 is not None:
        print("measured now    : %s" % fmt(now12[off:off + 6]))
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
