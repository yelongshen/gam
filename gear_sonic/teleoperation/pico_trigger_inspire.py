#!/usr/bin/env python3
"""
pico_trigger_inspire.py -- open/close the G1's Inspire hands with the PICO controller triggers.

  PICO controllers --> XRoboToolkit PC service (this PC) --> this script (xrobotoolkit_sdk)
       trigger L/R (0..1) --> grasp mapping --> speed limit --> 6 values per hand (API 1 = open, 0 = closed)
       --> UDP --> inspire_relay.py (robot PC) --> DDS rt/inspire/cmd --> inspire_g1 --> Inspire hands

Order of the 6 values per hand: [pinky, ring, middle, index, thumb_bend, thumb_rotation].
The four fingers close to --closed-value (default 0.2) and the thumb closes to thumb_bend 0.4 / thumb_rotation 0 (bent, and
rotated across the palm) with the same trigger. Use --no-thumb to keep the thumb open. A closed thumb together with
closed fingers may collide mechanically: pull the trigger slowly the first time, and raise --thumb-bend-closed /
--thumb-rot-closed if the thumb and fingers touch.

Safety:
  * starts from the REAL hand pose (read back through the relay) and never sends anything before that is known
    (--assume-open overrides, assuming open hands);
  * ARMING: a hand is only driven once its trigger has been released (< --arm-below) after the script started, so a trigger
    that happens to be pulled at start-up cannot slam the hand shut;
  * the command moves at most --max-speed per second, and never below --closed-value (default 0.2) on the fingers;
  * if the controller data stops (headset not sending / paired off), the hands ramp open over --off-ramp seconds;
  * Ctrl+C ramps both hands open. The relay itself stops publishing 0.5 s after the last command.

    python gear_sonic/teleoperation/pico_trigger_inspire.py --print-only                  # just show the trigger values
    python gear_sonic/teleoperation/pico_trigger_inspire.py --relay 192.168.8.227 --seconds 60
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import xrobotoolkit_sdk as xrt
except ImportError:  # pragma: no cover
    xrt = None

FINGERS = ["pinky", "ring", "middle", "index"]


def smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def shaped(t, deadzone, gamma):
    """trigger 0..1 -> closure 0..1 with a dead zone at the start and an optional curve."""
    c = (min(max(t, 0.0), 1.0) - deadzone) / max(1e-6, 1.0 - deadzone)
    return min(max(c, 0.0), 1.0) ** gamma


def main():
    ap = argparse.ArgumentParser(description="PICO triggers -> Inspire hand open/close")
    ap.add_argument("--relay", default=None, metavar="HOST[:PORT]", help="inspire_relay.py on the robot PC (port 9871)")
    ap.add_argument("--print-only", action="store_true", help="read the triggers and print the commands; send nothing")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--closed-value", type=float, default=0.2,
                    help="API value of the four fingers at a fully pulled trigger (0 = fist, 1 = open); 0.2 leaves the "
                         "fingers a little open so objects are not crushed")
    ap.add_argument("--no-thumb", action="store_true", help="keep the thumb open (only the four fingers move)")
    ap.add_argument("--thumb-bend-closed", type=float, default=0.4,
                    help="thumb_bend value at a fully pulled trigger (0 = fully bent, 1 = open); default 0.4")
    ap.add_argument("--thumb-rot-closed", type=float, default=0.0,
                    help="thumb_rotation value at a fully pulled trigger (0 = rotated across the palm, 1 = open); default 0.0")
    ap.add_argument("--deadzone", type=float, default=0.05, help="trigger below this does nothing")
    ap.add_argument("--gamma", type=float, default=1.0, help=">1 = gentler start of the pull")
    ap.add_argument("--max-speed", type=float, default=1.0, help="max change of any value per second (1.0 = full range in 1 s)")
    ap.add_argument("--arm-below", type=float, default=0.1, help="a trigger must be below this before its hand is driven")
    ap.add_argument("--stale", type=float, default=0.7, help="seconds without new controller data before the hands open")
    ap.add_argument("--off-ramp", type=float, default=2.0, help="seconds to open when data is lost / on exit")
    ap.add_argument("--assume-open", action="store_true", help="start even if the real hand pose cannot be read")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = until Ctrl+C)")
    a = ap.parse_args()
    if xrt is None:
        sys.exit("xrobotoolkit_sdk not importable (use .venv_teleop)")
    if not a.print_only and not a.relay:
        sys.exit("give --relay HOST[:PORT] (or --print-only)")

    link = None
    cmd = {"left": np.ones(6), "right": np.ones(6)}
    if not a.print_only:
        from avp_g1_inspire_teleop import InspireRelay
        host, _, p = a.relay.partition(":")
        link = InspireRelay(host, int(p) if p else 9871)
        state = link.wait_state(2.5)
        if state is None:
            if not a.assume_open:
                sys.exit(f"[trigger] no hand state from the relay at {a.relay}: is inspire_relay.py running on the robot "
                         "PC (with --allow for this PC's IP) and inspire_g1 behind it? (--assume-open to ignore)")
            print("[trigger] WARNING: no hand state; assuming the hands are OPEN")
        else:
            cmd = {s: np.clip(state[s], 0.0, 1.0) for s in ("left", "right")}
            print("[trigger] hands now: right [%s]  left [%s]" % (" ".join(f"{v:.2f}" for v in cmd["right"]),
                                                                  " ".join(f"{v:.2f}" for v in cmd["left"])))

    open_pose = np.ones(6)
    closed_pose = np.array([a.closed_value] * 4 + [1.0 if a.no_thumb else a.thumb_bend_closed,
                                                   1.0 if a.no_thumb else a.thumb_rot_closed])
    xrt.init()
    print("[trigger] waiting for controller data from the headset (XRoboToolkit: Controller ON, Send ON) ...")
    armed = {"left": False, "right": False}
    last_ts, last_change = None, time.time()
    have_data, t_start, t_print = False, time.time(), 0.0
    dt = 1.0 / a.hz
    off_t0, off_from, was_off = 0.0, None, False
    try:
        while a.seconds <= 0 or time.time() - t_start < a.seconds:
            t0 = time.time()
            ts = int(xrt.get_time_stamp_ns())
            if ts != last_ts:
                last_ts, last_change = ts, t0
            fresh = ts != 0 and (t0 - last_change) < a.stale
            trig = {"left": float(xrt.get_left_trigger()), "right": float(xrt.get_right_trigger())}
            if fresh and not have_data:
                have_data = True
                print("[trigger] controller data is flowing")
            if fresh:
                was_off = False
                for side in ("left", "right"):
                    if not armed[side]:
                        if trig[side] < a.arm_below:
                            armed[side] = True
                            print(f"[trigger] {side} hand armed (trigger released)")
                        continue
                    c = shaped(trig[side], a.deadzone, a.gamma)
                    target = open_pose + c * (closed_pose - open_pose)
                    step = a.max_speed * dt
                    cmd[side] = cmd[side] + np.clip(target - cmd[side], -step, step)
            elif have_data:
                if not was_off:
                    was_off, off_t0, off_from = True, t0, {s: cmd[s].copy() for s in cmd}
                    armed = {"left": False, "right": False}
                    print(f"[trigger] controller data stopped -> opening the hands over {a.off_ramp:.1f}s")
                w = 1.0 if a.off_ramp <= 0 else smoothstep((t0 - off_t0) / a.off_ramp)
                for side in cmd:
                    cmd[side] = (1 - w) * off_from[side] + w * open_pose
            if link is not None and (have_data):   # nothing is sent until real controller data has been seen
                link.send(cmd)
            if t0 - t_print > 0.5:
                t_print = t0
                f = lambda v: " ".join(f"{x:4.2f}" for x in v)  # noqa: E731
                act = link.latest() if link is not None else None
                print(f"trig L {trig['left']:.2f} R {trig['right']:.2f} {'ok ' if fresh else 'STALE'} "
                      f"armed L{int(armed['left'])} R{int(armed['right'])} | cmd L [{f(cmd['left'])}] R [{f(cmd['right'])}]"
                      + (f" | actual L [{f(act['left'])}] R [{f(act['right'])}]" if act is not None else ""))
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if link is not None and have_data:
            frm = {s: cmd[s].copy() for s in cmd}
            n = int(a.off_ramp * a.hz)
            try:
                for i in range(1, n + 1):
                    w = smoothstep(i / n)
                    link.send({s: (1 - w) * frm[s] + w * open_pose for s in frm})
                    time.sleep(dt)
            except KeyboardInterrupt:
                pass
            link.send({"left": open_pose, "right": open_pose})
            print("[trigger] hands commanded open, bye")
        try:
            xrt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
