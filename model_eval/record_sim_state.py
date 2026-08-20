"""
Record the true simulated robot state directly from the MuJoCo sim over DDS.

The deploy CSV logs contain base ORIENTATION but not base POSITION, so a video
reconstructed from them cannot show world translation (the robot appears to
walk in place). The simulator itself publishes full odometry on `rt/odostate`
(position + orientation) and joint angles on `rt/lowstate`, so we subscribe to
those instead and get the exact state MuJoCo simulated.

Writes an .npz with:
    t          (T,)        seconds since start
    base_pos   (T, 3)      world position of the floating base
    base_quat  (T, 4)      w, x, y, z
    q          (T, 29)     joint positions

Usage:
    .venv_sim/bin/python record_sim_state.py --out /tmp/state.npz --duration 20
"""
import argparse
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, OdoState_

NUM_JOINTS = 29


class SimStateRecorder:
    def __init__(self, interface="lo"):
        ChannelFactoryInitialize(0, interface)
        self.odo = None
        self.low = None
        self._odo_sub = ChannelSubscriber("rt/odostate", OdoState_)
        self._odo_sub.Init(self._on_odo, 10)
        self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_sub.Init(self._on_low, 10)

    def _on_odo(self, msg):
        self.odo = msg

    def _on_low(self, msg):
        self.low = msg

    def record(self, duration, rate=50.0):
        dt = 1.0 / rate
        t0 = time.time()
        ts, pos, quat, qs = [], [], [], []
        # wait for first messages
        while self.odo is None or self.low is None:
            if time.time() - t0 > 15:
                raise SystemExit("No DDS data — is the sim running on 'lo'?")
            time.sleep(0.1)
        print("[rec] receiving sim state...", flush=True)
        t0 = time.time()
        while time.time() - t0 < duration:
            loop = time.time()
            o, l = self.odo, self.low
            ts.append(time.time() - t0)
            pos.append(list(o.position[:3]))
            quat.append(list(o.orientation[:4]))
            qs.append([l.motor_state[i].q for i in range(NUM_JOINTS)])
            d = dt - (time.time() - loop)
            if d > 0:
                time.sleep(d)
        return (np.asarray(ts), np.asarray(pos, dtype=np.float64),
                np.asarray(quat, dtype=np.float64), np.asarray(qs, dtype=np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/sim_state.npz')
    ap.add_argument('--duration', type=float, default=20.0)
    ap.add_argument('--interface', default='lo')
    ap.add_argument('--rate', type=float, default=50.0)
    args = ap.parse_args()

    rec = SimStateRecorder(args.interface)
    t, pos, quat, q = rec.record(args.duration, args.rate)
    np.savez(args.out, t=t, base_pos=pos, base_quat=quat, q=q)
    print(f"[rec] saved {len(t)} frames -> {args.out}")
    print(f"[rec] base height {pos[:,2].min():.3f}..{pos[:,2].max():.3f} m, "
          f"travel {np.linalg.norm(pos[-1,:2]-pos[0,:2]):.2f} m")


if __name__ == "__main__":
    main()
