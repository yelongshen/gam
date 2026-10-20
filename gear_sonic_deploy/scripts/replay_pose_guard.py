"""replay_pose_guard.py - run pose_guard.PoseGuard over recorded terrain messages, offline.

For tuning the guard's thresholds against real recordings (record_terrain.py npz files) before
changing them on the robot: how much of a recording it rejects, and when each rejection episode
started and why. Use clean walks as well as diverged ones -- a threshold change should leave the
clean ones untouched.

    python3 replay_pose_guard.py walk3_terrain.npz walk5_terrain.npz ...

On the gantry recordings of 2026-10-20 (thresholds as committed): a clean walk (run 3) and a
close box approach 0.0%; DLIO diverging to 59 m (run 5) and glitching to 3.5 m (run 6) both caught.
"""
import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_guard import PoseGuard  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--utc-offset", type=float, default=8.0, help="hours, for the printed clock times")
    a = ap.parse_args()

    def clock(t):
        return (datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
                + datetime.timedelta(hours=a.utc_offset)).strftime("%H:%M:%S")

    for path in a.recordings:
        T = dict(np.load(path))
        n, res, o = T["height_grid"].shape[1], float(T["grid_resolution"]), T["grid_origin"]
        gx, gy = np.meshgrid(o[0] + res * np.arange(n), o[1] + res * np.arange(n))
        xy = np.stack([gx.ravel(), gy.ravel()], 1)
        g, ts = PoseGuard(), T["pose_stamp"]
        bad, episodes = 0, []
        for i in range(len(ts)):
            g.odom(ts[i], T["torso_pos"][i], T["torso_quat"][i])
            g.floor(ts[i], T["height_grid"][i], T["height_grid_valid"][i], xy)
            ok, why = g.state(ts[i])
            if g.take_new_episode():
                episodes.append(f"{clock(T['recv_time'][i])} {why[0]}")
            bad += not ok
        print(f"{Path(path).name}: {len(ts)} messages, rejected {bad / len(ts):.1%}, {len(episodes)} episodes")
        for e in episodes[:8]:
            print(f"    {e}")
        if len(episodes) > 8:
            print(f"    ... {len(episodes) - 8} more")


if __name__ == "__main__":
    main()
