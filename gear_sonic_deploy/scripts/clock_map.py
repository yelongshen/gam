"""clock_map.py - map a remote sensor clock onto this machine's clock without jumps (numpy-free).

g1_zmq_bridge.py stamps the MID-360's cloud and IMU messages with the sensor's own clock (~18 min off
the Jetson's), mapped onto the local clock by an offset: the least-delayed message defines it, since
its transport delay is closest to zero, so local = remote + min(receive time - remote time).

The first version took that minimum over a sliding window of the last 400 messages. Whenever the
fastest message aged out of the window, the offset stepped up to the next-fastest one -- by however
much more that one had been delayed -- and every later cloud and IMU stamp shifted by the same step.
Under CPU load the delays vary by tens of ms (IMU arriving in bursts, 200-330 Hz instead of ~205),
so the steps did too; DLIO integrates the IMU by stamp, and on the gantry (2026-10-20) it then
estimated metre-per-second velocities for a robot standing still and drifted off by up to 59 m.

Here the offset follows the window minimum but slews: it may move by at most `max_slew` seconds per
second of local time (default 0.5 ms/s, far above the drift between two crystal clocks, ~0.05 ms/s,
and far below the steps above). Until `warmup` samples have arrived it jumps straight to the running
minimum, so the start-up estimate settles quickly.

  cm = ClockMap(); local_stamp = cm(remote_stamp)        # recv time defaults to time.time()
"""
import collections
import threading
import time


class ClockMap:
    def __init__(self, window=400, max_slew=0.5e-3, warmup=200):
        self.samples = collections.deque(maxlen=window)
        self.max_slew = max_slew
        self.warmup = warmup
        self.offset = None
        self.n = 0
        self.last_local = None
        self.lock = threading.Lock()

    def __call__(self, remote_t, local_now=None):
        now = time.time() if local_now is None else local_now
        with self.lock:
            self.samples.append(now - remote_t)
            target = min(self.samples)
            self.n += 1
            if self.offset is None or self.n <= self.warmup:
                self.offset = target if self.offset is None else min(self.offset, target)
            else:
                step = self.max_slew * max(now - self.last_local, 0.0)
                self.offset += min(max(target - self.offset, -step), step)
            self.last_local = now
            return remote_t + self.offset


class StampMonitor:
    """Counts stamps that go backwards or leave gaps larger than `gap` seconds (per stream)."""

    def __init__(self, gap=0.015):
        self.gap = gap
        self.last = None
        self.backwards = 0
        self.gaps = 0

    def __call__(self, t):
        if self.last is not None:
            if t < self.last:
                self.backwards += 1
            elif t - self.last > self.gap:
                self.gaps += 1
        self.last = t if self.last is None else max(self.last, t)

    def take(self):
        out = (self.backwards, self.gaps)
        self.backwards = self.gaps = 0
        return out
