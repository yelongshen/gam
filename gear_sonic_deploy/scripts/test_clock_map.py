"""Tests for clock_map.py: python3 -m pytest test_clock_map.py"""
import collections

import numpy as np

from clock_map import ClockMap, StampMonitor


def bursty_stream(seconds=60.0, rate=205.0, skew_ppm=50.0, seed=0):
    """IMU-like stamps on a sensor clock 18 min off, delivered with delays that come and go in bursts
    (1-3 ms normally, 20-40 ms for a few hundred ms at a time), as under CPU load on the Jetson."""
    rng = np.random.default_rng(seed)
    remote = np.arange(0.0, seconds, 1.0 / rate) + 1080.0
    true_local = (remote - 1080.0) * (1 + skew_ppm * 1e-6) + 1000.0       # where each sample really belongs
    delay = rng.uniform(0.001, 0.003, len(remote))
    t = 0
    while t < len(remote):
        t += int(rng.integers(100, 600))
        n = int(rng.integers(20, 80))
        delay[t:t + n] += rng.uniform(0.02, 0.04)
    recv = true_local + delay
    return remote, true_local, recv


def old_clock_map(remote, recv, window=400):
    s = collections.deque(maxlen=window)
    out = []
    for r, l in zip(remote, recv):
        s.append(l - r)
        out.append(r + min(s))
    return np.array(out)


def test_old_window_minimum_steps_under_bursty_delays():
    remote, true_local, recv = bursty_stream()
    out = old_clock_map(remote, recv)
    err = np.diff(out) - np.diff(remote)
    assert np.abs(err).max() > 5e-4          # the jumps the fix removes (for the record)


def test_spacing_is_preserved_and_never_goes_backwards():
    remote, true_local, recv = bursty_stream()
    cm, mon = ClockMap(), StampMonitor(gap=0.015)
    out = np.array([cm(r, l) for r, l in zip(remote, recv)])
    for t in out:
        mon(t)
    backwards, gaps = mon.take()
    assert backwards == 0 and gaps == 0
    d = np.diff(out[300:]) - np.diff(remote[300:])          # after warm-up
    assert np.abs(d).max() < 2e-5                            # 0.5 ms/s slew x ~5 ms spacing, plus skew


def test_tracks_clock_drift_and_stays_near_the_fastest_delivery():
    remote, true_local, recv = bursty_stream(seconds=120.0, skew_ppm=50.0)
    cm = ClockMap()
    out = np.array([cm(r, l) for r, l in zip(remote, recv)])
    err = out - true_local                                   # >= 0: never ahead of the true time by much
    assert np.abs(err[2000:]).max() < 0.004                  # within the 1-3 ms base delay, despite 50 ppm skew


def test_stamp_monitor_counts_backwards_and_gaps():
    mon = StampMonitor(gap=0.015)
    for t in (0.0, 0.005, 0.004, 0.010, 0.040, 0.045):
        mon(t)
    assert mon.take() == (1, 1)
    assert mon.take() == (0, 0)
