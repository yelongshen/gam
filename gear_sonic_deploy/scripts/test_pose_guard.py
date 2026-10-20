"""Tests for pose_guard.py: python3 -m pytest test_pose_guard.py"""
import numpy as np

from pose_guard import PoseGuard


def quat_yaw(a):
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])


def walk(guard, t0, t1, rate=100.0, speed=1.2, bob=0.04, yaw_rate=0.8):
    """A G1-like walk: forward at `speed`, torso bobbing 4 cm at 2 Hz, turning slowly."""
    for t in np.arange(t0, t1, 1.0 / rate):
        guard.odom(t, [speed * t, 0.0, 0.8 + bob * np.sin(2 * np.pi * 2 * t)], quat_yaw(yaw_rate * t))


def flat_grid(above=0.8, n=50, res=0.04):
    c = (np.arange(n) - (n - 1) / 2.0) * res
    gx, gy = np.meshgrid(c, c)
    return np.full(n * n, -above), np.ones(n * n, bool), np.stack([gx.ravel(), gy.ravel()], 1)


def test_walking_is_plausible():
    g = PoseGuard()
    walk(g, 0.0, 10.0)
    h, v, xy = flat_grid(0.8)
    assert g.floor(10.0, h, v, xy) == 0.8
    assert g.state(10.0) == (True, [])
    assert not g.take_new_episode()


def test_jump_is_rejected_and_held_for_two_seconds():
    g = PoseGuard()
    walk(g, 0.0, 5.0)
    g.odom(5.01, [6.0 + 1.0, -2.0, 0.8], quat_yaw(4.0))     # 2 m sideways in 10 ms: a divergence
    ok, why = g.state(5.02)
    assert not ok and any("speed" in w for w in why)
    assert g.take_new_episode() and not g.take_new_episode()  # one clear request per episode
    walk(g, 5.02, 7.5)                                         # plausible again (relative motion)
    assert not g.state(6.9)[0]                                 # still inside the hold
    assert g.state(7.6)[0]                                     # trusted again


def test_vertical_jump_and_fast_turn_are_rejected():
    g = PoseGuard()
    walk(g, 0.0, 2.0)
    g.odom(2.01, [2.4, 0.0, 1.5], quat_yaw(1.6))               # +0.7 m up in 10 ms
    assert any("climb" in w for w in g.state(2.02)[1])
    g = PoseGuard()
    walk(g, 0.0, 2.0, yaw_rate=0.0)
    for k, t in enumerate(np.arange(2.0, 2.2, 0.01)):          # 10 rad/s spin
        g.odom(t, [2.4, 0.0, 0.8], quat_yaw(10.0 * (t - 2.0)))
    assert any("turn" in w for w in g.state(2.2)[1])


def test_floor_height_check():
    g = PoseGuard()
    walk(g, 0.0, 1.0)
    for above, ok in ((0.8, True), (0.45, True), (2.3, False), (0.1, False)):
        g = PoseGuard()
        h, v, xy = flat_grid(above)
        g.floor(1.0, h, v, xy)
        assert g.state(1.0)[0] == ok, above
    g = PoseGuard()
    h, v, xy = flat_grid(2.3)
    v[:] = False                                               # nothing observed: no verdict
    assert g.floor(1.0, h, v, xy) is None and g.state(1.0)[0]
