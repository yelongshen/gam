#!/usr/bin/env python3
"""
visualize_avp_stream.py
=======================
Linux-friendly live viewer for the Apple Vision Pro hand stream (AVPHandStreamer app -> UDP :9870).
Needs only numpy + matplotlib (no dex_retargeting / pinocchio / DDS / robot).

Shows both hands as 3D skeletons (OpenXR world frame, y up, metres), the PC IP to type into the headset,
and live stats: packets/s per hand, apparent latency (arrival time - packet "t"; includes clock skew
between the headset and this PC) and time since the last packet.

Packet (one UDP datagram per hand): {"hand": "left"|"right", "joints": [[x,y,z], ...25 or 27], "t": unix_s}

Usage
-----
  python gear_sonic/teleoperation/visualize_avp_stream.py              # GUI, listen on :9870
  python gear_sonic/teleoperation/visualize_avp_stream.py --no-gui     # print stats only (SSH / debugging)
  python gear_sonic/teleoperation/visualize_avp_stream.py --sim        # built-in fake hands, no headset needed
  python gear_sonic/teleoperation/visualize_avp_stream.py --wrist-local   # draw each hand relative to its wrist
"""
import argparse
import json
import socket
import threading
import time

import numpy as np

# WebXR 25-joint layout: 0 wrist | 1-4 thumb | 5-9 index | 10-14 middle | 15-19 ring | 20-24 little
FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8, 9), "middle": (10, 11, 12, 13, 14),
           "ring": (15, 16, 17, 18, 19), "little": (20, 21, 22, 23, 24)}
COLORS = {"thumb": "tab:red", "index": "tab:orange", "middle": "tab:green", "ring": "tab:blue", "little": "tab:purple"}
N_JOINTS = 25
STALE_S = 0.5  # hand shown as "lost" after this long without packets


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


class Receiver(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.lock = threading.Lock()
        self.joints = {"left": None, "right": None}
        self.last_rx = {"left": 0.0, "right": 0.0}
        self.count = {"left": 0, "right": 0}
        self.latency = {"left": 0.0, "right": 0.0}
        self.bad = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.2)
        self.running = True
        self.sender_ip = None

    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            now = time.time()
            try:
                msg = json.loads(data.decode())
                hand = msg["hand"]
                j = np.asarray(msg["joints"], dtype=float)
                if hand not in self.joints or j.ndim != 2 or j.shape[1] != 3 or len(j) < N_JOINTS:
                    raise ValueError("bad packet")
            except Exception:
                self.bad += 1
                continue
            with self.lock:
                self.joints[hand] = j[:N_JOINTS]
                self.last_rx[hand] = now
                self.count[hand] += 1
                self.latency[hand] = now - float(msg.get("t", now))
                self.sender_ip = addr[0]

    def snapshot(self):
        with self.lock:
            return ({k: (None if v is None else v.copy()) for k, v in self.joints.items()},
                    dict(self.last_rx), dict(self.count), dict(self.latency), self.sender_ip, self.bad)

    def stop(self):
        self.running = False
        self.sock.close()


def sim_sender(port, stop):
    """Fake AVP: two waving hands with curling fingers, sent to localhost like the real app."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()
    while not stop.is_set():
        t = time.time() - t0
        for hand, sx in (("left", -0.2), ("right", 0.2)):
            joints = np.zeros((N_JOINTS, 3))
            joints[0] = [sx + 0.05 * np.sin(t), 1.1 + 0.05 * np.cos(1.3 * t), -0.4]
            for fi, (_, idx) in enumerate(FINGERS.items()):
                base = joints[0] + np.array([0.02 * (fi - 2), 0.0, -0.03]) * (1 if fi else 0.5)
                curl = 0.5 * (1 + np.sin(2 * t + fi))
                p = base.copy()
                ang = 0.0
                for k, ji in enumerate(idx):
                    ang += curl * 0.7 * (k > 0)
                    p = p + 0.025 * np.array([0.0, np.cos(ang), -np.sin(ang)]) * (1.0 if k else 0.8)
                    joints[ji] = p
            s.sendto(json.dumps({"hand": hand, "joints": joints.tolist(), "t": time.time()}).encode(),
                     ("127.0.0.1", port))
        time.sleep(1 / 60)


def main():
    ap = argparse.ArgumentParser(description="Live viewer for the AVP hand stream (UDP)")
    ap.add_argument("--port", type=int, default=9870)
    ap.add_argument("--no-gui", action="store_true", help="print stats only")
    ap.add_argument("--sim", action="store_true", help="generate fake hands locally (no headset)")
    ap.add_argument("--wrist-local", action="store_true", help="draw each hand relative to its own wrist")
    ap.add_argument("--seconds", type=float, default=0.0, help="exit after N seconds (0 = run until closed)")
    a = ap.parse_args()

    ip = local_ip()
    rx = Receiver(a.port)
    rx.start()
    stop = threading.Event()
    if a.sim:
        threading.Thread(target=sim_sender, args=(a.port, stop), daemon=True).start()
    print(f"[AVP] PC IP: {ip}   UDP port: {a.port}")
    print("[AVP] In the headset app: enter this IP and port, tap Start Streaming. Ctrl+C / close window to quit.")
    t_start = time.time()

    def stats_line(cnt0, t0, snap):
        _, last, cnt, lat, sender, bad = snap
        dt = max(time.time() - t0, 1e-6)
        now = time.time()
        parts = []
        for h in ("left", "right"):
            age = now - last[h] if last[h] else float("inf")
            state = "OK  " if age < STALE_S else "LOST"
            parts.append(f"{h[0].upper()}:{state} {((cnt[h] - cnt0[h]) / dt):5.1f}Hz lat {lat[h] * 1000:6.0f}ms")
        return "  ".join(parts) + f"  from {sender or '-'}  bad_pkts {bad}"

    try:
        if a.no_gui:
            cnt0, t0 = dict(rx.snapshot()[2]), time.time()
            while a.seconds <= 0 or time.time() - t_start < a.seconds:
                time.sleep(1.0)
                snap = rx.snapshot()
                print(stats_line(cnt0, t0, snap))
                cnt0, t0 = dict(snap[2]), time.time()
            return

        import matplotlib
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure("AVP hand stream", figsize=(11, 6))
        axes = {"left": fig.add_subplot(1, 3, 1, projection="3d"), "right": fig.add_subplot(1, 3, 2, projection="3d"),
                "both": fig.add_subplot(1, 3, 3, projection="3d")}
        title = fig.suptitle("", fontsize=10, family="monospace")

        def disp(p):  # OpenXR (x right, y up, -z forward) -> matplotlib (x, depth, up)
            return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)

        def draw(ax, hands, label, span=None):
            ax.cla()
            ax.set_title(label, fontsize=9)
            pts_all = []
            for h, j in hands:
                if j is None:
                    continue
                d = disp(j - j[0]) if a.wrist_local else disp(j)
                pts_all.append(d)
                for f, idx in FINGERS.items():
                    ch = [0] + list(idx)
                    ax.plot(d[ch, 0], d[ch, 1], d[ch, 2], "-o", color=COLORS[f], ms=2.5, lw=1.6)
                ax.scatter(*d[0], color="k", s=30)
            if pts_all:
                P = np.vstack(pts_all)
                c = P.mean(0)
                r = span or max(0.08, float(np.abs(P - c).max()) * 1.15)
                ax.set_xlim(c[0] - r, c[0] + r)
                ax.set_ylim(c[1] - r, c[1] + r)
                ax.set_zlim(c[2] - r, c[2] + r)
            ax.set_xlabel("x", fontsize=7)
            ax.set_ylabel("depth", fontsize=7)
            ax.set_zlabel("up", fontsize=7)
            ax.tick_params(labelsize=6)

        cnt0, t0 = dict(rx.snapshot()[2]), time.time()
        plt.ion()
        plt.show(block=False)
        last_stat = time.time()
        while plt.fignum_exists(fig.number) and (a.seconds <= 0 or time.time() - t_start < a.seconds):
            joints, last, cnt, lat, sender, bad = rx.snapshot()
            now = time.time()
            live = {h: joints[h] if (last[h] and now - last[h] < STALE_S) else None for h in joints}
            draw(axes["left"], [("left", live["left"])], "left" + ("" if live["left"] is not None else "  (no data)"), 0.12)
            draw(axes["right"], [("right", live["right"])], "right" + ("" if live["right"] is not None else "  (no data)"), 0.12)
            draw(axes["both"], [("left", live["left"]), ("right", live["right"])], "both hands")
            if now - last_stat >= 0.5:
                title.set_text(f"PC {ip}:{a.port}   " + stats_line(cnt0, t0, rx.snapshot()))
                cnt0, t0, last_stat = dict(cnt), now, now
            plt.pause(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        rx.stop()


if __name__ == "__main__":
    main()
