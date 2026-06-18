#!/usr/bin/env python3
"""
SONIC Episode Visualizer
========================
Renders one episode of the trained SONIC combined policy in the MuJoCo
viewer side-by-side with the reference motion (ghost skeleton in red).

Usage
-----
# Visualize latest checkpoint, random reference motion:
    python gear_sonic/training/visualize_episode.py

# Specific checkpoint and motion:
    python gear_sonic/training/visualize_episode.py \\
        --ckpt outputs/sonic_combined/best_combined.pt \\
        --motion /path/to/body_check_001__A548.npz \\
        --speed 0.5

Keyboard controls (MuJoCo viewer)
----------------------------------
  SPACE   pause / resume
  R       reset episode (new random reference)
  N       next reference motion
  +/-     speed up / slow down
  Q / ESC quit
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import mujoco.viewer

from gear_sonic.training.encoders       import SonicEncoderDecoder
from gear_sonic.training.ppo_trainer    import PolicyHead
from gear_sonic.training.g1_mujoco_env import (
    G1MuJoCoEnv, N_JOINTS, JOINT_NAMES, DEFAULT_Q, SCENE_XML, OBS_SCALE
)

# ── Globals (shared with key callback) ────────────────────────────────────────
_pause     = False
_reset     = False
_next      = False
_speed     = 1.0          # 1.0 = real-time


def _key_cb(keycode):
    global _pause, _reset, _next, _speed
    try:
        c = chr(keycode)
    except Exception:
        c = ""
    if c == " ":
        _pause = not _pause
        print(f"{'Paused' if _pause else 'Resumed'}")
    elif c.upper() == "R":
        _reset = True
        print("Reset requested")
    elif c.upper() == "N":
        _next = True
        print("Next motion requested")
    elif c in ("+", "="):
        _speed = min(_speed * 1.5, 4.0)
        print(f"Speed: {_speed:.1f}x")
    elif c in ("-", "_"):
        _speed = max(_speed / 1.5, 0.1)
        print(f"Speed: {_speed:.1f}x")


# ── Ghost body (red tint for reference skeleton) ──────────────────────────────

def _add_ghost_geoms(model: mujoco.MjModel, data: mujoco.MjData):
    """Return a second MjData initialized at the reference pose."""
    # We visualise the reference as a semi-transparent model clone.
    # The simplest approach: a separate MjModel loaded from the same XML
    # with all geom rgba set to (1, 0, 0, 0.3).
    ghost_m = mujoco.MjModel.from_xml_path(SCENE_XML)
    for i in range(ghost_m.ngeom):
        ghost_m.geom_rgba[i] = [1.0, 0.1, 0.1, 0.25]
    ghost_d = mujoco.MjData(ghost_m)
    return ghost_m, ghost_d


# ── Observation normaliser (matches combined trainer) ─────────────────────────
_OBS_SCALE_T = None   # filled once


def _norm(obs_np: np.ndarray, device) -> torch.Tensor:
    global _OBS_SCALE_T
    if _OBS_SCALE_T is None:
        _OBS_SCALE_T = torch.from_numpy(OBS_SCALE).to(device)
    t = torch.from_numpy(obs_np.astype(np.float32)).to(device)
    return (t * _OBS_SCALE_T).clamp(-10., 10.).unsqueeze(0)


# ── Reference window helper ────────────────────────────────────────────────────
def _ref_win(ref_rad: np.ndarray, step: int, W: int) -> torch.Tensor:
    T   = len(ref_rad)
    end = min(step, T - 1)
    s   = max(end - W + 1, 0)
    win = ref_rad[s: end + 1]
    if len(win) < W:
        win = np.concatenate([np.tile(win[0], (W - len(win), 1)), win])
    return torch.from_numpy(np.degrees(win).astype(np.float32)).unsqueeze(0)


# ── Load model from checkpoint ────────────────────────────────────────────────
def _load_model(ckpt_path: str, device, W: int, token_dim: int, hidden: int):
    encoder = SonicEncoderDecoder(window=W, token_dim=token_dim,
                                   hidden_dim=hidden).to(device)
    policy  = PolicyHead(G1MuJoCoEnv.OBS_DIM, token_dim, N_JOINTS, hidden).to(device)

    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        if "encoder" in ck:
            encoder.load_state_dict(ck["encoder"])
        elif "model_state" in ck:
            encoder.load_state_dict(ck["model_state"])
        if "policy" in ck:
            policy.load_state_dict(ck["policy"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint — using random weights")

    encoder.eval(); policy.eval()
    return encoder, policy


# ── Main visualizer ───────────────────────────────────────────────────────────
def run(args):
    global _pause, _reset, _next, _speed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load checkpoint ────────────────────────────────────────────────────
    W         = args.window
    token_dim = args.token_dim
    hidden    = args.hidden
    encoder, policy = _load_model(args.ckpt, device, W, token_dim, hidden)

    # ── load reference motions ─────────────────────────────────────────────
    if args.motion and os.path.exists(args.motion):
        motion_files = [args.motion]
    else:
        pattern = os.path.join(args.data_dir, "*.npz")
        motion_files = sorted(f for f in glob.glob(pattern)
                              if not _has_nan(f))
    if not motion_files:
        raise FileNotFoundError(f"No valid NPZ files in {args.data_dir}")
    print(f"Available motions: {len(motion_files)}")

    motion_idx = 0

    # ── MuJoCo sim + ghost ─────────────────────────────────────────────────
    env     = G1MuJoCoEnv(
        sim_dt=0.005, control_hz=50.0,
        max_episode_frames=args.max_frames,
        min_height=0.3,
    )
    ghost_m, ghost_d = _add_ghost_geoms(env.model, env.data)

    # ── episode state ──────────────────────────────────────────────────────
    ref_rad   = None
    step_ep   = 0
    obs_np    = None
    done      = False
    ep_reward = 0.0
    ep_steps  = 0

    def _start_episode(midx: int):
        nonlocal ref_rad, step_ep, obs_np, done, ep_reward, ep_steps
        f = motion_files[midx % len(motion_files)]
        g_r = np.load(f)["g_r"].astype(np.float32)
        obs_np   = env.reset(g_r)
        ref_rad  = env._ref_traj
        step_ep  = 0
        done     = False
        ep_reward = 0.0
        ep_steps  = 0
        print(f"\nEpisode start — motion: {os.path.basename(f)}  ({len(g_r)} frames)")
        return obs_np

    obs_np = _start_episode(motion_idx)

    # ── viewer ─────────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=_key_cb
    ) as viewer:

        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -15
        viewer.cam.distance  = 3.5
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]

        print("\nControls:  SPACE=pause  R=reset  N=next motion  +/-=speed  Q=quit")

        ctrl_dt  = 1.0 / 50.0   # 50 Hz
        t_last   = time.time()

        while viewer.is_running():

            # handle keyboard events
            if _reset:
                _reset = False
                obs_np = _start_episode(motion_idx)
            if _next:
                _next = False
                motion_idx = (motion_idx + 1) % len(motion_files)
                obs_np = _start_episode(motion_idx)

            if _pause:
                viewer.sync()
                time.sleep(0.02)
                continue

            # ── policy step ────────────────────────────────────────────────
            with torch.no_grad():
                obs_t = _norm(obs_np, device)
                g_r_w = _ref_win(ref_rad, step_ep, W).to(device)
                z     = encoder.E_r(g_r_w)
                mean, lstd = policy(obs_t, z)
                action_np  = mean.squeeze(0).cpu().numpy()   # deterministic (mean)

            obs_np2, reward, done, _ = env.step(action_np)
            ep_reward += reward
            ep_steps  += 1
            step_ep   += 1

            # ── set ghost to reference pose ────────────────────────────────
            ref_q_rad = ref_rad[min(step_ep, len(ref_rad) - 1)]
            mujoco.mj_resetData(ghost_m, ghost_d)
            ghost_d.qpos[7: 7 + N_JOINTS] = ref_q_rad
            ghost_d.qpos[2] = env.data.qpos[2]   # match height
            mujoco.mj_forward(ghost_m, ghost_d)

            # overlay ghost into scene
            mujoco.mj_forward(env.model, env.data)
            viewer.sync()

            # ── status line ────────────────────────────────────────────────
            if ep_steps % 50 == 0:
                T  = len(ref_rad)
                pct = min(step_ep, T) / T * 100
                print(f"  step {ep_steps:4d}  ref {pct:5.1f}%  "
                      f"reward {ep_reward/ep_steps:.3f}  "
                      f"height {env.data.qpos[2]:.2f}m")

            # ── episode end ────────────────────────────────────────────────
            if done:
                outcome = "FELL" if env.data.qpos[2] < 0.35 else "completed"
                print(f"  → Episode {outcome}  "
                      f"steps={ep_steps}  avg_rew={ep_reward/ep_steps:.3f}")
                if args.loop:
                    obs_np = _start_episode(motion_idx)
                else:
                    print("  (press R to restart, N for next motion)")
                    _pause = True
                    obs_np = obs_np2
                    continue
            else:
                obs_np = obs_np2

            # ── real-time pacing ───────────────────────────────────────────
            t_now    = time.time()
            elapsed  = t_now - t_last
            t_last   = t_now
            sleep_t  = ctrl_dt / _speed - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


def _has_nan(path: str) -> bool:
    d = np.load(path)
    return np.isnan(d["g_r"]).any() or np.isnan(d["g_h"]).any()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="SONIC episode visualizer")
    p.add_argument("--ckpt",       default="outputs/sonic_combined/best_combined.pt",
                   help="Path to combined trainer checkpoint (.pt)")
    p.add_argument("--data-dir",   default="/home/grease/ego_dataset/work_bearlu/data/bones-studio-processed",
                   help="Directory of processed NPZ files")
    p.add_argument("--motion",     default=None,
                   help="Specific NPZ file to visualize (overrides --data-dir random)")
    p.add_argument("--max-frames", type=int, default=600,
                   help="Max frames per episode (default 600 = 12s @ 50Hz)")
    p.add_argument("--speed",      type=float, default=1.0,
                   help="Playback speed multiplier (default 1.0 = real-time)")
    p.add_argument("--loop",       action="store_true",
                   help="Auto-restart episode on completion")
    p.add_argument("--window",     type=int, default=8)
    p.add_argument("--token-dim",  type=int, default=64)
    p.add_argument("--hidden",     type=int, default=256)
    args = p.parse_args()
    _speed = args.speed   # apply initial speed
    run(args)


if __name__ == "__main__":
    main()
