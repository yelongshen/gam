"""
SONIC PPO Trainer
=================
Trains a policy decoder on top of the pre-trained SONIC encoders
using Proximal Policy Optimization (PPO) with MuJoCo physics simulation.

Architecture:
  Observation  → [E_r | z_r (frozen)]  +  robot_proprioception
                                              ↓
                                       PolicyHead (MLP)
                                              ↓
                               29-DoF joint position action

Training loop (per PPO iteration):
  1. Roll out N steps in MuJoCo using current policy
  2. Compute GAE advantages from rewards
  3. PPO clipped surrogate update (K mini-epochs)
  4. Combined loss includes supervised alignment losses from Stage 1
     (optional — can keep encoders frozen in Stage 2)

Usage:
    python gear_sonic/training/ppo_trainer.py \
        --config gear_sonic/training/config_ppo.yaml
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.training.encoders  import SonicEncoderDecoder
from gear_sonic.training.g1_mujoco_env import G1MuJoCoEnv, N_JOINTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Policy network ────────────────────────────────────────────────────────────

class PolicyHead(nn.Module):
    """
    Policy decoder: z_token + proprioception → 29-DoF joint position action.

    Input : cat(z, q, dq, root_rpy, root_vel, ref_q, ref_dq, phase)
            = token_dim + 29 + 29 + 3 + 3 + 6 + 29 + 29 + 2
    Output: mean (29,)  and  log_std (29,)  of Gaussian action distribution
    """

    def __init__(self, obs_dim: int, token_dim: int, action_dim: int = 29,
                 hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + token_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.mean_head    = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)
        # initialise output layer small for stable early training
        nn.init.uniform_(self.mean_head.weight,  -0.01, 0.01)
        nn.init.uniform_(self.log_std_head.weight, -0.01, 0.01)

    def forward(self, obs: torch.Tensor, z: torch.Tensor):
        x    = torch.cat([obs, z], dim=-1)
        feat = self.net(x)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat).clamp(-4, 2)
        return mean, log_std

    def get_action(self, obs: torch.Tensor, z: torch.Tensor):
        mean, log_std = self(obs, z)
        std  = log_std.exp().clamp(1e-4, 2.0)  # prevent std collapse / explosion
        dist = Normal(mean, std, validate_args=False)
        action = dist.rsample()
        log_p  = dist.log_prob(action).sum(-1)
        return action, log_p, dist.entropy().sum(-1)


class ValueHead(nn.Module):
    """Critic: obs + z → scalar value."""

    def __init__(self, obs_dim: int, token_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + token_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, z], dim=-1)).squeeze(-1)


# ── PPO rollout buffer ────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, token_dim: int, act_dim: int):
        self.obs      = np.zeros((n_steps, obs_dim),   dtype=np.float32)
        self.z        = np.zeros((n_steps, token_dim), dtype=np.float32)
        self.actions  = np.zeros((n_steps, act_dim),   dtype=np.float32)
        self.log_probs= np.zeros(n_steps, dtype=np.float32)
        self.rewards  = np.zeros(n_steps, dtype=np.float32)
        self.values   = np.zeros(n_steps, dtype=np.float32)
        self.dones    = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, z, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i]      = obs
        self.z[i]        = z
        self.actions[i]  = action
        self.log_probs[i]= log_prob
        self.rewards[i]  = reward
        self.values[i]   = value
        self.dones[i]    = done
        self.ptr += 1

    def full(self): return self.ptr >= len(self.rewards)

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        adv = np.zeros_like(self.rewards)
        last_adv = 0.0
        n = len(self.rewards)
        vals = np.append(self.values, last_value)
        for t in reversed(range(n)):
            delta    = self.rewards[t] + gamma * vals[t+1] * (1 - self.dones[t]) - vals[t]
            adv[t]   = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_adv
            last_adv = adv[t]
        self.advantages = adv
        self.returns    = adv + self.values
        # normalise advantages
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)


# ── PPO Trainer ───────────────────────────────────────────────────────────────

def ppo_train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    log_fh = open(out_dir / "ppo_train.log", "w")

    def _log(msg):
        logger.info(msg); log_fh.write(msg + "\n"); log_fh.flush()

    # ── Load pre-trained encoders (frozen) ────────────────────────────────────
    encoder = SonicEncoderDecoder(
        window=cfg["encoder_window"],
        token_dim=cfg["token_dim"],
        hidden_dim=cfg["encoder_hidden"],
    ).to(device)
    if cfg.get("encoder_ckpt"):
        ckpt = torch.load(cfg["encoder_ckpt"], map_location=device)
        encoder.load_state_dict(ckpt["model_state"])
        _log(f"Loaded encoder checkpoint: {cfg['encoder_ckpt']}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ── Policy and value networks ─────────────────────────────────────────────
    OBS_DIM   = G1MuJoCoEnv.OBS_DIM
    TOKEN_DIM = cfg["token_dim"]
    ACT_DIM   = N_JOINTS

    policy = PolicyHead(OBS_DIM, TOKEN_DIM, ACT_DIM, cfg["policy_hidden"]).to(device)
    critic = ValueHead(OBS_DIM, TOKEN_DIM, cfg["policy_hidden"]).to(device)

    params = list(policy.parameters()) + list(critic.parameters())
    optim  = torch.optim.Adam(params, lr=cfg["lr"], eps=1e-5)

    n_params = sum(p.numel() for p in params)
    _log(f"Policy+Critic parameters: {n_params:,}")

    # ── Reference motion library ──────────────────────────────────────────────
    npz_files = sorted(glob.glob(os.path.join(cfg["data_dir"], "*.npz")))
    npz_files = [f for f in npz_files if not _has_nan(f)]
    _log(f"Reference motions: {len(npz_files)}")

    # ── MuJoCo environment ────────────────────────────────────────────────────
    env = G1MuJoCoEnv(
        sim_dt=cfg.get("sim_dt", 0.005),
        control_hz=cfg.get("control_hz", 50.0),
        max_episode_frames=cfg.get("episode_frames", 300),
        min_height=cfg.get("min_height", 0.3),
    )

    # ── Main PPO loop ─────────────────────────────────────────────────────────
    total_steps = 0
    best_ep_reward = -np.inf
    buf = RolloutBuffer(cfg["n_steps"], OBS_DIM, TOKEN_DIM, ACT_DIM)

    obs_np = _reset_random(env, npz_files, encoder, device, cfg)
    ep_rewards = []

    for iteration in range(1, cfg["n_iterations"] + 1):
        buf.ptr = 0
        ep_rew = 0.0
        t0 = time.time()

        # ── Collect rollout ───────────────────────────────────────────────────
        while not buf.full():
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            z_t   = _encode_obs(obs_np, env, encoder, device, cfg)

            with torch.no_grad():
                action_t, log_p_t, _ = policy.get_action(obs_t, z_t)
                value_t = critic(obs_t, z_t)

            action_np = action_t.squeeze(0).cpu().numpy()
            obs_np2, reward, done, _ = env.step(action_np)
            ep_rew += reward

            buf.add(
                obs_np, z_t.squeeze(0).cpu().numpy(),
                action_np, log_p_t.item(), reward,
                value_t.item(), float(done),
            )
            obs_np = obs_np2

            if done:
                ep_rewards.append(ep_rew); ep_rew = 0.0
                obs_np = _reset_random(env, npz_files, encoder, device, cfg)

        # Bootstrap last value
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            z_t   = _encode_obs(obs_np, env, encoder, device, cfg)
            last_val = critic(obs_t, z_t).item()
        buf.compute_gae(last_val, cfg["gamma"], cfg["gae_lambda"])
        total_steps += cfg["n_steps"]

        # ── PPO update ────────────────────────────────────────────────────────
        obs_b  = torch.from_numpy(buf.obs).to(device)
        z_b    = torch.from_numpy(buf.z).to(device)
        act_b  = torch.from_numpy(buf.actions).to(device)
        old_lp = torch.from_numpy(buf.log_probs).to(device)
        adv_b  = torch.from_numpy(buf.advantages.astype(np.float32)).to(device)
        ret_b  = torch.from_numpy(buf.returns.astype(np.float32)).to(device)

        pl_total = vl_total = ent_total = 0.0
        n_mini = cfg["n_steps"] // cfg["mini_batch"]

        for _ in range(cfg["k_epochs"]):
            idx = torch.randperm(cfg["n_steps"])
            for start in range(0, cfg["n_steps"], cfg["mini_batch"]):
                mb = idx[start: start + cfg["mini_batch"]]
                mean, log_std = policy(obs_b[mb], z_b[mb])
                dist   = Normal(mean, log_std.exp())
                new_lp = dist.log_prob(act_b[mb]).sum(-1)
                ent    = dist.entropy().sum(-1).mean()

                ratio  = (new_lp - old_lp[mb]).exp()
                a      = adv_b[mb]
                pg1    = ratio * a
                pg2    = ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * a
                pl     = -torch.min(pg1, pg2).mean()

                vl = F.mse_loss(critic(obs_b[mb], z_b[mb]), ret_b[mb])
                loss = pl + cfg["vf_coef"] * vl - cfg["ent_coef"] * ent

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                optim.step()

                pl_total += pl.item(); vl_total += vl.item(); ent_total += ent.item()

        mean_ep = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
        elapsed = time.time() - t0
        msg = (f"Iter {iteration:4d}  steps={total_steps:7,}  "
               f"rew={mean_ep:.3f}  pl={pl_total/cfg['k_epochs']/n_mini:.4f}  "
               f"vl={vl_total/cfg['k_epochs']/n_mini:.4f}  {elapsed:.1f}s")
        _log(msg)

        ckpt = {"policy": policy.state_dict(), "critic": critic.state_dict(),
                "optim": optim.state_dict(), "iteration": iteration}
        torch.save(ckpt, out_dir / "last_ppo.pt")
        if mean_ep > best_ep_reward:
            best_ep_reward = mean_ep
            torch.save(ckpt, out_dir / "best_ppo.pt")
            _log(f"  ✅  New best reward: {best_ep_reward:.3f}")

    log_fh.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_nan(path: str) -> bool:
    d = np.load(path)
    return np.isnan(d["g_r"]).any() or np.isnan(d["g_h"]).any()

def _reset_random(env, files, encoder, device, cfg) -> np.ndarray:
    """Pick a random reference motion and reset the environment."""
    f    = files[np.random.randint(len(files))]
    d    = np.load(f)
    g_r  = d["g_r"].astype(np.float32)   # (T, 29) degrees
    return env.reset(g_r)

def _encode_obs(obs_np, env, encoder, device, cfg) -> torch.Tensor:
    """Extract the g_r window from the observation and encode it."""
    W = cfg["encoder_window"]
    # q is in obs[0:29] (radians), convert back to degrees for encoder
    q_rad = obs_np[:N_JOINTS]
    q_deg = np.degrees(q_rad)
    # build a (1, W, 29) window by repeating current frame
    window = torch.from_numpy(
        np.tile(q_deg, (1, W, 1)).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        z = encoder.E_r(window)
    return z


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gear_sonic/training/config_ppo.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ppo_train(cfg)


if __name__ == "__main__":
    main()
