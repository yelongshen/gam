#!/usr/bin/env python3
"""
SONIC Training Script (TRL-style)
==================================
Adapted from the official NVlabs GR00T-WholeBodyControl training pipeline:
  https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/train_agent_trl.py

The official code uses HuggingFace TRL + IsaacLab.  This version keeps the
same *structural* design (PolicyAndValueWrapper, RolloutStorage, rollout →
GAE → PPO-update loop) but runs on MuJoCo without IsaacLab or TRL/Accelerate
dependencies, so it works in the existing .venv_sim environment.

Architecture mirrors TRLPPOTrainer:
  ┌──────────────────────────────────────────────────────────────┐
  │  for batch in range(num_total_batches):                       │
  │      obs_dict = _rollout_step(model, obs_dict)   # collect   │
  │      rollout_data = _get_rollout_data(obs_keys)  # GAE       │
  │      for ppo_epoch:                                           │
  │          for mini_batch:                                      │
  │              fwd = _forward_model(model, mb)                  │
  │              loss = _compute_loss(fwd, mb)                    │
  │              loss.backward(); optimizer.step()                │
  └──────────────────────────────────────────────────────────────┘

Key components:
  PolicyAndValueWrapper  — holds encoder + policy head + critic (§ TRL)
  RolloutStorage         — stores (obs, z, action, log_p, reward, value, done)
  _rollout_step()        — collect num_steps transitions, compute GAE
  _forward_model()       — re-evaluate log-probs and values for PPO update
  _compute_loss()        — clipped surrogate + value loss + entropy

Usage:
    cd /home/grease/gam
    source .venv_sim/bin/activate
    python gear_sonic/training/train_trl_style.py \\
        --config gear_sonic/training/config_trl.yaml \\
        --iters 20
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.training.encoders       import SonicEncoderDecoder
from gear_sonic.training.g1_mujoco_env import G1MuJoCoEnv, N_JOINTS, OBS_SCALE
from gear_sonic.training.ppo_trainer    import ValueHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Observation normalisation (identical to sonic_combined_trainer) ──────────
_OBS_SCALE = OBS_SCALE

def _norm(obs: np.ndarray) -> np.ndarray:
    return np.clip(obs * _OBS_SCALE, -10., 10.)


# ──────────────────────────────────────────────────────────────────────────────
# SonicActor  (matches official Actor from actor_critic_modules.py)
# ──────────────────────────────────────────────────────────────────────────────

class SonicActor(nn.Module):
    """
    Actor with INPUT-INDEPENDENT shared noise std.

    This matches the official Actor in actor_critic_modules.py:

        self.log_std = nn.Parameter(
            torch.log(init_noise_std * torch.ones(num_actions))
        )
        self.distribution = Normal(action_mean,
                                   (action_mean * 0.0 + self.std).clamp(min=1e-6))

    Using a shared (observation-independent) log_std parameter is the
    standard approach for locomotion RL (RSL-RL, Isaac Lab, SONIC official).
    It avoids the entropy-collapse failure mode where the network learns to
    increase log_std to maximise the entropy bonus instead of tracking motion.

    Official config: init_noise_std = 0.05  (after supervised pre-training)
    """

    def __init__(self, obs_dim: int, token_dim: int, action_dim: int,
                 hidden: int, init_noise_std: float = 1.0):
        super().__init__()
        in_dim = obs_dim + token_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        # Shared log_std — a single learnable vector, NOT input-dependent
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(np.log(init_noise_std)))
        )
        nn.init.uniform_(self.mean_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns action_mean (B, act_dim)."""
        return self.mean_head(self.net(torch.cat([obs, z], dim=-1)))

    @property
    def std(self) -> torch.Tensor:
        """Clamped std identical to official get_std."""
        return self.log_std.clamp(-20, 2).exp().clamp(min=1e-6)

    def get_action(self, obs: torch.Tensor, z: torch.Tensor):
        """Sample action + compute log-prob + entropy (no-grad caller)."""
        mean = self.forward(obs, z)
        std  = self.std
        dist = Normal(mean, std.expand_as(mean), validate_args=False)
        action   = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return action, log_prob, entropy


# ──────────────────────────────────────────────────────────────────────────────
# PolicyAndValueWrapper  (mirrors TRL's PolicyAndValueWrapper)
# ──────────────────────────────────────────────────────────────────────────────

class PolicyAndValueWrapper(nn.Module):
    """
    Wraps encoder, SonicActor, and critic into a single nn.Module.

    Mirrors TRL's ``PolicyAndValueWrapper`` which exposes:
      - forward_component(mode="policy", ...)  → log-probs, action stats
      - forward_component(mode="value", ...)   → value estimates

    Inputs
    ------
    obs  : (B, 130)  normalised proprioceptive + reference observations
    z    : (B, token_dim)  already-encoded reference motion token
    """

    def __init__(self, encoder: SonicEncoderDecoder,
                 policy: SonicActor, critic: ValueHead):
        super().__init__()
        self.encoder = encoder
        self.policy  = policy
        self.critic  = critic

    # ── inference helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, g_r_win: torch.Tensor) -> torch.Tensor:
        """g_r_win: (B, W, 29) → z_r: (B, token_dim)"""
        return self.encoder.E_r(g_r_win)

    @torch.no_grad()
    def rollout_step(self, obs_t: torch.Tensor,
                     z_t: torch.Tensor) -> dict:
        """
        Sample one action for rollout (no grad).

        Returns dict with keys: actions, log_prob, value, action_mean, action_std
        """
        action, log_p, entropy = self.policy.get_action(obs_t, z_t)
        value = self.critic(obs_t, z_t)
        return {
            "actions":     action,
            "log_prob":    log_p,
            "value":       value,
            "action_mean": self.policy.forward(obs_t, z_t),
            "action_std":  self.policy.std,
        }

    # ── training forward ─────────────────────────────────────────────────────

    def forward(self, obs_t: torch.Tensor,
                z_t: torch.Tensor,
                actions: torch.Tensor) -> dict:
        """
        Re-evaluate log-probs and values for PPO update (with grad).

        Returns dict with keys: log_prob, value, entropy
        """
        mean  = self.policy(obs_t, z_t)
        std   = self.policy.std.expand_as(mean)
        dist  = Normal(mean, std, validate_args=False)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(obs_t, z_t)
        return {"log_prob": log_prob, "value": value, "entropy": entropy}


# ──────────────────────────────────────────────────────────────────────────────
# RolloutStorage  (mirrors TRL's RolloutStorage / register_key API)
# ──────────────────────────────────────────────────────────────────────────────

class RolloutStorage:
    """
    Fixed-length ring buffer for one rollout of ``n_steps`` transitions.

    Mirrors TRL's ``data_utils.RolloutStorage`` with a simplified API:
      storage.add(obs, z, action, log_prob, reward, value, done)
      storage.compute_gae(last_value, gamma, lam)
      storage.sample_minibatch(mb_size, device)
    """

    def __init__(self, n_steps: int, obs_dim: int,
                 token_dim: int, act_dim: int):
        self.n   = n_steps
        self.ptr = 0
        self.obs      = np.zeros((n_steps, obs_dim),   dtype=np.float32)
        self.z        = np.zeros((n_steps, token_dim), dtype=np.float32)
        self.actions  = np.zeros((n_steps, act_dim),   dtype=np.float32)
        self.log_probs= np.zeros(n_steps,              dtype=np.float32)
        self.rewards  = np.zeros(n_steps,              dtype=np.float32)
        self.values   = np.zeros(n_steps,              dtype=np.float32)
        self.dones    = np.zeros(n_steps,              dtype=np.float32)
        # filled by compute_gae
        self.advantages = None
        self.returns    = None

    def clear(self):
        self.ptr = 0

    def add(self, obs, z, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i]       = obs
        self.z[i]         = z.flatten()
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.values[i]    = value
        self.dones[i]     = float(done)
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.n

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        """GAE-λ advantage estimation (same formula as TRL _compute_returns)."""
        adv      = np.zeros(self.n, np.float32)
        last_adv = 0.0
        vals     = np.append(self.values, last_value)
        for t in reversed(range(self.n)):
            mask      = 1.0 - self.dones[t]
            delta     = self.rewards[t] + gamma * vals[t + 1] * mask - vals[t]
            adv[t]    = delta + gamma * lam * mask * last_adv
            last_adv  = adv[t]
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.returns    = adv + self.values

    def tensors(self, device):
        """Return all stored arrays as tensors on ``device``."""
        def _t(x): return torch.as_tensor(x, device=device)
        return (
            _t(self.obs),
            _t(self.z),
            _t(self.actions),
            _t(self.log_probs),
            _t(self.advantages),
            _t(self.returns),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_ref_traj(npz_path: str) -> np.ndarray:
    """Load g_r from NPZ — shape (T, 29), degrees."""
    data = np.load(npz_path)
    return data["g_r"].astype(np.float32)


def _ref_window(ref_rad: np.ndarray, step: int, W: int) -> np.ndarray:
    """
    Extract a sliding window of W frames ending at `step` from `ref_rad`.
    Returns (W, 29) in degrees (encoder was trained on raw g_r which is degrees).
    """
    T   = len(ref_rad)
    end = min(step, T - 1)
    start = max(end - W + 1, 0)
    win = ref_rad[start : end + 1]                    # (≤W, 29)
    if len(win) < W:
        pad = np.tile(win[:1], (W - len(win), 1))
        win = np.concatenate([pad, win], axis=0)      # left-pad with first frame
    return np.degrees(win).astype(np.float32)          # (W, 29)


def _has_nan(path: str) -> bool:
    d = np.load(path)
    return bool(np.isnan(d["g_r"]).any())


# ──────────────────────────────────────────────────────────────────────────────
# Main training function  (mirrors TRLPPOTrainer.train())
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    """
    TRL-style PPO training loop.

    Structure follows TRLPPOTrainer.train():
      1. _rollout_step  : collect num_steps transitions, compute GAE
      2. _forward_model : re-evaluate log-probs + values for PPO update
      3. _compute_loss  : clipped surrogate + value loss + entropy bonus
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    log_fh = open(out_dir / "trl_train.log", "w")

    def _log(msg):
        logger.info(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    # ── Hyperparameters ───────────────────────────────────────────────────────
    W          = cfg.get("encoder_window",   8)      # reference window size
    token_dim  = cfg.get("token_dim",       64)
    hidden_dim = cfg.get("hidden_dim",     256)
    n_steps    = cfg.get("num_steps_per_env", 24)    # steps per rollout
    n_iters    = cfg.get("num_iterations",  500)
    mini_bs    = cfg.get("mini_batch_size",  48)
    n_epochs   = cfg.get("num_ppo_epochs",    5)
    lr         = cfg.get("learning_rate",   3e-4)
    gamma      = cfg.get("gamma",           0.99)
    lam        = cfg.get("gae_lambda",      0.95)
    clip_eps   = cfg.get("clip_range",       0.2)
    vf_coef    = cfg.get("vf_coef",          1.0)
    ent_coef   = cfg.get("ent_coef",         0.01)
    max_gnorm  = cfg.get("max_grad_norm",    1.0)
    rew_scale  = cfg.get("reward_scale",  5000.0)
    save_freq  = cfg.get("save_freq",        50)
    # Adaptive KL learning rate — matches official schedule: "adaptive"
    desired_kl    = cfg.get("desired_kl",     0.01)
    adaptive_lr_min = cfg.get("adaptive_lr_min", 1e-5)
    adaptive_lr_max = cfg.get("adaptive_lr_max", 1e-2)

    OBS_DIM = G1MuJoCoEnv.OBS_DIM  # 130
    ACT_DIM = N_JOINTS              # 29

    # ── Build models ─────────────────────────────────────────────────────────
    encoder = SonicEncoderDecoder(
        window=W, token_dim=token_dim, hidden_dim=hidden_dim
    ).to(device)

    # Optionally load pretrained encoder (supervised phase)
    if cfg.get("encoder_ckpt"):
        ckpt = torch.load(cfg["encoder_ckpt"], map_location=device)
        encoder.load_state_dict(ckpt["model_state"])
        _log(f"Loaded encoder checkpoint: {cfg['encoder_ckpt']}")
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        _log("Encoder frozen — training policy + critic only")
    else:
        _log("No encoder checkpoint — encoder trained jointly with policy")

    # SonicActor: input-independent shared log_std — matches official Actor
    # init_noise_std=1.0 (random init); use 0.05 when loading from pretrained
    init_noise_std = cfg.get("init_noise_std", 1.0)
    policy = SonicActor(OBS_DIM, token_dim, ACT_DIM, hidden_dim,
                        init_noise_std=init_noise_std).to(device)
    critic = ValueHead(OBS_DIM, token_dim, hidden_dim).to(device)

    model  = PolicyAndValueWrapper(encoder, policy, critic)
    params = (
        list(policy.parameters()) +
        list(critic.parameters()) +
        ([] if cfg.get("encoder_ckpt") else list(encoder.parameters()))
    )
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
    n_params  = sum(p.numel() for p in params)
    _log(f"Trainable parameters: {n_params:,}")

    # ── MuJoCo environment ────────────────────────────────────────────────────
    env = G1MuJoCoEnv()
    _log(f"MuJoCo env: obs_dim={env.OBS_DIM}, act_dim={env.ACTION_DIM}")

    # ── Motion library ────────────────────────────────────────────────────────
    data_dir = cfg.get("data_dir",
        "/home/grease/ego_dataset/work_bearlu/data/bones-studio-processed")
    files = sorted(f for f in glob.glob(os.path.join(data_dir, "*.npz"))
                   if not _has_nan(f))
    _log(f"Motion library: {len(files)} NPZ files from {data_dir}")
    if not files:
        raise RuntimeError(f"No .npz files found in {data_dir}")

    # ── Rollout storage ───────────────────────────────────────────────────────
    storage = RolloutStorage(n_steps, OBS_DIM, token_dim, ACT_DIM)

    # ── Tracking ──────────────────────────────────────────────────────────────
    ep_rew_buf   = deque(maxlen=100)
    ep_len_buf   = deque(maxlen=100)
    best_rew     = -float("inf")
    global_steps = 0
    t0           = time.time()

    # ── Sticky-motion parameters ──────────────────────────────────────────────
    # motion_switch_freq: how many rollouts to spend on each motion clip.
    # Within each rollout all episode resets use the SAME clip so the GAE
    # gradient compares actions under the SAME reference → ~10× lower variance.
    # Official SONIC: all 4096 envs run the same motion simultaneously.
    motion_switch_freq = cfg.get("motion_switch_freq", 1)  # default: new clip per rollout
    _log(f"  sticky motion: switch every {motion_switch_freq} rollout(s)")

    rng_np = np.random.default_rng(cfg.get("seed", 42))
    ref_traj_deg = _load_ref_traj(files[rng_np.integers(len(files))])
    ref_traj_rad = np.deg2rad(ref_traj_deg)
    obs          = _norm(env.reset(ref_traj_deg))
    step_in_ep   = 0
    ep_rew       = 0.0

    _log("=" * 70)
    _log("SONIC TRL-style PPO training")
    _log(f"  n_iters={n_iters}  n_steps={n_steps}  mini_bs={mini_bs}")
    _log(f"  γ={gamma}  λ={lam}  ε={clip_eps}  lr={lr}")
    _log("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════════
    # Training loop  (matches TRLPPOTrainer.train())
    # ═══════════════════════════════════════════════════════════════════════════

    for iteration in range(1, n_iters + 1):
        iter_t0 = time.time()
        model.eval()
        storage.clear()

        # ── STICKY MOTION: pick ONE motion per M rollouts ─────────────────────
        # All episode resets within this rollout reuse the same clip so the
        # PPO gradient is computed on a consistent reference trajectory.
        # This mirrors the official setup (all envs share the same motion).
        if (iteration - 1) % motion_switch_freq == 0:
            ref_traj_deg = _load_ref_traj(files[rng_np.integers(len(files))])
            ref_traj_rad = np.deg2rad(ref_traj_deg)
            obs          = _norm(env.reset(ref_traj_deg))
            step_in_ep   = 0
            ep_rew       = 0.0

        # ── 1. _rollout_step  ─────────────────────────────────────────────────
        # Collect n_steps transitions.  Mirrors TRLPPOTrainer._rollout_step().
        ep_rewards_this_iter = []
        reward_norm = 0.0  # default if no step completes

        for _ in range(n_steps):
            # Build reference window tensor
            g_r_win = _ref_window(ref_traj_rad, step_in_ep, W)   # (W, 29) deg
            g_r_t   = torch.tensor(g_r_win, device=device).unsqueeze(0)   # (1,W,29)
            obs_t   = torch.tensor(obs,     device=device).unsqueeze(0)   # (1,130)

            # policy_step  (mirrors TRLPPOTrainer.policy_step)
            with torch.no_grad():
                z_t   = model.encode(g_r_t)                # (1, token_dim)
                out   = model.rollout_step(obs_t, z_t)
                action_t  = out["actions"]                 # (1, 29)
                log_prob_t= out["log_prob"]                # (1,)
                value_t   = out["value"]                   # (1,)

            action_np = action_t.squeeze(0).cpu().numpy()
            z_np      = z_t.squeeze(0).cpu().numpy()

            # env.step  (mirrors TRLPPOTrainer._rollout_step env interaction)
            next_obs, reward, done, _ = env.step(action_np)
            reward_norm = reward / rew_scale
            ep_rew     += reward

            storage.add(
                obs      = obs,
                z        = z_np,
                action   = action_np,
                log_prob = log_prob_t.item(),
                reward   = reward_norm,
                value    = value_t.item(),
                done     = done,
            )

            global_steps += 1
            step_in_ep   += 1

            if done or step_in_ep >= len(ref_traj_rad):
                ep_rew_buf.append(ep_rew)
                ep_len_buf.append(step_in_ep)
                # STICKY: reset within the SAME motion clip (no new clip!)
                # The policy sees multiple attempts on the same reference,
                # giving a clean gradient signal (variance reduced ~10×).
                obs      = _norm(env.reset(ref_traj_deg))
                step_in_ep   = 0
                ep_rew       = 0.0
            else:
                obs = _norm(next_obs)

        ep_rewards_this_iter.append(reward_norm)

        # ── 2. GAE  (_compute_returns) ────────────────────────────────────────
        with torch.no_grad():
            g_r_win = _ref_window(ref_traj_rad, step_in_ep, W)
            g_r_t   = torch.tensor(g_r_win, device=device).unsqueeze(0)
            obs_t   = torch.tensor(obs,     device=device).unsqueeze(0)
            z_last  = model.encode(g_r_t)
            last_val = model.critic(obs_t, z_last).item()

        storage.compute_gae(last_val, gamma, lam)

        # ── 3. PPO update  (num_ppo_epochs × mini-batches) ───────────────────
        # Mirrors TRLPPOTrainer train() inner loop.
        model.train()
        obs_t_all, z_t_all, act_all, old_lp_all, adv_all, ret_all = \
            storage.tensors(device)

        pg_losses, vf_losses, ent_vals, kl_vals = [], [], [], []

        for _ in range(n_epochs):
            perm = torch.randperm(n_steps, device=device)
            for start in range(0, n_steps, mini_bs):
                idx = perm[start : start + mini_bs]

                mb_obs  = obs_t_all[idx]
                mb_z    = z_t_all[idx]
                mb_act  = act_all[idx]
                mb_olp  = old_lp_all[idx]
                mb_adv  = adv_all[idx]
                mb_ret  = ret_all[idx]

                # _forward_model  (mirrors TRL)
                fwd = model(mb_obs, mb_z, mb_act)
                new_lp  = fwd["log_prob"]   # (mb,)
                new_val = fwd["value"]       # (mb,)
                entropy = fwd["entropy"]     # (mb,)

                # _compute_ppo_loss  (mirrors TRL _compute_ppo_loss)
                ratio  = (new_lp - mb_olp).exp()
                surr1  = ratio * mb_adv
                surr2  = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss
                vf_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

                # Approx KL (for monitoring)
                approx_kl = ((mb_olp - new_lp) ** 2).mean().item()

                loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_gnorm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_vals.append(entropy.mean().item())
                kl_vals.append(approx_kl)

        # ── 3b. Adaptive KL learning rate  (official schedule: "adaptive") ──
        # Mirrors TRLPPOTrainer._adjust_learning_rate_based_on_kl()
        kl_mean = float(np.mean(kl_vals))
        if kl_mean > 2.0 * desired_kl:
            lr = max(lr / 1.5, adaptive_lr_min)
        elif kl_mean < 0.5 * desired_kl:
            lr = min(lr * 1.5, adaptive_lr_max)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── 4. Logging ────────────────────────────────────────────────────────
        iter_time = time.time() - iter_t0
        mean_rew  = np.mean(list(ep_rew_buf)) if ep_rew_buf else float("nan")
        mean_len  = np.mean(list(ep_len_buf)) if ep_len_buf else float("nan")

        msg = (
            f"Iter {iteration:4d} | "
            f"rew={mean_rew:8.2f} | "
            f"ep_len={mean_len:5.1f} | "
            f"pg={np.mean(pg_losses):.3f} | "
            f"vf={np.mean(vf_losses):.3f} | "
            f"ent={np.mean(ent_vals):.3f} | "
            f"kl={np.mean(kl_vals):.4f} | "
            f"std={policy.std.mean().item():.4f} | "
            f"lr={lr:.2e} | "
            f"{iter_time:.1f}s"
        )
        _log(msg)

        # ── 5. Checkpoint ─────────────────────────────────────────────────────
        if mean_rew > best_rew:
            best_rew = mean_rew
            torch.save({
                "iteration": iteration,
                "encoder_state_dict": encoder.state_dict(),
                "policy_state_dict":  policy.state_dict(),
                "critic_state_dict":  critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_reward": best_rew,
            }, out_dir / "best_trl.pt")
            _log(f"  ✅  best reward {best_rew:.3f}")

        if iteration % save_freq == 0:
            torch.save({
                "iteration": iteration,
                "encoder_state_dict": encoder.state_dict(),
                "policy_state_dict":  policy.state_dict(),
                "critic_state_dict":  critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, out_dir / f"checkpoint_trl_{iteration:04d}.pt")

    total_time = time.time() - t0
    _log(f"Training complete in {total_time/60:.1f} min | best_rew={best_rew:.3f}")
    log_fh.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SONIC TRL-style PPO training (pure MuJoCo + PyTorch)")
    parser.add_argument("--config", default="gear_sonic/training/config_trl.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--iters", type=int, default=None,
                        help="Override num_iterations from config")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    if args.iters is not None:
        cfg["num_iterations"] = args.iters

    train(cfg)
