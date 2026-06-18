"""
SONIC Combined Trainer — L_PPO + L_recon + L_token + L_cycle in one backward().
Run: python gear_sonic/training/sonic_combined_trainer.py --config gear_sonic/training/config_sonic_combined.yaml
"""

import argparse, glob, json, logging, os, sys, time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.training.encoders       import SonicEncoderDecoder
from gear_sonic.training.losses         import sonic_loss
from gear_sonic.training.g1_mujoco_env import G1MuJoCoEnv, N_JOINTS, OBS_SCALE
from gear_sonic.training.ppo_trainer    import PolicyHead, ValueHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Use shared OBS_SCALE from g1_mujoco_env
_OBS_SCALE = OBS_SCALE

def _norm(obs: np.ndarray) -> np.ndarray:
    return np.clip(obs * _OBS_SCALE, -10., 10.)


# ── Rollout buffer ────────────────────────────────────────────────────────────

class CombinedRolloutBuffer:
    def __init__(self, n, obs_dim, window, act_dim=N_JOINTS):
        self.obs      = np.zeros((n, obs_dim),          dtype=np.float32)
        self.g_r_win  = np.zeros((n, window, N_JOINTS), dtype=np.float32)
        self.actions  = np.zeros((n, act_dim),          dtype=np.float32)
        self.log_probs= np.zeros(n, dtype=np.float32)
        self.rewards  = np.zeros(n, dtype=np.float32)
        self.values   = np.zeros(n, dtype=np.float32)
        self.dones    = np.zeros(n, dtype=np.float32)
        self.ptr = 0; self.n = n

    def add(self, obs, g_r_win, action, lp, reward, value, done):
        i = self.ptr
        self.obs[i]      = obs;  self.g_r_win[i]  = g_r_win
        self.actions[i]  = action; self.log_probs[i]= lp
        self.rewards[i]  = reward; self.values[i]   = value
        self.dones[i]    = done; self.ptr += 1

    def full(self): return self.ptr >= self.n

    def compute_gae(self, last_val, gamma, lam):
        adv = np.zeros(self.n, np.float32); last = 0.
        vals = np.append(self.values, last_val)
        for t in reversed(range(self.n)):
            d = self.rewards[t] + gamma*vals[t+1]*(1-self.dones[t]) - vals[t]
            adv[t] = d + gamma*lam*(1-self.dones[t])*last; last=adv[t]
        self.advantages = (adv-adv.mean())/(adv.std()+1e-8)
        self.returns    = adv + self.values


# ── Supervised dataset ────────────────────────────────────────────────────────

class MotionWindowSampler:
    def __init__(self, data_dir, window, batch_size):
        files = sorted(f for f in glob.glob(os.path.join(data_dir,"*.npz"))
                       if not self._nan(f))
        logger.info(f"  Supervised dataset: {len(files)} files")
        self.bs = batch_size
        self.wins = []
        for f in files:
            d = np.load(f); gr=d["g_r"].astype(np.float32); gh=d["g_h"].astype(np.float32)
            T = min(len(gr),len(gh))
            for s in range(0, T-window+1, window//2):
                self.wins.append((gr[s:s+window], gh[s:s+window]))
        logger.info(f"  Supervised windows: {len(self.wins)}")
        all_gr = np.concatenate([w[0] for w in self.wins[:3000]])
        all_gh = np.concatenate([w[1] for w in self.wins[:3000]])
        self.gr_mu,self.gr_sig = all_gr.mean(0),np.where(all_gr.std(0)<1e-3,1.,all_gr.std(0))
        self.gh_mu,self.gh_sig = all_gh.mean(0),np.where(all_gh.std(0)<1e-3,1.,all_gh.std(0))

    @staticmethod
    def _nan(f):
        d = np.load(f)
        return np.isnan(d["g_r"]).any() or np.isnan(d["g_h"]).any()

    def sample(self, device):
        idx = np.random.randint(0,len(self.wins),self.bs)
        grs = np.stack([(self.wins[i][0]-self.gr_mu)/self.gr_sig for i in idx])
        ghs = np.stack([(self.wins[i][1]-self.gh_mu)/self.gh_sig for i in idx])
        return torch.from_numpy(grs).to(device), torch.from_numpy(ghs).to(device)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ref_window(ref_rad, step, W):
    T=len(ref_rad); end=min(step,T-1); start=max(end-W+1,0)
    win=ref_rad[start:end+1]
    if len(win)<W: win=np.concatenate([np.tile(win[0],(W-len(win),1)),win])
    return np.degrees(win).astype(np.float32)

def _reset(env, files, device, W):
    f=files[np.random.randint(len(files))]
    return env.reset(np.load(f)["g_r"].astype(np.float32))


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"config.json").write_text(json.dumps(cfg,indent=2))
    fh = open(out_dir/"combined_train.log","w")

    def _log(m): logger.info(m); fh.write(m+"\n"); fh.flush()

    W = cfg["encoder_window"]

    # Networks
    encoder = SonicEncoderDecoder(window=W, token_dim=cfg["token_dim"],
                                   hidden_dim=cfg["encoder_hidden"]).to(device)
    policy  = PolicyHead(G1MuJoCoEnv.OBS_DIM, cfg["token_dim"],
                         N_JOINTS, cfg["policy_hidden"]).to(device)
    critic  = ValueHead(G1MuJoCoEnv.OBS_DIM, cfg["token_dim"],
                        cfg["policy_hidden"]).to(device)

    if cfg.get("encoder_ckpt"):
        ck = torch.load(cfg["encoder_ckpt"],map_location=device)
        encoder.load_state_dict(ck["model_state"]); _log(f"Encoder ckpt: {cfg['encoder_ckpt']}")
    if cfg.get("ppo_ckpt"):
        ck = torch.load(cfg["ppo_ckpt"],map_location=device)
        policy.load_state_dict(ck["policy"]); critic.load_state_dict(ck["critic"])

    all_params = list(encoder.parameters())+list(policy.parameters())+list(critic.parameters())
    optim = torch.optim.Adam(all_params, lr=cfg["lr"], eps=1e-5)
    _log(f"Total parameters: {sum(p.numel() for p in all_params):,}")

    sampler  = MotionWindowSampler(cfg["data_dir"], W, cfg["sup_batch"])
    npz_files = sorted(f for f in glob.glob(os.path.join(cfg["data_dir"],"*.npz"))
                        if not MotionWindowSampler._nan(f))
    env  = G1MuJoCoEnv(sim_dt=cfg.get("sim_dt",0.005), control_hz=cfg.get("control_hz",50.),
                       max_episode_frames=cfg.get("episode_frames",300),
                       min_height=cfg.get("min_height",0.3))
    buf  = CombinedRolloutBuffer(cfg["n_steps"], G1MuJoCoEnv.OBS_DIM, W)

    def _ot(obs_np):   # obs → normalised cuda tensor (1, OBS_DIM)
        return torch.from_numpy(_norm(obs_np)).unsqueeze(0).to(device)

    obs_np       = _reset(env, npz_files, device, W)
    ref_rad      = env._ref_traj; step_ep = 0
    ep_rewards   = deque(maxlen=100); ep_rew = 0.; best = -np.inf

    _log("="*65)
    _log("SONIC Combined  L_total = L_PPO + L_recon + L_token + L_cycle")
    _log("="*65)

    for itr in range(1, cfg["n_iterations"]+1):
        buf.ptr = 0; t0 = time.time()
        encoder.eval()

        # ── collect rollout ───────────────────────────────────────────────────
        while not buf.full():
            g_r_win = _ref_window(ref_rad, step_ep, W)
            obs_t   = _ot(obs_np)
            g_r_t   = torch.from_numpy(g_r_win).unsqueeze(0).to(device)
            with torch.no_grad():
                z              = encoder.E_r(g_r_t)
                act, lp, _     = policy.get_action(obs_t, z)
                val            = critic(obs_t, z)
            action_np = act.squeeze(0).cpu().numpy()
            action_np = act.squeeze(0).cpu().numpy()
            obs_np2, rew, done, _ = env.step(action_np)
            ep_rew += rew
            buf.add(_norm(obs_np), g_r_win, action_np, lp.item(), rew, val.item(), float(done))
            obs_np = obs_np2; step_ep += 1
            if done:
                ep_rewards.append(ep_rew); ep_rew = 0.
                obs_np = _reset(env, npz_files, device, W)
                ref_rad = env._ref_traj; step_ep = 0

        with torch.no_grad():
            g_r_t = torch.from_numpy(_ref_window(ref_rad, step_ep, W)).unsqueeze(0).to(device)
            last_val = critic(_ot(obs_np), encoder.E_r(g_r_t)).item()
        buf.compute_gae(last_val, cfg["gamma"], cfg["gae_lambda"])

        # ── PPO + supervised update ───────────────────────────────────────────
        encoder.train()
        obs_b  = torch.from_numpy(buf.obs).to(device)
        grw_b  = torch.from_numpy(buf.g_r_win).to(device)
        act_b  = torch.from_numpy(buf.actions).to(device)
        olp_b  = torch.from_numpy(buf.log_probs).to(device).clamp(-20., 0.)  # prevent extreme ratios
        adv_b  = torch.from_numpy(buf.advantages).to(device)
        ret_b  = torch.from_numpy(buf.returns.astype(np.float32)).to(device)

        st = {k:0. for k in ("PPO","recon","token","cycle","total")}
        n_mini = cfg["n_steps"]//cfg["ppo_mini_batch"]

        for _ in range(cfg["k_epochs"]):
            idx = torch.randperm(cfg["n_steps"], device=device)
            for s in range(0, cfg["n_steps"], cfg["ppo_mini_batch"]):
                mb = idx[s:s+cfg["ppo_mini_batch"]]

                # PPO (re-encode with grad so E_r gets gradient from policy)
                z_p          = encoder.E_r(grw_b[mb])
                mean, lstd   = policy(obs_b[mb], z_p)
                std          = lstd.exp().clamp(1e-4, 2.0)   # prevent std collapse / explosion
                dist         = Normal(mean, std, validate_args=False)
                nlp          = dist.log_prob(act_b[mb]).sum(-1)
                ent          = dist.entropy().sum(-1).mean()
                ratio        = (nlp - olp_b[mb]).exp().clamp(0., 10.)  # prevent ratio explosion
                a            = adv_b[mb]
                L_ppo        = -torch.min(ratio*a, ratio.clamp(1-cfg["clip_eps"],1+cfg["clip_eps"])*a).mean()
                L_val        = F.mse_loss(critic(obs_b[mb], z_p.detach()), ret_b[mb])

                # Supervised (independent mini-batch from dataset)
                gr_s, gh_s   = sampler.sample(device)
                enc_out      = encoder(gr_s, gh_s)
                L_sup, comp  = sonic_loss(enc_out, gr_s,
                                          lambda_recon=cfg["lambda_recon"],
                                          lambda_token=cfg["lambda_token"],
                                          lambda_cycle=cfg["lambda_cycle"])

                loss = L_ppo + cfg["vf_coef"]*L_val - cfg["ent_coef"]*ent + L_sup
                if torch.isnan(loss):
                    _log(f"  ⚠ NaN loss in iter {itr} — skipping mini-batch")
                    continue
                optim.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(all_params, cfg["max_grad_norm"]); optim.step()

                # Reset any NaN weights (can happen early in training)
                for p in all_params:
                    if p.data.isnan().any():
                        nn.init.zeros_(p.data); _log("  ⚠ NaN weight reset")

                st["PPO"]  += L_ppo.item(); st["recon"] += comp["L_recon"]
                st["token"]+= comp["L_token"]; st["cycle"]+= comp["L_cycle"]
                st["total"]+= loss.item()

        denom = cfg["k_epochs"]*n_mini
        for k in st: st[k] /= denom
        mr = float(np.mean(ep_rewards)) if ep_rewards else 0.
        _log(f"Iter {itr:4d} | rew={mr:.3f} | PPO={st['PPO']:.3f} "
             f"recon={st['recon']:.3f} token={st['token']:.4f} "
             f"cycle={st['cycle']:.4f} | {time.time()-t0:.1f}s")

        ck = {"encoder":encoder.state_dict(),"policy":policy.state_dict(),
              "critic":critic.state_dict(),"optim":optim.state_dict(),"itr":itr}
        torch.save(ck, out_dir/"last_combined.pt")
        if mr > best: best=mr; torch.save(ck, out_dir/"best_combined.pt"); _log(f"  ✅  best reward {best:.3f}")

    fh.close()


def main():
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="gear_sonic/training/config_sonic_combined.yaml")
    p.add_argument("--iters", type=int)
    args = p.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    if args.iters: cfg["n_iterations"] = args.iters
    train(cfg)

if __name__ == "__main__":
    main()
