"""
SONIC supervised training experiment (no RL/PPO).

Implements Steps 1–2 of the SONIC training flow with:
  L_total = L_recon + λ_token * L_token + λ_cycl                loss, comp = sonic_loss(out, g_r,
                                         lambda_recon=cfg.get("lambda_recon", 1.0),
                                         lambda_token=cfg["lambda_token"],
                                         lambda_cycle=cfg["lambda_cycle"])L_cycle

No physics simulation is used — this is the encoder-alignment phase.
Run with:
    python gear_sonic/training/sonic_train_exp.py \
        --config gear_sonic/training/config_sonic_exp.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.training.encoders import SonicEncoderDecoder
from gear_sonic.training.losses import sonic_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Inline sliding-window dataset (avoids version conflicts) ──────────────────
import glob
import numpy as np

class _WindowDataset(torch.utils.data.Dataset):
    """Simple sliding-window dataset over processed NPZ files."""

    def __init__(self, data_dir, window, stride, val_ratio, split, stats=None):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        n_val = max(1, int(len(files) * val_ratio))
        files  = files[:n_val] if split == "val" else files[n_val:]
        logger.info(f"  [{split}] {len(files)} files")

        raws = {"g_r": [], "g_h": [], "g_m": []}
        skipped = 0
        for f in files:
            d = np.load(f)
            T = min(len(d["g_r"]), len(d["g_h"]), len(d["g_m"]))
            g_r_f = d["g_r"][:T].astype(np.float32)
            g_h_f = d["g_h"][:T].astype(np.float32)
            g_m_f = d["g_m"][:T].astype(np.float32)
            # Skip files that still have NaN (e.g. pre-FK-fix leftovers)
            if np.isnan(g_r_f).any() or np.isnan(g_h_f).any() or np.isnan(g_m_f).any():
                skipped += 1
                continue
            raws["g_r"].append(g_r_f)
            raws["g_h"].append(g_h_f)
            raws["g_m"].append(g_m_f)
        if skipped:
            logger.warning(f"  [{split}] Skipped {skipped} files with NaN values")

        if stats is None:   # compute on training split
            self.stats = {}
            for k in raws:
                arr = np.concatenate(raws[k])
                mu  = np.nan_to_num(np.nanmean(arr, axis=0), nan=0.0)
                sig = np.nan_to_num(np.nanstd(arr, axis=0),  nan=1.0)
                sig = np.where(sig < 1e-3, 1.0, sig)   # avoid ÷0 for constant features
                self.stats[k] = (mu.astype(np.float32), sig.astype(np.float32))
        else:
            self.stats = stats

        # normalise
        for k in raws:
            raws[k] = [(x - self.stats[k][0]) / self.stats[k][1] for x in raws[k]]

        # build windows
        self.wins = []
        for gr, gh, gm in zip(raws["g_r"], raws["g_h"], raws["g_m"]):
            for s in range(0, len(gr) - window + 1, stride):
                self.wins.append((gr[s:s+window], gh[s:s+window], gm[s:s+window]))
        logger.info(f"  [{split}] {len(self.wins)} windows")

    def __len__(self):  return len(self.wins)
    def __getitem__(self, i):
        return tuple(torch.from_numpy(x) for x in self.wins[i])


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ── Data ─────────────────────────────────────────────────────────────────
    W       = cfg["window"]
    stride  = cfg.get("stride", W // 2)
    ds_tr   = _WindowDataset(cfg["data_dir"], W, stride, cfg["val_ratio"], "train")
    ds_val  = _WindowDataset(cfg["data_dir"], W, stride, cfg["val_ratio"], "val",
                             stats=ds_tr.stats)

    dl_tr   = DataLoader(ds_tr,  batch_size=cfg["batch_size"], shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)
    dl_val  = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SonicEncoderDecoder(
        window=W,
        token_dim=cfg["token_dim"],
        hidden_dim=cfg["hidden_dim"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                               weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )

    best_val = float("inf")
    log_path = out_dir / "train.log"
    log_fh   = open(log_path, "w")

    def _log(msg):
        logger.info(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    _log(f"Starting SONIC supervised experiment — {cfg['epochs']} epochs")
    _log(f"Train windows: {len(ds_tr)}  Val windows: {len(ds_val)}")

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_totals = {"L_recon": 0., "L_token": 0., "L_cycle": 0., "L_total": 0.}
        t0 = time.time()

        for g_r, g_h, g_m in dl_tr:
            g_r, g_h, g_m = g_r.to(device), g_h.to(device), g_m.to(device)
            optim.zero_grad()
            out  = model(g_r, g_h, g_m)
            loss, comp = sonic_loss(out, g_r,
                                    lambda_recon=cfg.get("lambda_recon", 1.0),
                                    lambda_token=cfg["lambda_token"],
                                    lambda_cycle=cfg["lambda_cycle"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            optim.step()
            for k in tr_totals: tr_totals[k] += comp[k]

        n_batch = len(dl_tr)
        tr_avg  = {k: v / n_batch for k, v in tr_totals.items()}

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_total = 0.
        with torch.no_grad():
            for g_r, g_h, g_m in dl_val:
                g_r, g_h, g_m = g_r.to(device), g_h.to(device), g_m.to(device)
                out  = model(g_r, g_h, g_m)
                loss, _ = sonic_loss(out, g_r,
                                     lambda_token=cfg["lambda_token"],
                                     lambda_cycle=cfg["lambda_cycle"])
                val_total += loss.item()
        val_loss = val_total / max(len(dl_val), 1)

        scheduler.step()
        elapsed = time.time() - t0

        msg = (f"Epoch {epoch:>3}/{cfg['epochs']}  "
               f"train={tr_avg['L_total']:.4f} "
               f"(recon={tr_avg['L_recon']:.4f} "
               f"token={tr_avg['L_token']:.4f} "
               f"cycle={tr_avg['L_cycle']:.4f})  "
               f"val={val_loss:.4f}  "
               f"lr={scheduler.get_last_lr()[0]:.2e}  "
               f"{elapsed:.1f}s")
        _log(msg)

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
            "norm_stats": {k: (v[0].tolist(), v[1].tolist())
                           for k, v in ds_tr.stats.items()},
        }
        torch.save(ckpt, out_dir / "last_model.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best_model.pt")
            _log(f"  ✅  New best val loss: {best_val:.4f}")

    _log(f"\nDone. Best val loss: {best_val:.4f}")
    _log(f"Checkpoints: {out_dir}/best_model.pt")
    log_fh.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import yaml

    parser = argparse.ArgumentParser(description="SONIC supervised encoder experiment")
    parser.add_argument("--config", default="gear_sonic/training/config_sonic_exp.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:     cfg["epochs"]     = args.epochs
    if args.batch_size: cfg["batch_size"] = args.batch_size
    if args.lr:         cfg["lr"]         = args.lr

    train(cfg)


if __name__ == "__main__":
    main()
