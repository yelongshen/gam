"""
SONIC combined loss functions.

Step 3 of the SONIC training flow (from TRAINING_README.md):

    L_total = L_PPO + λ_recon * L_recon + λ_token * L_token + λ_cycle * L_cycle

What is implemented here (supervised phase, no physics sim):
    L_total = λ_recon * L_recon + λ_token * L_token + λ_cycle * L_cycle

What is NOT yet implemented:
    L_PPO — PPO surrogate loss.  Requires executing decoded motor actions in
            MuJoCo/Isaac physics simulation and computing reward.
            See TRAINING_README.md § Item 3 (ppo_trainer.py).

Loss definitions (exactly as specified):

    L_recon = ||D_r(z_r) - g_r||² + ||D_r(z_h) - g_r||² + ||D_r(z_m) - g_r||²
              ← all three tokens must reconstruct g_r via the shared decoder.
              ← D_r(z_h) term implicitly teaches E_h the human→robot mapping.

    L_token = ||z_r - z_h||²
              ← pulls robot and human tokens together in latent space.
              ← bidirectional: gradients flow into BOTH E_r and E_h.

    L_cycle = ||E_r(D_r(z_h)) - z_r||²
              ← E_r applied to a z_h-decoded motion must recover z_r.
              ← z_r target is detached: gradient flows only through E_r + D_r,
                not back into E_r a second time (avoids double-gradient collapse).
"""

import torch
import torch.nn.functional as F


_mse = F.mse_loss


def sonic_loss(
    out: dict,
    g_r: torch.Tensor,
    lambda_recon: float = 1.0,
    lambda_token: float = 0.5,
    lambda_cycle: float = 0.1,
) -> tuple:
    """
    Compute the combined SONIC supervised loss (no PPO).

    Parameters
    ----------
    out           : dict returned by SonicEncoderDecoder.forward()
    g_r           : target robot trajectory  (B, W, 29)
    lambda_recon  : weight for L_recon  (default 1.0)
    lambda_token  : weight for L_token  (default 0.5)
    lambda_cycle  : weight for L_cycle  (default 0.1)

    Returns
    -------
    loss       : scalar tensor (back-propagate this)
    components : dict of individual float values for logging
    """

    # ── L_recon ───────────────────────────────────────────────────────────────
    # Phase 1 (no g_m): L_recon = ||D_r(z_r)-g_r||² + ||D_r(z_h)-g_r||²
    # Phase 2 (use_mixed): adds       + ||D_r(z_m)-g_r||²  when "g_r_from_m" in out
    L_recon = (
        _mse(out["g_r_from_r"], g_r)
        + _mse(out["g_r_from_h"], g_r)
    )
    if "g_r_from_m" in out:
        L_recon = L_recon + _mse(out["g_r_from_m"], g_r)

    # ── L_token ───────────────────────────────────────────────────────────────
    # Spec: L_token = ||z_r - z_h||²
    # Both sides are live tensors → gradients flow into E_r AND E_h.
    # (No .detach() — bidirectional supervision as specified.)
    L_token = _mse(out["z_r"], out["z_h"])

    # ── L_cycle ───────────────────────────────────────────────────────────────
    # Spec: L_cycle = ||E_r(D_r(z_h)) - z_r||²
    # z_r TARGET is detached so the gradient only flows through E_r and D_r,
    # not back through E_r a second time (which would cause gradient collapse).
    L_cycle = _mse(out["z_r_cycle"], out["z_r"].detach())

    # ── L_total (no PPO term — supervised phase) ──────────────────────────────
    loss = lambda_recon * L_recon + lambda_token * L_token + lambda_cycle * L_cycle

    return loss, {
        "L_recon": L_recon.item(),
        "L_token": L_token.item(),
        "L_cycle": L_cycle.item(),
        "L_total": loss.item(),
        # L_PPO: not implemented — requires physics simulation (MuJoCo/Isaac)
    }
