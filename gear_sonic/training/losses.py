"""
SONIC combined loss functions (supervised phase — no RL/PPO).

L_total = L_recon + λ_token * L_token + λ_cycle * L_cycle

L_recon : all three tokens must reconstruct g_r via the shared decoder D_r
L_token : direct alignment — pulls z_r and z_h together in token space
L_cycle : E_r(D_r(z_h)) must recover z_r  (cycle-consistency)
"""

import torch
import torch.nn as nn
from typing import Optional


_mse = nn.functional.mse_loss


def sonic_loss(
    out: dict,
    g_r: torch.Tensor,
    lambda_token: float = 0.5,
    lambda_cycle: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the combined SONIC supervised loss.

    Parameters
    ----------
    out          : dict returned by SonicEncoderDecoder.forward()
    g_r          : target robot trajectory (B, W, 29)
    lambda_token : weight for L_token
    lambda_cycle : weight for L_cycle

    Returns
    -------
    loss         : scalar tensor
    components   : dict of individual loss values (for logging)
    """
    # ── L_recon ───────────────────────────────────────────────────────────────
    # All three tokens must reconstruct g_r via the shared decoder.
    # This implicitly retargets: gradients flow through E_h teaching it the
    # human→robot mapping without explicit retargeting supervision.
    L_recon = (
        _mse(out["g_r_from_r"], g_r)
        + _mse(out["g_r_from_h"], g_r)
        + _mse(out["g_r_from_m"], g_r)
    )

    # ── L_token ───────────────────────────────────────────────────────────────
    # Direct cross-modal supervision: pull z_r and z_h together.
    L_token = _mse(out["z_r"], out["z_h"].detach())

    # ── L_cycle ───────────────────────────────────────────────────────────────
    # E_r(D_r(z_h)) must recover z_r.
    # Detach z_r target so the cycle gradient flows only through E_r and D_r.
    L_cycle = _mse(out["z_r_cycle"], out["z_r"].detach())

    loss = L_recon + lambda_token * L_token + lambda_cycle * L_cycle

    return loss, {
        "L_recon": L_recon.item(),
        "L_token": L_token.item(),
        "L_cycle": L_cycle.item(),
        "L_total": loss.item(),
    }
