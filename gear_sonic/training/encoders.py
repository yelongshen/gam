"""
SONIC Encoder / Decoder Architecture
=====================================

Three encoders (E_r, E_h, E_m) each compress a short temporal window of
a different motion representation into a shared fixed-size latent token z.

One shared decoder D_r reconstructs g_r (robot joint trajectory) from any
token, enabling the reconstruction and cycle-consistency losses.

All tokens share the same output dimension (token_dim) so the downstream
policy decoder is agnostic to which encoder is used at deployment time.
"""

import torch
import torch.nn as nn
from typing import Optional


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mlp(dims: list, act=nn.GELU, norm=True) -> nn.Sequential:
    """Build a simple MLP with optional LayerNorm after each hidden layer."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:          # no norm/act on final layer
            if norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


# ── Encoders ─────────────────────────────────────────────────────────────────

class MotionEncoder(nn.Module):
    """
    Generic temporal window → latent token encoder.

    Input:  (B, W, feature_dim)
    Output: (B, token_dim)

    Architecture: flatten window → 3-layer MLP → token.
    Shared by E_r, E_h, E_m — only the input dimension differs.
    """

    def __init__(
        self,
        feature_dim: int,
        window: int,
        token_dim: int = 64,
        hidden_dim: int = 256,
        name: str = "encoder",
    ):
        super().__init__()
        self.name = name
        self.feature_dim = feature_dim
        self.window = window
        self.token_dim = token_dim

        in_dim = feature_dim * window
        self.net = _mlp([in_dim, hidden_dim, hidden_dim, token_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, F) → z: (B, token_dim)"""
        B = x.shape[0]
        return self.net(x.reshape(B, -1))


class RobotEncoder(MotionEncoder):
    """E_r: g_r (29-DoF joint angles) → z_r"""
    def __init__(self, window: int, token_dim: int = 64, hidden_dim: int = 256):
        super().__init__(29, window, token_dim, hidden_dim, name="E_r")


class HumanEncoder(MotionEncoder):
    """E_h: g_h (72-dim SMPL joint positions, 24×3) → z_h"""
    def __init__(self, window: int, token_dim: int = 64, hidden_dim: int = 256):
        super().__init__(72, window, token_dim, hidden_dim, name="E_h")


class MixedEncoder(MotionEncoder):
    """E_m: g_m (11-dim: 3 VR tracker XYZ + 2 lower-body scalars) → z_m"""
    def __init__(self, window: int, token_dim: int = 64, hidden_dim: int = 256):
        super().__init__(11, window, token_dim, hidden_dim, name="E_m")


# ── Decoder ───────────────────────────────────────────────────────────────────

class MotionDecoder(nn.Module):
    """
    Shared motion decoder D_r.

    Input:  (B, token_dim)  — any of z_r, z_h, z_m
    Output: (B, W, 29)      — reconstructed g_r window

    This decoder is SHARED across all three tokens during training so that
    the reconstruction loss forces each encoder to produce a token that lies
    in the same semantic space as z_r.
    """

    def __init__(
        self,
        token_dim: int = 64,
        window: int = 8,
        g_r_dim: int = 29,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.window = window
        self.g_r_dim = g_r_dim
        out_dim = g_r_dim * window
        self.net = _mlp([token_dim, hidden_dim, hidden_dim, out_dim])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, token_dim) → g_r_hat: (B, W, 29)"""
        return self.net(z).reshape(z.shape[0], self.window, self.g_r_dim)


# ── Combined SONIC encoder–decoder model ──────────────────────────────────────

class SonicEncoderDecoder(nn.Module):
    """
    SONIC encoder–decoder — Phase 1: robot ↔ human only (no g_m / E_m).

    Two encoders share one decoder D_r.  E_m is excluded to simplify the
    first training phase; add it back by setting use_mixed=True once the
    robot↔human alignment is stable.

    Step 2 of the SONIC training flow:
      1. Encode g_r → z_r  and  g_h → z_h
      2. Decode from both tokens → g_r_hat  (for L_recon)
      3. Cycle: D_r(z_h) → E_r → z_r_cycle  (for L_cycle)
    """

    def __init__(
        self,
        window: int = 8,
        token_dim: int = 64,
        hidden_dim: int = 256,
        use_mixed: bool = False,   # set True to re-enable E_m
    ):
        super().__init__()
        self.window     = window
        self.token_dim  = token_dim
        self.use_mixed  = use_mixed

        self.E_r = RobotEncoder(window, token_dim, hidden_dim)
        self.E_h = HumanEncoder(window, token_dim, hidden_dim)
        self.D_r = MotionDecoder(token_dim, window, 29, hidden_dim)

        if use_mixed:
            self.E_m = MixedEncoder(window, token_dim, hidden_dim)

    def forward(
        self,
        g_r: torch.Tensor,
        g_h: torch.Tensor,
        g_m: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        g_r : (B, W, 29)  — robot joint angles  [required]
        g_h : (B, W, 72)  — SMPL joint positions [required]
        g_m : (B, W, 11)  — VR mixed input       [optional, only if use_mixed=True]
        """
        # ── encode ───────────────────────────────────────────────────────────
        z_r = self.E_r(g_r)   # (B, token_dim)
        z_h = self.E_h(g_h)   # (B, token_dim)

        # ── decode — reconstruct g_r from both tokens ─────────────────────
        g_r_from_r = self.D_r(z_r)   # (B, W, 29)
        g_r_from_h = self.D_r(z_h)   # (B, W, 29)

        # ── cycle: D_r(z_h) → E_r → z_r_cycle ───────────────────────────
        z_r_cycle = self.E_r(g_r_from_h)   # (B, token_dim)

        out = {
            "z_r": z_r,
            "z_h": z_h,
            "g_r_from_r": g_r_from_r,
            "g_r_from_h": g_r_from_h,
            "z_r_cycle":  z_r_cycle,
        }

        # optional E_m path (Phase 2)
        if self.use_mixed and g_m is not None:
            z_m = self.E_m(g_m)
            out["z_m"]       = z_m
            out["g_r_from_m"] = self.D_r(z_m)

        return out

    @torch.no_grad()
    def encode(
        self,
        g_r: Optional[torch.Tensor] = None,
        g_h: Optional[torch.Tensor] = None,
        g_m: Optional[torch.Tensor] = None,
    ) -> dict:
        """Inference helper: encode one or more modalities."""
        out = {}
        if g_r is not None: out["z_r"] = self.E_r(g_r)
        if g_h is not None: out["z_h"] = self.E_h(g_h)
        if g_m is not None and self.use_mixed: out["z_m"] = self.E_m(g_m)
        return out
