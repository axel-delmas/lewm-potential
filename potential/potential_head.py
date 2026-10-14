"""Per-pair scalar potential head U(x_tau, tau | h_t, z_t, a_t) (no attention).

SinusoidalEmb, AdaLNZeroBlock and AdaLNFinal are copied verbatim from the bridge repo's
fm/flow_head.py (github.com/axel-delmas/fm-jepa) so that the conditioning trunk is identical to
VelocityHead's. Only the output layer differs: Linear(W, 1) instead of Linear(W, d).

Every nonlinearity on the x path is smooth (LayerNorm, GELU; SiLU on the conditioning path), so
grad_x U is continuous. With ReLU, grad_x U would be piecewise constant in x.
"""

import math

import torch
from torch import nn


class SinusoidalEmb(nn.Module):
    """Fixed sin/cos embedding of tau in [0, 1] (diffusion-style, tau scaled by 1000)."""

    def __init__(self, dim=128, max_period=10000.0, scale=1000.0):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.scale = scale

    def forward(self, tau):
        args = tau.reshape(-1, 1).float() * self.scale * self.freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLNZeroBlock(nn.Module):
    """x + gate * MLP(LN(x) * (1 + scale) + shift), modulation zero-initialised."""

    def __init__(self, W, mlp_ratio=4, p_drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(W, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(W, mlp_ratio * W),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(mlp_ratio * W, W),
            nn.Dropout(p_drop),
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(W, 3 * W))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, c):
        shift, scale, gate = self.mod(c).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class AdaLNFinal(nn.Module):
    """LN(x) * (1 + scale) + shift, modulation zero-initialised."""

    def __init__(self, W):
        super().__init__()
        self.norm = nn.LayerNorm(W, elementwise_affine=False, eps=1e-6)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(W, 2 * W))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, c):
        shift, scale = self.mod(c).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class PotentialHead(nn.Module):
    """U(x_tau, tau | h_t, z_t, a_t) -> (N,). All inputs are flattened pairs (N, .).

    Same trunk as VelocityHead (x_in, tau embedding, conditioning MLP, AdaLN-zero blocks, AdaLN
    final), but a scalar output layer, zero-initialised: U == 0 and v = -grad_x U == 0 at init.
    There is no out_norm option: normalising a scalar potential (or rescaling its gradient per
    dimension) would make the field non-conservative.
    """

    def __init__(self, d=192, W=256, n_blocks=4, mlp_ratio=4, a_dim=192, p_drop=0.0, t_dim=128):
        super().__init__()
        self.x_in = nn.Linear(d, W)
        self.t_emb = nn.Sequential(SinusoidalEmb(t_dim), nn.Linear(t_dim, W), nn.SiLU(), nn.Linear(W, W))
        self.cond = nn.Sequential(nn.Linear(d + d + a_dim + W, W), nn.SiLU(), nn.Linear(W, W))
        self.blocks = nn.ModuleList([AdaLNZeroBlock(W, mlp_ratio, p_drop) for _ in range(n_blocks)])
        self.final = AdaLNFinal(W)
        self.out = nn.Linear(W, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_tau, tau, h_t, z_t, a_emb):
        t = self.t_emb(tau)
        # under bf16 autocast the inputs can mix fp32/bf16; cat promotes, the next Linear recasts
        c = self.cond(torch.cat([h_t, z_t, a_emb, t], dim=-1))
        x = self.x_in(x_tau)
        for blk in self.blocks:
            x = blk(x, c)
        return self.out(self.final(x, c)).squeeze(-1)
