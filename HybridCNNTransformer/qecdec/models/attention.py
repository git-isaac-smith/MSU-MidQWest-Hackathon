"""Token features, learned relative space-time attention bias, and the transformer block."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """Fixed sin/cos features of integer (x, y, t) grid coordinates.

        phi(c) = [sin(w_k c), cos(w_k c)]_{k=0..K-1},   w_k = pi * 2^k / 16

    Wavelengths run from 32 down to 1 stabiliser spacings, independent of d, so the same
    features are valid at every distance. Output dim = 3 coords * 2 * K.
    """

    def __init__(self, n_freq: int = 6):
        super().__init__()
        w = math.pi * (2.0 ** torch.arange(n_freq)) / 16.0
        self.register_buffer("w", w, persistent=False)
        self.out_dim = 3 * 2 * n_freq

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:  # [..., 3] -> [..., 6K]
        a = xyt.float().unsqueeze(-1) * self.w                  # [..., 3, K]
        return torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)


class RelativeBias(nn.Module):
    """Learned additive attention bias, one value per head.

    event <-> event : b(dx, dy, dt) = MLP(clip(dx), clip(dy), clip(dt)) with offsets clipped to
                      [-C, C]. Because the clipped offsets take only (2C+1)^3 values we evaluate
                      the MLP on that whole table once and gather.
    event <-> boundary token j : learned table indexed by the event's clipped distance to boundary j.
    everything else (global token, virtual<->virtual) : one learned scalar per head.
    Padded keys get -inf.
    """

    N_BOUNDARIES = 4

    def __init__(self, heads: int, rel_clip: int, hidden: int, boundary_clip: int):
        super().__init__()
        self.heads, self.C = heads, rel_clip
        n = 2 * rel_clip + 1
        r = torch.arange(-rel_clip, rel_clip + 1).float()
        offs = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).view(-1, 3) / rel_clip
        self.register_buffer("offs", offs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, heads))
        self.vb = nn.Parameter(torch.zeros(self.N_BOUNDARIES, boundary_clip + 1, heads))
        self.special = nn.Parameter(torch.zeros(heads))
        self.n = n

    def pair_index(self, xyt: torch.Tensor) -> torch.Tensor:
        """xyt [B, E, 3] long -> flat table index [B, E, E] of the clipped offset x_i - x_j."""
        rel = (xyt.unsqueeze(2) - xyt.unsqueeze(1)).clamp(-self.C, self.C) + self.C
        return (rel[..., 0] * self.n + rel[..., 1]) * self.n + rel[..., 2]

    def forward(self, pair_idx: torch.Tensor, tok_bdist: torch.Tensor, n_virtual: int,
                key_pad: torch.Tensor) -> torch.Tensor:
        """Return bias [B, heads, S, S] with S = 1 + n_virtual + E. key_pad [B, S] True = padded."""
        B, E = tok_bdist.shape[:2]
        nsp = 1 + n_virtual
        S, H = nsp + E, self.heads
        bias = self.special.view(1, H, 1, 1).expand(B, H, S, S).clone()
        if E > 0:
            table = self.mlp(self.offs)                           # [(2C+1)^3, H]
            bias[:, :, nsp:, nsp:] = table[pair_idx].permute(0, 3, 1, 2)
            if n_virtual > 0:
                bidx = torch.arange(self.N_BOUNDARIES, device=tok_bdist.device)
                vbe = self.vb[bidx, tok_bdist]                    # [B, E, 4, H]
                vbe = vbe.permute(0, 3, 2, 1)                     # [B, H, 4, E]
                bias[:, :, 1:1 + n_virtual, nsp:] = vbe           # boundary-token queries -> event keys
                bias[:, :, nsp:, 1:1 + n_virtual] = vbe.transpose(2, 3)  # event queries -> boundary keys
        return bias.masked_fill(key_pad[:, None, None, :], float("-inf"))


class AttnBlock(nn.Module):
    """Pre-LayerNorm transformer layer with the relative bias added to the attention logits."""

    def __init__(self, width: int, heads: int, mlp_ratio: int, rel_clip: int, bias_hidden: int,
                 boundary_clip: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_ratio * width), nn.GELU(),
                                 nn.Linear(mlp_ratio * width, width))
        self.bias = RelativeBias(heads, rel_clip, bias_hidden, boundary_clip)

    def forward(self, x, pair_idx, tok_bdist, n_virtual, key_pad):
        B, S, W = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, self.heads, W // self.heads).permute(2, 0, 3, 1, 4)
        bias = self.bias(pair_idx, tok_bdist, n_virtual, key_pad).to(q.dtype)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)   # softmax(QK^T/sqrt(dh) + b) V
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, W))
        return x + self.mlp(self.ln2(x))
