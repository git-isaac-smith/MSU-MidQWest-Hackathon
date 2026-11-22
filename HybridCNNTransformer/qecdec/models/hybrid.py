"""Hybrid decoder: mask-aware CNN stem -> event tokens (+ boundary & global tokens) -> attention -> logit.

Forward pass
------------
1. Stem: `stem_layers` mask-aware 3x3x3 convs (GroupNorm + GELU) on the dense grid -> F [B, W, T, H, W].
2. Tokens: for every fired detector gather F at its (t, y, x), concatenate
   Fourier features of (x, y, t) and (if use_boundary_channels) the 8 static boundary features,
   and project to the model width.
3. Sequence = [global | 4 virtual boundary tokens (if use_boundary_tokens) | event tokens],
   right-padded to the batch's max event count; padded keys are masked with -inf.
4. `attn_layers` pre-LN transformer layers with a learned relative space-time bias added to the
   attention logits (see attention.py).
5. Readout: LayerNorm(global token) -> Linear -> one logit (BCE loss outside).

A shot with no events has only the global (+ boundary) tokens, which still attend to each other.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.geometry import C_IN, N_BASE, N_BFEAT, N_BOUNDARY
from .attention import AttnBlock, FourierFeatures
from .maskconv import MaskedConv3d


class HybridDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.n_in = C_IN if cfg.use_boundary_channels else N_BASE
        self.n_virtual = N_BOUNDARY if cfg.use_boundary_tokens else 0
        self.mask_aware = cfg.mask_aware_conv

        self.convs = nn.ModuleList(
            MaskedConv3d(self.n_in if i == 0 else w, w, k=3, mask_aware=cfg.mask_aware_conv)
            for i in range(cfg.stem_layers))
        self.norms = nn.ModuleList(nn.GroupNorm(8, w) for _ in range(cfg.stem_layers))

        self.fourier = FourierFeatures(cfg.n_fourier)
        self.n_bfeat = N_BFEAT if cfg.use_boundary_channels else 0
        self.tok_proj = nn.Linear(w + self.fourier.out_dim + self.n_bfeat, w)
        self.special = nn.Parameter(torch.randn(1 + self.n_virtual, w) * 0.02)   # global, then 4 boundaries

        self.blocks = nn.ModuleList(
            AttnBlock(w, cfg.heads, cfg.mlp_ratio, cfg.rel_clip, cfg.bias_hidden, cfg.boundary_clip)
            for _ in range(cfg.attn_layers))
        self.out_ln = nn.LayerNorm(w)
        self.head = nn.Linear(w, 1)

    def stem(self, grid: torch.Tensor) -> torch.Tensor:
        mask = grid[:, 1:2]
        x = grid[:, :self.n_in]
        for conv, norm in zip(self.convs, self.norms):
            x = F.gelu(norm(conv(x, mask)))
            if self.mask_aware:
                x = x * mask.to(x.dtype)      # GroupNorm re-fills invalid sites; zero them again
        return x

    def forward(self, batch) -> torch.Tensor:
        feat = self.stem(batch.grid)                                   # [B, W, T, H, W]
        B = feat.shape[0]
        xyt, valid = batch.tok_xyt, batch.tok_valid                    # [B, E, 3], [B, E]
        bi = torch.arange(B, device=feat.device)[:, None]
        f = feat.permute(0, 2, 3, 4, 1)[bi, xyt[..., 2], xyt[..., 1], xyt[..., 0]]   # [B, E, W]

        parts = [f, self.fourier(xyt).to(f.dtype)]
        if self.n_bfeat:
            parts.append(batch.tok_bfeat.to(f.dtype))
        h = self.tok_proj(torch.cat(parts, dim=-1))                    # [B, E, W]
        h = h * valid.unsqueeze(-1).to(h.dtype)

        special = self.special.unsqueeze(0).expand(B, -1, -1).to(h.dtype)
        x = torch.cat([special, h], dim=1)                             # [B, 1 + n_virtual + E, W]
        nsp = special.shape[1]
        key_pad = torch.cat([torch.zeros(B, nsp, dtype=torch.bool, device=x.device), ~valid], dim=1)

        pair_idx = self.blocks[0].bias.pair_index(xyt)                 # shared by all layers
        for blk in self.blocks:
            x = blk(x, pair_idx, batch.tok_bdist, self.n_virtual, key_pad)
        return self.head(self.out_ln(x[:, 0])).squeeze(-1).float()
