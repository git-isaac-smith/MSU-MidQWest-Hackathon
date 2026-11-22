"""Plain dense 3D CNN baseline: dilated residual convs -> masked global pooling -> logit."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.geometry import C_IN, N_BASE
from .maskconv import MaskedConv3d


def receptive_field(dilations: Sequence[int], k: int = 3) -> int:
    """Receptive field (per axis, in grid cells) of the input conv (dilation 1) plus the dilated
    blocks:  RF = 1 + (k - 1) * (1 + sum(dilations))."""
    return 1 + (k - 1) * (1 + sum(dilations))


class _Block(nn.Module):
    def __init__(self, w: int, dilation: int, mask_aware: bool):
        super().__init__()
        self.conv = MaskedConv3d(w, w, k=3, dilation=dilation, mask_aware=mask_aware)
        self.norm = nn.GroupNorm(8, w)
        self.mask_aware = mask_aware

    def forward(self, x, mask):
        y = F.gelu(self.norm(self.conv(x, mask)))
        if self.mask_aware:
            y = y * mask.to(y.dtype)   # GroupNorm re-fills invalid sites; zero them again
        return x + y


class CNNBaseline(nn.Module):
    def __init__(self, cfg: Config, max_d: int = 9):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.n_in = C_IN if cfg.use_boundary_channels else N_BASE
        self.rf = receptive_field(cfg.dilations)
        assert self.rf >= max_d, (
            f"receptive field {self.rf} < d={max_d}; lengthen cfg.dilations (currently {cfg.dilations})")
        self.inp = MaskedConv3d(self.n_in, w, k=3, dilation=1, mask_aware=cfg.mask_aware_conv)
        self.blocks = nn.ModuleList(_Block(w, dil, cfg.mask_aware_conv) for dil in cfg.dilations)
        self.head = nn.Sequential(nn.Linear(2 * w, w), nn.GELU(), nn.Linear(w, 1))

    def forward(self, batch) -> torch.Tensor:
        grid = batch.grid
        mask = grid[:, 1:2]
        x = F.gelu(self.inp(grid[:, :self.n_in], mask))
        for blk in self.blocks:
            x = blk(x, mask)
        m = mask.to(x.dtype)
        mean = (x * m).sum((2, 3, 4)) / m.sum((2, 3, 4)).clamp(min=1.0)           # masked mean pool
        mx = x.masked_fill(m == 0, float("-inf")).amax((2, 3, 4))                  # masked max pool
        return self.head(torch.cat([mean, mx], dim=1)).squeeze(-1).float()
