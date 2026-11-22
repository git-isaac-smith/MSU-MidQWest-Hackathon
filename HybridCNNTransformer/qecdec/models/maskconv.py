"""Mask-aware 3D convolution.

Plain zero padding silently treats missing sites (outside the code, or X-type sites in the
first/last round) as "no detection event". Here we instead renormalise by the number of
valid taps:

    y(v) = bias + (K^3 / n(v)) * sum_{u in N(v)} m(u) * W[u - v] * x(u),
    n(v) = sum_{u in N(v)} m(u),   K = kernel size,

so a site on the boundary (few valid neighbours) produces an output on the same scale as an
interior site, and the layer output is multiplied by m(v), keeping invalid sites exactly zero.
No reflect padding anywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv3d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, dilation: int = 1, mask_aware: bool = True):
        super().__init__()
        self.k, self.dilation, self.mask_aware = k, dilation, mask_aware
        self.padding = dilation * (k // 2)
        self.weight = nn.Parameter(torch.empty(cout, cin, k, k, k))
        self.bias = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.register_buffer("ones", torch.ones(1, 1, k, k, k), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W]; mask: [B or 1, 1, T, H, W] float in {0, 1}."""
        if not self.mask_aware:
            return F.conv3d(x, self.weight.to(x.dtype), self.bias.to(x.dtype),
                            padding=self.padding, dilation=self.dilation)
        y = F.conv3d(x * mask.to(x.dtype), self.weight.to(x.dtype), None,
                     padding=self.padding, dilation=self.dilation)
        with torch.no_grad():
            n = F.conv3d(mask.float(), self.ones, padding=self.padding, dilation=self.dilation)
            scale = (self.k ** 3) / n.clamp(min=1.0)
        y = y * scale.to(y.dtype) + self.bias.view(1, -1, 1, 1, 1).to(y.dtype)
        return y * mask.to(y.dtype)
