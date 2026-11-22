"""Turn raw Stim shots into model inputs: a dense grid plus padded event tokens."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import Geometry


@dataclass
class Batch:
    grid: torch.Tensor        # float [B, C_IN, T, H, W]   (channel 0 events, 1 mask, 2.. boundary)
    labels: torch.Tensor      # float [B]                  observable flip
    tok_xyt: torch.Tensor     # long  [B, E, 3]            (gx, gy, gt) of fired detectors, padded
    tok_valid: torch.Tensor   # bool  [B, E]               False = padding
    tok_bfeat: torch.Tensor   # float [B, E, N_BFEAT]      static boundary features at the token
    tok_bdist: torch.Tensor   # long  [B, E, 4]            clipped distance to each boundary
    d: int


def build_batch(det: np.ndarray, obs: np.ndarray, geom: Geometry, device,
                channels_last: bool = True) -> Batch:
    """det bool [B, ND], obs bool [B] -> Batch. E = max events in the batch (0 is allowed).

    Tokens are the fired detectors of each shot in detector-index order, right-padded.
    Everything after the host->device copy runs on `device` with no Python loops.
    """
    dev = torch.device(device)
    det_t = torch.from_numpy(np.ascontiguousarray(det)).to(dev)
    B = det_t.shape[0]
    T, H, W = geom.T, geom.H, geom.W

    ev = torch.zeros(B, T * H * W, device=dev)
    ev[:, geom.det_flat] = det_t.float()
    grid = torch.cat([ev.view(B, 1, T, H, W),
                      geom.static.unsqueeze(0).expand(B, -1, -1, -1, -1)], dim=1)
    if channels_last:
        grid = grid.contiguous(memory_format=torch.channels_last_3d)

    counts = det_t.sum(1)                                           # [B]
    E = int(counts.max().item())
    b_idx, k_idx = det_t.nonzero(as_tuple=True)                     # row-major: sorted by shot
    starts = counts.cumsum(0) - counts
    rank = torch.arange(b_idx.numel(), device=dev) - starts[b_idx]  # position of each event in its shot
    tok_det = torch.zeros(B, E, dtype=torch.long, device=dev)
    tok_det[b_idx, rank] = k_idx
    valid = torch.zeros(B, E, dtype=torch.bool, device=dev)
    valid[b_idx, rank] = True

    labels = torch.from_numpy(np.ascontiguousarray(obs).astype(np.float32)).to(dev)
    return Batch(grid=grid, labels=labels, tok_xyt=geom.det_xyt[tok_det], tok_valid=valid,
                 tok_bfeat=geom.det_bfeat[tok_det], tok_bdist=geom.det_bdist[tok_det], d=geom.d)
