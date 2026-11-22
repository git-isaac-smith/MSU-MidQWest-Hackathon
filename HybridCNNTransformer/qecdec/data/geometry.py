"""Detector coordinates -> grid, stabiliser type/weight, and the static boundary features.

Coordinate convention (Stim rotated surface code, distance d, rounds = d)
-------------------------------------------------------------------------
Data qubits sit at odd-odd (x, y); stabiliser ancillas (and therefore detectors) sit at
even-even (x, y) with 0 <= x, y <= 2d. A detector has coordinates (x, y, t), t = 0..rounds.
We map to integer grid cells

    gx = x / 2,   gy = y / 2,   gt = t,           gx, gy in [0, d],  gt in [0, d]

so the dense grid has shape [T, H, W] = [d+1, d+1, d+1] (axes t, y, x).

Derived stabiliser properties (checked against the circuit in tests/test_geometry.py)
------------------------------------------------------------------------------------
  type   : X if (gx + gy) is odd, else Z.            [= ((x + y) / 2) mod 2]
  weight : 2 if gx in {0, d} or gy in {0, d} (site on the outer ring), else 4.
Z-type weight-2 checks live on the left/right edges (gx in {0, d}); X-type weight-2 checks on the
bottom/top edges (gy in {0, d}).
Time: in a Z-memory experiment the first (t = 0) and last (t = d) detector layers exist only for
Z-type stabilisers, so the validity mask is a function of t as well as (x, y).

Grid channels  [C_IN = 10]
--------------------------
  0       detection event (0/1)                                  (dynamic, per shot)
  1       validity mask m: 1 at real detector sites, else 0      (static)
  2..5    clipped distance to boundary left/right/bottom/top:    (static)
            min(gx, C) / C, min(d - gx, C) / C, min(gy, C) / C, min(d - gy, C) / C
  6       stabiliser type: +1 X, -1 Z
  7       stabiliser weight / 4  (0.5 or 1.0)
  8       first-round flag (gt == 0)
  9       last-round flag  (gt == d)
All static feature channels are multiplied by m, so invalid sites are exactly zero. Everything is
defined from distances to the boundaries, never from absolute positions, so it is valid at any d.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .stim_source import make_circuit

N_BASE = 2                 # event channel + validity-mask channel
N_BFEAT = 8                # static boundary channels
C_IN = N_BASE + N_BFEAT
N_BOUNDARY = 4             # left, right, bottom, top
BOUNDARY_NAMES = ("left", "right", "bottom", "top")


# ---------------------------------------------------------------- pure functions of coordinates
def stabiliser_type(gx, gy):
    """1 for X-type, 0 for Z-type, from grid coordinates (gx = x/2, gy = y/2)."""
    return (np.asarray(gx) + np.asarray(gy)) % 2


def stabiliser_weight(gx, gy, d: int):
    """2 on the outer ring of the (d+1)x(d+1) site grid, else 4."""
    gx, gy = np.asarray(gx), np.asarray(gy)
    ring = (gx == 0) | (gx == d) | (gy == 0) | (gy == d)
    return np.where(ring, 2, 4)


def boundary_distances(d: int, clip: int) -> np.ndarray:
    """int64 [4, H, W]: clipped distance (in stabiliser spacings) to left/right/bottom/top."""
    gx = np.arange(d + 1)[None, :]
    gy = np.arange(d + 1)[:, None]
    full = np.broadcast_to
    dist = np.stack([full(gx, (d + 1, d + 1)), full(d - gx, (d + 1, d + 1)),
                     full(gy, (d + 1, d + 1)), full(d - gy, (d + 1, d + 1))])
    return np.minimum(dist, clip).astype(np.int64)


def site_features(d: int, clip: int, mask: np.ndarray) -> np.ndarray:
    """float32 [N_BFEAT, T, H, W] boundary channels (see module docstring), zeroed where mask == 0."""
    T = mask.shape[0]
    gy, gx = np.meshgrid(np.arange(d + 1), np.arange(d + 1), indexing="ij")     # [H, W]
    bd = boundary_distances(d, clip).astype(np.float32) / clip                  # [4, H, W]
    typ = np.where(stabiliser_type(gx, gy) == 1, 1.0, -1.0).astype(np.float32)
    wt = (stabiliser_weight(gx, gy, d) / 4.0).astype(np.float32)
    feats = np.zeros((N_BFEAT, T, d + 1, d + 1), np.float32)
    feats[0:4] = bd[:, None]
    feats[4] = typ[None]
    feats[5] = wt[None]
    feats[6, 0] = 1.0
    feats[7, T - 1] = 1.0
    return feats * mask[None].astype(np.float32)


@dataclass
class NumpyGeometry:
    d: int
    T: int
    H: int
    W: int
    num_detectors: int
    det_xyt: np.ndarray       # int64 [ND, 3]  (gx, gy, gt)
    det_stab_type: np.ndarray  # int64 [ND]    1 = X, 0 = Z
    det_weight: np.ndarray    # int64 [ND]
    static: np.ndarray        # float32 [1 + N_BFEAT, T, H, W]  (mask, boundary channels)
    det_bdist: np.ndarray     # int64 [ND, 4] clipped boundary distances (integers)


def build_numpy_geometry(d: int, boundary_clip: int, rounds: Optional[int] = None) -> NumpyGeometry:
    rounds = d if rounds is None else rounds
    circuit = make_circuit(d, 0.0, rounds)
    coords = circuit.get_detector_coordinates()
    nd = circuit.num_detectors
    arr = np.array([coords[i] for i in range(nd)], dtype=np.float64)           # [ND, 3]
    if arr.shape[1] != 3:
        raise ValueError(f"expected (x, y, t) detector coordinates, got {arr.shape[1]} values")
    if not (np.all(arr[:, :2] % 2 == 0) and np.all(arr[:, 2] == np.round(arr[:, 2]))):
        raise ValueError("unexpected detector coordinates: x, y must be even and t integer")
    gx, gy, gt = (arr[:, 0] // 2).astype(np.int64), (arr[:, 1] // 2).astype(np.int64), arr[:, 2].astype(np.int64)
    T, H, W = rounds + 1, d + 1, d + 1
    if gx.max() > d or gy.max() > d or gt.max() > rounds or min(gx.min(), gy.min(), gt.min()) < 0:
        raise ValueError("detector coordinates fall outside the expected grid")
    mask = np.zeros((T, H, W), np.float32)
    mask[gt, gy, gx] = 1.0
    if mask.sum() != nd:
        raise ValueError("two detectors map to the same grid cell")
    feats = site_features(d, boundary_clip, mask)
    static = np.concatenate([mask[None], feats], axis=0)
    bd = boundary_distances(d, boundary_clip)[:, gy, gx].T.copy()               # [ND, 4]
    return NumpyGeometry(d, T, H, W, nd, np.stack([gx, gy, gt], 1),
                         stabiliser_type(gx, gy).astype(np.int64),
                         stabiliser_weight(gx, gy, d).astype(np.int64), static, bd)


# ---------------------------------------------------------------- torch-side cache
@dataclass
class Geometry:
    d: int
    T: int
    H: int
    W: int
    num_detectors: int
    static: torch.Tensor      # float [1 + N_BFEAT, T, H, W]
    det_xyt: torch.Tensor     # long  [ND, 3]  (gx, gy, gt)
    det_flat: torch.Tensor    # long  [ND]     flat index into T*H*W
    det_bfeat: torch.Tensor   # float [ND, N_BFEAT]
    det_bdist: torch.Tensor   # long  [ND, 4]


class GeometryCache:
    """Per-distance geometry on `device`, built lazily: geom = cache(d)."""

    def __init__(self, boundary_clip: int, device):
        self.clip, self.device = boundary_clip, torch.device(device)
        self._cache: Dict[int, Geometry] = {}

    def __call__(self, d: int) -> Geometry:
        if d not in self._cache:
            g = build_numpy_geometry(d, self.clip)
            dev = self.device
            xyt = torch.from_numpy(g.det_xyt).to(dev)
            flat = (xyt[:, 2] * g.H + xyt[:, 1]) * g.W + xyt[:, 0]
            static = torch.from_numpy(g.static).to(dev)
            bfeat = static[1:, xyt[:, 2], xyt[:, 1], xyt[:, 0]].T.contiguous()
            self._cache[d] = Geometry(d, g.T, g.H, g.W, g.num_detectors, static, xyt, flat, bfeat,
                                      torch.from_numpy(g.det_bdist).to(dev))
        return self._cache[d]
