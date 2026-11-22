"""Seeds, device selection and the statistics used in the eval tables."""
from __future__ import annotations

import math
import os
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def new_run_dir(root: str) -> str:
    """Create <root>/<timestamp>; if that exists (same second) append _1, _2, ... Never reuses a folder."""
    os.makedirs(root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    i = 0
    while True:
        path = os.path.join(root, stamp if i == 0 else f"{stamp}_{i}")
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            i += 1


# ---------------- statistics ----------------
def eps_round(p_shot: float, rounds: int) -> float:
    """Per-round logical error rate.

    P_L = (1 - (1 - 2 eps)^rounds) / 2   =>   eps = (1 - (1 - 2 P_L)^(1/rounds)) / 2.
    P_L >= 0.5 means the decoder is no better than a coin: return 0.5.
    """
    if p_shot >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * p_shot) ** (1.0 / rounds))


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (well behaved at k = 0)."""
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    denom = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / denom
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def shot_and_round_rates(k: int, n: int, rounds: int):
    """Return (P, P_err, eps, eps_err); err = half-width of the 95% Wilson interval.

    The interval endpoints are pushed through the monotone map P -> eps, so the
    per-round error bar is consistent with the per-shot one.
    """
    p = k / n
    lo, hi = wilson_interval(k, n)
    e, elo, ehi = eps_round(p, rounds), eps_round(lo, rounds), eps_round(hi, rounds)
    return p, 0.5 * (hi - lo), e, 0.5 * (ehi - elo)
