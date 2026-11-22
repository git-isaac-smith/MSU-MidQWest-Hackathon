"""Shared helpers for train.py / eval.py / benchmark.py / tests."""
from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .data import GeometryCache, ShotSource, build_batch


def autocast_ctx(device: torch.device, enabled: bool):
    """bf16 autocast on CUDA or CPU; a no-op context when disabled."""
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


class TrainData:
    """Infinite stream of fresh batches; the distance of each batch is drawn from cfg.d_set."""

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg, self.device = cfg, device
        self.rng = np.random.default_rng(cfg.seed)
        self.geom = GeometryCache(cfg.boundary_clip, device)
        self.sources: Dict[int, ShotSource] = {}

    def next_batch(self, d: int = None):
        d = int(self.rng.choice(self.cfg.d_set)) if d is None else d
        if d not in self.sources:
            self.sources[d] = ShotSource(d, self.cfg.p, seed=self.cfg.seed * 1_000_003 + d)
        det, obs = self.sources[d].sample(self.cfg.batch_size)
        return build_batch(det, obs, self.geom(d), self.device, self.cfg.channels_last)


def make_val_sets(cfg: Config, p: float = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Fixed validation shots per distance (own seeds, disjoint from the training streams)."""
    p = cfg.p if p is None else p
    out = {}
    for d in cfg.d_set:
        src = ShotSource(d, p, seed=777_000_000 + d)
        out[d] = src.sample(cfg.val_shots)
    return out


@torch.no_grad()
def neural_predict(model: nn.Module, det: np.ndarray, obs: np.ndarray, d: int, geom: GeometryCache,
                   device: torch.device, batch_size: int, bf16: bool, channels_last: bool) -> np.ndarray:
    """Return the model's predicted observable flips (bool [N])."""
    model.eval()
    preds = []
    for i in range(0, len(det), batch_size):
        b = build_batch(det[i:i + batch_size], obs[i:i + batch_size], geom(d), device, channels_last)
        with autocast_ctx(device, bf16):
            logits = model(b)
        preds.append((logits > 0).cpu().numpy())
    return np.concatenate(preds)


def make_optimizer(model: nn.Module, cfg: Config):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.99))


def make_scheduler(opt, cfg: Config):
    """Linear warm-up then cosine decay to min_lr_frac * lr."""
    def f(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_step(model, opt, sched, batch, cfg: Config, device: torch.device):
    """One optimisation step. Returns (loss, batch logical error rate) as detached GPU tensors."""
    with autocast_ctx(device, cfg.bf16):
        logits = model(batch)
    loss = F.binary_cross_entropy_with_logits(logits.float(), batch.labels)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    sched.step()
    ler = ((logits.detach() > 0).float() != batch.labels).float().mean()
    return loss.detach(), ler
