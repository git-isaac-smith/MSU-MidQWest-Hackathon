"""Shared test helpers."""
import numpy as np

from qecdec.config import Config
from qecdec.data import GeometryCache, ShotSource, build_batch


def make_batch(d=3, p=0.01, n=8, seed=0, clip=3, device="cpu", channels_last=False):
    det, obs = ShotSource(d, p, seed=seed).sample(n)
    geom = GeometryCache(clip, device)(d)
    return build_batch(det, obs, geom, device, channels_last), det, obs


def tiny_cfg(**kw) -> Config:
    base = dict(width=16, stem_layers=2, attn_layers=2, heads=2, bf16=False, channels_last=False,
                d_set=[3, 5], bias_hidden=8)
    base.update(kw)
    cfg = Config(**base)
    cfg.validate()
    return cfg
