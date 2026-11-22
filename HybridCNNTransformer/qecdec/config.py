"""Config dataclass. One flat config drives model, data, training and eval.

Load from YAML, then optionally override with ``key=value`` strings
(values are parsed as YAML, so ``d_set=[3,5]`` and ``compile=true`` work).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Config:
    # ---------------- model ----------------
    model: str = "hybrid"                 # "hybrid" | "cnn_baseline"
    use_boundary_channels: bool = True    # ablation flag: 8 boundary input channels (+ token boundary feats)
    use_boundary_tokens: bool = True      # ablation flag: 4 learned virtual boundary tokens
    mask_aware_conv: bool = True          # ablation flag: renormalised conv vs plain zero-padded conv
    width: int = 64
    stem_layers: int = 2
    attn_layers: int = 3
    heads: int = 4
    mlp_ratio: int = 2
    n_fourier: int = 6                    # Fourier frequencies per coordinate
    boundary_clip: int = 3                # clip for boundary distances (stabiliser-spacing units)
    rel_clip: int = 4                     # clip for relative (dx,dy,dt) offsets in the attention bias
    bias_hidden: int = 32                 # hidden width of the relative-bias MLP
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 1])  # CNN baseline schedule

    # ---------------- data / noise ----------------
    p: float = 0.005                      # physical error rate used for training
    d_set: List[int] = field(default_factory=lambda: [3, 5, 7])  # rounds = d for every d

    # ---------------- training ----------------
    batch_size: int = 256
    steps: int = 20000
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    min_lr_frac: float = 0.02
    grad_clip: float = 1.0
    bf16: bool = True
    channels_last: bool = True
    compile: bool = False
    seed: int = 0
    log_every: int = 100
    val_every: int = 1000
    val_shots: int = 8192                 # per distance, fixed seed
    out_dir: str = "runs/hybrid"
    device: str = "auto"                  # "auto" | "cpu" | "cuda"

    def validate(self) -> None:
        assert self.model in ("hybrid", "cnn_baseline"), self.model
        assert all(d % 2 == 1 and d >= 3 for d in self.d_set), "only odd d >= 3 are supported"
        assert self.width % self.heads == 0, "width must be divisible by heads"
        assert self.stem_layers >= 1 and self.attn_layers >= 1


def from_dict(d: Dict[str, Any]) -> Config:
    names = {f.name for f in dataclasses.fields(Config)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    cfg = Config(**d)
    cfg.validate()
    return cfg


def load_config(path: Optional[str], overrides: Optional[List[str]] = None) -> Config:
    raw: Dict[str, Any] = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for item in overrides or []:
        k, v = item.split("=", 1)
        raw[k.strip()] = yaml.safe_load(v)
    return from_dict(raw)


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    return asdict(cfg)
