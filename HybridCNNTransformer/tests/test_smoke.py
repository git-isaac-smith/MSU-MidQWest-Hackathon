"""d=3 end-to-end smoke test: train a small hybrid on fresh Stim shots and check that it learns.

Random guessing has LER 0.5, but the meaningful bar is the trivial "never flip" decoder (LER equal to
the observable flip rate, ~0.10 at p = 0.005, d = 3). The trained model must get below half of it.
About 1 minute on a laptop CPU for 1500 steps; faster on a GPU.
"""
import os

import numpy as np
import pytest

from qecdec.config import load_config
from qecdec.data import ShotSource
from train import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.smoke
def test_d3_hybrid_beats_random_and_trivial(tmp_path):
    cfg = load_config(os.path.join(ROOT, "configs", "smoke_d3.yaml"), [f"out_dir={tmp_path.as_posix()}"])
    res = run(cfg)
    val = res["last_val"][3]
    _, obs = ShotSource(3, cfg.p, seed=777_000_000 + 3).sample(cfg.val_shots)   # the validation shots
    trivial = float(np.mean(obs))
    assert val < 0.5, "no better than random guessing"
    assert val < 0.5 * trivial, f"val LER {val:.4f} not clearly below trivial decoder {trivial:.4f}"
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
