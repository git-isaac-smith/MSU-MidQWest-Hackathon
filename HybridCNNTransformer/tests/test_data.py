"""build_batch: shapes, token/grid consistency, zero-event shots."""
import numpy as np
import torch

from qecdec.data import C_IN, GeometryCache, ShotSource, build_batch, build_mwpm, mwpm_predict
from helpers import make_batch


def test_batch_shapes_and_consistency():
    d, n = 5, 16
    b, det, obs = make_batch(d=d, p=0.01, n=n)
    T = H = W = d + 1
    E = int(det.sum(1).max())
    assert b.grid.shape == (n, C_IN, T, H, W)
    assert b.labels.shape == (n,) and b.tok_xyt.shape == (n, E, 3)
    assert b.tok_valid.shape == (n, E) and b.tok_bfeat.shape == (n, E, 8) and b.tok_bdist.shape == (n, E, 4)
    assert (b.tok_valid.sum(1).numpy() == det.sum(1)).all()
    assert (b.labels.numpy() == obs).all()
    # event channel carries exactly the fired detectors, and tokens point at those cells
    assert b.grid[:, 0].sum() == det.sum()
    for i in range(n):
        v = b.tok_valid[i]
        x, y, t = b.tok_xyt[i, v].T
        assert (b.grid[i, 0, t, y, x] == 1).all()
        assert (b.grid[i, 1, t, y, x] == 1).all()                  # events only at valid sites
    assert (b.grid[:, 0] * (1 - b.grid[:, 1])).sum() == 0


def test_channels_last_layout():
    b, _, _ = make_batch(d=3, channels_last=True)
    assert b.grid.is_contiguous(memory_format=torch.channels_last_3d)


def test_zero_event_batch():
    d = 3
    geom = GeometryCache(3, "cpu")(d)
    det = np.zeros((4, geom.num_detectors), bool)
    b = build_batch(det, np.zeros(4, bool), geom, "cpu", False)
    assert b.tok_xyt.shape == (4, 0, 3) and b.tok_valid.shape == (4, 0)


def test_shot_source_is_seeded_and_matches_mwpm_shapes():
    a = ShotSource(3, 0.01, seed=1).sample(64)
    b = ShotSource(3, 0.01, seed=1).sample(64)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    src = ShotSource(3, 0.005, seed=2)
    det, obs = src.sample(2000)
    pred = mwpm_predict(build_mwpm(src.circuit), det)
    assert pred.shape == obs.shape and pred.dtype == bool
    assert (pred != obs).mean() < obs.mean() + 1e-9               # MWPM no worse than "never flip"
