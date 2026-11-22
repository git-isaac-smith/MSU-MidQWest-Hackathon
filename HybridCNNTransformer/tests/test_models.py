"""Model shapes, flags, mask-aware conv, zero-event shots, key-padding invariance, d generalisation."""
import numpy as np
import pytest
import torch

from qecdec.data import GeometryCache, build_batch
from qecdec.models import build_model
from qecdec.models.cnn_baseline import receptive_field
from qecdec.models.maskconv import MaskedConv3d
from helpers import make_batch, tiny_cfg

FLAGS = [dict(), dict(use_boundary_channels=False), dict(use_boundary_tokens=False),
         dict(mask_aware_conv=False),
         dict(use_boundary_channels=False, use_boundary_tokens=False, mask_aware_conv=False)]


@pytest.mark.parametrize("model", ["hybrid", "cnn_baseline"])
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("d", [3, 5, 9])
def test_forward_shape_and_backward(model, flags, d):
    cfg = tiny_cfg(model=model, **flags)
    net = build_model(cfg, max_d=9)
    b, _, _ = make_batch(d=d, n=6)
    out = net(b)
    assert out.shape == (6,) and out.dtype == torch.float32 and torch.isfinite(out).all()
    torch.nn.functional.binary_cross_entropy_with_logits(out, b.labels).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("model", ["hybrid", "cnn_baseline"])
@pytest.mark.parametrize("tokens", [True, False])
def test_zero_event_shots(model, tokens):
    cfg = tiny_cfg(model=model, use_boundary_tokens=tokens)
    net = build_model(cfg).eval()
    d = 5
    geom = GeometryCache(cfg.boundary_clip, "cpu")(d)
    # (a) whole batch without events (E = 0)
    zeros = np.zeros((3, geom.num_detectors), bool)
    out = net(build_batch(zeros, np.zeros(3, bool), geom, "cpu", False))
    assert out.shape == (3,) and torch.isfinite(out).all()
    # (b) one empty shot inside a batch with events: same logit as when run alone
    b, det, obs = make_batch(d=d, p=0.02, n=5, seed=3)
    det[2] = False
    both = net(build_batch(det, obs, geom, "cpu", False))
    alone = net(build_batch(det[2:3], obs[2:3], geom, "cpu", False))
    assert torch.allclose(both[2], alone[0], atol=1e-4)


def test_padding_does_not_change_logits():
    cfg = tiny_cfg()
    net = build_model(cfg).eval()
    d = 5
    b, det, obs = make_batch(d=d, p=0.02, n=8, seed=5)
    assert len(set(det.sum(1))) > 1, "need shots with different event counts"
    geom = GeometryCache(cfg.boundary_clip, "cpu")(d)
    with torch.no_grad():
        full = net(b)
        for i in range(len(det)):
            one = net(build_batch(det[i:i + 1], obs[i:i + 1], geom, "cpu", False))
            assert torch.allclose(full[i], one[0], atol=1e-4), i


def test_one_model_runs_at_every_distance():
    net = build_model(tiny_cfg(), max_d=9).eval()          # parameters are independent of d
    for d in (3, 5, 7, 9, 11):
        b, _, _ = make_batch(d=d, n=2)
        assert net(b).shape == (2,)


def test_virtual_tokens_change_sequence_length():
    on = build_model(tiny_cfg(use_boundary_tokens=True))
    off = build_model(tiny_cfg(use_boundary_tokens=False))
    assert on.special.shape[0] == 5 and off.special.shape[0] == 1


def test_masked_conv_renormalises_boundary():
    conv = MaskedConv3d(1, 1, k=3, mask_aware=True)
    with torch.no_grad():
        conv.weight.fill_(1.0)
        conv.bias.zero_()
    mask = torch.zeros(1, 1, 5, 5, 5)
    mask[..., 1:4, 1:4, 1:4] = 1
    y = conv(mask.clone(), mask)
    assert torch.allclose(y[mask == 1], torch.full_like(y[mask == 1], 27.0))   # corner == interior
    assert (y[mask == 0] == 0).all()
    plain = MaskedConv3d(1, 1, k=3, mask_aware=False)
    with torch.no_grad():
        plain.weight.fill_(1.0)
    assert plain(mask.clone(), mask)[0, 0, 1, 1, 1] == 8.0                     # zero padding: only 8 taps


def test_cnn_receptive_field_assertion():
    assert receptive_field([1, 2, 4, 1]) == 19
    with pytest.raises(AssertionError):
        build_model(tiny_cfg(model="cnn_baseline", dilations=[1]), max_d=9)
