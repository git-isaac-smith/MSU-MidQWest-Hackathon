"""Detector-coordinate mapping, stabiliser type/weight derivation, X/Z balance, boundary channels."""
import numpy as np
import pytest

from qecdec.data import (C_IN, GeometryCache, build_numpy_geometry, make_circuit, stabiliser_type,
                         stabiliser_weight)
from qecdec.data.geometry import boundary_distances

DS = [3, 5, 7, 9]


@pytest.mark.parametrize("d", DS)
def test_circuit_basics(d):
    c = make_circuit(d, 0.001)
    assert c.num_observables == 1
    # (d+1) layers: first/last hold (d^2-1)/2 Z checks, the d-1 middle ones hold d^2-1 checks
    assert c.num_detectors == (d * d - 1) * d


@pytest.mark.parametrize("d", DS)
def test_detector_coordinates_map_to_unique_grid_cells(d):
    c = make_circuit(d, 0.0)
    coords = c.get_detector_coordinates()
    assert len(coords) == c.num_detectors
    g = build_numpy_geometry(d, boundary_clip=3)
    assert (g.T, g.H, g.W) == (d + 1, d + 1, d + 1)
    for i, (x, y, t) in coords.items():
        assert x % 2 == 0 and y % 2 == 0 and t == int(t)
        assert tuple(g.det_xyt[i]) == (int(x // 2), int(y // 2), int(t))
    assert len({tuple(r) for r in g.det_xyt}) == c.num_detectors
    assert g.static[0].sum() == c.num_detectors                     # mask marks exactly the detectors
    # Z-memory: first and last layers hold only Z checks, middle layers hold all d^2-1 checks.
    per_layer = g.static[0].sum((1, 2))
    assert per_layer[0] == per_layer[-1] == (d * d - 1) // 2
    assert np.all(per_layer[1:-1] == d * d - 1)


@pytest.mark.parametrize("d", DS)
def test_type_and_weight_match_the_circuit(d):
    """Ground truth from the gate structure: weight = CX count per round on the ancilla,
    X-type <=> the ancilla is hit by Hadamards."""
    c = make_circuit(d, 0.0)
    qcoords = c.get_final_qubit_coordinates()
    ancillas = {q: xy for q, xy in qcoords.items() if xy[0] % 2 == 0 and xy[1] % 2 == 0}
    cx_count = {q: 0 for q in ancillas}
    h_count = {q: 0 for q in ancillas}
    for ins in c.flattened():
        tg = [t.value for t in ins.targets_copy()]
        if ins.name == "CX":
            for q in tg:
                if q in cx_count:
                    cx_count[q] += 1
        elif ins.name == "H":
            for q in tg:
                if q in h_count:
                    h_count[q] += 1
    rounds = d
    g = build_numpy_geometry(d, 3)
    seen = set()
    for q, (x, y) in ancillas.items():
        gx, gy = int(x // 2), int(y // 2)
        weight = cx_count[q] / rounds
        is_x = h_count[q] > 0
        # only checks that actually own a detector are part of the code (ancilla coords may also
        # include none other); compare to the derivation
        assert weight == stabiliser_weight(gx, gy, d), (q, x, y)
        assert int(is_x) == int(stabiliser_type(gx, gy)), (q, x, y)
        seen.add((gx, gy))
    assert len(ancillas) == d * d - 1
    # the per-detector arrays agree with the circuit-derived values
    for (gx, gy, gt), typ, wt in zip(g.det_xyt, g.det_stab_type, g.det_weight):
        assert (gx, gy) in seen
        assert typ == stabiliser_type(gx, gy) and wt == stabiliser_weight(gx, gy, d)


@pytest.mark.parametrize("d", DS)
def test_xz_balance_and_weights(d):
    g = build_numpy_geometry(d, 3)
    mid = (g.det_xyt[:, 2] > 0) & (g.det_xyt[:, 2] < d)
    for layer in range(1, d):
        sel = g.det_xyt[:, 2] == layer
        assert (g.det_stab_type[sel] == 1).sum() == (g.det_stab_type[sel] == 0).sum() == (d * d - 1) // 2
        assert (g.det_weight[sel] == 2).sum() == 2 * (d - 1)
        assert (g.det_weight[sel] == 4).sum() == (d - 1) ** 2
        # weight-2 Z checks sit on left/right edges, weight-2 X checks on bottom/top edges
        x, y = g.det_xyt[sel, 0], g.det_xyt[sel, 1]
        z2 = (g.det_weight[sel] == 2) & (g.det_stab_type[sel] == 0)
        x2 = (g.det_weight[sel] == 2) & (g.det_stab_type[sel] == 1)
        assert np.all((x[z2] == 0) | (x[z2] == d)) and np.all((y[x2] == 0) | (y[x2] == d))
        assert z2.sum() == x2.sum() == d - 1
    assert (g.det_stab_type[g.det_xyt[:, 2] == 0] == 0).all()      # first layer: Z only
    assert mid.sum() == (d - 1) * (d * d - 1)


@pytest.mark.parametrize("d", [3, 5, 9])
def test_boundary_channels(d):
    clip = 3
    g = build_numpy_geometry(d, clip)
    s = g.static
    assert s.shape == (C_IN - 1, d + 1, d + 1, d + 1)              # mask + 8 boundary channels
    mask = s[0]
    # invalid sites are exactly zero in every channel
    assert np.all(s[1:][:, mask == 0] == 0)
    for (gx, gy, gt) in g.det_xyt:
        v = s[1:, gt, gy, gx]
        assert v[0] == pytest.approx(min(gx, clip) / clip)          # left
        assert v[1] == pytest.approx(min(d - gx, clip) / clip)      # right
        assert v[2] == pytest.approx(min(gy, clip) / clip)          # bottom
        assert v[3] == pytest.approx(min(d - gy, clip) / clip)      # top
        assert v[4] == (1.0 if (gx + gy) % 2 else -1.0)             # type
        ring = gx in (0, d) or gy in (0, d)
        assert v[5] == (0.5 if ring else 1.0)                       # weight / 4
        assert v[6] == float(gt == 0) and v[7] == float(gt == d)    # first / last round
    # defined by distance: left/right and bottom/top channels are mirror images at every d
    bd = boundary_distances(d, clip)
    assert np.array_equal(bd[0], bd[1][:, ::-1]) and np.array_equal(bd[2], bd[3][::-1, :])
    assert bd.max() == min(d, clip) and bd.min() == 0


@pytest.mark.parametrize("d", [3, 7])
def test_token_features_match_grid(d):
    geom = GeometryCache(3, "cpu")(d)
    xyt = geom.det_xyt
    grid_vals = geom.static[1:, xyt[:, 2], xyt[:, 1], xyt[:, 0]].T
    assert (grid_vals == geom.det_bfeat).all()
    gx, gy = xyt[:, 0], xyt[:, 1]
    assert (geom.det_bdist[:, 0] == gx.clamp(max=3)).all()
    assert (geom.det_bdist[:, 3] == (d - gy).clamp(max=3)).all()
    assert (geom.det_flat == (xyt[:, 2] * (d + 1) + xyt[:, 1]) * (d + 1) + xyt[:, 0]).all()
