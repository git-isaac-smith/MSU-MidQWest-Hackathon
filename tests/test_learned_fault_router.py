import numpy as np
import pytest

from bascd.circuits import memory_circuit
from bascd.learned_fault_router import (
    LearnedFaultRoutedDecoder,
    LearnedFaultRouter,
    _compile_projections,
    _fields,
)


@pytest.mark.parametrize(("basis", "parity"), [("x", False), ("z", True)])
def test_frozen_policy_matches_scores_and_matching_branches(tmp_path, basis, parity):
    circuit = memory_circuit(basis, 3, 3, .005)
    model = circuit.detector_error_model(decompose_errors=True)
    policy = LearnedFaultRouter.fit_sampled(
        circuit, model, seeds=[9041090], shots_per_seed=4096, parity=parity)
    path = tmp_path / "policy.npz"
    policy.save(path)
    loaded = LearnedFaultRouter.load(path)

    packed = circuit.compile_detector_sampler(seed=9041091).sample(2048, bit_packed=True)
    random = np.random.default_rng(9041092).integers(0, 256, size=(256, packed.shape[1]), dtype=np.uint8)
    packed = np.concatenate([packed, random])
    score = policy.score_batch(packed)
    route = loaded.route_batch(packed)
    assert np.array_equal(loaded.score_batch(packed), score)
    assert np.array_equal(route, score > policy.cutoff)

    decoder = LearnedFaultRoutedDecoder(model, loaded)
    options = dict(bit_packed_shots=True, bit_packed_predictions=True)
    ordinary = decoder.ordinary.decode_batch(packed & decoder.mask, **options)
    correlated = decoder.correlated.decode_batch(packed, enable_correlations=True, **options)
    assert np.array_equal(decoder.decode_batch(packed), np.where(route[:, None], correlated, ordinary))
    assert decoder.decode_batch(packed[:0]).shape == (0, 1)

    other = memory_circuit(basis, 3, 3, .006).detector_error_model(decompose_errors=True)
    with pytest.raises(ValueError, match="does not match"):
        LearnedFaultRoutedDecoder(other, loaded)


def test_logical_lane_projection_preserves_time_and_parity():
    circuit = memory_circuit("z", 5, 5, .005)
    model = circuit.detector_error_model(decompose_errors=True)
    projections = _compile_projections(circuit, model, temporal=True, parity=True)
    by_coordinate = {
        tuple(coordinate): index for index, coordinate in circuit.get_detector_coordinates().items()
    }
    rows = np.zeros((3, circuit.num_detectors), dtype=np.uint8)
    rows[0, [by_coordinate[(2, 2, 1)], by_coordinate[(0, 4, 2)]]] = 1
    rows[1, [by_coordinate[(2, 2, 1)], by_coordinate[(0, 4, 4)]]] = 1
    rows[2, [by_coordinate[(2, 2, 1)], by_coordinate[(2, 2, 2)]]] = 1
    relevant, other, occupancy, adjacent, parity = _fields(
        np.packbits(rows, bitorder="little", axis=1), *projections)
    assert relevant.tolist() == [2, 2, 2]
    assert other.tolist() == [0, 0, 0]
    assert occupancy.tolist() == [3, 3, 1]
    assert adjacent.tolist() == [2, 1, 1]
    assert parity.tolist() == [1, 1, 0]


@pytest.mark.parametrize("fraction", [0, 1])
def test_explicit_routing_budget_endpoints(fraction):
    circuit = memory_circuit("z", 3, 3, .005)
    model = circuit.detector_error_model(decompose_errors=True)
    packed = np.zeros((16, 3), dtype=np.uint8)
    bits = np.zeros(16, dtype=np.uint8)
    policy = LearnedFaultRouter.fit(circuit, model, packed, bits, bits, bits, fraction=fraction)
    assert np.all(policy.route_batch(packed) == bool(fraction))
