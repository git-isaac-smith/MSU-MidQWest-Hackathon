import numpy as np
import pymatching
import pytest

from bascd.circuits import memory_circuit
from bascd.learned_fault_router import LearnedFaultRoutedDecoder, LearnedFaultRouter, _compile_projections, _model_digest
from bascd.scalar_routing import ScalarLearnedDecoder, ScalarMatchingDecoder


@pytest.mark.parametrize(("distance", "temporal", "parity"), [(5, True, True), (7, True, False), (3, False, True)])
def test_scalar_routes_and_predictions_match_frozen_batch(tmp_path, distance, temporal, parity):
    circuit = memory_circuit("z", distance, distance, .005)
    model = circuit.detector_error_model(decompose_errors=True)
    activity, lanes, parities, lane_count, time_count = _compile_projections(circuit, model, temporal, parity)
    shape = (24, 24, 1 << lane_count, lane_count + 1 if temporal else 1,
             (lane_count + 1) // 2 + 1 if parity else 1)
    rng = np.random.default_rng(9270001)
    scores = rng.integers(-1, 2, size=shape, dtype=np.int8).astype(np.float32)
    # These count ranges exercise the cheap negative and positive decisions.
    scores[:3] = -1
    scores[10:] = 1
    policy = LearnedFaultRouter(_model_digest(model), activity, lanes, time_count, scores, 0., parities)
    path = tmp_path / "policy.npz"
    policy.save(path)
    frozen = LearnedFaultRouter.load(path)
    scalar = ScalarLearnedDecoder(model, frozen)
    batch = LearnedFaultRoutedDecoder(model, frozen)
    packed = circuit.compile_detector_sampler(seed=9270002).sample(512, bit_packed=True)
    random = rng.integers(0, 256, size=(512, packed.shape[1]), dtype=np.uint8)
    records = np.concatenate((packed, random))
    route = frozen.route_batch(records)
    assert np.any(route) and np.any(~route) and np.any(frozen._ambiguous)
    np.testing.assert_array_equal([scalar.route(row.tobytes()) for row in records], route)
    np.testing.assert_array_equal([scalar.decode(row.tobytes()) for row in packed], batch.decode_batch(packed)[:, 0])


@pytest.mark.parametrize("correlated", [False, True])
def test_scalar_matching_agrees_with_batch_and_rejects_bad_records(correlated):
    circuit = memory_circuit("z", 4, 4, .005)
    model = circuit.detector_error_model(decompose_errors=True)
    decoder = ScalarMatchingDecoder(model, correlated=correlated)
    matcher = pymatching.Matching.from_detector_error_model(model, enable_correlations=correlated)
    packed = circuit.compile_detector_sampler(seed=9270003).sample(128, bit_packed=True)
    expected = matcher.decode_batch(packed, enable_correlations=correlated, bit_packed_shots=True,
                                    bit_packed_predictions=True)[:, 0]
    np.testing.assert_array_equal([decoder.decode(row.tobytes()) for row in packed], expected)
    with pytest.raises(ValueError, match="byte detector record"):
        decoder.decode(bytes(decoder.record_bytes - 1))
    with pytest.raises(ValueError, match="byte detector record"):
        decoder.decode(bytearray(decoder.record_bytes))
    assert model.num_detectors % 8
    with pytest.raises(ValueError, match="padding"):
        decoder.decode(bytes(decoder.record_bytes - 1) + b"\x80")
