import numpy as np
import pytest
import stim

from bascd.circuits import memory_circuit
from bascd.hypergraph import HypergraphSpec, compile_hypergraph

pytest.importorskip("stabilizer_qec")


def test_search_preserves_joint_fault_across_decomposition():
    model = stim.DetectorErrorModel("""
        error(0.06) D0 D1 ^ D2 D3 L0
        error(0.08) D0 D1
        error(0.08) D2 D3
    """)
    decoder = compile_hypergraph(model)
    syndrome = np.ones((1, 4), dtype=bool)
    predictions, give_up = decoder.predictions_and_low_confidence(syndrome)
    np.testing.assert_array_equal(predictions, [[1]])
    np.testing.assert_array_equal(give_up, [False])
    np.testing.assert_array_equal(decoder.decode_one(syndrome[0]), [1])


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_search_batch_matches_chunked_and_single_inference(distance):
    circuit = memory_circuit("x", distance, distance, 0.003)
    model = circuit.detector_error_model(decompose_errors=True)
    detectors = circuit.compile_detector_sampler(seed=42).sample(20)
    detectors.flags.writeable = False
    decoder = compile_hypergraph(model)
    predictions = decoder.decode_batch(detectors)
    assert predictions.shape == (20, model.num_observables)
    assert predictions.dtype == np.uint8
    np.testing.assert_array_equal(
        predictions, np.concatenate([decoder.decode_batch(chunk) for chunk in np.array_split(detectors, 3)]),
    )
    np.testing.assert_array_equal(predictions, np.stack([decoder.decode_one(row) for row in detectors]))
    assert decoder.decode_batch(detectors[:0]).shape == (0, model.num_observables)


def test_queue_exhaustion_uses_correlated_matching():
    circuit = memory_circuit("z", 3, 3, 0.01)
    model = circuit.detector_error_model(decompose_errors=True)
    detectors = circuit.compile_detector_sampler(seed=123).sample(32)
    decoder = compile_hypergraph(model, HypergraphSpec(pqlimit=1))
    raw, give_up = decoder.predictions_and_low_confidence(detectors)
    assert give_up.any()
    predictions, fallback = decoder.decode_batch_with_fallback(detectors)
    np.testing.assert_array_equal(fallback, give_up)
    np.testing.assert_array_equal(predictions[~give_up], raw[~give_up])
    np.testing.assert_array_equal(predictions[give_up], decoder.fallback.decode_batch(detectors[give_up]))
    index = np.flatnonzero(give_up)[0]
    np.testing.assert_array_equal(decoder.decode_one(detectors[index]), predictions[index])
