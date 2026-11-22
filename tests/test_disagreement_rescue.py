import numpy as np
import pytest

from bascd.circuits import memory_circuit
from bascd.disagreement_rescue import DisagreementRescue


pytest.importorskip("stabilizer_qec")
PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)


def test_full_cascade_matches_independent_packed_votes():
    circuit = memory_circuit("z", 3, 3, .01)
    model = circuit.detector_error_model(decompose_errors=True)
    decoder = DisagreementRescue(model)
    records = circuit.compile_detector_sampler(seed=9042390).sample(384, bit_packed=True)
    ordinary = decoder.ordinary(records)
    expected = decoder.correlated.decode_batch(records, enable_correlations=True, **PACKED)
    disputes = ordinary[:, 0] != expected[:, 0]
    assert disputes.any() and not disputes.all()
    expected[disputes] = decoder.belief.decode_batch(records[disputes], threads=1, **PACKED)
    actual, used_correlated, used_belief = decoder.decode_batch_with_routes(records)
    assert used_correlated.all()
    np.testing.assert_array_equal(used_belief, disputes)
    np.testing.assert_array_equal(actual, expected)
    assert decoder.decode_batch(records[:0]).shape == (0, 1)


@pytest.mark.parametrize("converged, final_vote", [(True, 1), (False, 0)])
def test_activity_thresholds_and_convergence_gate(converged, final_vote):
    circuit = memory_circuit("z", 3, 3, .005)
    model = circuit.detector_error_model(decompose_errors=True)
    records = np.zeros((5, 3), dtype=np.uint8)
    records[:, 0] = [0, 1, 3, 7, 15]
    baseline = np.array([[0], [0], [0], [1], [1]], dtype=np.uint8)
    decoder = DisagreementRescue(model, ordinary=lambda _: baseline,
                                 correlated_min_detectors=2, belief_min_detectors=3,
                                 require_convergence=True)

    class Correlated:
        def decode_batch(self, packed, **kwargs):
            np.testing.assert_array_equal(packed, records[2:])
            assert kwargs == dict(enable_correlations=True, **PACKED)
            return np.array([[1], [0], [1]], dtype=np.uint8)

    class Belief:
        def decode_batch(self, packed, **kwargs):
            np.testing.assert_array_equal(packed, records[3:4])
            assert kwargs == dict(threads=1, return_converged=True, **PACKED)
            return np.ones((1, 1), dtype=np.uint8), np.array([converged])

    decoder.correlated = Correlated()
    decoder.belief = Belief()
    actual, used_correlated, used_belief = decoder.decode_batch_with_routes(records)
    np.testing.assert_array_equal(used_correlated, [False, False, True, True, True])
    np.testing.assert_array_equal(used_belief, [False, False, False, True, False])
    np.testing.assert_array_equal(actual.ravel(), [0, 0, 1, final_vote, 1])
    np.testing.assert_array_equal(baseline.ravel(), [0, 0, 0, 1, 1])
