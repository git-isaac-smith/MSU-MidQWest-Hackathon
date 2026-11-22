import numpy as np
import stim

from bascd.ambiguity_dispatch import AmbiguityDispatcher, _score_table
from bascd.circuits import memory_circuit


PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)


def test_coincidence_distinguishes_same_activity_in_the_two_components():
    model = stim.DetectorErrorModel("""
        error(0.01) D0 L0
        error(0.02) D1
        error(0.02) D2
        error(0.009) D0 L0 ^ D1
    """)
    table = _score_table(model, np.array([0b001], dtype=np.uint8))
    assert table[0, 0b011] > table[0, 0b101]
    assert table[0, 0b001] > table[0, 0b010]
    assert table[0, 0b110] == 32


def test_dispatch_preserves_native_branches_and_batch_order():
    circuit = memory_circuit("z", 5, 5, 0.005)
    model = circuit.detector_error_model(decompose_errors=True)
    decoder = AmbiguityDispatcher(circuit, model, calibration_seed=9042900, fraction=0.12)
    shots = circuit.compile_detector_sampler(seed=9042901).sample(512, bit_packed=True)
    use_correlations = decoder.route(shots)
    assert use_correlations.any() and not use_correlations.all()
    expected = decoder.ordinary.decode_batch(shots & decoder.logical_mask, **PACKED)
    expected[use_correlations] = decoder.correlated.decode_batch(
        shots[use_correlations], enable_correlations=True, **PACKED)
    np.testing.assert_array_equal(decoder.decode_batch(shots), expected)
    assert decoder.decode_batch(shots[:0]).shape == (0, 1)
