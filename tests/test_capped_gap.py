import numpy as np
import pymatching
import pytest
import stim

from bascd.capped_gap import CappedGapProbe, CappedGapRoutedDecoder
from bascd.circuits import memory_circuit
from bascd.logical_gap_route import LogicalGapProbe, PACKED
from bascd.projected_router import ProjectedOrdinary


@pytest.mark.parametrize("distance", [5, 7])
def test_capped_gap_matches_forced_gap_away_from_threshold(distance):
    circuit = memory_circuit("z", distance, distance, .005)
    model = circuit.detector_error_model(decompose_errors=True)
    packed = circuit.compile_detector_sampler(seed=9040311).sample(512, bit_packed=True)
    packed.flags.writeable = False
    cutoff = 5
    zero, one = LogicalGapProbe(model).costs(packed)
    gap = np.abs(zero - one)
    expected_route = gap < cutoff
    distant = gap > cutoff + .003
    separate = np.abs(gap - cutoff) > .003
    assert distant.any() and expected_route.any()

    probe = CappedGapProbe(model, cutoff)
    ordinary = ProjectedOrdinary(circuit, model).decode_batch(packed)
    for prediction in (None, ordinary):
        bits, route = probe.classify(packed, prediction)
        np.testing.assert_array_equal(route[separate], expected_route[separate])
        np.testing.assert_array_equal(bits[distant, 0], (one < zero)[distant])

    correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
    expected = np.where(expected_route[:, None], correlated.decode_batch(packed, enable_correlations=True, **PACKED),
                        (one < zero)[:, None])
    decoder = CappedGapRoutedDecoder(circuit, model, cutoff)
    result = decoder.decode_batch(packed)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[separate], expected[separate])
    assert decoder.decode_batch(packed[:0]).shape == (0, 1)


def test_capped_gap_appends_a_byte_when_all_detector_bits_are_relevant():
    model = stim.DetectorErrorModel(
        "error(.1) D0 L0\n"
        + "\n".join(f"error(.1) D{i} D{i + 1}" for i in range(7))
        + "\nerror(.1) D7"
    )
    packed = np.asarray([[0], [1], [128], [255]], dtype=np.uint8)
    cutoff = 1
    zero, one = LogicalGapProbe(model).costs(packed)
    probe = CappedGapProbe(model, cutoff)
    bits, route = probe.classify(packed)
    np.testing.assert_array_equal(route, np.abs(zero - one) < cutoff)
    np.testing.assert_array_equal(bits[~route, 0], (one < zero)[~route])
