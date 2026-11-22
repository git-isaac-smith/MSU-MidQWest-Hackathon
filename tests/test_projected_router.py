import numpy as np
import pymatching
import pytest

from bascd.circuits import memory_circuit
from bascd.css_routed import observable_detector_mask
from bascd.projected_router import ProjectedOrdinary


@pytest.mark.parametrize(("basis", "distance", "p", "cached"), [
    ("x", 5, .005, True),
    ("z", 7, .001, True),
    ("z", 7, .005, False),
])
def test_projected_cache_preserves_masked_matching(basis, distance, p, cached):
    circuit = memory_circuit(basis, distance, distance, p)
    model = circuit.detector_error_model(decompose_errors=True)
    decoder = ProjectedOrdinary(circuit, model)
    assert (decoder._cache is not None) == cached

    sampled = circuit.compile_detector_sampler(seed=9040301).sample(1024, bit_packed=True)
    records = np.asfortranarray(sampled)
    records.flags.writeable = False
    baseline = pymatching.Matching.from_detector_error_model(model)
    mask = observable_detector_mask(baseline, model.num_detectors)
    expected = baseline.decode_batch(records & mask, bit_packed_shots=True, bit_packed_predictions=True)
    actual = decoder.decode_batch(records)
    assert actual.dtype == np.uint8
    assert actual.shape == (len(records), 1)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(decoder.decode_batch(records[:16]), expected[:16])
    np.testing.assert_array_equal(records, sampled)
    assert decoder.decode_batch(records[:0]).shape == (0, 1)
    with pytest.raises(ValueError, match="packed detector"):
        decoder.decode_batch(records[:, :-1])

    if cached:
        # Force all fingerprints to select a single entry. Full-key checks
        # must send every different signature back through masked matching.
        decoder._cache.entries.fill(0)
        np.testing.assert_array_equal(decoder.decode_batch(records), expected)
