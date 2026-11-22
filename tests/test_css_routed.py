import numpy as np
import pymatching
import pytest
import stim

from bascd.circuits import memory_circuit
from bascd.css_routed import CssRoutedDecoder, observable_detector_mask


PACKED = dict(bit_packed_predictions=True, bit_packed_shots=True)


def test_component_mask_uses_all_logically_relevant_components():
    dem = stim.DetectorErrorModel("error(.01) D0 D2\nerror(.02) D2 L0\nerror(.03) D1\nerror(.04) D3 L0")
    m = pymatching.Matching.from_detector_error_model(dem)
    np.testing.assert_array_equal(observable_detector_mask(m, 4), [0b1101])


@pytest.mark.parametrize("basis", ["x", "z"])
def test_routed_paths_see_the_correct_detector_record(basis):
    c = memory_circuit(basis, 3, 3, .005)
    dem = c.detector_error_model(decompose_errors=True)
    data = c.compile_detector_sampler(seed=46391).sample(1000, bit_packed=True)
    original = data.copy()
    routed = CssRoutedDecoder(c, dem, calibration_seed=14513, fraction=.20)
    pop = np.unpackbits(data, axis=1).sum(axis=1)
    selected = pop > routed.cutoff
    assert selected.any() and not selected.all()
    projected = pymatching.Matching.from_detector_error_model(dem).decode_batch(data & routed.mask, **PACKED)
    full = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    expected = np.where(selected[:, None], full.decode_batch(data, enable_correlations=True, **PACKED), projected)
    prediction = routed.decode_batch(data)
    np.testing.assert_array_equal(prediction, expected)
    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(prediction, np.concatenate([routed.decode_batch(a) for a in np.array_split(data, 4)]))
    assert routed.decode_batch(data[:0]).shape == (0, 1)


def test_invalid_width_rejected():
    c = memory_circuit("z", 3, 3, .005)
    routed = CssRoutedDecoder(c, c.detector_error_model(decompose_errors=True), calibration_seed=756)
    with pytest.raises(ValueError, match="packed"):
        routed.decode_batch(np.zeros((2, 4), dtype=np.uint8))
