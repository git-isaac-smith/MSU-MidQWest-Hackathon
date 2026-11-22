import numpy as np
import pytest

from logical_reference import build_reference
from native_posterior import NativePosteriorDecoder, native_library_path
from posterior_decoder import PosteriorDecoder
from stim_data import ROUNDS


@pytest.fixture
def native_library():
    library = native_library_path()
    if not library.is_file():
        pytest.skip("native posterior shared library has not been built")
    return library


@pytest.mark.parametrize("p_data,p_readout", [(0.008, 0.04), (0.73, 0.81), (0.08, 0), (1, 1)])
def test_native_posterior_agrees_with_reference_after_cache_eviction(native_library, p_data, p_readout):
    reference = build_reference(5, p_data, p_readout)
    detectors = reference.circuit.compile_detector_sampler(seed=875819).sample(32)
    observed, terminal = reference.public_checks(detectors)
    original = PosteriorDecoder(5, p_data, p_readout)
    native = NativePosteriorDecoder(5, p_data, p_readout, library_path=native_library, prefix_cache_size=1)
    expected = original.predict_proba(observed, terminal)

    np.testing.assert_allclose(native.predict_proba(observed, terminal), expected, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(native.predict_proba(observed[::-1], terminal[::-1]), expected[::-1],
                               rtol=2e-13, atol=2e-15)
    assert native.predict_proba(observed[:0], terminal[:0]).shape == (0,)


def test_native_rejects_impossible_histories(native_library):
    decoder = NativePosteriorDecoder(5, 0, 0, library_path=native_library)
    observed = np.zeros((1, ROUNDS, *decoder._check_mask.shape), dtype=bool)
    terminal = np.zeros((1, decoder._check_count), dtype=bool)
    observed[0, 0, decoder._check_mask] = True
    with pytest.raises(ValueError, match="zero probability"):
        decoder.predict_proba(observed, terminal)

    observed[0, -1, decoder._check_mask] = True
    with pytest.raises(ValueError, match="impossible readout fault"):
        decoder.predict_proba(observed, terminal)


@pytest.mark.parametrize("distance,p_data,p_readout", [(3, 0.008, 0.04), (5, 0.008, 1e-250)])
def test_native_uses_reference_for_smaller_states_and_extreme_rates(tmp_path, distance, p_data, p_readout):
    reference = build_reference(distance, p_data, p_readout)
    detectors = reference.circuit.compile_detector_sampler(seed=973145).sample(12)
    observed, terminal = reference.public_checks(detectors)
    native = NativePosteriorDecoder(distance, p_data, p_readout, library_path=tmp_path / "unavailable")
    original = PosteriorDecoder(distance, p_data, p_readout)
    np.testing.assert_array_equal(native.predict_proba(observed, terminal),
                                  original.predict_proba(observed, terminal))
