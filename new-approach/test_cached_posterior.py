import numpy as np
import pytest

from cached_posterior import CachedPosteriorDecoder
from logical_reference import build_reference
from posterior_decoder import PosteriorDecoder
from stim_data import ROUNDS, sample_shots


@pytest.mark.parametrize("distance,p_data,p_readout", [
    (3, 0.008, 0.04),
    (5, 0.008, 0.04),
    (3, 0.72, 0.86),
    (3, 0, 0.04),
    (3, 1, 0),
    (3, 0.08, 1),
    (3, 1e-12, 1e-120),
])
def test_cached_posterior_matches_reference_on_sampled_faults(distance, p_data, p_readout):
    reference = build_reference(distance, p_data, p_readout)
    shots = sample_shots(distance, 139, p_data, p_readout, seed=93457)
    terminal = reference.terminal_checks(shots.targets)
    history = np.concatenate((shots.observed, shots.observed[:29]))
    terminal = np.concatenate((terminal, terminal[:29]))
    decoder = CachedPosteriorDecoder(distance, p_data, p_readout)
    expected = PosteriorDecoder(distance, p_data, p_readout).predict_proba(history, terminal)
    actual = decoder.predict_proba(history, terminal)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(actual[:29], actual[-29:])


def test_time_reversal_and_final_noisy_observation_share_one_canonical_history():
    distance, p_data, p_readout = 5, 0.008, 0.04
    reference = build_reference(distance, p_data, p_readout)
    detectors = reference.circuit.compile_detector_sampler(seed=62367).sample(67)
    observed, terminal = reference.public_checks(detectors)
    reversed_observed = observed.copy()
    reversed_observed[:, :-1, reference.check_mask] = (
        observed[:, -2::-1, reference.check_mask] ^ terminal[:, None]
    )
    reversed_observed[:, -1] ^= True
    decoder = CachedPosteriorDecoder(distance, p_data, p_readout)
    posterior = decoder.predict_proba(observed, terminal)
    cache_size = len(decoder._history_cache)
    np.testing.assert_array_equal(decoder.predict_proba(reversed_observed, terminal), posterior)
    assert len(decoder._history_cache) == cache_size
    expected = PosteriorDecoder(distance, p_data, p_readout).predict_proba(reversed_observed, terminal)
    np.testing.assert_allclose(posterior, expected, rtol=1e-13, atol=1e-15)


def test_cache_eviction_and_batch_order_preserve_near_ties():
    distance, p_data, p_readout = 5, 0.49999, 0.04
    reference = build_reference(distance, p_data, p_readout)
    detectors = reference.circuit.compile_detector_sampler(seed=322342).sample(81)
    observed, terminal = reference.public_checks(detectors)
    decoder = CachedPosteriorDecoder(distance, p_data, p_readout)
    posterior = decoder.predict_proba(observed, terminal)
    assert np.max(np.abs(posterior - 0.5)) < 1e-14

    small = CachedPosteriorDecoder(distance, p_data, p_readout)
    small._SHOT_BATCH = 7
    small._single_cache.capacity = 2
    small._pair_cache.capacity = 3
    small._HISTORY_CACHE_SIZE = 5
    np.testing.assert_array_equal(small.predict_proba(observed[::-1], terminal[::-1])[::-1], posterior)
    assert len(small._single_cache.rows) == 2
    assert len(small._pair_cache.rows) == 3
    assert len(small._history_cache) == 5
    np.testing.assert_array_equal(small.predict_proba(observed, terminal), posterior)


def test_empty_and_impossible_records_preserve_the_reference_contract():
    observed = np.zeros((1, ROUNDS, 4, 4), dtype=bool)
    terminal = np.zeros((1, 4), dtype=bool)
    decoder = CachedPosteriorDecoder(3, 0, 0)
    assert decoder.predict_proba(observed[:0], terminal[:0]).shape == (0,)
    with pytest.raises(ValueError, match="five observed"):
        decoder.predict_proba(observed[:, :-1], terminal)
    terminal[0, 0] = True
    with pytest.raises(ValueError, match="impossible readout fault"):
        decoder.predict_proba(observed, terminal)
    reference = build_reference(3, 0, 0)
    observed[:, -1, reference.check_mask] = terminal
    with pytest.raises(ValueError, match="zero probability"):
        decoder.predict_proba(observed, terminal)
