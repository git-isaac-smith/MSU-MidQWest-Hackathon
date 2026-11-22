import numpy as np
import pytest

from logical_reference import build_reference
from posterior_decoder import PosteriorDecoder
from stim_data import ROUNDS, sample_shots


def _physical_state_posterior(reference, observed, terminal, p_data, p_readout):
    """Enumerate all distance-three physical frames and their transitions."""
    count = reference.data_ids.size
    states = np.arange(1 << count)
    frames = (states[:, None] >> np.arange(count) & 1).astype(bool)
    frames = frames.reshape(-1, *reference.data_ids.shape)
    parity = reference.observable_parity(frames)
    syndromes = reference.terminal_checks(frames)
    weights = frames.sum(axis=(1, 2))
    faults = p_data ** weights * (1 - p_data) ** (count - weights)
    transitions = faults[states[:, None] ^ states]
    distribution = np.zeros((len(observed), len(states)))
    distribution[:, 0] = 1
    for recorded in observed[:, :, reference.check_mask].transpose(1, 0, 2):
        distribution = distribution @ transitions
        flips = np.count_nonzero(recorded[:, None] ^ syndromes, axis=-1)
        distribution *= p_readout ** flips * (1 - p_readout) ** (syndromes.shape[-1] - flips)
    distribution *= np.all(terminal[:, None] == syndromes, axis=-1)
    return distribution[:, parity].sum(axis=1) / distribution.sum(axis=1)


@pytest.mark.parametrize("p_data,p_readout", [(0.008, 0.04), (0.73, 0.81), (0.08, 0)])
def test_posterior_agrees_with_uncompressed_frame_enumeration(p_data, p_readout):
    reference = build_reference(3, p_data, p_readout)
    batch = sample_shots(3, 19, p_data, p_readout, seed=31243)
    terminal = reference.terminal_checks(batch.targets)
    posterior = PosteriorDecoder(3, p_data, p_readout).predict_proba(batch.observed, terminal)
    expected = _physical_state_posterior(reference, batch.observed, terminal, p_data, p_readout)
    np.testing.assert_allclose(posterior, expected, rtol=1e-12, atol=1e-14)


def test_final_noisy_readout_cancels_and_inactive_cells_have_no_effect():
    p_data, p_readout = 0.008, 0.04
    reference = build_reference(5, p_data, p_readout)
    detectors = reference.circuit.compile_detector_sampler(seed=83721).sample(21)
    checks, terminal = reference.public_checks(detectors)
    decoder = PosteriorDecoder(5, p_data, p_readout)
    posterior = decoder.predict_proba(checks, terminal)
    checks[:, -1, reference.check_mask] ^= True
    checks[:, :, ~reference.check_mask] ^= True
    np.testing.assert_array_equal(decoder.predict_proba(checks, terminal), posterior)


@pytest.mark.parametrize("p_data,p_readout", [(0, 0.04), (0, 1), (1, 0), (1, 1), (0.5, 0.04)])
def test_known_logical_priors(p_data, p_readout):
    reference = build_reference(5, p_data, p_readout)
    batch = sample_shots(5, 12, p_data, p_readout, seed=341)
    terminal = reference.terminal_checks(batch.targets)
    expected = p_data if p_data in (0, 0.5) else (ROUNDS * reference.data_ids.shape[0]) % 2
    posterior = PosteriorDecoder(5, p_data, p_readout).predict_proba(batch.observed, terminal)
    np.testing.assert_array_equal(posterior, np.full(len(batch.observed), expected))


def test_shapes_empty_arrays_and_impossible_records():
    decoder = PosteriorDecoder(3, 0, 0)
    observed = np.zeros((1, ROUNDS, 4, 4), dtype=bool)
    terminal = np.zeros((1, 4), dtype=bool)
    assert decoder.predict_proba(observed[:0], terminal[:0]).shape == (0,)
    with pytest.raises(ValueError, match="five observed"):
        decoder.predict_proba(observed[:, :4], terminal)
    with pytest.raises(ValueError, match="terminal parity"):
        decoder.predict_proba(observed, terminal[:, :3])
    terminal[0, 0] = True
    with pytest.raises(ValueError, match="impossible readout fault"):
        decoder.predict_proba(observed, terminal)


def test_extremely_unlikely_readout_still_conditions_the_logical_probability():
    reference = build_reference(3, 0.5, 0.04)
    observed = np.zeros((1, ROUNDS, *reference.check_mask.shape), dtype=bool)
    observed[:, -1, reference.check_mask] = True
    terminal = np.zeros((1, int(reference.check_mask.sum())), dtype=bool)
    posterior = PosteriorDecoder(3, 0.5, 1e-250).predict_proba(observed, terminal)
    np.testing.assert_array_equal(posterior, [0.5])
