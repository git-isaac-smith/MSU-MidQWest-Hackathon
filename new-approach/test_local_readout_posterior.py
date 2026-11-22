"""Verify the local-readout approximation against small independent calculations."""

import numpy as np
import pytest

from local_readout_posterior import LocalReadoutPosterior
from logical_reference import build_reference
from posterior_decoder import PosteriorDecoder
from stim_data import ROUNDS


def _sample(distance, p_data, p_readout, shots=27):
    reference = build_reference(distance, p_data, p_readout)
    detectors = reference.circuit.compile_detector_sampler(seed=86131).sample(shots)
    observed, terminal = reference.public_checks(detectors)
    return reference, observed, terminal


def _physical_radius_posterior(reference, observed, terminal, p_data, p_readout, radius):
    count = reference.data_ids.size
    states = np.arange(1 << count)
    frames = (states[:, None] >> np.arange(count) & 1).astype(bool)
    frames = frames.reshape(-1, *reference.data_ids.shape)
    parity = reference.observable_parity(frames)
    syndromes = reference.terminal_checks(frames)
    weights = frames.sum(axis=(1, 2))
    faults = p_data**weights * (1 - p_data)**(count - weights)
    transitions = faults[states[:, None] ^ states]
    distribution = np.zeros((len(observed), len(states)))
    distribution[:, 0] = 1
    for round_number in range(ROUNDS):
        distribution = distribution @ transitions
        if round_number < ROUNDS - 1:
            flips = np.count_nonzero(observed[:, round_number, reference.check_mask][:, None] ^ syndromes, axis=-1)
            distribution *= np.where(flips <= radius, (p_readout / (1 - p_readout))**flips, 0)
    distribution *= np.all(terminal[:, None] == syndromes, axis=-1)
    return distribution[:, parity].sum(axis=1) / distribution.sum(axis=1)


@pytest.mark.parametrize("p_data,p_readout", [(0.008, 0.04), (0.17, 0.29)])
def test_unrestricted_distance_three_matches_exact_decoder(p_data, p_readout):
    reference, observed, terminal = _sample(3, p_data, p_readout)
    decoder = LocalReadoutPosterior(3, p_data, p_readout, radius=int(reference.check_mask.sum()))
    exact = PosteriorDecoder(3, p_data, p_readout)
    np.testing.assert_allclose(decoder.predict_proba(observed, terminal), exact.predict_proba(observed, terminal),
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_restricted_history_matches_uncompressed_frame_enumeration(radius):
    p_data, p_readout = 0.078, 0.18
    reference, observed, terminal = _sample(3, p_data, p_readout, shots=11)
    decoder = LocalReadoutPosterior(3, p_data, p_readout, radius=radius)
    expected = _physical_radius_posterior(reference, observed, terminal, p_data, p_readout, radius)
    np.testing.assert_allclose(decoder.predict_proba(observed, terminal), expected, rtol=1e-12, atol=1e-14)


def test_final_raw_readout_and_inactive_checks_do_not_affect_distance_five():
    p_data, p_readout = 0.008, 0.04
    reference, observed, terminal = _sample(5, p_data, p_readout, shots=89)
    decoder = LocalReadoutPosterior(5, p_data, p_readout, radius=2)
    expected = decoder.predict_proba(observed, terminal)
    observed[:, -1, reference.check_mask] ^= True
    observed[:, :, ~reference.check_mask] ^= True
    np.testing.assert_array_equal(decoder.predict_proba(observed, terminal), expected)


def test_optional_beam_keeps_the_highest_mass_syndrome_and_can_keep_all():
    p_data, p_readout = 0.008, 0.04
    _, observed, terminal = _sample(3, p_data, p_readout)
    all_candidates = LocalReadoutPosterior(3, p_data, p_readout, radius=2)
    wide_beam = LocalReadoutPosterior(3, p_data, p_readout, radius=2, beam_width=32)
    np.testing.assert_array_equal(wide_beam.predict_proba(observed, terminal),
                                  all_candidates.predict_proba(observed, terminal))

    # For an all-zero readout history the zero syndrome has the highest
    # forward weight each round at these probabilities.
    observed[:] = False
    no_readout_faults = LocalReadoutPosterior(3, p_data, p_readout, radius=0)
    one_candidate = LocalReadoutPosterior(3, p_data, p_readout, radius=2, beam_width=1)
    np.testing.assert_array_equal(one_candidate.predict_proba(observed, terminal),
                                  no_readout_faults.predict_proba(observed, terminal))


def test_rejects_invalid_configuration_and_shapes_and_accepts_empty_batch():
    with pytest.raises(ValueError, match="distances 3, 5, and 7"):
        LocalReadoutPosterior(9, 0.008, 0.04)
    with pytest.raises(ValueError, match="between zero and one half"):
        LocalReadoutPosterior(3, 0, 0.04)
    with pytest.raises(ValueError, match="radius must be between"):
        LocalReadoutPosterior(7, 0.008, 0.04, radius=3)
    with pytest.raises(ValueError, match="beam_width must be positive"):
        LocalReadoutPosterior(3, 0.008, 0.04, beam_width=0)

    _, observed, terminal = _sample(3, 0.008, 0.04)
    decoder = LocalReadoutPosterior(3, 0.008, 0.04)
    assert decoder.predict_proba(observed[:0], terminal[:0]).shape == (0,)
    with pytest.raises(ValueError, match="five observed check grids"):
        decoder.predict_proba(observed[:, :4], terminal)
    with pytest.raises(ValueError, match="terminal parity"):
        decoder.predict_proba(observed, terminal[:, :3])
