import numpy as np
import pytest

from logical_reference import build_reference
from matching_confidence import MatchingConfidence
from stim_data import ROUNDS


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_lower_sector_cost_agrees_with_ordinary_mwpm(distance):
    p_data, p_readout = 0.008, 0.04
    reference = build_reference(distance, p_data, p_readout)
    detectors = reference.circuit.compile_detector_sampler(seed=42).sample(1024)
    sector_costs = MatchingConfidence(distance, p_data, p_readout).features(detectors)
    prediction, ordinary_cost = reference._mwpm.decode_batch(detectors, return_weights=True)
    lower_cost = sector_costs.min(axis=1)
    np.testing.assert_allclose(lower_cost, ordinary_cost, atol=1e-5)
    np.testing.assert_allclose(sector_costs[np.arange(len(detectors)), prediction[:, 0]], lower_cost, atol=1e-5)

    gap = sector_costs[:, 0] - sector_costs[:, 1]
    untied = np.abs(gap) > 1e-5
    assert (gap > 1e-5).any() and (gap < -1e-5).any()
    np.testing.assert_array_equal(gap[untied] > 0, prediction[untied, 0])


def test_both_costs_agree_with_exhaustive_spatial_search():
    distance, p_data = 3, 0.008
    reference = build_reference(distance, p_data, 0)
    single_site = np.eye(distance**2, dtype=bool).reshape(-1, distance, distance)
    terminal = reference.terminal_checks(single_site)
    observed = np.zeros((len(single_site), ROUNDS, *reference.check_mask.shape), dtype=bool)
    observed[:, :, reference.check_mask] = terminal[:, None, :]
    detector_flips = reference.detector_records(observed, terminal)
    logical_flips = reference.observable_parity(single_site)

    # Stim merges faults that produce the same detector and logical flips.
    # The merged Bernoulli probability is the chance of an odd fault count.
    signatures, multiplicities = np.unique(
        np.column_stack((detector_flips, logical_flips)), axis=0, return_counts=True
    )
    probabilities = (1 - (1 - 2 * p_data) ** multiplicities) / 2
    weights = np.log1p(-probabilities) - np.log(probabilities)
    choices = (np.arange(1 << len(weights))[:, None] >> np.arange(len(weights))) & 1
    effects = choices @ signatures & 1
    syndromes, index = np.unique(effects[:, :-1], axis=0, return_inverse=True)
    expected = np.full((len(syndromes), 2), np.inf)
    np.minimum.at(expected, (index, effects[:, -1]), choices @ weights)

    # Readout is ideal, so a history outside the first round has zero
    # syndrome. Moving its edges into the first round preserves the logical
    # parity, and canceling repeated edges can only lower the cost.
    actual = MatchingConfidence(distance, p_data, 0).features(syndromes)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("p_data,p_readout", [(0, 0), (0, 0.04), (1, 0.04), (1, 1)])
def test_only_the_certain_logical_class_has_finite_cost(p_data, p_readout):
    reference = build_reference(5, p_data, p_readout)
    detectors, logical = reference.circuit.compile_detector_sampler(seed=20).sample(16, separate_observables=True)
    decoder = MatchingConfidence(5, p_data, p_readout)
    costs = decoder.features(detectors)
    np.testing.assert_array_equal(np.isfinite(costs), np.arange(2) == logical)
    assert decoder.features(detectors[:0]).shape == (0, 2)


def test_complementing_fault_probabilities_shifts_logical_costs():
    reference = build_reference(5, 0.008, 0.04)
    detectors = reference.circuit.compile_detector_sampler(seed=21).sample(128)
    frame = build_reference(5, 1, 1)
    shift, logical = frame.circuit.compile_detector_sampler().sample(1, separate_observables=True)
    assert logical[0, 0]
    costs = MatchingConfidence(5, 0.008, 0.04).features(detectors)
    shifted = MatchingConfidence(5, 0.992, 0.96).features(detectors ^ shift)
    np.testing.assert_allclose(shifted, costs[:, ::-1], atol=1e-5)


def test_tied_and_impossible_syndromes():
    reference = build_reference(3, 0.5, 0)
    detector_count = reference.circuit.num_detectors
    detectors = np.zeros((2, detector_count), dtype=bool)
    detectors[1, -1] = True
    decoder = MatchingConfidence(3, 0.5, 0)
    np.testing.assert_array_equal(decoder.features(detectors), [[0, 0], [np.inf, np.inf]])
    with pytest.raises(ValueError, match="unpacked detector bits"):
        decoder.features(detectors[:, :-1])
    with pytest.raises(ValueError, match="binary"):
        decoder.features(detectors.astype(np.uint8) * 2)
