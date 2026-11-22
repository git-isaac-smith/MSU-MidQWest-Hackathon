import numpy as np
import pytest

from list_matching import ListMatchingPosterior
from logical_reference import build_reference, stim_circuit
from matching_confidence import MatchingConfidence
from posterior_decoder import PosteriorDecoder


def _records_from_edges(decoder, edges):
    toggles = np.zeros((len(edges), decoder._detector_count + 2), dtype=np.uint8)
    for fault, (first, second) in enumerate(decoder._endpoints):
        toggles[:, first] ^= edges[:, fault]
        toggles[:, second] ^= edges[:, fault]
    return toggles


def test_distance_seven_lists_contain_valid_distinctly_scored_corrections():
    decoder = ListMatchingPosterior(7, 0.008, 0.04, members=49, seed=25)
    sampler = stim_circuit(7, 0.008, 0.04).compile_detector_sampler(seed=54321)
    records = sampler.sample(19).astype(np.uint8)
    masses = decoder.log_masses(records)
    unique_counts = []
    for parity in (0, 1):
        boundary = np.bitwise_xor.reduce(records, axis=1) ^ parity
        constraints = np.column_stack((records, boundary, np.full(len(records), parity, dtype=np.uint8)))
        candidates = decoder._candidate_vectors(constraints)
        edges = np.unpackbits(candidates, axis=2, bitorder="little", count=decoder._weights.size)
        reconstructed = _records_from_edges(decoder, edges.reshape(-1, edges.shape[-1]))
        np.testing.assert_array_equal(reconstructed, np.repeat(constraints, candidates.shape[1], axis=0))

        for row, candidate in enumerate(edges):
            unique = np.unique(candidate, axis=0)
            unique_counts.append(len(unique))
            expected = np.logaddexp.reduce(-(unique @ decoder._weights))
            np.testing.assert_allclose(masses[row, parity], expected, atol=1e-12)

    assert min(unique_counts) < 49
    assert max(unique_counts) > 1


def test_one_member_sums_the_original_minimum_correction_in_each_sector():
    distance, p_data, p_readout = 3, 0.008, 0.04
    records = stim_circuit(distance, p_data, p_readout).compile_detector_sampler(seed=9173).sample(29)
    decoder = ListMatchingPosterior(distance, p_data, p_readout, members=1)
    confidence = MatchingConfidence(distance, p_data, p_readout)
    np.testing.assert_allclose(decoder.log_masses(records), -confidence.features(records), atol=1e-5)


def test_matching_list_agrees_with_distance_three_posterior_on_simple_records():
    distance, p_data, p_readout = 3, 0.008, 0.0
    decoder = ListMatchingPosterior(distance, p_data, p_readout, seed=84310500)
    edges = np.zeros((2, len(decoder._weights)), dtype=np.uint8)
    edges[1, np.flatnonzero((decoder._endpoints == decoder._detector_count + 1).any(axis=1))[0]] = 1
    records = _records_from_edges(decoder, edges)[:, :-2]
    reference = build_reference(distance, p_data, p_readout)
    exact = PosteriorDecoder(distance, p_data, p_readout).predict_proba(*reference.public_checks(records))
    approximate = decoder.predict_proba(records)
    np.testing.assert_allclose(approximate, exact, atol=5e-5)
    np.testing.assert_array_equal(approximate > 0.5, [False, True])


def test_readout_only_graph_supports_many_flips_and_rejects_impossible_records():
    decoder = ListMatchingPosterior(7, 0, 0.04, members=7)
    edges = np.zeros((1, len(decoder._weights)), dtype=np.uint8)
    simultaneous = np.flatnonzero((decoder._endpoints < 24).any(axis=1))[:8]
    assert len(simultaneous) == 8
    edges[:, simultaneous] = 1
    syndrome = _records_from_edges(decoder, edges)
    records = syndrome[:, :-2]
    candidates = decoder._candidate_vectors(syndrome)
    np.testing.assert_array_equal(candidates, np.repeat(np.packbits(edges, bitorder="little", axis=1)[:, None], 7, axis=1))
    np.testing.assert_array_equal(decoder.predict_proba(records), [0])
    assert np.isneginf(decoder.log_masses(records)[0, 1])

    records[:, 0] ^= 1
    np.testing.assert_array_equal(decoder.log_masses(records), [[-np.inf, -np.inf]])
    with pytest.raises(ValueError, match="zero probability"):
        decoder.predict_proba(records)


def test_seeded_scores_are_independent_of_batches_and_validate_records():
    reference = stim_circuit(3, 0.008, 0.04)
    records = reference.compile_detector_sampler(seed=565).sample(38).astype(np.uint8)
    first = ListMatchingPosterior(3, 0.008, 0.04, members=11, seed=24)
    second = ListMatchingPosterior(3, 0.008, 0.04, members=11, seed=24)
    second._BATCH = 7
    np.testing.assert_array_equal(first.predict_proba(records), second.predict_proba(records[::-1])[::-1])
    assert first.predict_proba(records[:0]).shape == (0,)
    with pytest.raises(ValueError, match="unpacked detector bits"):
        first.predict_proba(np.packbits(records, axis=1))
    with pytest.raises(ValueError, match="binary"):
        first.predict_proba(records + 2)
    with pytest.raises(ValueError, match="at least one"):
        ListMatchingPosterior(3, 0.008, 0.04, members=0)
