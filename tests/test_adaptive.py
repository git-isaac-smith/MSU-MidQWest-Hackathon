from dataclasses import replace

import numpy as np
import pytest
import stim

from bascd.adaptive import AdaptiveDecoder, RoutingPolicy, SyndromeFeatures, fit_policy, model_id
from bascd.adaptive_experiment import select, summarize
from bascd.circuits import memory_circuit
from bascd.decoders import CHOICES, compile_decoder


def test_spacetime_buckets_respect_detector_order_and_round_gaps():
    model = stim.DetectorErrorModel(
        "detector(0, 0, 3) D0\ndetector(0, 0, 1) D1\ndetector(0, 0, 0) D2")
    matrix = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool)
    features = SyndromeFeatures(model, "spacetime")
    assert features.buckets(matrix).tolist() == [8, 9, 13]
    assert [features.bucket(event) for event in matrix] == [8, 9, 13]


def test_fit_uses_paired_gain_and_respects_budget():
    dem = stim.DetectorErrorModel("error(0.02) D0 L0")
    features = SyndromeFeatures(dem, "count")
    buckets = np.repeat([0, 1, 2], [100, 30, 20])
    ordinary = np.zeros(len(buckets), dtype=bool)
    correlated = ordinary.copy()
    ordinary[100:130] = True
    correlated[130:] = True
    policy = fit_policy(dem, features, buckets, ordinary, correlated, 0.25, "count")
    assert policy.correlated_bins == (1,)
    assert RoutingPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(ValueError, match="different detector error model"):
        AdaptiveDecoder(dem, replace(policy, dem_sha256="0" * 64))


@pytest.mark.parametrize("feature,bins", [("count", ()), ("count", (2, 3, 4)), ("spacetime", (8, 9, 12, 13))])
def test_router_preserves_native_prediction_and_order(feature, bins):
    circuit = memory_circuit("z", 3, 3, 0.01)
    model = circuit.detector_error_model(decompose_errors=True)
    events = circuit.compile_detector_sampler(seed=45).sample(512)
    decoder = AdaptiveDecoder(model, RoutingPolicy(model_id(model), feature, bins))
    routed = decoder.route(events)
    assert routed.any() == bool(bins)
    assert not routed.all()
    expected = compile_decoder(model, CHOICES[0]).decode_batch(events)
    correlated = compile_decoder(model, CHOICES[1]).decode_batch(events)
    expected[routed] = correlated[routed]
    np.testing.assert_array_equal(decoder.decode_batch(events), expected)
    np.testing.assert_array_equal(decoder.decode_batch(events[::2]), expected[::2])
    np.testing.assert_array_equal([decoder.decode_one(e) for e in events[:32]], expected[:32])
    assert decoder.decode_batch(events[:0]).shape == (0, 1)
    with pytest.raises(ValueError, match="matching the model"):
        decoder.decode_batch(events[:, :-1])


def test_selection_includes_references_and_paired_sign_is_candidate_minus_baseline():
    row = dict(decoder="count-50", shots=100, failures=20, median_batch_s=0.00011,
               correlated_fraction=0.3, mwpm_only_failed=24, candidate_only_failed_vs_mwpm=10,
               correlated_mwpm_only_failed=5, candidate_only_failed_vs_correlated_mwpm=10)
    result = summarize([row])["count-50"]
    assert result["paired_vs_mwpm"]["ler_difference"] == pytest.approx(-0.14)
    assert result["paired_vs_mwpm"]["approximate_95_ci"][1] < 0
    assert result["paired_vs_correlated_mwpm"]["approximate_95_ci"] is None
    assert select({"mwpm": {"failures": 34, "batch_us_per_shot": 1},
                   "correlated_mwpm": {"failures": 15, "batch_us_per_shot": 1.8},
                   "count-50": result}) == {"1.5": "count-50", "2.0": "correlated_mwpm"}
