from types import SimpleNamespace

import numpy as np
import pytest

from frequency import Passband
from hybrid_cnn import HybridCNN, make_features
from hybrid_policy import HybridDecoder, gap_priority, raw_ablation
from localizer import Localizer
from stim_data import geometry


def test_raw_ablation_preserves_public_checks_geometry_and_matching_features():
    mask = geometry(3).check_mask
    observed = np.zeros((2, 5, *mask.shape), dtype=bool)
    observed[:, 2:4, 1, 1] = True
    terminal = np.zeros((2, mask.sum()), dtype=bool)
    terminal[1, 0] = True
    encoded = make_features(
        observed,
        terminal,
        mask,
        [False, True],
        [-2.0, 3.0],
        mwpm_weight=[3.0, 4.0],
        fft_site_probabilities=np.full((2, 3, 3), 0.25, dtype=np.float32),
        matching_features=np.ones((2, 1), dtype=np.float32),
    )
    original_spatial = encoded.spatial.copy()
    original_matching = encoded.matching.copy()
    assert np.any(original_spatial[..., 5:11])
    encoded.spatial.setflags(write=False)
    encoded.matching.setflags(write=False)

    ablated = raw_ablation(encoded)
    np.testing.assert_array_equal(ablated.check_mask, mask)
    np.testing.assert_array_equal(ablated.spatial[..., :5], original_spatial[..., :5])
    np.testing.assert_array_equal(ablated.spatial[..., 11:18], original_spatial[..., 11:18])
    np.testing.assert_array_equal(ablated.matching[:, :-1], original_matching[:, :-1])
    assert not np.any(ablated.spatial[..., 5:11])
    assert not np.any(ablated.spatial[..., 18])
    assert not np.any(ablated.matching[:, -1])
    np.testing.assert_array_equal(encoded.spatial, original_spatial)
    np.testing.assert_array_equal(encoded.matching, original_matching)


@pytest.mark.parametrize(
    ("mode", "routed", "expected"),
    [
        ("learned", [], [True, False, False, True, False]),
        ("balanced", [0, 2], [False, False, True, True, False]),
        ("margin_only", [0, 1], [False, False, False, True, False]),
        ("accuracy", [0, 1, 2, 4], [False, False, True, True, True]),
    ],
)
def test_policy_routes_with_inclusive_budgets_and_strict_learned_override(mode, routed, expected):
    detectors = np.eye(5, dtype=bool)
    detectors.setflags(write=False)
    baseline = np.array([True, True, False, True, False])
    baseline.setflags(write=False)
    costs = np.array([[2, 3], [2, 2], [0, 2], [4, 6.5], [5, 7]], dtype=float)
    costs.setflags(write=False)
    eligible = [0, 1, 2, 4]
    encoded = object()
    encode_calls = []
    network_calls = []
    expert_calls = []

    def mwpm_parity(records):
        assert records is detectors
        return baseline

    def matching_cost(records):
        assert records is detectors
        return costs

    def encode(records, parity, matching):
        np.testing.assert_array_equal(records, detectors[eligible])
        np.testing.assert_array_equal(parity, baseline[eligible])
        np.testing.assert_array_equal(matching, costs[eligible])
        encode_calls.append(records)
        return encoded

    def score(inputs):
        assert inputs is encoded
        network_calls.append(inputs)
        return np.array([0.5, 0.75, 0.625, 0.25])

    def exact(records, parity):
        np.testing.assert_array_equal(records, detectors[routed])
        np.testing.assert_array_equal(parity, baseline[routed])
        expert_calls.append(records)
        return ~parity

    decoder = HybridDecoder.__new__(HybridDecoder)
    decoder.ref = SimpleNamespace(mwpm_parity=mwpm_parity)
    decoder.confidence = SimpleNamespace(features=matching_cost)
    decoder.encode = encode
    decoder.network = SimpleNamespace(predict=score)
    decoder.exact_decisions = exact
    decoder.policy = {
        "gap_limit": 2.0,
        "override_threshold": 0.625,
        "gap_route_at": gap_priority(detectors, costs)[0],
        "cnn_route_at": -0.125,
    }

    np.testing.assert_array_equal(decoder.predict(detectors, mode), expected)
    assert len(encode_calls) == len(network_calls) == (mode in ("learned", "balanced"))
    assert len(expert_calls) == bool(routed)
    np.testing.assert_array_equal(detectors, np.eye(5, dtype=bool))
    np.testing.assert_array_equal(baseline, [True, True, False, True, False])


def test_exact_posterior_ties_keep_mwpm_parity():
    detectors = np.eye(6, dtype=bool)
    observed, terminal = object(), object()
    baseline = np.array([True, False, False, True, False, True])
    probabilities = np.array([0.5, 0.5, 0.5 + 5e-13, 0.5 - 5e-13, 0.5 + 1e-9, 0.5 - 1e-9])

    def public_checks(records):
        assert records is detectors
        return observed, terminal

    def posterior(observed_checks, terminal_checks):
        assert observed_checks is observed and terminal_checks is terminal
        return probabilities

    decoder = HybridDecoder.__new__(HybridDecoder)
    decoder.ref = SimpleNamespace(public_checks=public_checks)
    decoder.posterior = SimpleNamespace(predict_proba=posterior)
    np.testing.assert_array_equal(
        decoder.exact_decisions(detectors, baseline), [True, False, False, True, True, False]
    )


def test_d3_model_predicts_from_fresh_public_detectors_without_mutating_inputs():
    site_model = Localizer(channels=7, seed=91)
    decoder = HybridDecoder((3, 0.02, 0.05), site_model, 0.5, Passband())
    sampler = decoder.ref.circuit.compile_detector_sampler(seed=92)
    detectors, _ = sampler.sample(24, separate_observables=True)
    baseline = decoder.ref.mwpm_parity(detectors)
    costs = decoder.confidence.features(detectors)
    inputs = decoder.encode(detectors, baseline, costs)
    decoder.network = HybridCNN(inputs, seed=93, conv_channels=4, hidden_channels=8)
    decoder.policy = {"gap_limit": 100.0, "override_threshold": 0.5}
    expected = baseline ^ (decoder.network.predict(inputs) > 0.5)
    fresh, _ = decoder.ref.circuit.compile_detector_sampler(seed=92).sample(24, separate_observables=True)
    fresh.setflags(write=False)

    np.testing.assert_array_equal(decoder.predict(fresh, mode="learned"), expected)
    np.testing.assert_array_equal(fresh, detectors)
    assert decoder.predict(fresh[:0], mode="learned").shape == (0,)
