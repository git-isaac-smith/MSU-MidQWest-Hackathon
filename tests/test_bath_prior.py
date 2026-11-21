import itertools
import json

import numpy as np
import pytest
import stim

from bascd.bath_prior import bath_prior_circuit, bath_prior_dem, bath_prior_from_profile, load_bath_prior
from bascd.circuits import memory_circuit


def profile(paulis=(0.8, 0.02, 0.06, 0.12)):
    return {
        "format_version": 1,
        "clock": {"conditional_paulis": dict(zip("IXYZ", paulis, strict=True))},
        "selected": {"conditional_paulis": dict(zip("IXYZ", (0.7, 0.15, 0.1, 0.05), strict=True))},
    }


def test_load_prior_uses_the_selected_profile_channel(tmp_path):
    path = tmp_path / "bath-profile.json"
    path.write_text(json.dumps(profile()), encoding="utf-8")
    expected = [("clock", (0.1, 0.3, 0.6)), ("selected", (0.5, 1 / 3, 1 / 6))]
    for channel, ratios in expected:
        prior = load_bath_prior(path, channel=channel)
        assert prior.channel == channel
        assert prior.ratios == pytest.approx(ratios)
    assert load_bath_prior(path).channel == "clock"


@pytest.mark.parametrize("paulis", [
    (0.9, 0.1, 0.1, 0.1), (0.8, -0.1, 0.2, 0.1), (1, 0, 0, 0),
    (0.8, float("nan"), 0.1, 0.1), (0.8, True, 0.1, 0.1),
])
def test_prior_rejects_invalid_pauli_probabilities(paulis):
    with pytest.raises(ValueError):
        bath_prior_from_profile(profile(paulis))


def test_prior_rejects_unsupported_version_and_channel():
    with pytest.raises(ValueError):
        bath_prior_from_profile({**profile(), "format_version": 2})
    with pytest.raises(ValueError):
        bath_prior_from_profile(profile(), channel="holdout")


def test_prior_rejects_an_unverified_data_site_mapping():
    prior = bath_prior_from_profile(profile())
    circuit = memory_circuit("x", 3, 4, 0.01)
    altered = circuit.copy()
    altered.append("TICK")
    with pytest.raises(ValueError):
        bath_prior_circuit(altered, basis="x", distance=3, rounds=4, p=0.01, prior=prior)
    with pytest.raises(ValueError):
        bath_prior_dem(circuit, basis="x", distance=3, rounds=4, p=0.02, prior=prior)


@pytest.mark.parametrize("p", [0, 0.01])
def test_isotropic_prior_is_identical_to_reference(p):
    prior = bath_prior_from_profile(profile((0.7, 0.1, 0.1, 0.1)))
    circuit = memory_circuit("z", 3, 4, p)
    kwargs = dict(basis="z", distance=3, rounds=4, p=p, prior=prior)
    assert bath_prior_circuit(circuit, **kwargs) == circuit
    assert bath_prior_dem(circuit, **kwargs) == circuit.detector_error_model(decompose_errors=True)


def test_prior_rejects_a_channel_requiring_disjoint_errors():
    prior = bath_prior_from_profile(profile((0.8, 0.1, 0, 0.1)))
    circuit = memory_circuit("z", 3, 1, 0.01)
    with pytest.raises(ValueError, match="independent"):
        bath_prior_dem(circuit, basis="z", distance=3, rounds=1, p=0.01, prior=prior)


def data_targets(instructions):
    measure = next(op for op in reversed(instructions) if op.name in ("M", "MX"))
    return measure.targets_copy()


def categorical_probabilities(axes):
    """Enumerate the eight possible independent-axis subsets on one qubit."""
    totals = np.zeros(4)
    for selected in itertools.product((False, True), repeat=3):
        probability = 1.0
        pauli = 0
        for mask, active, rate in zip((1, 2, 3), selected, axes, strict=True):
            probability *= rate if active else 1 - rate
            if active:
                pauli ^= mask
        totals[pauli] += probability
    return totals


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("expand_uniform", [False, True])
def test_prior_changes_only_round_start_data_noise_and_preserves_total_probability(basis, expand_uniform):
    p, rounds = 0.013, 4
    prior = bath_prior_from_profile(profile((.7, .1, .1, .1)) if expand_uniform else profile())
    circuit = memory_circuit(basis, 3, rounds, p)
    preserved = circuit.copy()
    revised = bath_prior_circuit(circuit, basis=basis, distance=3, rounds=rounds, p=p, prior=prior,
                                 expand_isotropic=expand_uniform)
    assert circuit == preserved
    assert revised != circuit
    assert revised.without_noise() == circuit.without_noise()

    original_steps = list(circuit.flattened())
    data = data_targets(original_steps)
    revised_steps = iter(revised.flattened())
    replacements = 0
    for op in original_steps:
        current = next(revised_steps)
        if op.name != "DEPOLARIZE1" or op.targets_copy() != data:
            assert current == op
            continue
        replacements += 1
        axes = [current, next(revised_steps), next(revised_steps)]
        assert [axis.name for axis in axes] == ["X_ERROR", "Y_ERROR", "Z_ERROR"]
        assert all(axis.targets_copy() == data for axis in axes)
        probabilities = categorical_probabilities([axis.gate_args_copy()[0] for axis in axes])
        np.testing.assert_allclose(probabilities, [1 - p, *(p * np.array(prior.ratios))], atol=1e-12)
    assert next(revised_steps, None) is None
    assert replacements == rounds


def parity_expectation(dem, parity_sets):
    """Return E[(-1)^parity] for arbitrary sets of detector/observable IDs."""
    result = np.ones(len(parity_sets))
    for op in dem.flattened():
        if op.type != "error":
            continue
        symptoms = set()
        for target in op.targets_copy():
            if target.is_separator():
                continue
            kind = "D" if target.is_relative_detector_id() else "L"
            symptoms.symmetric_difference_update({(kind, target.val)})
        for index, group in enumerate(parity_sets):
            if len(group & symptoms) % 2:
                result[index] *= 1 - 2 * op.args_copy()[0]
    return result


@pytest.mark.parametrize("basis", ["x", "z"])
def test_prior_dem_matches_independent_fault_propagation(basis):
    p = 0.019
    prior = bath_prior_from_profile(profile())
    circuit = memory_circuit(basis, 3, 1, p)
    flattened = circuit.flattened()
    steps = list(flattened)
    data = data_targets(steps)
    site = next(i for i, op in enumerate(steps) if op.name == "DEPOLARIZE1" and op.targets_copy() == data)

    before, after = flattened[:site], flattened[site + 1:]
    rest = before + after
    ids = [("D", i) for i in range(circuit.num_detectors)]
    ids += [("L", i) for i in range(circuit.num_observables)]
    parity_sets = [set(group) for size in (1, 2) for group in itertools.combinations(ids, size)]
    parity = parity_expectation(rest.detector_error_model(), parity_sets)

    for qubit in data:
        probability_of_flip = np.zeros(len(parity_sets))
        for pauli, ratio in zip("XYZ", prior.ratios, strict=True):
            # Propagate one injected Pauli without any of the original noise.
            propagated = before.without_noise()
            propagated.append(pauli + "_ERROR", [qubit], 1)
            propagated += after.without_noise()
            detectors, observables = propagated.compile_detector_sampler(seed=43).sample(
                1, separate_observables=True
            )
            symptoms = {key for key, value in zip(ids, np.r_[detectors[0], observables[0]], strict=True)
                        if value}
            probability_of_flip += p * ratio * np.array([len(group & symptoms) % 2 for group in parity_sets])
        parity *= 1 - 2 * probability_of_flip

    result = bath_prior_dem(circuit, basis=basis, distance=3, rounds=1, p=p, prior=prior)
    np.testing.assert_allclose(parity_expectation(result, parity_sets), parity, atol=2e-12)
