import numpy as np
import pytest

from bascd.burst_samples import BurstSampler, poisson_parity_probability
from bascd.circuits import memory_circuit


def _direct_sample(circuit, data_qubits, flips):
    # In the generated memory the first TICK after each reset or ancilla
    # readout starts the next round. Force the requested flips at that tick.
    forced = circuit.without_noise().flattened()
    instructions = forced.copy()
    forced.clear()
    round_index = 0
    at_boundary = False
    for instruction in instructions:
        forced.append(instruction)
        if instruction.name in ("R", "MR"):
            at_boundary = True
        if instruction.name == "TICK" and at_boundary:
            targets = [qubit for qubit, flip in zip(data_qubits, flips[round_index]) if flip]
            if targets:
                forced.append("X_ERROR", targets, 1.0)
            at_boundary = False
            round_index += 1
    detectors, observables = forced.compile_detector_sampler().sample(
        1, separate_observables=True, bit_packed=True
    )
    return detectors[0], observables[0, 0]


@pytest.mark.parametrize("distance,rounds", [(3, 1), (3, 4), (5, 3)])
def test_single_flip_response_matches_direct_stim(distance, rounds):
    circuit = memory_circuit("z", distance, rounds, 0.01)
    sampler = BurstSampler(circuit, rounds)
    assert sampler.n_sites == distance**2
    flips = np.eye(rounds * sampler.n_sites, dtype=bool).reshape(-1, rounds, sampler.n_sites)

    detectors, logical = sampler.delta(flips)
    expected_detectors, expected_logical = zip(
        *[_direct_sample(circuit, sampler.data_qubits, row) for row in flips]
    )
    np.testing.assert_array_equal(detectors, expected_detectors)
    np.testing.assert_array_equal(logical, expected_logical)
    assert logical.any()


def test_burst_overlap_cancels_and_responses_add_by_xor():
    rounds = 3
    circuit = memory_circuit("z", 3, rounds, 0.0)
    sampler = BurstSampler(circuit, rounds)
    first = np.zeros((rounds, sampler.n_sites), dtype=bool)
    first[0, [0, 1]] = True
    first[1, [0, 3]] = True
    second = np.zeros_like(first)
    second[0, [1, 2]] = True
    second[2, 1] = True
    flips = np.stack([first, second, first ^ second, first ^ first])

    detectors, logical = sampler.delta(flips)
    expected_detectors, expected_logical = _direct_sample(circuit, sampler.data_qubits, flips[2])
    np.testing.assert_array_equal(detectors[2], expected_detectors)
    assert logical[2] == expected_logical
    np.testing.assert_array_equal(detectors[0] ^ detectors[1], detectors[2])
    assert (logical[0] ^ logical[1]) == logical[2]
    assert not detectors[3].any()
    assert not logical[3]


def test_planar_kernel_and_poisson_parity():
    sampler = BurstSampler(memory_circuit("z", 3, 1, 0.0), 1)
    corner, neighbor, far_corner = 0, 1, 8
    np.testing.assert_allclose(sampler.distances[corner, [corner, neighbor, far_corner]], [0, 1, np.sqrt(8)])
    kernel = sampler.kernel(2.0)
    np.testing.assert_allclose(kernel[corner, [corner, neighbor]], [0.5, 0.5 * np.exp(-0.5)])
    np.testing.assert_allclose(kernel, kernel.T)
    assert kernel[4].sum() > kernel[corner].sum()
    expected_flips = np.array([0, 1e-20, 0.5, 100])
    np.testing.assert_allclose(
        poisson_parity_probability(expected_flips), [0, 1e-20, (1 - np.exp(-1)) / 2, 0.5], atol=0
    )
