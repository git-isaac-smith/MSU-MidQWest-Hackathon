import numpy as np
import pytest
import stim

from stim_data import ROUNDS, _layout, _simulate_batch, geometry, sample_shots


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_geometry_matches_stim_coordinates(distance):
    grid = geometry(distance)
    assert grid.check_mask.shape == (distance + 1, distance + 1)
    assert grid.check_mask.sum() == (distance**2 - 1) // 2
    assert grid.data_ids.shape == (distance, distance)
    coords = grid.circuit.get_final_qubit_coordinates()
    for (row, col), qubit in np.ndenumerate(grid.data_ids):
        assert coords[qubit] == [2 * col + 1, 2 * row + 1]


def test_sample_outcomes_and_labels():
    grid = geometry(5)
    empty = sample_shots(5, 20, 0, 0, 42)
    assert empty.observed.shape == empty.clean.shape == (20, 5, 6, 6)
    assert empty.targets.shape == (20, 5, 5)
    assert empty.observed.dtype == empty.clean.dtype == empty.targets.dtype == bool
    assert not empty.observed.any()
    assert not empty.clean.any()
    assert not empty.targets.any()

    data_only = sample_shots(5, 20, 0.2, 0, 42)
    np.testing.assert_array_equal(data_only.observed, data_only.clean)
    readout = sample_shots(5, 20, 0.2, 1, 42)
    np.testing.assert_array_equal(readout.clean, data_only.clean)
    np.testing.assert_array_equal(readout.targets, data_only.targets)
    np.testing.assert_array_equal(readout.observed, readout.clean ^ grid.check_mask)
    assert readout.targets.any()


def test_known_faults_match_stim_raw_measurements():
    layout = _layout(3)
    data = np.zeros((ROUNDS, 9, 1), dtype=bool)
    readout = np.zeros((ROUNDS, 8, 1), dtype=bool)
    data[1, 4, 0] = data[3, 4, 0] = data[3, 0, 0] = True
    readout[0, np.flatnonzero(layout.ancilla_ids == 11), 0] = True
    readout[2, np.flatnonzero(layout.ancilla_ids == 9), 0] = True
    shot = _simulate_batch(layout, data, readout)

    expected_clean = np.zeros((1, ROUNDS, 4, 4), dtype=bool)
    expected_clean[0, 1:, 1, 1] = True
    expected_clean[0, 1:3, 2, 2] = True
    np.testing.assert_array_equal(shot.clean, expected_clean)
    expected_observed = expected_clean.copy()
    expected_observed[0, 2, 1, 1] ^= True
    np.testing.assert_array_equal(shot.observed, expected_observed)
    expected_target = np.zeros((1, 3, 3), dtype=bool)
    expected_target[0, 0, 0] = True
    np.testing.assert_array_equal(shot.targets, expected_target)

    explicit_circuit = stim.Circuit()
    round_index = 0
    starting_round = True
    for instruction in layout.geometry.circuit.flattened():
        if instruction.name == "TICK" and starting_round:
            if round_index == 1:
                explicit_circuit.append("X_ERROR", [10], 1)
            elif round_index == 3:
                explicit_circuit.append("X_ERROR", [10, 1], 1)
            starting_round = False
        if instruction.name == "MR":
            if round_index == 0:
                explicit_circuit.append("X_ERROR", [11], 1)
            elif round_index == 2:
                explicit_circuit.append("X_ERROR", [9], 1)
            round_index += 1
            starting_round = True
        explicit_circuit.append(instruction)

    # The d=3 Stim MR order is 2, 9, 11, 13, 14, 16, 18, 25.
    samples = explicit_circuit.compile_sampler(seed=27).sample(4)
    z_measurements = samples[:, : ROUNDS * 8].reshape(4, ROUNDS, 8)[:, :, [1, 3, 4, 6]]
    expected_z = shot.observed[:, :, layout.geometry.check_mask]
    np.testing.assert_array_equal(z_measurements, np.repeat(expected_z, 4, axis=0))
