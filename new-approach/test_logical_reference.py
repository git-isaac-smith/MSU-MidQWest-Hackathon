import numpy as np
import pymatching
import pytest
import stim

from logical_reference import build_reference
from stim_data import ROUNDS, _layout, _simulate_batch


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_public_checks_match_compiled_stim_measurements(distance):
    layout = _layout(distance)
    reference = build_reference(distance, p_data=0.08, p_readout=0.15)
    circuit = reference.circuit
    measurements = circuit.compile_sampler(seed=27).sample(128)
    detectors, observable = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True
    )
    observed, terminal = reference.public_checks(detectors)
    targets = measurements[:, -reference.data_ids.size :].reshape(128, distance, distance)

    np.testing.assert_array_equal(observed[:, :, reference.check_mask], measurements[:, layout.check_records])
    assert not observed[:, :, ~reference.check_mask].any()
    np.testing.assert_array_equal(terminal, reference.terminal_checks(targets))
    np.testing.assert_array_equal(detectors, reference.detector_records(observed, terminal))
    np.testing.assert_array_equal(observable[:, 0], reference.observable_parity(targets))
    assert observable.any()

    matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    np.testing.assert_array_equal(reference.mwpm_parity(detectors), matching.decode_batch(detectors)[:, 0])


def test_simulated_archive_faults_match_explicit_circuit():
    layout = _layout(3)
    reference = build_reference(3, p_data=0.008, p_readout=0.04)
    # Include data across all three rows, repeated data faults, an X-ancilla
    # readout error, and a last-round Z-ancilla readout error.
    data_faults = {0: [1], 1: [8], 2: [10], 3: [10], 4: [15]}
    readout_faults = {0: [11], 2: [9], 4: [14]}
    data = np.zeros((ROUNDS, 9, 1), dtype=bool)
    readout = np.zeros((ROUNDS, 8, 1), dtype=bool)
    for turn, qubits in data_faults.items():
        data[turn, np.isin(layout.geometry.data_ids.ravel(), qubits), 0] = True
    for turn, qubits in readout_faults.items():
        readout[turn, np.isin(layout.ancilla_ids, qubits), 0] = True
    shot = _simulate_batch(layout, data, readout)

    circuit = stim.Circuit()
    current_round = 0
    before_round = True
    for instruction in layout.geometry.circuit.flattened():
        if before_round and instruction.name == "TICK":
            circuit.append("X_ERROR", data_faults[current_round], 1)
            before_round = False
        if instruction.name == "MR":
            if current_round in readout_faults:
                circuit.append("X_ERROR", readout_faults[current_round], 1)
            current_round += 1
            before_round = True
        circuit.append(instruction)

    detectors, observable = circuit.compile_detector_sampler(seed=47).sample(1, separate_observables=True)
    coordinates = circuit.get_detector_coordinates()
    active_detectors = [number for number, (x, y, _) in coordinates.items()
                        if reference.check_mask[int(y) // 2, int(x) // 2]]
    terminal = reference.terminal_checks(shot.targets)
    np.testing.assert_array_equal(reference.detector_records(shot.observed, terminal), detectors[:, active_detectors])
    np.testing.assert_array_equal(reference.observable_parity(shot.targets), observable[:, 0])
    assert observable[0, 0]


def test_completion_uses_only_fft_calls_and_public_terminal_syndrome():
    reference = build_reference(3, p_data=0.008, p_readout=0.04)
    detectors, observable = reference.circuit.compile_detector_sampler(seed=45).sample(
        256, separate_observables=True
    )
    _, terminal = reference.public_checks(detectors)
    site_calls = np.random.default_rng(88).random((256, 3, 3)) < 0.3
    correction = reference.complete_fft(site_calls, terminal)
    np.testing.assert_array_equal(reference.terminal_checks(correction), terminal)
    assert reference.mwpm_parity(detectors).shape == observable[:, 0].shape

    # A logical X string has exactly zero syndrome. Completion must preserve
    # its action on the logical bit, since the residual is identical.
    logical_x = np.zeros((3, 3), dtype=bool)
    logical_x[:, 0] = True
    altered = reference.complete_fft(site_calls ^ logical_x, terminal)
    np.testing.assert_array_equal(altered, correction ^ logical_x)
    np.testing.assert_array_equal(reference.observable_parity(altered), ~reference.observable_parity(correction))


def test_no_faults_decode_to_no_correction():
    reference = build_reference(3, p_data=0, p_readout=0)
    detectors, observable = reference.circuit.compile_detector_sampler().sample(4, separate_observables=True)
    observed, terminal = reference.public_checks(detectors)
    assert not (detectors.any() or observable.any() or observed.any() or terminal.any())
    assert not reference.mwpm_parity(detectors).any()
    assert not reference.complete_fft(np.zeros((4, 3, 3), dtype=bool), terminal).any()


@pytest.mark.parametrize("p_data,p_readout", [(1, 1), (1, 0.04), (0, 1)])
def test_mwpm_accounts_for_certain_faults(p_data, p_readout):
    reference = build_reference(3, p_data, p_readout)
    detectors, observable = reference.circuit.compile_detector_sampler(seed=71).sample(32, separate_observables=True)
    np.testing.assert_array_equal(reference.mwpm_parity(detectors), observable[:, 0])
