"""Compare frozen site calls and standard MWPM on a common Stim Z memory."""

from dataclasses import dataclass

import numpy as np
import pymatching
import stim

from stim_data import ROUNDS, Geometry, geometry


def _stim_circuit(layout: Geometry, p_data: float, p_readout: float) -> stim.Circuit:
    if not (0 <= p_data <= 1 and 0 <= p_readout <= 1):
        raise ValueError("fault probabilities must be between zero and one")

    circuit = stim.Circuit()
    completed_rounds = 0
    start_round = True
    for instruction in layout.circuit.flattened():
        if instruction.name == "DETECTOR":
            x, y, _ = instruction.gate_args_copy()
            if not layout.check_mask[int(y) // 2, int(x) // 2]:
                continue
        if instruction.name == "TICK" and start_round and completed_rounds < ROUNDS:
            circuit.append("X_ERROR", layout.data_ids.ravel(), p_data)
            start_round = False
        if instruction.name == "MR":
            circuit.append("X_ERROR", instruction.targets_copy(), p_readout)
            completed_rounds += 1
            start_round = True
        circuit.append(instruction)
    return circuit


def stim_circuit(distance: int, p_data: float, p_readout: float) -> stim.Circuit:
    """Build the archived noise model on Stim's rotated Z-memory circuit.

    Data X faults precede each syndrome round and ancilla X readout faults
    precede each MR. Final data measurements are ideal. Only Z detectors are
    retained. The omitted X-check detector differences depend on independent
    X-ancilla readout faults and carry no information about the Z logical bit.
    """
    return _stim_circuit(geometry(distance), p_data, p_readout)


def _terminal_support(circuit: stim.Circuit, layout: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Extract Z-check and logical observable support from Stim's final M records."""
    sites = {tuple(site): col for col, site in enumerate(np.argwhere(layout.check_mask))}
    data_columns = {qubit: col for col, qubit in enumerate(layout.data_ids.ravel())}
    checks = np.zeros((len(sites), len(data_columns)), dtype=np.uint8)
    observable = np.zeros(len(data_columns), dtype=np.uint8)
    final_record_columns = []
    for instruction in circuit:
        if instruction.name == "M":
            final_record_columns = [data_columns[target.value] for target in instruction.targets_copy()]
        if instruction.name == "DETECTOR" and final_record_columns:
            x, y, _ = instruction.gate_args_copy()
            check = checks[sites[(int(y) // 2, int(x) // 2)]]
            for target in instruction.targets_copy():
                if target.value >= -len(final_record_columns):
                    check[final_record_columns[target.value]] ^= 1
        if instruction.name == "OBSERVABLE_INCLUDE":
            for target in instruction.targets_copy():
                observable[final_record_columns[target.value]] ^= 1
    return checks, observable


@dataclass(frozen=True)
class LogicalReference:
    circuit: stim.Circuit
    check_mask: np.ndarray
    data_ids: np.ndarray
    _checks: np.ndarray
    _observable: np.ndarray
    _detector_times: np.ndarray
    _detector_checks: np.ndarray
    _mwpm: pymatching.Matching
    _completion: pymatching.Matching
    _detector_shift: np.ndarray
    _observable_shift: bool

    def _data_rows(self, data_bits: np.ndarray) -> np.ndarray:
        data_bits = np.asarray(data_bits, dtype=np.uint8)
        if data_bits.ndim != 3 or data_bits.shape[1:] != self.data_ids.shape:
            raise ValueError("expected a shot-by-row-by-column data parity array")
        return data_bits.reshape(len(data_bits), self.data_ids.size)

    def _terminal_rows(self, checks: np.ndarray, shots: int) -> np.ndarray:
        checks = np.asarray(checks, dtype=bool)
        if checks.shape != (shots, len(self._checks)):
            raise ValueError("expected one terminal parity per shot and active Z check")
        return checks

    def terminal_checks(self, data_bits: np.ndarray) -> np.ndarray:
        """At the sampler boundary, project ideal terminal data onto public Z checks.

        In this noise model, archive targets and ideal terminal measurements
        agree on every Z stabilizer parity. Only these parities leave the
        sampler for decoding. Check columns follow the row-major active mask.
        """
        return (self._data_rows(data_bits) @ self._checks.T & 1).astype(bool)

    def detector_records(self, observed: np.ndarray, terminal_checks: np.ndarray) -> np.ndarray:
        """Assemble public Z detector records in the circuit's Stim order."""
        observed = np.asarray(observed, dtype=bool)
        if observed.ndim != 4 or observed.shape[1:] != (ROUNDS, *self.check_mask.shape):
            raise ValueError("expected five observed check grids per shot")
        terminal = self._terminal_rows(terminal_checks, len(observed))
        raw = observed[:, :, self.check_mask]
        differences = raw.copy()
        differences[:, 1:] ^= raw[:, :-1]
        differences = np.concatenate((differences, (raw[:, -1] ^ terminal)[:, None]), axis=1)
        return differences[:, self._detector_times, self._detector_checks]

    def public_checks(self, detectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recover the raw Z checks and ideal terminal checks from sampled detectors.

        Stim's Z checks start at zero. Cumulative XOR of successive detection
        events recovers each check outcome, including the terminal parity.
        """
        detectors = np.asarray(detectors, dtype=bool)
        if detectors.ndim != 2 or detectors.shape[1] != len(self._detector_times):
            raise ValueError("expected one column per circuit Z detector")
        checks = np.zeros((len(detectors), ROUNDS + 1, len(self._checks)), dtype=bool)
        checks[:, self._detector_times, self._detector_checks] = detectors
        checks = np.logical_xor.accumulate(checks, axis=1)
        observed = np.zeros((len(detectors), ROUNDS, *self.check_mask.shape), dtype=bool)
        observed[:, :, self.check_mask] = checks[:, :-1]
        return observed, checks[:, -1]

    def observable_parity(self, data_bits: np.ndarray) -> np.ndarray:
        """Calculate Stim's logical Z parity of a correction or evaluation data."""
        return (self._data_rows(data_bits) @ self._observable & 1).astype(bool)

    def mwpm_parity(self, detectors: np.ndarray) -> np.ndarray:
        """Predict observable flips using ordinary PyMatching on the Stim DEM."""
        predictions = self._mwpm.decode_batch(np.asarray(detectors, dtype=bool) ^ self._detector_shift)
        return predictions[:, 0].astype(bool) ^ self._observable_shift

    def complete_fft(self, site_calls: np.ndarray, terminal_checks: np.ndarray) -> np.ndarray:
        """XOR frozen hard calls with a minimum-weight residual spatial correction.

        Each data-site edge has fixed unit weight. Matching uses only the
        difference between the calls' syndrome and the public terminal checks.
        The final correction matches the measured terminal syndrome by design.
        """
        site_calls = np.asarray(site_calls, dtype=bool)
        terminal = self._terminal_rows(terminal_checks, len(site_calls))
        residual = self.terminal_checks(site_calls) ^ terminal
        completion = self._completion.decode_batch(residual).reshape(site_calls.shape)
        return site_calls ^ completion.astype(bool)


def build_reference(distance: int, p_data: float, p_readout: float) -> LogicalReference:
    """Build the shared observation map, the Stim baseline, and the FFT completion."""
    layout = geometry(distance)
    circuit = _stim_circuit(layout, p_data, p_readout)
    checks, observable = _terminal_support(circuit, layout)
    sites = {tuple(site): col for col, site in enumerate(np.argwhere(layout.check_mask))}
    times, columns = [], []
    for x, y, time in circuit.get_detector_coordinates().values():
        times.append(int(time))
        columns.append(sites[(int(y) // 2, int(x) // 2)])

    matching_circuit = circuit
    detector_shift = np.zeros(circuit.num_detectors, dtype=bool)
    observable_shift = False
    if 1 in (p_data, p_readout):
        # A certain fault has a known detector/observable shift and would
        # otherwise give the matching graph an infinite log-odds weight.
        deterministic = _stim_circuit(layout, p_data == 1, p_readout == 1)
        detectors, observables = deterministic.compile_detector_sampler().sample(1, separate_observables=True)
        detector_shift = detectors[0]
        observable_shift = bool(observables[0, 0])
        matching_circuit = _stim_circuit(layout, p_data if p_data < 1 else 0, p_readout if p_readout < 1 else 0)
    dem = matching_circuit.detector_error_model(decompose_errors=True)
    mwpm = pymatching.Matching.from_detector_error_model(dem)
    mwpm.ensure_num_fault_ids(circuit.num_observables)
    completion = pymatching.Matching.from_check_matrix(checks, weights=1.0)
    completion.ensure_num_fault_ids(layout.data_ids.size)
    return LogicalReference(
        circuit, layout.check_mask, layout.data_ids, checks, observable,
        np.asarray(times), np.asarray(columns), mwpm, completion,
        detector_shift, observable_shift,
    )
