"""Draw labeled X-fault histories from Stim's rotated Z-memory circuit."""

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

import numpy as np
import stim


ROUNDS = 5
_BATCH_SIZE = 8192


@dataclass(frozen=True)
class Geometry:
    check_mask: np.ndarray
    data_ids: np.ndarray
    circuit: stim.Circuit


@dataclass(frozen=True)
class SampledShots:
    observed: np.ndarray
    clean: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class _Layout:
    geometry: Geometry
    initialization: stim.Circuit
    rounds: Sequence[tuple[stim.Circuit, stim.CircuitInstruction]]
    ancilla_ids: np.ndarray
    check_records: np.ndarray
    check_reference: np.ndarray


def _layout(distance: int) -> _Layout:
    distance = index(distance)
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer of at least 3")

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=ROUNDS
    )
    coordinates = circuit.get_final_qubit_coordinates()
    data_ids = np.empty((distance, distance), dtype=int)
    for qubit, (x, y) in coordinates.items():
        x, y = int(x), int(y)
        if x % 2 and y % 2:
            data_ids[(y - 1) // 2, (x - 1) // 2] = qubit

    initialization = stim.Circuit()
    gates = stim.Circuit()
    rounds = []
    initialized = False
    for instruction in circuit.flattened():
        if not initialized:
            initialization.append(instruction)
            initialized = instruction.name == "R"
        elif instruction.name == "MR":
            rounds.append((gates, instruction))
            gates = stim.Circuit()
            if len(rounds) == ROUNDS:
                break
        elif instruction.name != "DETECTOR":
            gates.append(instruction)

    ancilla_ids = np.asarray([target.value for target in rounds[0][1].targets_copy()])
    # Stim emits first-round detectors only for Z checks in the Z-memory circuit.
    z_check_sites = {
        (int(y) // 2, int(x) // 2)
        for x, y, time in circuit.get_detector_coordinates().values()
        if time == 0
    }
    check_mask = np.zeros((distance + 1, distance + 1), dtype=bool)
    record_for_site = {}
    for record, qubit in enumerate(ancilla_ids):
        x, y = coordinates[qubit]
        site = (int(y) // 2, int(x) // 2)
        if site in z_check_sites:
            check_mask[site] = True
            record_for_site[site] = record
    check_offsets = np.asarray([record_for_site[site] for site in sorted(z_check_sites)])
    check_records = np.arange(ROUNDS)[:, None] * len(ancilla_ids) + check_offsets

    return _Layout(
        Geometry(check_mask, data_ids, circuit),
        initialization,
        rounds,
        ancilla_ids,
        check_records,
        circuit.reference_sample()[check_records],
    )


def geometry(distance: int) -> Geometry:
    """Return the noiseless circuit and its row-major data and check grids."""
    return _layout(distance).geometry


def _simulate_batch(
    layout: _Layout, data_faults: np.ndarray, readout_faults: np.ndarray
) -> SampledShots:
    # Fault arrays use Stim's qubit-major broadcast order within each round.
    shots = data_faults.shape[-1]
    shape = (shots, ROUNDS, *layout.geometry.check_mask.shape)
    num_qubits = layout.geometry.circuit.num_qubits
    data_ids = layout.geometry.data_ids.ravel()

    observed_sim, clean_sim = (
        stim.FlipSimulator(
            batch_size=shots, num_qubits=num_qubits, disable_stabilizer_randomization=True
        )
        for _ in range(2)
    )
    for simulator in (observed_sim, clean_sim):
        simulator.do(layout.initialization)

    data_mask = np.zeros((num_qubits, shots), dtype=bool)
    readout_mask = np.zeros_like(data_mask)
    for time, (gates, measure_reset) in enumerate(layout.rounds):
        data_mask[data_ids] = data_faults[time]
        readout_mask[layout.ancilla_ids] = readout_faults[time]
        for simulator in (observed_sim, clean_sim):
            simulator.broadcast_pauli_errors(pauli="X", mask=data_mask)
            simulator.do(gates)
        observed_sim.broadcast_pauli_errors(pauli="X", mask=readout_mask)
        for simulator in (observed_sim, clean_sim):
            simulator.do(measure_reset)

    outcomes = []
    for simulator in (observed_sim, clean_sim):
        flips = simulator.get_measurement_flips()[layout.check_records]
        checks = flips ^ layout.check_reference[:, :, None]
        grid = np.zeros(shape, dtype=bool)
        grid[:, :, layout.geometry.check_mask] = checks.transpose(2, 0, 1)
        outcomes.append(grid)

    targets = np.logical_xor.reduce(data_faults, axis=0).T
    targets = targets.reshape(shots, *layout.geometry.data_ids.shape)
    return SampledShots(*outcomes, targets)


def sample_shots(
    distance: int, shots: int, p_data: float, p_readout: float, seed: int
) -> SampledShots:
    """Sample five rounds of raw Z-check outcomes and final data X parities.

    Each round independently applies X to each data qubit with probability
    p_data before the first stabilizer gate and X to each ancilla immediately
    before its Z-basis measurement/reset with probability p_readout. State
    preparation, Clifford gates, and resets are ideal. Final data measurements
    are excluded from the features. Inactive check-grid sites are zero.

    The clean outcomes use the same data faults with ideal ancilla readout and
    are intended for evaluation only. All arrays are boolean. Observed and clean
    have shape (shots, 5, distance + 1, distance + 1), and targets have shape
    (shots, distance, distance).
    """
    shots = index(shots)
    if shots < 0:
        raise ValueError("shots must be nonnegative")
    if not (0 <= p_data <= 1 and 0 <= p_readout <= 1):
        raise ValueError("fault probabilities must be between zero and one")

    layout = _layout(distance)
    shape = (shots, ROUNDS, *layout.geometry.check_mask.shape)
    observed = np.empty(shape, dtype=bool)
    clean = np.empty_like(observed)
    targets = np.empty((shots, *layout.geometry.data_ids.shape), dtype=bool)
    data_seed, readout_seed = np.random.SeedSequence(seed).spawn(2)
    data_rng = np.random.default_rng(data_seed)
    readout_rng = np.random.default_rng(readout_seed)
    for start in range(0, shots, _BATCH_SIZE):
        stop = min(start + _BATCH_SIZE, shots)
        batch_size = stop - start
        data_faults = data_rng.random((ROUNDS, layout.geometry.data_ids.size, batch_size)) < p_data
        readout_faults = readout_rng.random((ROUNDS, len(layout.ancilla_ids), batch_size)) < p_readout
        batch = _simulate_batch(layout, data_faults, readout_faults)
        observed[start:stop] = batch.observed
        clean[start:stop] = batch.clean
        targets[start:stop] = batch.targets
    return SampledShots(observed, clean, targets)
