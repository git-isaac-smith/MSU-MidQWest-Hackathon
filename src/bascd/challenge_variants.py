"""Fixed additions to Stim's simple_uniform rotated-memory circuit."""

import stim

from .circuits import memory_circuit


VARIANT_NAMES = (
    "ordinary",
    "final_round_measurement",
    "measurement_burst",
    "spatial_correlated",
)
VARIANT_VERSION = 1
FINAL_ROUND_FLIP_MULTIPLIER = 4
MEASUREMENT_BURST_MULTIPLIER = 2
SPATIAL_PAIR_MULTIPLIER = 0.5


def _neighbor_pairs(
    targets: list[int], coordinates: dict[int, list[float]], offsets: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    at = {tuple(coordinates[target]): target for target in targets}
    pairs = []
    for (x, y), first in sorted(at.items()):
        for dx, dy in offsets:
            second = at.get((x + dx, y + dy))
            if second is not None:
                pairs.append((first, second))
    return pairs


def generate_variant_circuit(
    variant: str, *, basis: str = "z", distance: int, rounds: int, p: float,
) -> stim.Circuit:
    """Generate a benchmark with identical detector/observable definitions across variants.

    Final-round measurement adds independent ancilla flips with probability 4p immediately
    before the last syndrome extraction's readout/reset. Measurement bursts add a joint
    X flip with probability 2p on each diagonal nearest-neighbor ancilla pair immediately
    before the middle round's readout/reset (round (rounds + 1) // 2, one-indexed).
    Spatial correlation adds independent joint XX and ZZ events of probability p/2 per
    horizontal/vertical nearest-neighbor data pair before the first Hadamards of each round.
    Each event augments the ordinary noise and its probability saturates at 1/2.
    """
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown circuit variant {variant!r}")
    circuit = memory_circuit(basis.removeprefix("memory_"), distance, rounds, p)
    if variant == "ordinary" or p == 0:
        return circuit

    coordinates = circuit.get_final_qubit_coordinates()
    data_qubits = [qubit for qubit, (x, y) in coordinates.items() if x % 2 and y % 2]
    data_pairs = _neighbor_pairs(data_qubits, coordinates, ((2, 0), (0, 2)))
    result = stim.Circuit()
    round_number = 1
    started_round = False
    for instruction in circuit.flattened():
        if instruction.name == "H" and not started_round:
            started_round = True
            if variant == "spatial_correlated":
                for first, second in data_pairs:
                    for pauli in (stim.target_x, stim.target_z):
                        result.append(
                            "E", [pauli(first), pauli(second)],
                            min(0.5, SPATIAL_PAIR_MULTIPLIER * p),
                        )
        if instruction.name == "MR":
            ancillas = [target.value for target in instruction.targets_copy()]
            if variant == "final_round_measurement" and round_number == rounds:
                result.append("X_ERROR", ancillas, min(0.5, FINAL_ROUND_FLIP_MULTIPLIER * p))
            if variant == "measurement_burst" and round_number == (rounds + 1) // 2:
                for first, second in _neighbor_pairs(ancillas, coordinates, ((2, 2), (2, -2))):
                    result.append(
                        "E", [stim.target_x(first), stim.target_x(second)],
                        min(0.5, MEASUREMENT_BURST_MULTIPLIER * p),
                    )
            round_number += 1
            started_round = False
        result.append(instruction)
    return result
