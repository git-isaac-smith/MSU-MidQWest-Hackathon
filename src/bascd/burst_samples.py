"""Sample the response of a rotated Stim Z memory to added data-qubit X flips."""

import numpy as np
import stim


class BurstSampler:
    """Propagate X flips inserted immediately before the first H of each round.

    Each column of an input flip mask corresponds to a data qubit, ordered as
    in the final Z measurement. Coordinates are in data-qubit lattice spacings.
    The response stores little-endian detector bytes followed by one logical
    byte for each (round, data qubit) pair.
    """

    def __init__(self, circuit: stim.Circuit, rounds: int):
        if rounds < 1 or circuit.num_observables != 1:
            raise ValueError("expected at least one round and one logical observable")

        noiseless = circuit.without_noise().flattened()
        readouts = [instruction for instruction in noiseless if instruction.name == "M"]
        if len(readouts) != 1:
            raise ValueError("expected one final Z data-qubit readout")
        self.data_qubits = tuple(target.value for target in readouts[0].targets_copy())
        self.rounds = rounds
        self.n_sites = len(self.data_qubits)

        coords = noiseless.get_final_qubit_coordinates()
        if any(len(coords.get(qubit, ())) != 2 for qubit in self.data_qubits):
            raise ValueError("expected two-dimensional coordinates for data qubits")
        self.coordinates = np.array([coords[qubit] for qubit in self.data_qubits]) / 2
        offsets = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        self.distances = np.linalg.norm(offsets, axis=2)
        self.full_response = self._build_response(noiseless)
        self.full_response.setflags(write=False)

        # One injection affects only a few bytes even in a long memory.
        width = np.count_nonzero(self.full_response, axis=2).max()
        self._response_columns = np.zeros((rounds, self.n_sites, width), dtype=np.intp)
        self._response_values = np.zeros((rounds, self.n_sites, width), dtype=np.uint8)
        for round_index, site_index in np.ndindex(rounds, self.n_sites):
            row = self.full_response[round_index, site_index]
            columns = np.flatnonzero(row)
            self._response_columns[round_index, site_index, :len(columns)] = columns
            self._response_values[round_index, site_index, :len(columns)] = row[columns]

    def kernel(self, xi: float) -> np.ndarray:
        """Return probability of an X flip for each (burst center, site)."""
        if not np.isfinite(xi) or xi <= 0:
            raise ValueError("xi must be finite and positive")
        return 0.5 * np.exp(-self.distances / xi)

    def _build_response(self, circuit: stim.Circuit) -> np.ndarray:
        # Each instance has one injection, so its frame is empty at its slot.
        sim = stim.FlipSimulator(
            batch_size=self.rounds * self.n_sites,
            num_qubits=circuit.num_qubits,
            disable_stabilizer_randomization=True,
        )
        round_index = 0
        waiting_for_first_h = True
        for instruction in circuit:
            if instruction.name == "H" and waiting_for_first_h:
                if round_index >= self.rounds:
                    raise ValueError("circuit has more round starts than requested")
                for site_index, qubit in enumerate(self.data_qubits):
                    sim.set_pauli_flip(
                        "X", qubit_index=qubit, instance_index=round_index * self.n_sites + site_index
                    )
                waiting_for_first_h = False
            elif instruction.name == "MR":
                if waiting_for_first_h:
                    raise ValueError("expected a Hadamard layer before each ancilla readout")
                waiting_for_first_h = True
                round_index += 1
            sim.do(instruction)
        if round_index != self.rounds or not waiting_for_first_h:
            raise ValueError("circuit round count does not match the requested rounds")

        detectors = np.packbits(sim.get_detector_flips().T, axis=1, bitorder="little")
        logical = sim.get_observable_flips().T.astype(np.uint8)
        response = np.concatenate((detectors, logical), axis=1)
        return response.reshape(self.rounds, self.n_sites, response.shape[1])

    def delta(self, flips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return packed detector flips and a logical flip per shot.

        The input has shape (shots, rounds, data qubits) and contains the XOR
        of all burst flips at each site. The resulting arrays can be XORed
        with an independent Stim sample of the original circuit's noise.
        """
        flips = np.asarray(flips)
        if flips.ndim != 3 or flips.shape[1:] != (self.rounds, self.n_sites):
            raise ValueError("expected flip shape (shots, rounds, data qubits)")
        response = np.zeros((len(flips), self.full_response.shape[-1]), dtype=np.uint8)
        shots, rounds, sites = np.nonzero(flips)
        np.bitwise_xor.at(
            response,
            (shots[:, None], self._response_columns[rounds, sites]),
            self._response_values[rounds, sites],
        )
        return response[:, :-1], response[:, -1].astype(bool)


def poisson_parity_probability(expected_flips: np.ndarray | float) -> np.ndarray | np.float64:
    """Return the probability of odd parity for a Poisson number of flips."""
    return -0.5 * np.expm1(-2.0 * np.asarray(expected_flips))
