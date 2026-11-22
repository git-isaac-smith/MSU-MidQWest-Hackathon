"""Approximate logical inference by keeping nearby ideal check syndromes."""

from collections import Counter
from itertools import combinations
from operator import index

import numpy as np

from logical_reference import _stim_circuit, _terminal_support
from stim_data import ROUNDS, geometry


_MIX_CHUNK = 1 << 18


def _reduce(word, basis):
    """Express a joint parity word in the current independent-fault basis."""
    coordinates = 0
    while word:
        pivot = word.bit_length() - 1
        if pivot not in basis:
            break
        row, row_coordinates = basis[pivot]
        word ^= row
        coordinates ^= row_coordinates
    return word, coordinates


def _mix_dependent_fault(weights, shift, probability):
    # Each XOR pair has one index with the shift's pivot bit clear. Insert
    # that zero bit into the compressed indices so each pair is mixed once.
    pivot = 1 << (shift.bit_length() - 1)
    low_mask = pivot - 1
    pair_count = len(weights) // 2
    offsets = np.arange(min(_MIX_CHUNK, pair_count), dtype=np.intp)
    for start in range(0, pair_count, _MIX_CHUNK):
        compressed = start + offsets[:pair_count - start]
        left = ((compressed & ~low_mask) << 1) | (compressed & low_mask)
        right = left ^ shift
        previous_left, previous_right = weights[left], weights[right]
        weights[left] = (1 - probability) * previous_left + probability * previous_right
        weights[right] = probability * previous_left + (1 - probability) * previous_right


def _one_round(shifts, check_count, p_data):
    basis = {}
    independent, dependent = [], []
    for shift, copies in Counter(map(int, shifts)).items():
        if shift == 0:
            continue
        probability = p_data
        for _ in range(copies - 1):
            probability = p_data + (1 - 2 * p_data) * probability
        remainder, coordinates = _reduce(shift, basis)
        if remainder:
            basis[remainder.bit_length() - 1] = remainder, coordinates ^ (1 << len(basis))
            independent.append(probability)
        else:
            dependent.append((coordinates, probability))
    if len(basis) != check_count + 1:
        raise ValueError("data faults do not span the joint syndrome and logical parity")

    weights = np.empty(1 << len(basis), dtype=np.float64)
    weights[0] = 1
    for bit, probability in enumerate(independent):
        size = 1 << bit
        np.multiply(weights[:size], probability, out=weights[size:2 * size])
        weights[:size] *= 1 - probability
    for shift, probability in dependent:
        _mix_dependent_fault(weights, shift, probability)
    coordinate_shifts = np.array([_reduce(1 << bit, basis)[1] for bit in range(check_count + 1)], dtype=np.uint32)
    return weights, coordinate_shifts


class LocalReadoutPosterior:
    """Condition the five-round posterior on readout errors within a radius.

    The restriction applies separately to each of the first four rounds.
    The final syndrome comes from the terminal detectors. One-round data
    transitions sum over every physical fault pattern in the model.

    If beam_width is supplied, retain up to that many ideal syndromes after
    each noisy check update, based on their forward mass summed over parity.
    """

    def __init__(self, distance: int, p_data: float, p_readout: float, radius: int = 1,
                 beam_width: int | None = None):
        distance, radius = index(distance), index(radius)
        if distance not in (3, 5, 7):
            raise ValueError("local readout posterior supports distances 3, 5, and 7")
        if not (0 < p_data < 0.5 and 0 < p_readout < 0.5):
            raise ValueError("data and readout fault probabilities must lie strictly between zero and one half")
        if beam_width is not None and index(beam_width) < 1:
            raise ValueError("beam_width must be positive")

        layout = geometry(distance)
        self._check_mask = layout.check_mask
        self._check_count = int(layout.check_mask.sum())
        max_radius = self._check_count if distance == 3 else 2
        if not 0 <= radius <= max_radius:
            raise ValueError(f"radius must be between zero and {max_radius} for distance {distance}")

        checks, logical = _terminal_support(_stim_circuit(layout, 0, 0), layout)
        physical_bits = 1 << np.arange(self._check_count, dtype=np.intp)
        joint_shifts = physical_bits @ checks + (1 << self._check_count) * logical.astype(np.intp)
        self._transition, shifts = _one_round(joint_shifts, self._check_count, p_data)
        self._check_shifts, self._odd_shift = shifts[:-1], shifts[-1]

        masks, likelihoods = [], []
        readout_odds = p_readout / (1 - p_readout)
        for count in range(radius + 1):
            for bits in combinations(self._check_shifts, count):
                masks.append(np.bitwise_xor.reduce(bits, dtype=np.uint32))
                likelihoods.append(readout_odds ** count)
        self._masks = np.array(masks, dtype=np.uint32)
        self._readout = np.array(likelihoods, dtype=np.float64)
        candidate_count = len(self._masks)
        self._beam_width = candidate_count if beam_width is None else min(candidate_count, index(beam_width))
        self._shot_batch = min(512, 524288 // (candidate_count * self._beam_width))

    def _words(self, checks):
        return np.bitwise_xor.reduce(np.where(checks, self._check_shifts, np.uint32(0)), axis=-1)

    def _retain(self, even, odd):
        retained_masks = self._masks[None, :]
        if self._beam_width < len(self._masks):
            keep = np.argpartition(even + odd, -self._beam_width, axis=1)[:, -self._beam_width:]
            even = np.take_along_axis(even, keep, axis=1)
            odd = np.take_along_axis(odd, keep, axis=1)
            retained_masks = self._masks[keep]
        scale = even.sum(axis=1, keepdims=True) + odd.sum(axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError("retained history has zero probability at float64 precision")
        even /= scale
        odd /= scale
        return even, odd, retained_masks

    def _predict_batch(self, observed, terminal):
        delta = observed[:, 0, None] ^ self._masks
        even = self._transition[delta] * self._readout
        odd = self._transition[delta ^ self._odd_shift] * self._readout
        even, odd, retained_masks = self._retain(even, odd)

        for n in range(1, ROUNDS - 1):
            change = observed[:, n - 1] ^ observed[:, n]
            delta = change[:, None, None] ^ retained_masks[:, :, None] ^ self._masks
            same = self._transition[delta]
            flip = self._transition[delta ^ self._odd_shift]
            next_even = np.einsum("bi,bij->bj", even, same) + np.einsum("bi,bij->bj", odd, flip)
            next_odd = np.einsum("bi,bij->bj", odd, same) + np.einsum("bi,bij->bj", even, flip)
            even, odd, retained_masks = self._retain(next_even * self._readout, next_odd * self._readout)

        delta = (observed[:, -1] ^ terminal)[:, None] ^ retained_masks
        same = self._transition[delta]
        flip = self._transition[delta ^ self._odd_shift]
        logical_zero = np.einsum("bi,bi->b", even, same) + np.einsum("bi,bi->b", odd, flip)
        logical_one = np.einsum("bi,bi->b", odd, same) + np.einsum("bi,bi->b", even, flip)
        total = logical_zero + logical_one
        if np.any(total == 0):
            raise ValueError("retained terminal history has zero probability at float64 precision")
        return logical_one / total

    def predict_proba(self, observed: np.ndarray, terminal: np.ndarray) -> np.ndarray:
        """Return P(final logical bit = 1) for independent data and readout noise."""
        observed = np.asarray(observed, dtype=bool)
        terminal = np.asarray(terminal, dtype=bool)
        if observed.ndim != 4 or observed.shape[1:] != (ROUNDS, *self._check_mask.shape):
            raise ValueError("expected five observed check grids per shot")
        if terminal.shape != (len(observed), self._check_count):
            raise ValueError("expected one terminal parity per shot and active Z check")

        posterior = np.empty(len(observed), dtype=np.float64)
        for start in range(0, len(posterior), self._shot_batch):
            part = slice(start, start + self._shot_batch)
            # Given the terminal syndrome, the fifth readout likelihood is
            # a common factor for every history and both logical parities.
            raw = self._words(observed[part, :-1, self._check_mask])
            posterior[part] = self._predict_batch(raw, self._words(terminal[part]))
        return posterior
