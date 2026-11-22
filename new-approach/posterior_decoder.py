"""Exact logical inference for the controlled independent X-noise experiment."""

from collections import Counter
from math import log, log1p
from operator import index

import numpy as np

from logical_reference import _stim_circuit, _terminal_support
from stim_data import ROUNDS, geometry


class PosteriorDecoder:
    """Condition the final logical Z bit on the public five-round history.

    A data frame is represented by its Z syndrome and one logical bit. Each
    data X fault XORs a fixed mask into this state, and its effect on all later
    observations is determined by that state. The algorithm sums over all
    fault histories with independent data and readout rates and ideal initial
    checks and terminal syndrome. State size grows exponentially with distance.
    """

    _SHOT_BATCH = 1024
    _STATE_BATCH = 48

    def __init__(self, distance: int, p_data: float, p_readout: float):
        distance = index(distance)
        if distance not in (3, 5):
            raise ValueError("posterior decoding supports distances 3 and 5")
        if not (0 <= p_data <= 1 and 0 <= p_readout <= 1):
            raise ValueError("fault probabilities must be between zero and one")

        layout = geometry(distance)
        checks, logical = _terminal_support(_stim_circuit(layout, 0, 0), layout)
        self._check_mask = layout.check_mask
        self._check_count = len(checks)
        self._words = 1 << self._check_count
        self._check_weights = 1 << np.arange(self._check_count, dtype=np.intp)
        self._syndromes = np.arange(self._words, dtype=np.intp)
        states = np.arange(2 * self._words, dtype=np.intp)
        state_shifts = self._check_weights @ checks + self._words * logical.astype(np.intp)
        data_minority = min(p_data, 1 - p_data)
        self._faults = []
        if p_data > 0.5:
            certain_shift = np.bitwise_xor.reduce(state_shifts)
            if certain_shift:
                self._faults.append((states ^ certain_shift, 1))
        for shift, copies in Counter(state_shifts).items():
            prob = data_minority
            for _ in range(copies - 1):
                prob = prob * (1 - data_minority) + (1 - prob) * data_minority
            if shift and prob:
                self._faults.append((states ^ shift, prob))

        hamming_weight = np.array([syndrome.bit_count() for syndrome in range(self._words)])
        self._readout = (1 - p_readout) ** (self._check_count - hamming_weight) * p_readout ** hamming_weight
        self._impossible = (self._readout == 0) & (p_readout in (0, 1))

        # Any supported step can be constructed with at most one minority
        # data fault per state bit by choosing a basis of the qubit masks.
        # Use log space if a possible whole history can leave float64's
        # normal range, even though its conditional parity may be ordinary.
        rank_bound = self._check_count + 1
        readout_minority = min(p_readout, 1 - p_readout)
        min_log_weight = 0.0
        if data_minority:
            min_log_weight += ROUNDS * (rank_bound * log(data_minority)
                                        + (len(state_shifts) - rank_bound) * log1p(-data_minority))
        if readout_minority:
            min_log_weight += (ROUNDS - 1) * self._check_count * log(readout_minority)
        self._use_log = min_log_weight < log(np.finfo(np.float64).tiny)
        if self._use_log:
            with np.errstate(divide="ignore"):
                self._log_faults = [(permutation, np.log(prob), np.log1p(-prob))
                                    for permutation, prob in self._faults]
                if p_readout in (0, 1):
                    self._log_readout = np.log(self._readout)
                else:
                    self._log_readout = (hamming_weight * log(p_readout)
                                         + (self._check_count - hamming_weight) * log1p(-p_readout))
        else:
            initial_state = np.zeros((1, len(states)), dtype=np.float64)
            initial_state[0, 0] = 1
            self._prior = self._transition(initial_state).reshape(2, self._words)

    def _transition(self, weights: np.ndarray) -> np.ndarray:
        """XOR-convolve with the per-round data-fault distribution in place."""
        shifted = np.empty_like(weights)
        for permutation, probability in self._faults:
            np.take(weights, permutation, axis=1, mode="clip", out=shifted)
            shifted *= probability
            weights *= 1 - probability
            weights += shifted
        return weights

    def _emission(self, observed: np.ndarray) -> np.ndarray:
        return self._readout[observed[:, None] ^ self._syndromes]

    def _update(self, incoming: np.ndarray, observations: np.ndarray,
                rows: np.ndarray | None = None) -> np.ndarray:
        """Apply the noisy check likelihood followed by one data-fault round."""
        result = np.empty((len(observations), 2, self._words), dtype=np.float64)
        for start in range(0, len(result), self._STATE_BATCH):
            part = slice(start, start + self._STATE_BATCH)
            previous = incoming if rows is None else incoming[rows[part]]
            result[part] = previous * self._emission(observations[part])[:, None, :]
            # These likelihoods are used up to an independent scale for each row.
            weights = result[part].reshape(-1, 2 * self._words)
            scale = weights.max(axis=1, keepdims=True)
            np.divide(weights, scale, where=scale > 0, out=weights)
            self._transition(weights)
        return result

    def _predict_batch(self, observed: np.ndarray, terminal: np.ndarray) -> np.ndarray:
        # Split the five-round chain at round three. XOR symmetry lets the
        # message from the terminal state share an update with the first round.
        # The fifth raw check is conditionally independent given the terminal.
        observations, inverse = np.unique(np.r_[observed[:, 0], observed[:, 3] ^ terminal],
                                          return_inverse=True)
        first_rows, fourth_rows = inverse.reshape(2, -1)
        single_message = self._update(self._prior, observations)
        prefix, prefix_rows = np.unique(first_rows * self._words + observed[:, 1], return_inverse=True)
        forward = self._update(single_message, prefix % self._words, rows=prefix // self._words)

        posterior = np.empty(len(observed), dtype=np.float64)
        for start in range(0, len(observed), self._STATE_BATCH):
            part = slice(start, start + self._STATE_BATCH)
            middle = forward[prefix_rows[part]] * self._emission(observed[part, 2])[:, None, :]
            translated = terminal[part, None] ^ self._syndromes
            backward = single_message[fourth_rows[part, None], :, translated]
            logical_zero = np.einsum("bjs,bsj->b", middle, backward)
            logical_one = np.einsum("bjs,bsj->b", middle[:, ::-1], backward)
            total = logical_zero + logical_one
            if np.any(total == 0):
                raise ValueError("check history has zero probability under the configured noise")
            posterior[part] = logical_one / total
        return posterior

    def _predict_log(self, observed: np.ndarray, terminal: np.ndarray) -> np.ndarray:
        """Run forward inference in log space for extreme fault probabilities."""
        weights = np.full((len(observed), 2 * self._words), -np.inf)
        weights[:, 0] = 0
        shifted = np.empty_like(weights)
        frames = weights.reshape(-1, 2, self._words)
        for time in range(ROUNDS):
            for permutation, log_fault, log_no_fault in self._log_faults:
                np.take(weights, permutation, axis=1, mode="clip", out=shifted)
                shifted += log_fault
                weights += log_no_fault
                np.logaddexp(weights, shifted, out=weights)
            if time < ROUNDS - 1:
                frames += self._log_readout[observed[:, time, None] ^ self._syndromes][:, None, :]
        choices = frames[np.arange(len(observed)), :, terminal]
        scale = choices.max(axis=1, keepdims=True)
        if np.any(~np.isfinite(scale)):
            raise ValueError("check history has zero probability under the configured noise")
        relative = np.exp(choices - scale)
        return relative[:, 1] / relative.sum(axis=1)

    def predict_proba(self, observed: np.ndarray, terminal: np.ndarray) -> np.ndarray:
        """Return a probability of final logical parity one per shot.

        The observed grids follow Stim's five-round geometry, and terminal
        columns follow the row-major active Z-check mask. Histories impossible
        under the supplied probabilities raise ValueError.
        """
        observed = np.asarray(observed, dtype=bool)
        terminal = np.asarray(terminal, dtype=bool)
        if observed.ndim != 4 or observed.shape[1:] != (ROUNDS, *self._check_mask.shape):
            raise ValueError("expected five observed check grids per shot")
        if terminal.shape != (len(observed), self._check_count):
            raise ValueError("expected one terminal parity per shot and active Z check")
        raw = observed[:, :, self._check_mask] @ self._check_weights
        final = terminal @ self._check_weights
        if np.any(self._impossible[raw[:, -1] ^ final]):
            raise ValueError("final check differs from the terminal syndrome by an impossible readout fault")

        posterior = np.empty(len(observed), dtype=np.float64)
        predict = self._predict_log if self._use_log else self._predict_batch
        batch_size = self._STATE_BATCH if self._use_log else self._SHOT_BATCH
        for start in range(0, len(observed), batch_size):
            part = slice(start, start + batch_size)
            posterior[part] = predict(raw[part], final[part])
        return posterior
