"""Bounded, exact all-shot logical inference for the five-round experiment."""

from collections import OrderedDict
from threading import Lock

import numpy as np

from posterior_decoder import PosteriorDecoder


class _MessageCache:
    """Bound the number of stored state vectors and evict the least recent."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.rows = OrderedDict()

    def gather(self, keys: np.ndarray, compute) -> list[np.ndarray]:
        missing = [key for key in keys if key not in self.rows]
        fresh = dict(zip(missing, compute(np.asarray(missing, dtype=np.intp)))) if missing else {}
        result = [fresh[key] if key in fresh else self.rows[key] for key in keys]
        for key in keys:
            if key in fresh and self.capacity:
                if len(self.rows) == self.capacity:
                    self.rows.popitem(last=False)
                self.rows[key] = fresh[key].copy()
            elif key in self.rows:
                self.rows.move_to_end(key)
        return result


class CachedPosteriorDecoder(PosteriorDecoder):
    """Compute the exact binary posterior with reusable time-reversed messages.

    All five data transitions have the same symmetric XOR kernel. After
    conditioning on the ideal terminal syndrome, the first four noisy checks
    can therefore be read in either direction, with the reverse checks XORed
    by the terminal syndrome. The fifth noisy check is already validated by
    the base decoder. Its likelihood cancels out of the posterior.
    """

    _SHOT_BATCH = 16384
    _SINGLE_CACHE_BYTES = 96 * 1024**2
    _PAIR_CACHE_BYTES = 128 * 1024**2
    _HISTORY_CACHE_SIZE = 262144

    def __init__(self, distance: int, p_data: float, p_readout: float):
        super().__init__(distance, p_data, p_readout)
        row_bytes = 2 * self._words * np.dtype(np.float64).itemsize
        self._single_cache = _MessageCache(min(self._words, self._SINGLE_CACHE_BYTES // row_bytes))
        self._pair_cache = _MessageCache(min(self._words**2, self._PAIR_CACHE_BYTES // row_bytes))
        self._history_cache = OrderedDict()
        self._cache_lock = Lock()
        self._population = np.array([word.bit_count() for word in range(self._words)])
        self._fallback = p_data in (0, 1) or p_readout in (0, 1)

    def _singles(self, word: np.ndarray) -> list[np.ndarray]:
        return self._single_cache.gather(word, lambda keys: self._update(self._prior, keys))

    def _new_pairs(self, pairs: np.ndarray) -> np.ndarray:
        word, rows = np.unique(pairs // self._words, return_inverse=True)
        return self._update(np.stack(self._singles(word)), pairs % self._words, rows=rows)

    def _predict_batch(self, observed: np.ndarray, terminal: np.ndarray) -> np.ndarray:
        if self._fallback:
            result = np.empty(len(observed), dtype=np.float64)
            for start in range(0, len(observed), PosteriorDecoder._SHOT_BATCH):
                part = slice(start, start + PosteriorDecoder._SHOT_BATCH)
                result[part] = super()._predict_batch(observed[part], terminal[part])
            return result

        n = self._words
        left = observed[:, 0].astype(np.int64) * n + observed[:, 1]
        right = (observed[:, 3] ^ terminal).astype(np.int64) * n + (observed[:, 2] ^ terminal)
        # Use a 64-bit canonical history key. Distance five has 12
        # checks, so two pairs and the terminal fit into 60 bits.
        key = (np.minimum(left, right) * n**2 + np.maximum(left, right)) * n + terminal
        key, inverse = np.unique(key, return_inverse=True)
        with self._cache_lock:
            posterior = self._predict_unique(key)
        return posterior[inverse]

    def _predict_unique(self, key: np.ndarray) -> np.ndarray:
        posterior = np.empty(len(key), dtype=np.float64)
        missing = []
        for row, history in enumerate(key):
            value = self._history_cache.get(history)
            if value is None:
                missing.append(row)
            else:
                posterior[row] = value
                self._history_cache.move_to_end(history)
        if missing:
            histories = key[missing]
            values = self._evaluate(histories)
            posterior[missing] = values
            if self._HISTORY_CACHE_SIZE:
                for history, value in zip(histories, values):
                    if len(self._history_cache) == self._HISTORY_CACHE_SIZE:
                        self._history_cache.popitem(last=False)
                    self._history_cache[history] = value
        return posterior

    def _evaluate(self, key: np.ndarray) -> np.ndarray:
        n = self._words
        terminal = key % n
        left = key // n**3
        right = key // n % n**2

        # Prefer to advance the pair with fewer detection events. That pair
        # is likelier to recur. A fixed orientation keeps float64 reduction
        # order independent of the batch and of cache hits.
        forward_first = self._event_count(left) <= self._event_count(right)
        forward = np.where(forward_first, left, right)
        backward = np.where(forward_first, right, left)

        posterior = np.empty(len(key), dtype=np.float64)
        order = np.argsort(forward, kind="stable")
        translated_suffix = np.empty((2, n), dtype=np.float64)
        for start in range(0, len(order), self._STATE_BATCH):
            selection = order[start:start + self._STATE_BATCH]
            prefix, pair_rows = np.unique(forward[selection], return_inverse=True)
            advanced = self._pair_cache.gather(prefix, self._new_pairs)
            suffix = backward[selection]
            words, single_rows = np.unique(suffix // n, return_inverse=True)
            singles = self._singles(words)
            for row, pair, single, word in zip(selection, pair_rows, single_rows, suffix % n):
                translated = terminal[row] ^ self._syndromes
                np.take(singles[single], translated, axis=1, mode="clip", out=translated_suffix)
                translated_suffix *= self._readout[word ^ translated]
                # Sum over equal center syndromes. A terminal logical one
                # swaps the translated suffix's two logical components.
                parity = advanced[pair] @ translated_suffix.T
                logical_zero = parity[0, 0] + parity[1, 1]
                logical_one = parity[0, 1] + parity[1, 0]
                total = logical_zero + logical_one
                if total == 0:
                    raise ValueError("check history has zero probability under the configured noise")
                posterior[row] = logical_one / total
        return posterior

    def _event_count(self, pair: np.ndarray) -> np.ndarray:
        first = pair // self._words
        second = pair % self._words
        return self._population[first] + self._population[first ^ second]
