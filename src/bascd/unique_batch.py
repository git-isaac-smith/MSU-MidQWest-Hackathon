"""Decode repeated detector records once within each batch chunk."""

from typing import Protocol

import numpy as np


class _Decoder(Protocol):
    def decode_batch(self, detectors: np.ndarray) -> np.ndarray: ...

    def decode_one(self, detectors: np.ndarray) -> np.ndarray: ...


def _row_keys(detectors: np.ndarray) -> np.ndarray:
    """Build exact row keys, packing Boolean detectors for the sort."""
    if not detectors.shape[1]:
        return np.zeros(len(detectors), dtype=np.uint8)
    packed = np.packbits(detectors, axis=1) if detectors.dtype == np.bool_ else detectors
    contiguous = np.ascontiguousarray(packed)
    row_bytes = contiguous.shape[1] * contiguous.dtype.itemsize
    return contiguous.view(np.dtype((np.void, row_bytes))).ravel()


class UniqueBatchDecoder:
    """Reuse deterministic per-shot predictions for identical detector rows."""

    def __init__(
        self, decoder: _Decoder, num_detectors: int, *, chunk_size: int = 32_768,
        probe_shots: int = 1_024, min_duplicate_fraction: float = 0.01,
    ):
        if type(num_detectors) is not int or num_detectors < 0:
            raise ValueError("num_detectors must be a nonnegative integer")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if type(probe_shots) is not int or probe_shots < 2:
            raise ValueError("probe_shots must be an integer of at least two")
        if not 0 <= min_duplicate_fraction <= 1:
            raise ValueError("min_duplicate_fraction must be in [0, 1]")
        self.decoder = decoder
        self.num_detectors = num_detectors
        self.chunk_size = chunk_size
        self.probe_shots = probe_shots
        self.min_duplicate_fraction = min_duplicate_fraction

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError("expected a (shots, detectors) array matching the model")
        if len(detectors) < 2 or detectors.dtype.kind not in "biu":
            return self.decoder.decode_batch(detectors)

        probe = detectors[:self.probe_shots]
        duplicates = len(probe) - len(np.unique(_row_keys(probe)))
        if duplicates < self.min_duplicate_fraction * len(probe):
            return self.decoder.decode_batch(detectors)
        if len(detectors) <= self.chunk_size:
            return self._decode_chunk(detectors)
        chunks = range(0, len(detectors), self.chunk_size)
        return np.concatenate([
            self._decode_chunk(detectors[start:start + self.chunk_size]) for start in chunks
        ])

    def _decode_chunk(self, detectors: np.ndarray) -> np.ndarray:
        _, first, inverse = np.unique(_row_keys(detectors), return_index=True, return_inverse=True)
        if len(first) == len(detectors):
            return self.decoder.decode_batch(detectors)
        order = np.argsort(first)
        predictions = self.decoder.decode_batch(detectors[first[order]])
        key_order = np.empty_like(predictions)
        key_order[order] = predictions
        return key_order[inverse]

    def decode_one(self, detectors: np.ndarray) -> np.ndarray:
        if detectors.ndim != 1 or detectors.shape[0] != self.num_detectors:
            raise ValueError("expected a detector vector matching the model")
        return self.decoder.decode_one(detectors)
