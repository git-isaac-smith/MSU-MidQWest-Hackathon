"""Decode one completed packed detector record without batch routing overhead."""

from functools import partial
from pathlib import Path

import numpy as np
import pymatching

from .css_routed import _PACKED
from .learned_fault_router import LearnedFaultRoutedDecoder, LearnedFaultRouter


def _validate_record(record, width, remainder):
    if not isinstance(record, bytes) or len(record) != width:
        raise ValueError(f"expected a {width}-byte detector record")
    if remainder and record[-1] >> remainder:
        raise ValueError("nonzero detector padding")


def _decode_packed(decode, record):
    packed = np.frombuffer(record, dtype=np.uint8).reshape(1, -1)
    return int(decode(packed)[0, 0])


def _lane_crossings(parity, lane_count):
    prefix = parity ^ (parity << 1)
    prefix ^= prefix << 2
    prefix ^= prefix << 4
    crossings = (prefix & ((1 << lane_count) - 1)).bit_count()
    return min(crossings, lane_count + 1 - crossings)


class ScalarMatchingDecoder:
    """Decode a packed record with ordinary or correlated PyMatching."""

    def __init__(self, model, *, correlated=False):
        if model.num_observables != 1 or model.num_detectors < 1:
            raise ValueError("expected at least one detector and one observable")
        self.record_bytes = (model.num_detectors + 7) // 8
        self._remainder = model.num_detectors % 8
        matcher = pymatching.Matching.from_detector_error_model(model, enable_correlations=correlated)
        self._decode = partial(matcher.decode_batch, enable_correlations=correlated, **_PACKED)

    def decode(self, record: bytes) -> int:
        """Return the predicted logical bit for one little-endian record."""
        _validate_record(record, self.record_bytes, self._remainder)
        return _decode_packed(self._decode, record)


class ScalarLearnedDecoder:
    """Apply a frozen learned gate and call only the chosen PyMatching branch."""

    def __init__(self, model, policy: LearnedFaultRouter):
        if model.num_observables != 1 or model.num_detectors < 1:
            raise ValueError("expected at least one detector and one observable")
        self.record_bytes = (model.num_detectors + 7) // 8
        self._remainder = model.num_detectors % 8
        if len(policy.activity_lookup) != self.record_bytes:
            raise ValueError("policy detector width differs from model")
        batch = LearnedFaultRoutedDecoder(model, policy)
        self._observable_mask = int.from_bytes(batch.mask, "little")
        self._ordinary = partial(batch.ordinary.decode_batch, enable_correlations=False, **_PACKED)
        self._correlated = partial(batch.correlated.decode_batch, enable_correlations=True, **_PACKED)

        self._relevant_size, self._other_size, self._lane_size, self._adjacent_size, self._parity_size = (
            policy.scores.shape
        )
        # Reuse the batch policy's feasible count-pair decisions and its NumPy
        # comparison against the cutoff, including its score dtype semantics.
        self._count_routes = np.where(policy._ambiguous, 2, policy._always).astype(np.uint8).tobytes()
        self._score_routes = (policy.scores > policy.cutoff).tobytes()
        self._lanes = policy.lane_lookup.tolist()
        self._adjacent_pairs = range(policy.time_count - 1)
        self._parities = None
        if policy.parity_lookup is not None:
            self._parities = policy.parity_lookup.tolist()
            self._crossings = bytes(_lane_crossings(p, policy.lane_count) for p in range(256))

    @classmethod
    def from_case(cls, case: Path):
        """Open a frozen compute-study case and check its artifact digests."""
        from .compute_replay import _frozen_circuit

        case = Path(case)
        circuit = _frozen_circuit(case)
        model = circuit.detector_error_model(decompose_errors=True)
        return cls(model, LearnedFaultRouter.load(case / "policy.npz"))

    def _use_correlated(self, record, value, relevant):
        logical_count = relevant.bit_count()
        other_count = value.bit_count() - logical_count
        count_pair = (min(logical_count, self._relevant_size - 1) * self._other_size
                      + min(other_count, self._other_size - 1))
        route = self._count_routes[count_pair]
        if route != 2:
            return bool(route)

        projection, parity = 0, 0
        if self._parities is None:
            for byte, lanes in zip(record, self._lanes):
                projection |= lanes[byte]
        else:
            for byte, lanes, parities in zip(record, self._lanes, self._parities):
                projection |= lanes[byte]
                parity ^= parities[byte]
            parity = self._crossings[parity]

        adjacent = 0
        if self._adjacent_pairs:
            combined = projection | (projection >> 8)
            for _ in self._adjacent_pairs:
                adjacent = max(adjacent, (combined & 255).bit_count())
                combined >>= 8
        projection |= projection >> 32
        projection |= projection >> 16
        projection |= projection >> 8
        lane_mask = projection & 255
        index = (((count_pair * self._lane_size + lane_mask) * self._adjacent_size + adjacent)
                 * self._parity_size + parity)
        return bool(self._score_routes[index])

    def route(self, record: bytes) -> bool:
        """Return whether the frozen policy selects correlated matching."""
        _validate_record(record, self.record_bytes, self._remainder)
        value = int.from_bytes(record, "little")
        return self._use_correlated(record, value, value & self._observable_mask)

    def decode(self, record: bytes) -> int:
        """Return the predicted logical bit for one little-endian record."""
        _validate_record(record, self.record_bytes, self._remainder)
        value = int.from_bytes(record, "little")
        relevant = value & self._observable_mask
        if self._use_correlated(record, value, relevant):
            return _decode_packed(self._correlated, record)
        return _decode_packed(self._ordinary, relevant.to_bytes(self.record_bytes, "little"))
