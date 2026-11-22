"""Packed detector-only decoding with a bounded DEM posterior cache."""

import numpy as np
import pymatching
import stim

from .exact_small import MAX_DETECTORS, compile_exact_small


_MIX = np.random.default_rng(5).integers(1, 2**64, size=64, dtype=np.uint64) | np.uint64(1)
_POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def _fingerprint(records: np.ndarray) -> np.ndarray:
    """Project packed records into table buckets, verifying all hits separately."""
    width = records.shape[1]
    words = width // 8
    fingerprint = np.zeros(len(records), dtype=np.uint64)
    packed_words = records[:, :words * 8].view("<u8")
    for index in range(words):
        value = packed_words[:, index] * _MIX[index % len(_MIX)]
        rotation = np.uint64(1 + (13 * index) % 63)
        fingerprint ^= (value << rotation) | (value >> (np.uint64(64) - rotation))
    for index in range(words * 8, width):
        fingerprint ^= records[:, index].astype(np.uint64) * _MIX[index % len(_MIX)]
    fingerprint ^= fingerprint >> np.uint64(32)
    fingerprint *= np.uint64(0xd6e8feb86659fd93)
    return fingerprint ^ (fingerprint >> np.uint64(32))


def _faults(model: stim.DetectorErrorModel) -> tuple[np.ndarray, np.ndarray]:
    width = (model.num_detectors + 7) // 8
    mechanisms = []
    for error in model.flattened():
        if error.type != "error":
            continue
        probability, = error.args_copy()
        if probability >= 0.5:
            raise ValueError("posterior cache requires error probabilities below one half")
        if probability == 0:
            continue
        detector_mask, logical_bit = 0, 0
        for target in error.targets_copy():
            if target.is_relative_detector_id():
                detector_mask ^= 1 << target.val
            elif target.is_logical_observable_id():
                logical_bit ^= 1
        if detector_mask or logical_bit:
            odds = probability / (1 - probability)
            mechanisms.append((detector_mask, -odds if logical_bit else odds))
    mechanisms.sort(key=lambda item: abs(item[1]), reverse=True)
    packed = b"".join(mask.to_bytes(width, "little") for mask, _ in mechanisms)
    return (np.frombuffer(packed, dtype=np.uint8).reshape(len(mechanisms), width),
            np.array([weight for _, weight in mechanisms]))


class _PosteriorCache:
    """Sum signed odds for zero, one, and a bounded set of two-fault histories."""

    def __init__(self, model: stim.DetectorErrorModel, pair_fault_limit: int):
        faults, signed_odds = _faults(model)
        first, second = np.triu_indices(min(len(faults), pair_fault_limit), k=1)
        syndromes = np.concatenate((np.zeros((1, faults.shape[1]), dtype=np.uint8),
                                    faults, faults[first] ^ faults[second]))
        signed = np.concatenate(([1.0], signed_odds, signed_odds[first] * signed_odds[second]))
        fingerprints = _fingerprint(syndromes)
        order = np.argsort(fingerprints, kind="stable")
        fingerprints = fingerprints[order]
        groups = np.r_[0, np.flatnonzero(fingerprints[1:] != fingerprints[:-1]) + 1]
        keys = syndromes[order[groups]]
        grouped_signed = np.add.reduceat(signed[order], groups)
        grouped_mass = np.add.reduceat(np.abs(signed[order]), groups)
        # A fingerprint collision may combine different syndromes. Discard all
        # colliding records before the posterior signs enter the inference cache.
        ordered_syndromes = syndromes[order]
        valid = abs(grouped_signed) > 0.1 * grouped_mass
        for index in range(syndromes.shape[1]):
            column = ordered_syndromes[:, index]
            valid &= (np.minimum.reduceat(column, groups) == np.maximum.reduceat(column, groups))
        self.keys = keys[valid]
        self.predictions = (grouped_signed[valid] < 0).astype(np.uint8)
        fingerprints = fingerprints[groups][valid]
        buckets = 1 << (max(1, 4 * len(fingerprints)) - 1).bit_length()
        self.mask = np.uint64(buckets - 1)
        self.indices = np.full(buckets, -1, dtype=np.int32)
        slots = (fingerprints & self.mask).astype(np.intp)
        # Keep the highest estimated mass at a bucket when slots compete.
        ranked = np.argsort(grouped_mass[valid], kind="stable")
        slot_winners = np.full(buckets, -1, dtype=np.int32)
        np.maximum.at(slot_winners, slots[ranked], np.arange(len(ranked), dtype=np.int32))
        occupied = slot_winners >= 0
        self.indices[occupied] = ranked[slot_winners[occupied]]

    def lookup(self, records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        entries = self.indices[_fingerprint(records) & self.mask]
        rows = np.flatnonzero(entries >= 0)
        candidates = entries[rows]
        verified = np.all(records[rows] == self.keys[candidates], axis=1)
        return rows[verified], self.predictions[candidates[verified]]


class FastHybridDecoder:
    """Decode packed or unpacked detectors with an offline posterior lookup.

    Models with at most 24 detectors use the full posterior. Larger models
    use a bounded cache of DEM fault combinations and decode misses by matching.
    Supply a matching-compatible decomposed DEM for these larger models.
    Correlated matching can be enabled for all misses or for those with a
    minimum number of detector events. With one logical observable both output
    formats have shape ``(shots, 1)`` and dtype ``uint8``.
    """

    def __init__(self, model: stim.DetectorErrorModel, *, pair_fault_limit: int = 1400,
                 correlated_fallback: bool = False, correlated_min_detectors: int = 0):
        if model.num_observables != 1:
            raise ValueError("fast hybrid requires one logical observable")
        if type(pair_fault_limit) is not int or pair_fault_limit < 0:
            raise ValueError("pair_fault_limit must be a nonnegative integer")
        if type(correlated_min_detectors) is not int or correlated_min_detectors < 0:
            raise ValueError("correlated_min_detectors must be a nonnegative integer")
        self.num_detectors = model.num_detectors
        self.num_observables = 1
        self._width = (self.num_detectors + 7) // 8
        self._exact = None
        self._cache = None
        self._fallback = None
        self._correlated = None
        self._correlated_min_detectors = correlated_min_detectors
        if self.num_detectors <= MAX_DETECTORS:
            # Separators describe suggested graph decompositions of mechanisms;
            # removing them leaves each mechanism and its probability intact.
            undecomposed = stim.DetectorErrorModel()
            for instruction in model.flattened():
                if instruction.type == "error":
                    targets = [target for target in instruction.targets_copy() if not target.is_separator()]
                    undecomposed.append("error", instruction.args_copy(), targets)
                else:
                    undecomposed.append(instruction)
            self._exact = compile_exact_small(undecomposed)
        else:
            self._cache = _PosteriorCache(model, pair_fault_limit)
            if not correlated_fallback or correlated_min_detectors:
                self._fallback = pymatching.Matching.from_detector_error_model(model)
            if correlated_fallback:
                self._correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)

    def _decode_misses(self, records: np.ndarray, bit_packed_predictions: bool) -> np.ndarray:
        options = dict(bit_packed_shots=True, bit_packed_predictions=bit_packed_predictions)
        if self._correlated is None:
            return self._fallback.decode_batch(records, **options)
        if not self._correlated_min_detectors:
            return self._correlated.decode_batch(records, enable_correlations=True, **options)
        correlated = _POPCOUNT[records].sum(axis=1) >= self._correlated_min_detectors
        predictions = np.empty((len(records), 1), dtype=np.uint8)
        if np.any(correlated):
            predictions[correlated] = self._correlated.decode_batch(
                records[correlated], enable_correlations=True, **options)
        if not np.all(correlated):
            predictions[~correlated] = self._fallback.decode_batch(records[~correlated], **options)
        return predictions

    def decode_batch(self, shots: np.ndarray, *, bit_packed_shots: bool = False,
                     bit_packed_predictions: bool = False) -> np.ndarray:
        shots = np.asarray(shots)
        expected = self._width if bit_packed_shots else self.num_detectors
        if shots.ndim != 2 or shots.shape[1] != expected:
            raise ValueError(f"expected {expected} detector columns per shot")
        if bit_packed_shots:
            if shots.dtype != np.uint8:
                raise ValueError("packed detectors must have dtype uint8")
            packed = np.ascontiguousarray(shots)
        else:
            if shots.dtype.kind not in "biu" or (shots.size and (shots.min() < 0 or shots.max() > 1)):
                raise ValueError("unpacked detectors must contain only bits")
            packed = np.packbits(shots, axis=1, bitorder="little")
        if self._exact is not None:
            indices = np.zeros((len(packed), 4), dtype=np.uint8)
            indices[:, :self._width] = packed
            mask = (1 << self.num_detectors) - 1
            lookup = (indices.view("<u4").ravel() & mask) ^ self._exact._detector_offset
            prediction = self._exact._negative[lookup] ^ self._exact._observable_offset
            return prediction.astype(np.uint8)[:, None]
        if self.num_detectors % 8:
            mask = (1 << (self.num_detectors % 8)) - 1
            if np.any(packed[:, -1] & (255 ^ mask)):
                packed = packed.copy()
                packed[:, -1] &= mask
        hits, predictions = self._cache.lookup(packed)
        output = np.empty((len(packed), 1), dtype=np.uint8)
        output[hits, 0] = predictions
        if len(hits) < len(packed):
            misses = np.ones(len(packed), dtype=bool)
            misses[hits] = False
            output[misses] = self._decode_misses(packed[misses], bit_packed_predictions)
        return output

    def decode_one(self, detectors: np.ndarray) -> np.ndarray:
        return self.decode_batch(np.asarray(detectors)[None, :])[0]
