"""Select a precompiled matcher using observable syndrome features."""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import stim

from .decoders import CHOICES, compile_decoder

COUNT_BINS = 16
TIME_BINS = 4


def model_id(model: stim.DetectorErrorModel) -> str:
    return hashlib.sha256(str(model).encode()).hexdigest()


@dataclass(frozen=True)
class RoutingPolicy:
    dem_sha256: str
    feature: str
    correlated_bins: tuple[int, ...]

    def __post_init__(self):
        size = {"count": COUNT_BINS, "spacetime": COUNT_BINS * TIME_BINS}.get(self.feature)
        if size is None:
            raise ValueError("unsupported routing feature")
        if (len(self.dem_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in self.dem_sha256)):
            raise ValueError("invalid detector error model hash")
        if (any(type(i) is not int or not 0 <= i < size for i in self.correlated_bins)
                or tuple(sorted(set(self.correlated_bins))) != self.correlated_bins):
            raise ValueError("expected sorted unique correlated bins in range")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(value["dem_sha256"], value["feature"], tuple(value["correlated_bins"]))


class SyndromeFeatures:
    def __init__(self, model, feature):
        self.size = COUNT_BINS if feature == "count" else COUNT_BINS * TIME_BINS
        self.boundaries = None
        self.round_bits = None
        if feature == "spacetime":
            coords = model.get_detector_coordinates()
            if (not model.num_detectors or len(coords) != model.num_detectors or
                    any(len(c) < 3 or not np.isfinite(c[2]) for c in coords.values())):
                raise ValueError("spacetime routing requires a time coordinate for every detector")
            times = np.array([coords[i][2] for i in range(model.num_detectors)])
            self.order = np.argsort(times, kind="stable")
            times = times[self.order]
            self.boundaries = np.r_[0, np.flatnonzero(np.diff(times)) + 1]
            self.adjacent = np.diff(times[self.boundaries]) == 1
            self.reorder = not np.array_equal(self.order, np.arange(len(times)))
            if len(self.boundaries) <= 64:
                rounds = np.searchsorted(self.boundaries, np.arange(len(times)), side="right") - 1
                self.round_bits = np.empty(len(times), dtype=np.uint64)
                self.round_bits[self.order] = np.left_shift(np.uint64(1), rounds.astype(np.uint64))
                self.adjacent_mask = sum(1 << int(i) for i in np.flatnonzero(self.adjacent))

    def buckets(self, detectors):
        if self.boundaries is None:
            return np.minimum(np.count_nonzero(detectors, axis=1), COUNT_BINS - 1)
        ordered = detectors[:, self.order] if self.reorder else detectors
        rounds = np.add.reduceat(ordered, self.boundaries, axis=1, dtype=np.int32)
        occupied = rounds > 0
        pairs = np.count_nonzero(occupied[:, :-1] & occupied[:, 1:] & self.adjacent, axis=1)
        return (np.minimum(rounds.sum(axis=1), COUNT_BINS - 1) * TIME_BINS +
                np.minimum(pairs, TIME_BINS - 1))

    def bucket(self, detectors):
        if self.boundaries is None:
            return min(np.count_nonzero(detectors), COUNT_BINS - 1)
        if self.round_bits is None:
            return int(self.buckets(detectors[None, :])[0])
        active = np.flatnonzero(detectors)
        occupied = int(np.bitwise_or.reduce(self.round_bits[active], initial=np.uint64(0)))
        pairs = (occupied & (occupied >> 1) & self.adjacent_mask).bit_count()
        return min(active.size, COUNT_BINS - 1) * TIME_BINS + min(pairs, TIME_BINS - 1)


class AdaptiveDecoder:
    """Route each shot once. Mixed batches are scattered back into input order."""

    def __init__(self, model: stim.DetectorErrorModel, policy: RoutingPolicy):
        if policy.dem_sha256 != model_id(model):
            raise ValueError("routing policy belongs to a different detector error model")
        self.num_detectors = model.num_detectors
        self.num_observables = model.num_observables
        self.ordinary = compile_decoder(model, CHOICES[0])
        self.correlated = compile_decoder(model, CHOICES[1]) if policy.correlated_bins else None
        self.features = SyndromeFeatures(model, policy.feature) if self.correlated else None
        if self.features is not None:
            self.lookup = np.zeros(self.features.size, dtype=bool)
            self.lookup[list(policy.correlated_bins)] = True

    def route(self, detectors):
        if self.features is None:
            return np.zeros(len(detectors), dtype=bool)
        return self.lookup[self.features.buckets(detectors)]

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError("expected a (shots, detectors) array matching the model")
        if not len(detectors):
            return np.empty((0, self.num_observables), dtype=np.uint8)
        if self.correlated is None:
            return self.ordinary.decode_batch(detectors)
        routed = self.route(detectors)
        if not routed.any():
            return self.ordinary.decode_batch(detectors)
        if routed.all():
            return self.correlated.decode_batch(detectors)
        predictions = np.empty((len(detectors), self.num_observables), dtype=np.uint8)
        predictions[~routed] = self.ordinary.decode_batch(detectors[~routed])
        predictions[routed] = self.correlated.decode_batch(detectors[routed])
        return predictions

    def decode_one(self, detectors: np.ndarray) -> np.ndarray:
        if detectors.ndim != 1 or len(detectors) != self.num_detectors:
            raise ValueError("expected a detector vector matching the model")
        if self.features is not None and self.lookup[self.features.bucket(detectors)]:
            return self.correlated.decode_one(detectors)
        return self.ordinary.decode_one(detectors)


def fit_policy(model, features, buckets, ordinary_failed, correlated_failed, cap, feature):
    """Rank buckets by smoothed paired net recovery per routed shot."""
    shots = np.bincount(buckets, minlength=features.size)
    recovered = np.bincount(buckets[ordinary_failed & ~correlated_failed], minlength=features.size)
    lost = np.bincount(buckets[~ordinary_failed & correlated_failed], minlength=features.size)
    gain = recovered - lost
    rank = np.argsort(-gain / (shots + 256), kind="stable")
    remaining = int(cap * len(buckets))
    selected = []
    for index in rank:
        if recovered[index] + lost[index] >= 12 and gain[index] > 0 and shots[index] <= remaining:
            selected.append(int(index))
            remaining -= shots[index]
    return RoutingPolicy(model_id(model), feature, tuple(sorted(selected)))
