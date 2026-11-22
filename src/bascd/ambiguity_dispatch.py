"""Dispatch packed CSS memory syndromes using detector fault coincidences."""

from itertools import combinations
from math import sqrt

import numpy as np
import pymatching

from .css_routed import observable_detector_mask


_PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)
_BYTE_BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little")
_SCORE_SCALE = 16


def _score_table(model, logical_mask):
    """Combine activity and within-byte coincidence into one lookup per byte."""
    logical = np.unpackbits(logical_mask, bitorder="little")
    valid = np.arange(len(logical)) < model.num_detectors
    detector_weights = (valid + 2 * logical).reshape(-1, 8)
    scores = (detector_weights @ _BYTE_BITS.T).astype(np.float64)

    marginal = np.zeros(model.num_detectors)
    coincident = {}
    for fault in model.flattened():
        if fault.type != "error":
            continue
        probability, = fault.args_copy()
        if not probability:
            continue
        active = set()
        for target in fault.targets_copy():
            if target.is_relative_detector_id():
                active.symmetric_difference_update((target.val,))
        detectors = sorted(active)
        marginal[detectors] += probability
        for first, second in combinations(detectors, 2):
            if first // 8 == second // 8 and logical[first] != logical[second]:
                pair = first, second
                coincident[pair] = coincident.get(pair, 0.0) + probability

    # Normalize joint mass by the detector marginals, with saturation so a
    # strong coincidence never adds more than two detector-count points.
    for (first, second), probability in coincident.items():
        relative_mass = probability / (sqrt(marginal[first]) * sqrt(marginal[second]))
        both_active = _BYTE_BITS[:, first % 8] & _BYTE_BITS[:, second % 8]
        scores[first // 8] += 2 * min(4 * relative_mass, 1) * both_active
    return np.rint(_SCORE_SCALE * scores).astype(np.int16)


class AmbiguityDispatcher:
    """Route high-score shots to full correlated MWPM after unlabeled calibration.

    An event in the logical graph component scores three points and an event
    outside it scores one. Opposite-component detector pairs in the same packed
    byte can also score when the DEM contains a fault that flips them together.
    The cutoff is a quantile of an independent calibration sample. Cutoff ties
    use ordinary matching on the logical component.
    """

    def __init__(self, circuit, model, *, calibration_seed, fraction=0.06, calibration_shots=8192):
        if model.num_observables != 1 or model.num_detectors < 1:
            raise ValueError("Expected one observable and at least one detector")
        if circuit.num_detectors != model.num_detectors or circuit.num_observables != model.num_observables:
            raise ValueError("Calibration circuit dimensions must match the model")
        if not 0 <= fraction <= 1 or calibration_shots < 1:
            raise ValueError("Expected a routing fraction in [0, 1] and positive calibration size")
        self.ordinary = pymatching.Matching.from_detector_error_model(model)
        self.correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
        self.logical_mask = observable_detector_mask(self.ordinary, model.num_detectors)
        self._table = _score_table(model, self.logical_mask)
        self._columns = np.arange(len(self.logical_mask))
        self._fraction = fraction
        if 0 < fraction < 1:
            sample = circuit.compile_detector_sampler(seed=calibration_seed).sample(
                calibration_shots, bit_packed=True)
            self.cutoff = int(np.quantile(self.score(sample), 1 - fraction, method="higher"))
        else:
            self.cutoff = int(self._table.max(axis=1).sum()) if fraction == 0 else -1

    def _validate(self, packed):
        if packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != len(self.logical_mask):
            raise ValueError("Expected little-endian packed detector rows")

    def _scores(self, packed):
        return self._table[self._columns, packed].sum(axis=1, dtype=np.int32)

    def score(self, packed):
        """Return calibrated-score units for each packed detector row."""
        self._validate(packed)
        return self._scores(packed)

    def route(self, packed):
        """Return the mask of shots requiring correlated matching."""
        self._validate(packed)
        return self._scores(packed) > self.cutoff

    def decode_batch(self, packed):
        """Return packed logical predictions in the original shot order."""
        self._validate(packed)
        if self._fraction == 0:
            return self.ordinary.decode_batch(packed & self.logical_mask, **_PACKED)
        if self._fraction == 1:
            return self.correlated.decode_batch(packed, enable_correlations=True, **_PACKED)
        escalate = self._scores(packed) > self.cutoff
        easy = packed & self.logical_mask
        easy[escalate] = 0
        predictions = self.ordinary.decode_batch(easy, **_PACKED)
        if np.any(escalate):
            predictions[escalate] = self.correlated.decode_batch(
                packed[escalate], enable_correlations=True, **_PACKED)
        return predictions
