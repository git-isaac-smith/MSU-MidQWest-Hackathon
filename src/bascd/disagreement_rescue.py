"""Resolve disagreements between ordinary and correlated matching with belief matching."""

import numpy as np
import pymatching

from .css_routed import _counts, observable_detector_mask


_PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)


class DisagreementRescue:
    """Use up to twelve belief-propagation iterations on disputed syndromes.

    The ordinary vote uses the observable-bearing graph components. Supply
    an optional packed ordinary decoder callable to change that implementation.
    By default all shots receive a correlated vote and all disagreements
    receive a belief vote. Optional detector-count gates set minimum counts
    for the correlated and belief stages. On disputed shots where a belief
    vote is skipped or rejected by the convergence gate, the correlated vote
    is returned.
    """

    def __init__(self, model, *, ordinary=None, correlated_min_detectors=0,
                 belief_min_detectors=0, require_convergence=False, threads=1):
        if model.num_observables != 1 or model.num_detectors < 1:
            raise ValueError("Expected one observable and at least one detector")
        for threshold in (correlated_min_detectors, belief_min_detectors):
            if type(threshold) is not int or threshold < 0:
                raise ValueError("Detector-count gates must be nonnegative integers")
        if type(threads) is not int or threads < 1:
            raise ValueError("Expected a positive thread count")
        try:
            from stabilizer_qec import BeliefMatching
        except ImportError as exc:
            raise ImportError("Disagreement rescue requires the bascd[hypergraph] extra") from exc

        self.width = (model.num_detectors + 7) // 8
        self.correlated_min_detectors = correlated_min_detectors
        self.belief_min_detectors = belief_min_detectors
        self.require_convergence = require_convergence
        self.threads = threads
        if ordinary is None:
            self._ordinary_matcher = pymatching.Matching.from_detector_error_model(model)
            self._mask = observable_detector_mask(self._ordinary_matcher, model.num_detectors)
            ordinary = self._masked_ordinary
        self.ordinary = ordinary
        self.correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
        self.belief = BeliefMatching(model, max_bp_iters=12, bp_method="product_sum")

    def _masked_ordinary(self, packed):
        return self._ordinary_matcher.decode_batch(packed & self._mask, **_PACKED)

    def decode_batch(self, packed):
        """Return a packed uint8 observable prediction per packed detector row."""
        return self.decode_batch_with_routes(packed)[0]

    def decode_batch_with_routes(self, packed):
        """Return predictions and masks identifying correlated and belief calls."""
        if packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != self.width:
            raise ValueError("Expected little-endian packed detector rows")
        ordinary = self.ordinary(packed)
        counts = _counts(packed) if self.correlated_min_detectors or self.belief_min_detectors else None
        if self.correlated_min_detectors:
            compared = counts >= self.correlated_min_detectors
            if not np.any(compared):
                return ordinary, compared, compared.copy()
            prediction = ordinary.copy()
            prediction[compared] = self.correlated.decode_batch(
                packed[compared], enable_correlations=True, **_PACKED)
        else:
            compared = np.ones(len(packed), dtype=bool)
            prediction = self.correlated.decode_batch(packed, enable_correlations=True, **_PACKED)
        disputed = (ordinary[:, 0] != prediction[:, 0])
        if self.belief_min_detectors:
            disputed &= counts >= self.belief_min_detectors
        if np.any(disputed):
            belief = self.belief.decode_batch(
                packed[disputed], threads=self.threads, return_converged=self.require_convergence, **_PACKED)
            if self.require_convergence:
                belief, converged = belief
                prediction[np.flatnonzero(disputed)[converged]] = belief[converged]
            else:
                prediction[disputed] = belief
        return prediction, compared, disputed
