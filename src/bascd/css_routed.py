"""One-observable CSS memory decoder with selective correlated matching."""

import numpy as np
import pymatching
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


_BIT_COUNTS = np.asarray([v.bit_count() for v in range(256)], dtype=np.uint8)
_PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)


def observable_detector_mask(matching, detector_count):
    """Find every component of the ordinary matching graph carrying an L0 edge."""
    edges = matching.edges()
    pairs = [(u, v) for u, v, _ in edges if v is not None]
    starts, ends = zip(*pairs) if pairs else ((), ())
    graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8), (starts, ends)),
                       shape=(detector_count, detector_count))
    _, component = connected_components(graph, directed=False)
    relevant = {component[u] for u, _, edge in edges if 0 in edge["fault_ids"]}
    mask = np.isin(component, list(relevant))
    return np.packbits(mask, bitorder="little")


def _counts(records):
    return _BIT_COUNTS[records].sum(axis=1, dtype=np.uint32)


class CssRoutedDecoder:
    """Decode the logical component, escalating high-activity shots to full correlated MWPM.

    The branch threshold comes from a fixed, independent unlabeled calibration sample.
    The full matching path receives the original detection events in both bases.
    """

    def __init__(self, circuit, model, *, calibration_seed, fraction=0.06, calibration_shots=8192):
        if model.num_observables != 1 or not 0 <= fraction <= 1 or calibration_shots < 1:
            raise ValueError("Requires one logical observable, a valid fraction and positive calibration size")
        self.ordinary = pymatching.Matching.from_detector_error_model(model)
        self.correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
        self.mask = observable_detector_mask(self.ordinary, model.num_detectors)
        calibration = circuit.compile_detector_sampler(seed=calibration_seed).sample(calibration_shots, bit_packed=True)
        counts = _counts(calibration)
        self.cutoff = int(np.quantile(counts, 1 - fraction, method="higher"))
        if fraction == 0:
            self.cutoff = model.num_detectors
        if fraction == 1:
            self.cutoff = -1

    def decode_batch(self, packed):
        if packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != len(self.mask):
            raise ValueError("Expected little-endian packed detector rows")
        escalate = _counts(packed) > self.cutoff
        easy = packed & self.mask
        easy[escalate] = 0
        prediction = self.ordinary.decode_batch(easy, **_PACKED)
        if np.any(escalate):
            prediction[escalate] = self.correlated.decode_batch(packed[escalate], enable_correlations=True, **_PACKED)
        return prediction
