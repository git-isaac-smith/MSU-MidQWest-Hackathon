"""Threshold tests for the complementary logical-sector matching cost."""

import numpy as np
import pymatching
from scipy.sparse import csc_matrix

from .css_routed import observable_detector_mask
from .projected_router import ProjectedOrdinary


_PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)


class CappedGapProbe:
    """Test whether two logical sectors are within a fixed matching-cost gap.

    For requested parity r, the augmented matcher minimizes the cost of sector
    r versus sector 1-r plus the cutoff. The extra boundary edge indicates
    which sector it actually selected.
    """

    def __init__(self, model, cutoff):
        if model.num_observables != 1:
            raise ValueError("requires one logical observable")
        if not np.isfinite(cutoff) or not 0 < cutoff <= (1 << 24) - 1:
            raise ValueError("cutoff must be a positive finite matching weight")
        self.cutoff = float(cutoff)
        ordinary = pymatching.Matching.from_detector_error_model(model)
        self.mask = observable_detector_mask(ordinary, model.num_detectors)
        included = np.unpackbits(self.mask, bitorder="little", count=model.num_detectors)
        discarded = np.flatnonzero(~included.astype(bool))
        parity = int(discarded[-1]) if len(discarded) else model.num_detectors
        self._parity_byte, parity_bit = divmod(parity, 8)
        self._parity_mask = np.uint8(1 << parity_bit)
        nodes = max(model.num_detectors, parity + 1)
        self._width = (nodes + 7) // 8
        self._matcher = pymatching.Matching(csc_matrix((nodes, 0), dtype=np.uint8))
        for start, end, edge in ordinary.edges():
            if not included[start]:
                continue
            weight = edge["weight"]
            if edge["fault_ids"]:
                if end is not None or edge["fault_ids"] != {0}:
                    raise ValueError("requires logical flips exclusively on boundary edges")
                self._matcher.add_edge(start, parity, weight=weight)
            elif end is None:
                self._matcher.add_boundary_edge(start, weight=weight)
            else:
                self._matcher.add_edge(start, end, weight=weight)
        self._matcher.add_boundary_edge(parity, fault_ids={0}, weight=self.cutoff)

    def _requests(self, packed, copies):
        requests = np.empty((copies * len(packed), self._width), dtype=np.uint8)
        input_width = len(self.mask)
        np.bitwise_and(packed, self.mask, out=requests[:len(packed), :input_width])
        requests[:len(packed), input_width:] = 0
        return requests

    def classify(self, packed: np.ndarray, ordinary: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return predictions and the mask of shots within the cutoff.

        Supply the ordinary prediction to test just its complementary sector.
        A shot exactly at the cutoff may route either way under matching ties.
        """
        if packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != len(self.mask):
            raise ValueError("expected little-endian packed detector rows")
        if ordinary is not None:
            if ordinary.shape != (len(packed), 1) or ordinary.dtype != np.uint8:
                raise ValueError("expected one packed uint8 prediction per detector row")
            requests = self._requests(packed, 1)
            requests[:, self._parity_byte] |= (ordinary[:, 0] ^ np.uint8(1)) * self._parity_mask
            shortcut = self._matcher.decode_batch(requests, **_PACKED)
            return ordinary.copy(), shortcut[:, 0] == 0

        n = len(packed)
        requests = self._requests(packed, 2)
        requests[n:] = requests[:n]
        requests[n:, self._parity_byte] |= self._parity_mask
        shortcuts = self._matcher.decode_batch(requests, **_PACKED)
        return shortcuts[:n], shortcuts[:n, 0] == shortcuts[n:, 0]


class CappedGapRoutedDecoder:
    """Send shots within a fixed logical gap to a correlated specialist."""

    def __init__(self, circuit, model, cutoff, *, mode="complement", specialist="correlated"):
        if mode not in ("complement", "biased"):
            raise ValueError("expected complement or biased gap probing")
        self.probe = CappedGapProbe(model, cutoff)
        self.ordinary = ProjectedOrdinary(circuit, model) if mode == "complement" else None
        if specialist == "correlated":
            self._backend = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
            self._options = {"enable_correlations": True}
        elif specialist == "belief":
            from stabilizer_qec import BeliefMatching
            self._backend = BeliefMatching(model, max_bp_iters=12, bp_method="product_sum")
            self._options = {"threads": 1}
        else:
            raise ValueError("expected correlated or belief specialist")

    def decode_batch(self, packed: np.ndarray) -> np.ndarray:
        ordinary = self.ordinary.decode_batch(packed) if self.ordinary is not None else None
        bits, routed = self.probe.classify(packed, ordinary)
        if np.any(routed):
            bits[routed] = self._backend.decode_batch(packed[routed], **_PACKED, **self._options)
        return bits
