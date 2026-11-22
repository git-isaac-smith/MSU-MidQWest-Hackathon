"""Two logical-sector matching costs for conditional decoding of a CSS memory."""

import numpy as np
import pymatching
from scipy.sparse import csc_matrix

from .css_routed import observable_detector_mask

PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)


class LogicalGapProbe:
    """Constrain the parity at the logical boundary of a single-observable graph."""

    def __init__(self, model):
        if model.num_observables != 1:
            raise ValueError("requires one logical observable")
        ordinary = pymatching.Matching.from_detector_error_model(model)
        mask = observable_detector_mask(ordinary, model.num_detectors)
        self.selected = np.flatnonzero(np.unpackbits(mask, bitorder="little", count=model.num_detectors))
        self.width = (model.num_detectors + 7) // 8
        observable_node = len(self.selected)
        renumber = {int(k): i for i, k in enumerate(self.selected)}
        self.matcher = pymatching.Matching(csc_matrix((observable_node + 1, 0), dtype=np.uint8))
        for u, v, edge in ordinary.edges():
            if u not in renumber:
                continue
            node = renumber[u]
            if edge["fault_ids"]:
                if v is not None or edge["fault_ids"] != {0}:
                    raise ValueError("requires logical flips exclusively on boundary edges")
                self.matcher.add_edge(node, observable_node, weight=edge["weight"])
            elif v is None:
                self.matcher.add_boundary_edge(node, weight=edge["weight"])
            else:
                self.matcher.add_edge(node, renumber[v], weight=edge["weight"])

    def costs(self, packed):
        """Return ordinary graph minimum weights for logical zero and one."""
        if packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != self.width:
            raise ValueError("expected packed detector records for the compiled graph")
        n = len(packed)
        k = len(self.selected)
        relevant = np.unpackbits(packed, axis=1, bitorder="little")[:, self.selected]
        width = (k + 8) // 8
        both = np.zeros((2 * n, width), dtype=np.uint8)
        both[:n, : (k + 7) // 8] = np.packbits(relevant, axis=1, bitorder="little")
        both[n:] = both[:n]
        both[n:, k // 8] |= np.uint8(1 << (k % 8))
        _, weights = self.matcher.decode_batch(both, return_weights=True, **PACKED)
        return weights[:n], weights[n:]


class GapRoutedDecoder:
    """Use the cheap logical class on confident shots and a specialist on near ties."""

    def __init__(self, model, cutoff, specialist="correlated"):
        self.probe = LogicalGapProbe(model)
        self.cutoff = float(cutoff)
        if not np.isfinite(self.cutoff) or self.cutoff < 0:
            raise ValueError("cutoff must be finite and nonnegative")
        self.specialist = specialist
        if specialist == "correlated":
            self.backend = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
        elif specialist == "belief":
            from stabilizer_qec import BeliefMatching

            self.backend = BeliefMatching(model, max_bp_iters=12, bp_method="product_sum")
        else:
            raise ValueError("expected correlated or belief specialist")

    def decode_batch(self, packed):
        zero, one = self.probe.costs(packed)
        bits = (one < zero).astype(np.uint8)[:, None]
        routed = np.abs(one - zero) <= self.cutoff
        if np.any(routed):
            options = {"enable_correlations": True} if self.specialist == "correlated" else {"threads": 1}
            bits[routed] = self.backend.decode_batch(packed[routed], **PACKED, **options)
        return bits
