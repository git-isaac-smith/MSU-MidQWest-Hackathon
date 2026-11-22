"""Weighted growth and peeling baseline using the stabilizer-qec implementation."""

import numpy as np
import stim


class UnionFindDecoder:
    """Compile Stim's decomposed model for single-threaded, packed decoding."""

    def __init__(self, model: stim.DetectorErrorModel):
        try:
            from stabilizer_qec import UnionFind
        except ImportError as exc:
            raise ImportError("Weighted union-find requires the bascd[hypergraph] extra") from exc

        self.decoder = UnionFind(model)

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        """Return packed observable flips from packed, little-endian detector rows."""
        return self.decoder.decode_batch(
            detectors, bit_packed_shots=True, bit_packed_predictions=True, threads=1,
        )


def compile_union_find(model: stim.DetectorErrorModel) -> UnionFindDecoder:
    return UnionFindDecoder(model)
