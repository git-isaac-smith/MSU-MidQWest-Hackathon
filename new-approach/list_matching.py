"""Approximate the logical posterior by summing distinct matching corrections."""

from operator import index

import numpy as np
import pymatching
from scipy.sparse import csc_matrix

from matching_confidence import MatchingConfidence


class ListMatchingPosterior(MatchingConfidence):
    """Find candidate fault sets in each logical sector and sum their masses.

    Each ensemble member solves the same constrained matching problem with
    its own weights. Every DEM edge has a separate fault ID, so each returned
    vector identifies a complete fault set. The original independent DEM
    probabilities score each distinct set once. The list can omit valid
    corrections, so the resulting posterior is an approximation.

    ``members`` includes the original, unperturbed graph. Subsequent graphs
    cycle through the perturbation scales using the supplied random seed.
    """

    _BATCH = 1024
    _SCALES = (0.015, 0.06, 0.18, 0.36, 0.60, 0.85)

    def __init__(self, distance: int, p_data: float, p_readout: float, *,
                 members: int = 49, seed: int = 0):
        members = index(members)
        if members < 1:
            raise ValueError("expected at least one matching member")
        super().__init__(distance, p_data, p_readout)

        rng = np.random.default_rng(seed)
        edges = self._matching.edges()
        self._endpoints = np.asarray([(first, second) for first, second, _ in edges], dtype=np.intp).reshape(-1, 2)
        self._weights = np.asarray([edge["weight"] for _, _, edge in edges], dtype=np.float64)
        self._edge_bytes = (len(edges) + 7) // 8
        byte_values = np.arange(256, dtype=np.uint8)[:, None]
        bits = np.unpackbits(byte_values, axis=1, bitorder="little")
        padded = np.zeros((self._edge_bytes, 8), dtype=np.float64)
        padded.ravel()[:len(edges)] = self._weights
        self._byte_weights = padded @ bits.T

        self._members = [self._graph(self._weights)]
        for member in range(members - 1):
            scale = self._SCALES[member % len(self._SCALES)]
            factors = rng.uniform(1 - scale, 1 + scale, size=len(edges))
            self._members.append(self._graph(self._weights * factors))

    def _graph(self, weights: np.ndarray) -> pymatching.Matching:
        matching = pymatching.Matching(csc_matrix((self._detector_count + 2, 0), dtype=np.uint8))
        for fault, ((first, second), weight) in enumerate(zip(self._endpoints, weights)):
            matching.add_edge(int(first), int(second), fault_ids=fault, weight=float(weight))
        return matching

    def _candidate_vectors(self, syndromes: np.ndarray) -> np.ndarray:
        """Return little-endian packed DEM-edge vectors, with one list per shot."""
        packed = np.empty((len(syndromes), len(self._members), self._edge_bytes), dtype=np.uint8)
        for index, matching in enumerate(self._members):
            packed[:, index] = matching.decode_batch(syndromes, bit_packed_predictions=True)
        return packed

    def _list_mass(self, syndromes: np.ndarray) -> np.ndarray:
        if not self._edge_bytes:
            return np.zeros(len(syndromes), dtype=np.float64)

        packed = self._candidate_vectors(syndromes)
        words = packed.view(np.dtype((np.void, self._edge_bytes)))[..., 0]
        words.sort(axis=1)
        unique = np.ones(words.shape, dtype=bool)
        unique[:, 1:] = words[:, 1:] != words[:, :-1]
        # The likelihood includes the same no-fault factor in every sector.
        # The score is therefore the negative sum of original edge log odds.
        rows, choices = np.nonzero(unique)
        candidates = packed[rows, choices]
        log_weights = -self._byte_weights[np.arange(self._edge_bytes), candidates].sum(axis=1)
        mass = np.full(len(syndromes), -np.inf, dtype=np.float64)
        np.logaddexp.at(mass, rows, log_weights)
        return mass

    def _batch_log_masses(self, detectors: np.ndarray) -> np.ndarray:
        packed = np.packbits(detectors, axis=1, bitorder="little")
        words = packed.view(np.dtype((np.void, packed.shape[1])))[:, 0]
        _, first, inverse = np.unique(words, return_index=True, return_inverse=True)
        syndromes = np.empty((len(first), self._detector_count + 2), dtype=np.uint8)
        syndromes[:, :-2] = detectors[first] ^ self._detector_offset
        detector_parity = np.bitwise_xor.reduce(syndromes[:, :-2], axis=1)
        masses = np.full((len(first), 2), -np.inf, dtype=np.float64)
        for logical in range(2):
            # Fix the logical boundary and balance the syndrome at the other
            # boundary. An odd component cannot be explained by any edge set.
            logical_parity = logical ^ self._logical_offset
            syndromes[:, -1] = logical_parity
            syndromes[:, -2] = detector_parity ^ logical_parity
            if self._component_order is None:
                masses[:, logical] = self._list_mass(syndromes)
            else:
                parity = np.bitwise_xor.reduceat(
                    syndromes[:, self._component_order], self._component_starts, axis=1
                )
                supported = ~parity.any(axis=1)
                if supported.any():
                    masses[supported, logical] = self._list_mass(syndromes[supported])
        return masses[inverse]

    def log_masses(self, detectors: np.ndarray) -> np.ndarray:
        """Return the log list masses for logical 0 and 1, up to a common factor."""
        detectors = np.asarray(detectors)
        if detectors.ndim != 2 or detectors.shape[1] != self._detector_count:
            raise ValueError(f"expected {self._detector_count} unpacked detector bits per shot")
        if detectors.dtype.kind not in "bui" or (detectors.size and (detectors.min() < 0 or detectors.max() > 1)):
            raise ValueError("detectors must contain binary integer or boolean values")

        masses = np.empty((len(detectors), 2), dtype=np.float64)
        for start in range(0, len(detectors), self._BATCH):
            part = slice(start, start + self._BATCH)
            masses[part] = self._batch_log_masses(detectors[part])
        return masses

    def predict_proba(self, detectors: np.ndarray) -> np.ndarray:
        """Return each shot's normalized list mass for logical one."""
        masses = self.log_masses(detectors)
        if np.any(np.isneginf(masses).all(axis=1)):
            raise ValueError("detector record has zero probability under the configured noise")
        delta = masses[:, 1] - masses[:, 0]
        return np.exp(-np.logaddexp(0, -delta))
