"""Compute constrained matching costs from the public Stim detector record."""

from math import log, log1p

import numpy as np
import pymatching
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from logical_reference import _stim_circuit
from stim_data import geometry


class MatchingConfidence:
    """Minimum Stim DEM matching cost for each logical class.

    This rotated Z memory has two spatial boundaries. The DEM labels edges
    ending at one of them with L0. There are no time boundaries because the
    initial and terminal Z checks are ideal. Making both spatial boundaries
    explicit matching nodes lets their incident parity be fixed per shot.
    """

    def __init__(self, distance: int, p_data: float, p_readout: float):
        if not (0 <= p_data <= 1 and 0 <= p_readout <= 1):
            raise ValueError("fault probabilities must be between zero and one")
        layout = geometry(distance)
        circuit = _stim_circuit(layout, min(p_data, 1 - p_data), min(p_readout, 1 - p_readout))
        self._detector_count = circuit.num_detectors
        self._detector_offset = np.zeros(self._detector_count, dtype=bool)
        self._logical_offset = False
        if p_data > 0.5 or p_readout > 0.5:
            frame = _stim_circuit(layout, p_data > 0.5, p_readout > 0.5)
            detectors, logicals = frame.compile_detector_sampler().sample(1, separate_observables=True)
            self._detector_offset = detectors[0]
            self._logical_offset = bool(logicals[0, 0])

        node_count = self._detector_count + 2
        self._matching = pymatching.Matching(csc_matrix((node_count, 0), dtype=np.uint8))
        edge_starts, edge_ends = [], []
        for instruction in circuit.detector_error_model().flattened():
            if instruction.type != "error":
                continue
            detectors = []
            logical = False
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detectors.append(target.val)
                elif target.is_logical_observable_id() and target.val == 0:
                    logical = not logical
                else:
                    raise ValueError("expected independent graphlike Stim DEM faults")
            if len(detectors) == 1:
                detectors.append(self._detector_count + logical)
            elif len(detectors) != 2 or logical:
                raise ValueError("expected only boundary edges to carry the logical label")
            first, second = detectors
            probability, = instruction.args_copy()
            self._matching.add_edge(first, second, weight=log1p(-probability) - log(probability))
            edge_starts.append(first)
            edge_ends.append(second)

        # With an absent fault type, some syndromes or logical classes can be
        # impossible. A graph supports exactly the even syndromes in each of
        # its connected components. With both fault types present there is
        # one component and the two boundary parities ensure evenness.
        adjacency = csr_matrix(
            (np.ones(len(edge_starts), dtype=np.uint8), (edge_starts, edge_ends)),
            shape=(node_count, node_count),
        )
        component_count, components = connected_components(adjacency, directed=False)
        self._component_order = self._component_starts = None
        if component_count > 1:
            self._component_order = np.argsort(components)
            self._component_starts = np.r_[0, np.cumsum(np.bincount(components)[:-1])]

    def features(self, detectors: np.ndarray) -> np.ndarray:
        """Return costs for logical 0 and 1 in columns 0 and 1, respectively.

        Costs sum the DEM log odds for selected edges, measured from the more
        likely frame. An impossible class costs infinity. A positive cost
        difference `costs[:, 0] - costs[:, 1]` favors logical 1. It compares the
        best history in each class. A posterior would sum all histories.
        """
        detectors = np.asarray(detectors)
        if detectors.ndim != 2 or detectors.shape[1] != self._detector_count:
            raise ValueError(f"expected {self._detector_count} unpacked detector bits per shot")
        if detectors.dtype.kind not in "bui" or (detectors.size and (detectors.min() < 0 or detectors.max() > 1)):
            raise ValueError("detectors must contain binary integer or boolean values")

        shot_count = len(detectors)
        costs = np.full((shot_count, 2), np.inf)
        if not shot_count:
            return costs

        syndromes = np.empty((shot_count, self._detector_count + 2), dtype=np.uint8)
        syndromes[:, :-2] = detectors ^ self._detector_offset
        detector_parity = np.bitwise_xor.reduce(syndromes[:, :-2], axis=1)
        for logical in range(2):
            # Each selected edge toggles two endpoints. The parity at the
            # L0 boundary fixes the other boundary once the syndrome is set.
            logical_parity = logical ^ self._logical_offset
            syndromes[:, -1] = logical_parity
            syndromes[:, -2] = detector_parity ^ logical_parity
            if self._component_order is None:
                _, costs[:, logical] = self._matching.decode_batch(syndromes, return_weights=True)
            else:
                component_parity = np.bitwise_xor.reduceat(
                    syndromes[:, self._component_order], self._component_starts, axis=1
                )
                supported = ~component_parity.any(axis=1)
                if supported.any():
                    _, costs[supported, logical] = self._matching.decode_batch(
                        syndromes[supported], return_weights=True
                    )
        return costs
