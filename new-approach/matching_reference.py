"""Ordinary MWPM for the same open-ended five-round checks used by the CNN."""

from math import log, log1p

import numpy as np
import pymatching
from scipy.sparse import csc_matrix

from stim_data import ROUNDS, _layout


class MatchingReference:
    """Decode raw Z-check grids into a minimum-weight history's final data parity.

    Each independent data or readout fault costs log((1-p)/p). The matching
    selects one optimum when histories tie, using PyMatching's tie resolution.
    Distinct physical parities can have the same observed checks and weight,
    including data sites with identical check support on a spatial boundary.
    This predicts a single correction, which can differ from the sampled faults.
    """

    def __init__(self, distance: int, p_data: float, p_readout: float):
        if not (0 <= p_data <= 1 and 0 <= p_readout <= 1):
            raise ValueError("fault probabilities must be between zero and one")
        layout = _layout(distance)
        self._check_mask = layout.geometry.check_mask
        self._reference = layout.check_reference
        self._data_shape = layout.geometry.data_ids.shape
        check_count = int(self._check_mask.sum())
        data_count = layout.geometry.data_ids.size
        detector_count = ROUNDS * check_count

        # A data X copies onto each Z ancilla it controls in the Stim circuit.
        # It persists in the data frame, so only its insertion appears in the
        # time-differenced check history.
        check_ancillas = layout.ancilla_ids[layout.check_records[0]]
        check_indices = {qubit: check for check, qubit in enumerate(check_ancillas)}
        data_indices = {qubit: site for site, qubit in enumerate(layout.geometry.data_ids.ravel())}
        check_support = [[] for _ in range(data_count)]
        for gate in layout.rounds[0][0]:
            if gate.name == "CX":
                targets = gate.targets_copy()
                for control, target in zip(targets[::2], targets[1::2]):
                    if target.value in check_indices:
                        check_support[data_indices[control.value]].append(check_indices[target.value])
        if any(len(support) not in (1, 2) for support in check_support):
            raise ValueError("a data X must flip one or two Z checks")
        for support in check_support:
            support.sort()

        # The empty matrix reserves the real detector indices, even at p=0.
        matching = pymatching.Matching(csc_matrix((detector_count, 0), dtype=np.uint8))
        self._detector_offset = np.zeros(detector_count, dtype=bool)
        self._data_offset = np.zeros(data_count, dtype=bool)
        next_node = detector_count

        def add_fault(first, second, probability, site=None):
            nonlocal next_node
            # Above 1/2, start from the more likely faulted frame and let the
            # matching choose deviations from it with nonnegative weights.
            if probability > 0.5:
                self._detector_offset[first] ^= True
                if second is not None:
                    self._detector_offset[second] ^= True
                if site is not None:
                    self._data_offset[site] ^= True
                probability = 1 - probability
            if probability == 0:
                return

            weight = log1p(-probability) - log(probability)
            if second is None:
                duplicate = matching.has_boundary_edge(first)
            else:
                duplicate = matching.has_edge(first, second)
            if duplicate:
                # Subdivide parallel mechanisms. Zero syndrome at the private
                # node forces both segments of this original edge to agree.
                # The site ID and full fault weight live on its second segment.
                matching.add_edge(first, next_node, weight=0)
                first = next_node
                next_node += 1
            if second is None:
                matching.add_boundary_edge(first, fault_ids=site, weight=weight)
            else:
                matching.add_edge(first, second, fault_ids=site, weight=weight)

        for time in range(ROUNDS):
            base = time * check_count
            for site, support in enumerate(check_support):
                first = base + support[0]
                second = base + support[1] if len(support) == 2 else None
                add_fault(first, second, p_data, site)
            for check in range(check_count):
                # A wrong measurement flips two successive differences, except
                # in the final round, where the history ends at the boundary.
                first = base + check
                second = first + check_count if time + 1 < ROUNDS else None
                add_fault(first, second, p_readout)

        matching.ensure_num_fault_ids(data_count)
        self._matching = matching

    def predict(self, observed: np.ndarray) -> np.ndarray:
        """Return boolean data corrections with shape (shots, distance, distance)."""
        observed = np.asarray(observed)
        if observed.ndim != 4 or observed.shape[1:] != (ROUNDS, *self._check_mask.shape):
            raise ValueError("observed must contain five check grids per shot")
        raw = observed[:, :, self._check_mask]
        if raw.dtype.kind not in "buif" or not np.all((raw == 0) | (raw == 1)):
            raise ValueError("observed checks must contain binary values")

        relative = raw.astype(bool, copy=False) ^ self._reference
        events = relative.copy()
        events[:, 1:] ^= relative[:, :-1]
        events = events.reshape(len(observed), self._detector_offset.size)
        syndromes = np.zeros((len(observed), self._matching.num_detectors), dtype=np.uint8)
        syndromes[:, :events.shape[1]] = events ^ self._detector_offset
        predictions = self._matching.decode_batch(syndromes).astype(bool)
        return (predictions ^ self._data_offset).reshape(len(observed), *self._data_shape)
