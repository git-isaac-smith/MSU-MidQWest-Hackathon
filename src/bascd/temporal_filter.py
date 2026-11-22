"""Temporal preprocessing controls for circuit-level rotated-memory syndromes."""

from collections import defaultdict
from math import inf, isfinite
from typing import Literal

import numpy as np
import pymatching
import stim
from scipy.fft import dct, idct


def _interior_traces(circuit: stim.Circuit) -> list[list[int]]:
    """Find contiguous same-check detector traces between the open ends."""
    coordinates = circuit.get_detector_coordinates()
    if not coordinates or any(len(coords) != 3 for coords in coordinates.values()):
        raise ValueError("expected detector coordinates (x, y, round)")
    first_round = min(coords[2] for coords in coordinates.values())
    final_round = max(coords[2] for coords in coordinates.values())
    checks = defaultdict(dict)
    for detector, (x, y, time) in coordinates.items():
        check = checks[x, y]
        if time in check:
            raise ValueError("multiple detectors at one check and round")
        check[time] = detector

    traces = []
    for check in checks.values():
        trace = []
        previous = -inf
        for time, detector in sorted(check.items()):
            if first_round < time < final_round:
                if time - previous != 1 and trace:
                    traces.append(trace)
                    trace = []
                trace.append(detector)
            previous = time
        if trace:
            traces.append(trace)
    return traces


class TemporalFilterDecoder:
    """Decode packed detections using temporal preprocessing and ordinary MWPM.

    The low-pass control uses a hard DCT-II cutoff in cycles per round followed
    by a threshold. The transform has reflective, rather than periodic, ends.
    Each contiguous interior trace is processed independently. First-round
    and final data-readout detectors remain untouched.

    Guarded mode is a local predecoder. An interior same-check pair is removed
    only when it has a matching edge and neither endpoint has another adjacent
    active detector. The edge's observable flips are applied to the final
    result. This guard is a heuristic, not a global optimality guarantee.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        matching: pymatching.Matching | None = None,
        *,
        cutoff: float = 0.25,
        threshold: float = 0.5,
        mode: Literal["lowpass", "guarded"] = "lowpass",
    ):
        if not 0 <= cutoff <= 0.5 or not 0 < threshold <= 1:
            raise ValueError("cutoff must be in [0, 0.5] and threshold in (0, 1]")
        if mode not in ("lowpass", "guarded"):
            raise ValueError("mode must be lowpass or guarded")
        if matching is None:
            model = circuit.detector_error_model(decompose_errors=True)
            matching = pymatching.Matching.from_detector_error_model(model)
        if matching.num_detectors != circuit.num_detectors:
            raise ValueError("matching and circuit detector counts differ")
        if matching.num_fault_ids > circuit.num_observables:
            raise ValueError("matching has fault IDs outside the circuit observables")

        self._matching = matching
        self._num_detectors = circuit.num_detectors
        self._num_observables = circuit.num_observables
        self._threshold = threshold
        self._mode = mode
        traces = _interior_traces(circuit)
        self._lowpass_groups = []
        self._pairs = []
        if mode == "lowpass":
            by_length = defaultdict(list)
            for trace in traces:
                by_length[len(trace)].append(trace)
            for length, group in by_length.items():
                kept = min(length, 1 + int(2 * length * cutoff))
                if kept < length:
                    self._lowpass_groups.append((np.asarray(group), kept))
        else:
            self._compile_pairs(traces)

    def _compile_pairs(self, traces: list[list[int]]) -> None:
        neighbors = defaultdict(set)
        boundary_weights = defaultdict(lambda: inf)
        pair_edges = {}
        for a, b, edge in self._matching.edges():
            if b is None:
                boundary_weights[a] = edge["weight"]
            elif a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)
                pair_edges[min(a, b), max(a, b)] = edge
        for trace in traces:
            for a, b in zip(trace, trace[1:]):
                edge = pair_edges.get((min(a, b), max(a, b)))
                if edge is None:
                    continue
                weight = edge["weight"]
                if not isfinite(weight) or not 0 < weight < boundary_weights[a] + boundary_weights[b]:
                    continue
                competing = sorted((neighbors[a] | neighbors[b]) - {a, b})
                self._pairs.append((a, b, competing, sorted(edge["fault_ids"])))

    def _check_packed(self, detections: np.ndarray) -> np.ndarray:
        detections = np.asarray(detections)
        width = (self._num_detectors + 7) // 8
        if detections.dtype != np.uint8 or detections.ndim != 2 or detections.shape[1] != width:
            raise ValueError(f"expected uint8 packed detections with {width} bytes per shot")
        return detections

    def _preprocess_block(self, detections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        original = np.unpackbits(detections, axis=1, count=self._num_detectors, bitorder="little")
        residual = original.copy()
        correction = np.zeros((len(detections), self._num_observables), dtype=np.uint8)
        if self._mode == "lowpass":
            for indices, kept in self._lowpass_groups:
                coefficients = dct(original[:, indices].astype(np.float64), axis=-1, norm="ortho")
                coefficients[..., kept:] = 0
                reconstructed = idct(coefficients, axis=-1, norm="ortho")
                comparison = self._threshold * (1 - 8 * np.finfo(float).eps)
                residual[:, indices] = reconstructed >= comparison
        else:
            for a, b, competing, fault_ids in self._pairs:
                shots = np.flatnonzero(original[:, a] & original[:, b])
                if competing:
                    shots = shots[~original[np.ix_(shots, competing)].any(axis=1)]
                residual[shots, a] = 0
                residual[shots, b] = 0
                if fault_ids:
                    correction[np.ix_(shots, fault_ids)] ^= 1
        return (
            np.packbits(residual, axis=1, bitorder="little"),
            np.packbits(correction, axis=1, bitorder="little"),
        )

    def preprocess(self, packed_detections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return packed residual detector bits and the packed logical correction."""
        detections = self._check_packed(packed_detections)
        residual = np.empty_like(detections, order="C")
        correction = np.empty((len(detections), (self._num_observables + 7) // 8), dtype=np.uint8)
        for start in range(0, len(detections), 4096):
            batch = slice(start, start + 4096)
            residual[batch], correction[batch] = self._preprocess_block(detections[batch])
        return residual, correction

    def decode_batch(
        self,
        packed_detections: np.ndarray,
        *,
        bit_packed_shots: bool = True,
        bit_packed_predictions: bool = True,
    ) -> np.ndarray:
        """Return little-endian packed predictions from little-endian packed shots."""
        if not bit_packed_shots or not bit_packed_predictions:
            raise ValueError("TemporalFilterDecoder requires packed shots and predictions")
        residual, correction = self.preprocess(packed_detections)
        predictions = self._matching.decode_batch(
            residual, bit_packed_shots=True, bit_packed_predictions=True
        )
        correction[:, : predictions.shape[1]] ^= predictions
        return correction
