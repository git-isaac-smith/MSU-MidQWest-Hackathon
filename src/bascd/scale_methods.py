"""Additional decoders for fixed rotated-memory circuits at larger distances."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
import pymatching
import stim

from .accuracy_decoders import BeliefDecoder
from .ambiguity_dispatch import AmbiguityDispatcher
from .capped_gap import CappedGapRoutedDecoder
from .css_routed import CssRoutedDecoder
from .disagreement_rescue import DisagreementRescue
from .ensemble import EnsembleMember, EnsembleSpec, PreparedEnsembleCase
from .hypergraph import HypergraphSpec, compile_hypergraph
from .logical_gap_route import GapRoutedDecoder
from .projected_router import ProjectedOrdinary
from .temporal_filter import TemporalFilterDecoder
from .union_find import compile_union_find


METHODS = (
    "union_find", "union_find_handwritten", "css_routed", "temporal_lowpass", "local_peel", "belief",
    "ensemble3_jitter", "hypergraph_search", "projected_ordinary", "capped_gap", "logical_gap",
    "disagreement_rescue", "ambiguity_dispatch",
)
_UNPACKED_METHODS = {"union_find_handwritten", "belief", "ensemble3_jitter", "hypergraph_search"}
CSS_FRACTION = 0.06
CSS_CALIBRATION_SHOTS = 8192
CSS_CALIBRATION_SEED = 50179000
TEMPORAL_CUTOFF = 0.20
TEMPORAL_THRESHOLD = 0.5
BELIEF_ITERATIONS = 12
GAP_CUTOFF = 2.0
HANDWRITTEN_WEIGHT_SCALE = 16
JITTER_ENSEMBLE = EnsembleSpec((
    EnsembleMember(),
    EnsembleMember(jitter=1.1, seed=11),
    EnsembleMember(jitter=1.1, seed=29),
))
SEARCH_SPEC = HypergraphSpec(threads=1)


def _handwritten_graph(model):
    """Build the notebook's graph from the decomposition, in DEM edge order."""
    boundary = model.num_detectors
    edges = {}
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        probability, = instruction.args_copy()
        parts = [[]]
        for target in instruction.targets_copy():
            if target.is_separator():
                parts.append([])
            else:
                parts[-1].append(target)
        for part in parts:
            detectors = sorted(target.val for target in part if target.is_relative_detector_id())
            if len(detectors) == 1:
                detectors.append(boundary)
            if len(detectors) != 2:
                continue
            key = tuple(detectors)
            flip = sum(target.is_logical_observable_id() for target in part) % 2
            if key in edges:
                old_probability, flip = edges[key]
                combined = old_probability * (1 - probability) + probability * (1 - old_probability)
                edges[key] = combined, flip
            else:
                edges[key] = probability, flip

    if not edges:
        raise ValueError("the handwritten decoder requires graphlike faults")
    start, end = np.asarray(list(edges), dtype=np.int64).T.copy()
    probability = np.array([edge[0] for edge in edges.values()])
    flips = np.array([edge[1] for edge in edges.values()], dtype=np.int64)
    if np.any((probability <= 0) | (probability >= 0.5)):
        raise ValueError("the handwritten decoder requires graphlike probabilities in (0, 0.5)")
    log_odds = np.log((1 - probability) / probability)
    weights = np.maximum(1, np.rint(HANDWRITTEN_WEIGHT_SCALE * log_odds / np.median(log_odds))).astype(np.int64)

    degrees = np.zeros(boundary + 2, dtype=np.int64)
    np.add.at(degrees, start + 1, 1)
    np.add.at(degrees, end + 1, 1)
    offsets = np.cumsum(degrees)
    adjacency = np.empty(offsets[-1], dtype=np.int64)
    position = offsets[:-1].copy()
    for edge, (first, second) in enumerate(zip(start, end)):
        adjacency[position[first]] = edge
        position[first] += 1
        adjacency[position[second]] = edge
        position[second] += 1
    return boundary, start, end, weights, flips, offsets, adjacency


def _find_root(parent, node):
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != root:
        next_node = parent[node]
        parent[node] = root
        node = next_node
    return root


def _growth_and_peel(detectors, boundary, start, end, weights, flips, offsets, adjacency, find_root):
    """Run the notebook's smallest-cluster growth and boundary-rooted peeling."""
    nodes = boundary + 1
    edges = len(start)
    output = np.zeros((len(detectors), 1), dtype=np.uint8)
    parent = np.empty(nodes, dtype=np.int64)
    size = np.empty(nodes, dtype=np.int64)
    odd = np.empty(nodes, dtype=np.uint8)
    at_boundary = np.empty(nodes, dtype=np.uint8)
    head = np.empty(nodes, dtype=np.int64)
    tail = np.empty(nodes, dtype=np.int64)
    next_member = np.empty(nodes, dtype=np.int64)
    initialized = np.zeros(nodes, dtype=np.int64)
    active_round = np.zeros(nodes, dtype=np.int64)
    growth = np.zeros(edges, dtype=np.int64)
    full = np.zeros(edges, dtype=np.bool_)
    touched = np.empty(edges, dtype=np.int64)
    newly_full = np.empty(edges, dtype=np.int64)
    defect_parity = np.empty(nodes, dtype=np.uint8)
    visited = np.empty(nodes, dtype=np.bool_)
    root = np.empty(nodes, dtype=np.bool_)
    parent_edge = np.empty(nodes, dtype=np.int64)
    tree_parent = np.empty(nodes, dtype=np.int64)
    order = np.empty(nodes, dtype=np.int64)
    active = np.empty(nodes, dtype=np.int64)
    lit = np.empty(nodes, dtype=np.int64)
    used = np.empty(nodes, dtype=np.int64)
    seeds = np.empty(nodes + 1, dtype=np.int64)
    round_number = 0

    for shot_index, shot in enumerate(detectors):
        stamp = shot_index + 1
        lit_count = used_count = touched_count = 0
        for detector in range(boundary):
            if shot[detector]:
                lit[lit_count] = detector
                lit_count += 1
                used[used_count] = detector
                used_count += 1
                initialized[detector] = stamp
                parent[detector] = head[detector] = tail[detector] = detector
                size[detector] = odd[detector] = 1
                at_boundary[detector] = 0
                next_member[detector] = -1
        if not lit_count:
            continue

        while True:
            round_number += 1
            active_count = 0
            smallest = nodes
            for index in range(lit_count):
                cluster = find_root(parent, lit[index])
                if odd[cluster] and not at_boundary[cluster] and active_round[cluster] != round_number:
                    active_round[cluster] = round_number
                    active[active_count] = cluster
                    active_count += 1
                    smallest = min(smallest, size[cluster])
            if not active_count:
                break

            full_count = 0
            for index in range(active_count):
                cluster = active[index]
                if size[cluster] != smallest:
                    continue
                node = head[cluster]
                while node != -1:
                    for position in range(offsets[node], offsets[node + 1]):
                        edge = adjacency[position]
                        if not full[edge]:
                            if growth[edge] == 0:
                                touched[touched_count] = edge
                                touched_count += 1
                            growth[edge] += 1
                            if growth[edge] >= weights[edge]:
                                full[edge] = True
                                newly_full[full_count] = edge
                                full_count += 1
                    node = next_member[node]

            for index in range(full_count):
                edge = newly_full[index]
                for node in (start[edge], end[edge]):
                    if initialized[node] != stamp:
                        initialized[node] = stamp
                        used[used_count] = node
                        used_count += 1
                        parent[node] = head[node] = tail[node] = node
                        size[node] = 1
                        odd[node] = 0
                        at_boundary[node] = node == boundary
                        next_member[node] = -1
                first = find_root(parent, start[edge])
                second = find_root(parent, end[edge])
                if first != second:
                    if size[first] < size[second]:
                        first, second = second, first
                    parent[second] = first
                    size[first] += size[second]
                    odd[first] ^= odd[second]
                    at_boundary[first] |= at_boundary[second]
                    next_member[tail[first]] = head[second]
                    tail[first] = tail[second]

        for index in range(used_count):
            node = used[index]
            defect_parity[node] = 0
            visited[node] = root[node] = False
        for index in range(lit_count):
            defect_parity[lit[index]] = 1
        seed_count = int(initialized[boundary] == stamp)
        seeds[0] = boundary
        seeds[seed_count:seed_count + lit_count] = lit[:lit_count]

        tree_count = 0
        for index in range(seed_count + lit_count):
            node = seeds[index]
            if visited[node]:
                continue
            root[node] = visited[node] = True
            order[tree_count] = node
            queue = tree_count
            tree_count += 1
            while queue < tree_count:
                node = order[queue]
                queue += 1
                for position in range(offsets[node], offsets[node + 1]):
                    edge = adjacency[position]
                    if full[edge]:
                        neighbor = start[edge] if end[edge] == node else end[edge]
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            tree_parent[neighbor] = node
                            parent_edge[neighbor] = edge
                            order[tree_count] = neighbor
                            tree_count += 1

        prediction = 0
        for index in range(tree_count - 1, -1, -1):
            node = order[index]
            if not root[node] and defect_parity[node]:
                prediction ^= flips[parent_edge[node]]
                defect_parity[tree_parent[node]] ^= 1
        output[shot_index, 0] = prediction
        for index in range(touched_count):
            edge = touched[index]
            growth[edge] = 0
            full[edge] = False
    return output


@cache
def _compile_handwritten_kernel():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError("the handwritten challenge decoder requires numba") from exc
    return njit(_growth_and_peel), njit(_find_root)


class _HandwrittenUnionFind:
    """Syndrome-only port of Quantum_Bears_Decoder_Challenge.ipynb, Parts C–D."""

    def __init__(self, model):
        self.graph = _handwritten_graph(model)
        self.kernel, self.find_root = _compile_handwritten_kernel()
        self.decode_batch(np.zeros((1, model.num_detectors), dtype=bool))

    def decode_batch(self, detectors):
        return self.kernel(np.ascontiguousarray(detectors), *self.graph, self.find_root)


@dataclass(frozen=True)
class _BatchDecoder:
    decode: Callable[[np.ndarray], np.ndarray]
    num_detectors: int
    packed: bool = False

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        if detectors.dtype != np.bool_ or detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError("expected a (shots, detectors) boolean array matching the circuit")
        if not len(detectors):
            return np.empty((0, 1), dtype=np.uint8)
        records = np.packbits(detectors, axis=1, bitorder="little") if self.packed else detectors
        predictions = self.decode(records)
        if predictions.shape != (len(detectors), 1):
            raise ValueError("expected one prediction per shot")
        return predictions & 1 if self.packed else predictions


def compile_method(
    name: str, circuit: stim.Circuit, distance: int, basis: str, p: float, *,
    calibration_seed: int = CSS_CALIBRATION_SEED,
) -> _BatchDecoder:
    """Compile a single-thread decoder for the supplied one-observable circuit.

    The circuit supplies the native noise model and detector order. The separate
    unlabeled CSS calibration sample sets the escalation threshold. Full lookup
    has a 24-detector limit and is handled outside this factory. Frozen small-
    circuit policies are tied to the circuits used for their calibration.
    """
    if name not in METHODS:
        raise ValueError(f"unknown scale method {name!r}")
    if circuit.num_observables != 1:
        raise ValueError("expected a single-observable memory circuit")
    model = circuit.detector_error_model(decompose_errors=True)
    if name == "union_find":
        decoder = compile_union_find(model)
    elif name == "union_find_handwritten":
        decoder = _HandwrittenUnionFind(model)
    elif name in ("css_routed", "ambiguity_dispatch"):
        router = CssRoutedDecoder if name == "css_routed" else AmbiguityDispatcher
        decoder = router(
            circuit, model, calibration_seed=calibration_seed,
            fraction=CSS_FRACTION, calibration_shots=CSS_CALIBRATION_SHOTS,
        )
    elif name in ("temporal_lowpass", "local_peel"):
        decoder = TemporalFilterDecoder(
            circuit, pymatching.Matching.from_detector_error_model(model),
            cutoff=TEMPORAL_CUTOFF, threshold=TEMPORAL_THRESHOLD,
            mode="lowpass" if name == "temporal_lowpass" else "guarded",
        )
    elif name == "belief":
        decoder = BeliefDecoder(model, iterations=BELIEF_ITERATIONS, threads=1)
    elif name == "ensemble3_jitter":
        prepared = PreparedEnsembleCase(circuit, model, basis=basis, distance=distance, rounds=distance, p=p, priors={})
        decoder = prepared.compile(JITTER_ENSEMBLE)
    elif name == "hypergraph_search":
        decoder = compile_hypergraph(model, SEARCH_SPEC)
    elif name == "projected_ordinary":
        decoder = ProjectedOrdinary(circuit, model)
    elif name == "capped_gap":
        decoder = CappedGapRoutedDecoder(circuit, model, GAP_CUTOFF)
    elif name == "logical_gap":
        decoder = GapRoutedDecoder(model, GAP_CUTOFF)
    elif name == "disagreement_rescue":
        ordinary = ProjectedOrdinary(circuit, model)
        decoder = DisagreementRescue(model, ordinary=ordinary.decode_batch)
    return _BatchDecoder(decoder.decode_batch, circuit.num_detectors, name not in _UNPACKED_METHODS)
