"""Syndrome-only decoders used in the accuracy campaign and frozen deployment."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .bath_prior import BathPrior
from .decoders import CHOICES, compile_decoder
from .ensemble import EnsembleMember, EnsembleSpec, PreparedEnsembleCase
from .exact_small import compile_exact_small
from .hypergraph import HypergraphSpec, compile_hypergraph


class BeliefDecoder:
    def __init__(self, model, iterations, threads):
        from stabilizer_qec import BeliefMatching
        self.matcher = BeliefMatching(model, max_bp_iters=iterations, bp_method="product_sum")
        self.threads = threads

    def decode_batch(self, detectors):
        return self.matcher.decode_batch(detectors, threads=self.threads)

    def decode_one(self, detectors):
        return self.matcher.decode_batch(detectors[None, :], threads=1)[0]


class SearchWithBeliefFallback:
    def __init__(self, search, belief):
        self.search = search
        self.belief = belief

    def decode_batch(self, detectors):
        result, exhausted = self.search.predictions_and_low_confidence(detectors)
        if exhausted.any():
            result[exhausted] = self.belief.decode_batch(detectors[exhausted])
        return result

    def decode_one(self, detectors):
        return self.decode_batch(detectors[None, :])[0]


class DisputeDecoder:
    """Resolve two decoders' disagreements with search, trusting the first on exhaustion."""

    def __init__(self, first, second, search):
        self.first = first
        self.second = second
        self.search = search

    def decode_batch(self, detectors):
        result = self.first.decode_batch(detectors)
        alternative = self.second.decode_batch(detectors)
        disputed = np.flatnonzero((result != alternative).any(axis=1))
        if len(disputed):
            search, exhausted = self.search.predictions_and_low_confidence(detectors[disputed])
            result[disputed[~exhausted]] = search[~exhausted]
        return result

    def decode_one(self, detectors):
        return self.decode_batch(detectors[None, :])[0]


def circuit_hash(circuit):
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def compile_candidate(circuit, scenario, recipe, name):
    """Compile from the physical circuit, public calibration and frozen settings."""
    if name == "exact_small":
        return compile_exact_small(circuit.detector_error_model(decompose_errors=False))
    model = circuit.detector_error_model(decompose_errors=True)
    if name == "native":
        return compile_decoder(model, CHOICES[1])
    priors = {key: BathPrior(key, tuple(ratios)) for key, ratios in recipe["bath_xyz"].items()}
    prepared = PreparedEnsembleCase(circuit, model, **scenario, priors=priors)
    if name == "bath_clock":
        return prepared.compile_member(EnsembleMember("clock"))
    if name in ("ensemble3", "ensemble5"):
        return prepared.compile(EnsembleSpec.from_dict(recipe[name]))
    if name == "belief":
        return BeliefDecoder(model, **recipe["belief"])
    if name == "bath_belief":
        return BeliefDecoder(prepared.model_for(EnsembleMember("clock")), **recipe["belief"])
    if name not in ("search_native", "search_belief", "dispute", "belief_pair"):
        raise ValueError(f"unknown candidate {name!r}")
    search = compile_hypergraph(model, HypergraphSpec(**recipe["search"]))
    if name == "search_native":
        return search
    belief = BeliefDecoder(model, **recipe["belief"])
    if name == "search_belief":
        return SearchWithBeliefFallback(search, belief)
    if name == "belief_pair":
        second = BeliefDecoder(prepared.model_for(EnsembleMember("clock")), **recipe["belief"])
    else:
        second = prepared.compile(EnsembleSpec.from_dict(recipe["ensemble5"]))
    return DisputeDecoder(belief, second, search)


def load_frozen_decoder(path: Path | str, circuit):
    """Load the winning policy for this exact circuit and compile it for inference."""
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    if frozen.get("format_version") != 1:
        raise ValueError("unsupported frozen accuracy policy")
    fingerprint = circuit_hash(circuit)
    matches = [s for s in frozen["circuits"] if s["sha256"] == fingerprint]
    if len(matches) != 1:
        raise ValueError("circuit does not match a unique frozen calibration")
    case = matches[0]["scenario"]
    name = frozen["by_distance"][str(case["distance"])]
    return compile_candidate(circuit, case, frozen["recipe"], name)
