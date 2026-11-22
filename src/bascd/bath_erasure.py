"""Research control for location heralds in an ideal rotated surface-code memory."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pymatching
import stim
from scipy.special import logsumexp
from scipy.stats import binom

from .stats import shot_failures, wilson95


@dataclass(frozen=True)
class BathNoise:
    """One effective twirled channel, including its public binary erasure flag."""

    time: float
    erasure_probability: float
    conditional_paulis: tuple[float, float, float, float]

    def __post_init__(self):
        if (not np.isfinite(self.time) or self.time < 0
                or not 0 <= self.erasure_probability <= 1):
            raise ValueError("invalid bath check time or erasure probability")
        probabilities = np.asarray(self.conditional_paulis)
        if (probabilities.shape != (4,) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or probabilities[1:3].sum() > 1
                or not np.isclose(probabilities.sum(), 1, atol=1e-12, rtol=0)):
            raise ValueError("expected normalized conditional probabilities in I, X, Y, Z order")

    @classmethod
    def from_profile_entry(cls, entry):
        paulis = entry["conditional_paulis"]
        probabilities = tuple(float(paulis[name]) for name in "IXYZ")
        return cls(float(entry["time"]), float(entry["erasure_probability"]), probabilities)

    @property
    def passed_flip_probability(self):
        return sum(self.conditional_paulis[1:3])

    @property
    def blind_flip_probability(self):
        erased = self.erasure_probability
        return erased / 2 + (1 - erased) * self.passed_flip_probability

    def sample_paulis(self, shots, sites, rng):
        """Return private Pauli labels (I, X, Y, Z as 0 through 3) and public flags."""
        flags = rng.random((shots, sites)) < self.erasure_probability
        labels = rng.choice(4, size=flags.shape, p=self.conditional_paulis)
        labels[flags] = rng.integers(0, 4, int(flags.sum()))
        return labels, flags


@dataclass(frozen=True)
class ZMemory:
    """Single-opportunity X/Y fault map from ideal Stim rotated memory in Z."""

    sites: tuple[int, ...]
    checks: np.ndarray
    logical_z: np.ndarray

    @classmethod
    def rotated(cls, distance):
        if distance < 3 or distance % 2 != 1:
            raise ValueError("expected an odd rotated memory distance of at least three")
        circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=1)
        final_measurement = next(instruction for instruction in circuit if instruction.name == "M")
        sites = tuple(target.value for target in final_measurement.targets_copy())
        checks = np.zeros((circuit.num_detectors, len(sites)), dtype=np.uint8)
        logical_z = np.zeros(len(sites), dtype=np.uint8)
        site_indices = {site: index for index, site in enumerate(sites)}
        # The first tick immediately follows preparation, where the bath acts.
        # At this tick a data X fault anticommutes with each Y or Z it touches.
        for target, regions in circuit.detecting_regions(ticks=[0]).items():
            support = [site_indices[site] for site in regions[0].pauli_indices("YZ") if site in site_indices]
            if target.is_relative_detector_id():
                checks[target.val, support] = 1
            elif target.is_logical_observable_id() and target.val == 0:
                logical_z[support] = 1
            else:
                raise ValueError("unexpected target in single-site fault map")
        # Final comparison detectors are identically zero for noise before the
        # ideal round. Only retain the detector coordinates that carry syndrome.
        checks = checks[np.any(checks, axis=1)]
        if len(sites) != distance**2 or len(checks) != (distance**2 - 1) // 2:
            raise ValueError("unexpected rotated memory dimensions")
        checks.setflags(write=False)
        logical_z.setflags(write=False)
        return cls(sites, checks, logical_z)

    def measure(self, paulis):
        """Return public syndrome and evaluator-only logical parity."""
        bit_flips = ((paulis == 1) | (paulis == 2)).astype(np.uint8)
        return (bit_flips @ self.checks.T) % 2, (bit_flips @ self.logical_z[:, None]) % 2


def _log_odds(probability):
    # Use finite weights for deterministic endpoint probabilities.
    floor = np.finfo(float).tiny
    return np.log(max(1 - probability, floor)) - np.log(max(probability, floor))


def _paired_log10_p_value(blind_only, heralded_only):
    discordant = blind_only + heralded_only
    smaller_tail = np.arange(min(blind_only, heralded_only) + 1)
    log_p = np.log(2) + logsumexp(binom.logpmf(smaller_tail, discordant, 0.5))
    return float(min(log_p, 0) / np.log(10))


class LocationDecoder:
    """MWPM on the same surface-code graph with or without erasure locations."""

    def __init__(self, code, noise):
        self._code = code
        self._edges = []
        boundaries = set()
        for index, support in enumerate(code.checks.T):
            detectors = np.flatnonzero(support)
            if len(detectors) == 1:
                # Distinct boundary nodes preserve distinct physical edges that
                # terminate at the same check but have their own location flags.
                boundary = len(code.checks) + len(boundaries)
                boundaries.add(boundary)
                ends = (int(detectors[0]), boundary)
            elif len(detectors) == 2:
                ends = tuple(int(detector) for detector in detectors)
            else:
                raise ValueError("each data fault must touch one or two checks")
            self._edges.append((*ends, {0} if code.logical_z[index] else set()))
        self._boundaries = boundaries
        self._pass_weight = _log_odds(noise.passed_flip_probability)
        blind_weights = np.full(len(self._edges), _log_odds(noise.blind_flip_probability))
        self._blind = self._matching(blind_weights)

    def _matching(self, weights):
        graph = pymatching.Matching()
        for (left, right, logical_ids), weight in zip(self._edges, np.abs(weights), strict=True):
            graph.add_edge(left, right, fault_ids=logical_ids, weight=float(weight))
        graph.set_boundary_nodes(self._boundaries)
        # When p > 1/2, match deviations from an X fault on that site. This
        # avoids negative weights on edges to explicit boundary nodes.
        prior_syndrome, prior_logical = self._code.measure((weights < 0).astype(np.uint8))
        return graph, prior_syndrome, prior_logical

    @staticmethod
    def _decode(compiled, syndrome):
        graph, prior_syndrome, prior_logical = compiled
        return graph.decode_batch(syndrome ^ prior_syndrome) ^ prior_logical

    def decode_blind(self, syndrome):
        return self._decode(self._blind, syndrome)

    def decode_heralded(self, syndrome, flags):
        if flags.shape != (len(syndrome), len(self._edges)) or flags.dtype != np.bool_:
            raise ValueError("expected one Boolean erasure flag per shot and data site")
        predictions = np.empty((len(syndrome), 1), dtype=np.uint8)
        patterns, groups, counts = np.unique(flags, axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(groups, kind="stable")
        offset = 0
        for pattern, count in zip(patterns, counts, strict=True):
            indices = order[offset:offset + count]
            compiled = self._matching(np.where(pattern, 0.0, self._pass_weight))
            predictions[indices] = self._decode(compiled, syndrome[indices])
            offset += count
        return predictions


def evaluate_clock(code, noise, shots, seed):
    """Compare two predictions on each shot with the same channel and syndrome."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    start = perf_counter()
    paulis, flags = noise.sample_paulis(shots, len(code.sites), np.random.default_rng(seed))
    syndrome, truth = code.measure(paulis)
    decoder = LocationDecoder(code, noise)
    blind = shot_failures(decoder.decode_blind(syndrome), truth)
    heralded = shot_failures(decoder.decode_heralded(syndrome, flags), truth)
    blind_only = int((blind & ~heralded).sum())
    heralded_only = int((~blind & heralded).sum())

    def summarize(failures):
        count = int(failures.sum())
        return {"failures": count, "logical_failure_fraction": count / shots,
                "wilson_95": wilson95(count, shots)}

    return {
        "time": noise.time,
        "seed": seed,
        "shots": shots,
        "erasure_probability": noise.erasure_probability,
        "conditional_paulis": dict(zip("IXYZ", noise.conditional_paulis, strict=True)),
        "bit_flip_probability_given_pass": noise.passed_flip_probability,
        "bit_flip_probability_marginal": noise.blind_flip_probability,
        "sampled_flags": int(flags.sum()),
        "shots_with_flags": int(flags.any(axis=1).sum()),
        "blind": summarize(blind),
        "heralded": summarize(heralded),
        "paired": {
            "both_correct": int((~blind & ~heralded).sum()),
            "blind_only_failed": blind_only,
            "heralded_only_failed": heralded_only,
            "both_failed": int((blind & heralded).sum()),
            "blind_minus_heralded_failure_fraction": (blind_only - heralded_only) / shots,
            "exact_two_sided_log10_p_value": _paired_log10_p_value(blind_only, heralded_only),
        },
        "elapsed_seconds": perf_counter() - start,
    }


def run_control(profile_path, *, shots=20_000, seed=31, distance=3):
    profile_bytes = Path(profile_path).read_bytes()
    profile = json.loads(profile_bytes)
    if profile.get("format_version") != 1:
        raise ValueError("expected version 1 of the archived bath profile")
    code = ZMemory.rotated(distance)
    return {
        "format_version": 1,
        "experiment": "research-only location-heralded bath erasure control",
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "profile_reference": profile["reference"],
        "time_unit": profile["time_unit"],
        "versions": {"numpy": np.__version__, "stim": stim.__version__, "pymatching": pymatching.__version__},
        "distance": distance,
        "basis": "Z memory, tracking X or Y data faults",
        "data_sites": len(code.sites),
        "syndrome_bits": len(code.checks),
        "assumptions": [
            "Independent copies of the effective Pauli-twirled bath channel act on individual data sites.",
            "Each data site has one bath opportunity followed by one ideal syndrome measurement.",
            "A flag identifies the erased data location, which is replaced by a uniform random Pauli in the model.",
            "The archived channel assumes a free clock-specific frame correction on pass.",
            "Preparation, checks, and final measurements are ideal.",
            "The two clocks define separate physical models, each with its own paired decoder comparison.",
        ],
        "clocks": {
            "fixed": evaluate_clock(code, BathNoise.from_profile_entry(profile["clock"]), shots, seed),
            "selected": evaluate_clock(code, BathNoise.from_profile_entry(profile["selected"]), shots, seed + 1),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path, help="Archived bath-profile.json")
    parser.add_argument("--output", required=True, type=Path, help="Path for the control result JSON")
    parser.add_argument("--shots", type=int, default=20_000, help="Number of paired shots per clock")
    parser.add_argument("--seed", type=int, default=31, help="Fixed-clock seed, incremented for the selected clock")
    parser.add_argument("--distance", type=int, default=3, help="Odd rotated memory distance of at least three")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = run_control(args.profile, shots=args.shots, seed=args.seed, distance=args.distance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
