"""Pair ordinary and correlated PyMatching on a fixed Stim stress sweep."""

import argparse
import hashlib
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pymatching
import stim

from bascd.circuits import memory_circuit
from bascd.measure import Scenario
from bascd.records import manifest, save_csv, save_json
from bascd.stats import paired_counts, paired_ler_difference, shot_failures, wilson95


DISTANCE = ROUNDS = 7
PROBABILITIES = (0.005, 0.01, 0.015, 0.02, 0.04)
DECODERS = {"ordinary": False, "correlated": True}


def score(ordinary, correlated):
    counts = paired_counts(ordinary, correlated)
    shots = len(ordinary)
    row = dict(shots=shots, **counts)
    for name, failures in (("ordinary", ordinary), ("correlated", correlated)):
        count = int(failures.sum())
        low, high = wilson95(count, shots)
        row.update({
            f"{name}_errors": count,
            f"{name}_ler": count / shots,
            f"{name}_wilson95_low": low,
            f"{name}_wilson95_high": high,
        })
    wins = counts["mwpm_only_failed"]
    losses = counts["correlated_only_failed"]
    row["correlated_minus_ordinary_ler"] = (losses - wins) / shots
    row["paired_approximate_95_ci"] = (
        paired_ler_difference([shots], [wins], [losses])["approximate_95_ci"]
        if shots >= 2 else None
    )
    return row


def run_case(basis, p, root_seed, shots, batch_size):
    scenario = Scenario(DISTANCE, p, basis)
    seed = scenario.seed(root_seed, "sweep")
    circuit = memory_circuit(basis, DISTANCE, ROUNDS, p)
    model = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, truth = sampler.sample(shots, separate_observables=True, bit_packed=True)
    detectors.flags.writeable = truth.flags.writeable = False
    digest = hashlib.sha256(detectors.tobytes() + truth.tobytes()).hexdigest()
    decode_kwargs = dict(bit_packed_shots=True, bit_packed_predictions=True)
    matchers = {
        name: pymatching.Matching.from_detector_error_model(model, enable_correlations=correlated)
        for name, correlated in DECODERS.items()
    }
    failures = {name: np.empty(shots, dtype=bool) for name in DECODERS}
    timings = {name: [] for name in DECODERS}

    for name, matcher in matchers.items():
        warmup = np.zeros((1, detectors.shape[1]), dtype=np.uint8)
        matcher.decode_batch(warmup, enable_correlations=DECODERS[name], **decode_kwargs)

    for index, start in enumerate(range(0, shots, batch_size)):
        batch = slice(start, start + batch_size)
        order = tuple(reversed(DECODERS)) if index % 2 else tuple(DECODERS)
        for name in order:
            start_time = perf_counter()
            predictions = matchers[name].decode_batch(
                detectors[batch], enable_correlations=DECODERS[name], **decode_kwargs
            )
            timings[name].append(perf_counter() - start_time)
            failures[name][batch] = shot_failures(predictions, truth[batch])

    bit_counts = np.unpackbits(detectors, count=circuit.num_detectors, axis=1, bitorder="little")
    density = bit_counts.mean(axis=1)
    info = dict(distance=DISTANCE, rounds=ROUNDS, basis=basis, p=p, seed=seed,
                num_detectors=circuit.num_detectors, sample_sha256=digest)
    row = dict(**info, **score(**failures), mean_detector_density=float(density.mean()))
    for name, times in timings.items():
        row[f"{name}_batch_times_s"] = times
        row[f"{name}_batch_us_per_shot"] = sum(times) * 1e6 / shots

    bins = (
        ("<10%", density < 0.10),
        ("10% to <25%", (density >= 0.10) & (density < 0.25)),
        (">=25%", density >= 0.25),
    )
    strata = []
    for label, mask in bins:
        count = int(mask.sum())
        if count:
            stratum = score(**{name: failed[mask] for name, failed in failures.items()})
            strata.append(dict(**info, density_bin=label, fraction_of_case=count / shots, **stratum))
    return row, strata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=28971314)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.shots < 2 or args.batch_size < 1:
        parser.error("need at least two shots and a positive batch size")

    args.out.mkdir(parents=True, exist_ok=True)
    protocol = manifest(["python", *sys.argv], args.seed, args.shots, 1)
    protocol.update(
        distance=DISTANCE, rounds=ROUNDS, probabilities=PROBABILITIES, bases=["x", "z"],
        batch_size=args.batch_size,
        timing="One inference pass per decoder after warm-up, on equal-size batches by default. "
               "Decoder order alternates by batch. Timing covers decode_batch only.",
        sampling="One bit-packed Stim sample call per circuit with separate observable truth.",
        stratification="Detector firing fraction per shot, using fixed <10%, 10% to <25%, "
                       "and >=25% bins. Every nonempty bin is reported.",
        sampling_seed_scope="Scenario.seed(root seed, sweep); sample hashes cover detectors and truth.",
    )
    save_json(args.out / "manifest.json", protocol)
    rows, density_rows = [], []
    for p in PROBABILITIES:
        for basis in ("x", "z"):
            row, strata = run_case(basis, p, args.seed, args.shots, args.batch_size)
            rows.append(row)
            density_rows.extend(strata)
            save_csv(args.out / "scenarios.csv", rows)
            save_csv(args.out / "density.csv", density_rows)
            print(f"p={p:g} {basis.upper()}: ordinary {row['ordinary_errors']}, "
                  f"correlated {row['correlated_errors']} / {args.shots}", flush=True)


if __name__ == "__main__":
    main()
