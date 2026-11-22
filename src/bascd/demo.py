"""Compare three presentation decoders on one shared Stim sample."""

import argparse
from functools import partial
from time import perf_counter

import pymatching

from .circuits import memory_circuit
from .css_routed import CssRoutedDecoder
from .stats import paired_counts, shot_failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=10_000, help="number of test shots (default: %(default)s)")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")

    distance, rounds, basis, p = 7, 7, "z", 0.005
    seed, calibration_seed = 954130, 50170002
    circuit = memory_circuit(basis, distance, rounds, p)
    model = circuit.detector_error_model(decompose_errors=True)
    ordinary = pymatching.Matching.from_detector_error_model(model)
    correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
    selective = CssRoutedDecoder(circuit, model, calibration_seed=calibration_seed)
    packed = dict(bit_packed_shots=True, bit_packed_predictions=True)
    methods = (
        ("Ordinary MWPM", partial(ordinary.decode_batch, **packed)),
        ("Correlated matching", partial(correlated.decode_batch, enable_correlations=True, **packed)),
        ("Selective correlated matching", selective.decode_batch),
    )

    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, truth = sampler.sample(args.shots, separate_observables=True, bit_packed=True)
    print(f"Distance {distance} {basis.upper()} memory, {rounds} rounds, p={p}")
    print(f"Shared sample size = {args.shots:,}, seed = {seed}")
    print(f"Selective matching uses full correlations above {selective.cutoff} detection events.")
    print("Batch time covers decoding only. Recoveries and regressions are relative to ordinary MWPM.")
    print()
    print(f"{'Decoder':<30} {'Failures':>8} {'Error rate':>10} {'us/shot':>9} {'Recovered':>9} {'Regressed':>9}")
    ordinary_failed = None
    for name, decode in methods:
        decode(detectors[:32])
        start = perf_counter()
        predictions = decode(detectors)
        elapsed = perf_counter() - start
        failed = shot_failures(predictions, truth)
        if ordinary_failed is None:
            ordinary_failed = failed
        paired = paired_counts(ordinary_failed, failed)
        print(
            f"{name:<30} {failed.sum():>8,} {failed.mean():>10.3%} {1e6 * elapsed / args.shots:>9.2f}"
            f" {paired['mwpm_only_failed']:>9,} {paired['correlated_only_failed']:>9,}"
        )


if __name__ == "__main__":
    main()
