"""Measure warmed, individual public decoder calls separately from batch throughput."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from distance7_study import DeployedDistanceSeven
from full_map_decoder import FullMapDecoder


def measure_one(distance, decoder, seed, n):
    sampler = decoder.ref.circuit.compile_detector_sampler(seed=seed)
    warm = sampler.sample(512)
    records = sampler.sample(n)
    methods = {"mwpm": decoder.ref.mwpm_parity, "candidate": decoder.predict}
    results = {}
    for name, predict in methods.items():
        predict(warm)
        elapsed = np.empty(n)
        decisions = np.empty(n, dtype=bool)
        for i in range(n):
            shot = records[i:i + 1]
            start = perf_counter_ns()
            prediction = predict(shot)
            elapsed[i] = (perf_counter_ns() - start) / 1000
            decisions[i] = prediction[0]
        np.testing.assert_array_equal(decisions, predict(records))
        median, p95, p99, p999 = np.percentile(elapsed, [50, 95, 99, 99.9])
        results[name] = dict(median_us=median, p95_us=p95, p99_us=p99, p99_9_us=p999,
                             mean_us=float(np.mean(elapsed)))
    return dict(distance=distance, seed=seed, shots=n, timings=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    results = Path(__file__).resolve().parent / "results"
    parser.add_argument("--distance5", type=Path, default=results / "hybrid-01")
    parser.add_argument("--distance7", type=Path, default=results / "list-matching-01")
    parser.add_argument("--backend", choices=("native", "cached", "reference"), default="cached")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    if args.out.exists():
        parser.error("Choose a new latency output file")
    five = measure_one(5, FullMapDecoder.load(args.distance5, args.backend), 84994151, args.shots)
    seven = measure_one(7, DeployedDistanceSeven(args.distance7), 84994171, args.shots)
    report = dict(method="One in-memory wall-clock call on each fresh shot after a separate 512-shot warm-up. "
                         "Excludes model construction, file I/O, and Stim sampling. All single-shot decisions "
                         "were compared with predictions for the same records in a batch.",
                  distance5_backend=args.backend, platform=platform.platform(), processor=platform.processor(),
                  measurements=[five, seven])
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
