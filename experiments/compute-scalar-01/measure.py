import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from bascd.compute_campaign import factory
from bascd.compute_replay import _frozen_circuit
from bascd.learned_fault_router import LearnedFaultRouter
from bascd.scalar_routing import ScalarLearnedDecoder, ScalarMatchingDecoder


def measure(case, distance):
    circuit = _frozen_circuit(case, distance)
    model = circuit.detector_error_model(decompose_errors=True)
    policy = LearnedFaultRouter.load(case / "policy.npz")
    scalar = ScalarLearnedDecoder(model, policy)
    deployed = factory("learned", case)

    def batch_single(record):
        packed = np.frombuffer(record, dtype=np.uint8).reshape(1, -1)
        return int(deployed(packed)[0, 0])

    methods = {
        "ordinary": ScalarMatchingDecoder(model).decode,
        "correlated": ScalarMatchingDecoder(model, correlated=True).decode,
        "learned_batch_single": batch_single,
        "learned_scalar": scalar.decode,
    }
    names = tuple(methods)
    calls = tuple(methods.values())
    base = 9270000 + distance * 100
    warm_seed, seeds = base, list(range(base + 1, base + 4))
    warm = circuit.compile_detector_sampler(seed=warm_seed).sample(128, bit_packed=True)
    for row in warm:
        record = row.tobytes()
        for decode in calls:
            decode(record)
    routed = 0
    latency = np.empty((len(names), len(seeds), 1024), dtype=np.uint64)
    for j, seed in enumerate(seeds):
        packed = circuit.compile_detector_sampler(seed=seed).sample(1024, bit_packed=True)
        answers = deployed(packed)[:, 0]
        predictions = np.empty(len(packed), dtype=np.uint8)
        records = [row.tobytes() for row in packed]
        order = np.argsort(np.random.default_rng(seed).random((len(records), len(calls))), axis=1)
        for i, record in enumerate(records):
            for method in order[i]:
                decode = calls[method]
                start = perf_counter_ns()
                prediction = decode(record)
                stop = perf_counter_ns()
                latency[method, j, i] = stop - start
                if names[method] == "learned_scalar":
                    predictions[i] = prediction
        branch = policy.route_batch(packed)
        routed += int(np.count_nonzero(branch))
        if not np.array_equal([scalar.route(record) for record in records], branch):
            raise ValueError(f"scalar learned routing differs at d={distance}, seed={seed}")
        if not np.array_equal(predictions, answers):
            raise ValueError(f"scalar learned predictions differ at d={distance}, seed={seed}")
    return {
        "distance": distance,
        "warmup_seed": warm_seed,
        "seeds": seeds,
        "shots_per_seed": latency.shape[2],
        "learned_outputs_match_batch": True,
        "learned_route_matches_batch": True,
        "routed_to_correlated": routed,
        "latency_microseconds": {
            name: dict(zip(("p50", "p95", "p99"), np.percentile(latency[i] / 1000, (50, 95, 99)).tolist()))
            for i, name in enumerate(names)
        },
        "median_per_seed_microseconds": {
            name: (np.median(latency[i], axis=1) / 1000).tolist()
            for i, name in enumerate(names)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {
        "clock": "time.perf_counter_ns",
        "timing_scope": "one synchronous bytes-to-int call from a completed detector record",
        "host": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pymatching", "stim")},
        "cases": [measure(args.run / f"d{distance}", distance) for distance in (5, 7)],
    }
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
