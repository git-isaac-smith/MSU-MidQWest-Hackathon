"""Time each complete controlled-model decoder on shared held-out chunks."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from run import (
    DISTANCES, HYBRID, LIST, PHYSICAL_NOISE, SITE, FullMapDecoder, HybridCNN, HybridDecoder,
    ListMatchingPosterior, Localizer, MARGIN_LIMIT, MatchingConfidence, Passband, build_reference,
    features, overridden, read_json,
)

from hybrid_study import decision_rule, gap, raw_ablation


def decoders(distance):
    physical = distance, *PHYSICAL_NOISE
    reference = build_reference(*physical)
    confidence = MatchingConfidence(*physical)
    site_spec = read_json(SITE / "metrics.json")
    band = Passband(**site_spec["passband"])
    list_spec = read_json(LIST / "policy.json")
    engine_spec = list_spec["list_matching"]
    lists = ListMatchingPosterior(*physical, members=engine_spec["members"], seed=engine_spec["seed"])
    methods = {"mwpm": reference.mwpm_parity}

    def site_method(model, mode, complete):
        def predict(detectors):
            observed, terminal = reference.public_checks(detectors)
            selected = model.predict(features(observed, reference.check_mask, band if mode == "fft" else None))
            selected = selected >= site_spec["thresholds"][mode]
            if complete:
                selected = reference.complete_fft(selected, terminal)
            return reference.observable_parity(selected)
        return predict

    for mode in ("raw", "fft"):
        model = Localizer.load(SITE / f"{mode}-cnn.npz")
        methods[f"{mode}_site_parity"] = site_method(model, mode, False)
        methods[f"{mode}_site_final"] = site_method(model, mode, True)

    def list_method(detectors):
        baseline = reference.mwpm_parity(detectors)
        route = gap(confidence.features(detectors)) <= MARGIN_LIMIT
        return overridden(baseline, route, lists.predict_proba(detectors[route]), list_spec["cutoff"])

    methods["list_matching"] = list_method
    if distance == 5:
        hybrid = HybridDecoder.load(HYBRID)
        raw_net = HybridCNN.load(HYBRID / "raw-cnn.npz")

        def raw_override(detectors):
            baseline = hybrid.ref.mwpm_parity(detectors)
            costs = hybrid.confidence.features(detectors)
            route = gap(costs) <= hybrid.policy["gap_limit"]
            if route.any():
                inputs = raw_ablation(hybrid.encode(detectors[route], baseline[route], costs[route]))
                baseline[route] ^= decision_rule(raw_net.predict(inputs), hybrid.policy["raw_override_threshold"])
            return baseline

        methods["hybrid_raw_learned"] = raw_override
        for mode in ("learned", "margin_only", "balanced", "accuracy"):
            from functools import partial
            methods[f"hybrid_{mode}"] = partial(hybrid.predict, mode=mode)
        methods["full_posterior"] = FullMapDecoder(physical, backend="native").predict
    return reference.circuit.num_detectors, methods


def benchmark(source):
    rows = []
    for distance in DISTANCES:
        detector_count, methods = decoders(distance)
        with np.load(source / f"d{distance}.npz", allow_pickle=False) as records:
            packed = records["detectors"]
            holders = {name: records[name] for name in methods}
        width = 64
        timings = {name: [] for name in methods}
        order = np.random.default_rng(1975000 + distance)
        for repeat in range(6):
            shot_slice = slice(repeat * width, (repeat + 1) * width)
            detectors = np.unpackbits(packed[shot_slice], count=detector_count, axis=1, bitorder="little").view(bool)
            for name in order.permutation(list(methods)):
                start = perf_counter_ns()
                bits = methods[name](detectors)
                elapsed = perf_counter_ns() - start
                np.testing.assert_array_equal(bits, holders[name][shot_slice])
                if repeat:
                    timings[name].append(elapsed / (1_000 * width))
        rows.extend(dict(distance=distance, method=name, batch_shots=width, measured_batches=5,
                         median_batch_us_per_shot=median(times), batch_us_per_shot=json.dumps(times))
                    for name, times in timings.items())
        print(f"d{distance}: verified and timed {len(methods)} methods", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("x", encoding="utf-8", newline="") as stream:
        rows = benchmark(args.run)
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

