"""Train, select, and evaluate a fixed-noise syndrome router."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import stim

from .adaptive import AdaptiveDecoder, RoutingPolicy, SyndromeFeatures, fit_policy
from .circuits import memory_circuit
from .decoders import CHOICES, compile_decoder
from .measure import Scenario, measure, measure_decoder, sample_circuit
from .records import manifest, save_csv, save_json, scenarios
from .stats import paired_ler_difference, shot_failures

BASELINES = tuple(s.decoder_type for s in CHOICES[:2])
ROUTE_CAPS = (0.25, 0.50, 0.75)
TIME_CAPS = (1.5, 2.0)


def source(circuit_file):
    if circuit_file is not None:
        circuit = stim.Circuit.from_file(circuit_file)
        case = Scenario(0, 0.0, "external")
        yield case, circuit, {"circuit": circuit_file.name, "noise_model": "supplied Stim circuit"}
        return
    for case in scenarios():
        circuit = memory_circuit(case.basis, case.distance, case.distance, case.p)
        yield case, circuit, dict(**asdict(case), rounds=case.distance, noise_model="simple_uniform")


def fit(data):
    failed = []
    for spec in CHOICES[:2]:
        decoder = compile_decoder(data.model, spec)
        failed.append(shot_failures(decoder.decode_batch(data.detectors), data.truth))
    routes = {}
    for feature in ("count", "spacetime"):
        try:
            extractor = SyndromeFeatures(data.model, feature)
        except ValueError:
            continue
        buckets = extractor.buckets(data.detectors)
        for cap in ROUTE_CAPS:
            routes[f"{feature}-{int(100 * cap)}"] = fit_policy(
                data.model, extractor, buckets, *failed, cap, feature)
    return routes


def compare(data, routes, repeats, latency_shots, index):
    names = (*BASELINES, *routes)
    index %= len(names)
    rows, masks = {}, {}
    for name in names[index:] + names[:index]:
        if name in BASELINES:
            row, failed = measure(data, CHOICES[BASELINES.index(name)], repeats, latency_shots)
            rate = float(name == BASELINES[1])
        else:
            policy = routes[name]
            def build():
                return AdaptiveDecoder(data.model, policy)
            row, failed = measure_decoder(data, build, name, policy.to_dict(), repeats, latency_shots)
            rate = float(build().route(data.detectors).mean())
        row["correlated_fraction"] = rate
        rows[name], masks[name] = row, failed
    for name, row in rows.items():
        for baseline in BASELINES:
            row[f"{baseline}_only_failed"] = int((masks[baseline] & ~masks[name]).sum())
            row[f"candidate_only_failed_vs_{baseline}"] = int((masks[name] & ~masks[baseline]).sum())
    return list(rows.values())


def summarize(rows):
    summary = {}
    for name in sorted({r["decoder"] for r in rows}):
        cases = [r for r in rows if r["decoder"] == name]
        shots = sum(r["shots"] for r in cases)
        failures = sum(r["failures"] for r in cases)
        point = dict(shots=shots, failures=failures, ler=failures / shots,
                     batch_us_per_shot=1e6 * sum(r["median_batch_s"] for r in cases) / shots,
                     correlated_fraction=sum(r["shots"] * r["correlated_fraction"] for r in cases) / shots)
        for baseline in BASELINES:
            recovered = [r[f"{baseline}_only_failed"] for r in cases]
            lost = [r[f"candidate_only_failed_vs_{baseline}"] for r in cases]
            point[f"paired_vs_{baseline}"] = paired_ler_difference(
                (r["shots"] for r in cases), recovered, lost)
        summary[name] = point
    return summary


def select(validation):
    ordinary = validation["mwpm"]["batch_us_per_shot"]
    return {
        str(cap): min((name for name, score in validation.items()
                       if score["batch_us_per_shot"] <= ordinary * cap),
                      key=lambda name: (validation[name]["failures"],
                                        validation[name]["batch_us_per_shot"], name))
        for cap in TIME_CAPS
    }


def run(args):
    if min(args.train_shots, args.validation_shots, args.test_shots) < 2 or args.repeats < 1:
        raise ValueError("each split needs at least two shots and timing needs at least one repeat")
    if not 0 <= args.seed < 2**64 or not 0 <= args.latency_shots <= args.test_shots:
        raise ValueError("invalid seed or latency sample size")
    inputs = list(source(args.circuit))
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = manifest(["python", "-m", "bascd.adaptive_experiment", *sys.argv[1:]],
                        args.seed, args.test_shots, args.repeats)
    protocol.update(train_shots=args.train_shots, validation_shots=args.validation_shots,
                    test_shots=args.test_shots, latency_shots=args.latency_shots,
                    train_route_caps=ROUTE_CAPS, validation_runtime_caps=TIME_CAPS,
                    paired_interval="stratified normal approximation; omitted below 30 disagreements",
                    scenarios=[metadata for _, _, metadata in inputs])
    if args.circuit is not None:
        protocol["noise"] = {"name": "supplied Stim circuit", "source": str(args.circuit)}
        inputs[0][1].to_file(args.out / "input.stim")
    save_json(args.out / "manifest.json", protocol)

    def draw(item, split, shots):
        case, circuit, metadata = item
        return sample_circuit(lambda: circuit, case.seed(args.seed, split), split, shots, metadata)

    policies, train_samples, validation_rows = [], [], []
    for index, item in enumerate(inputs):
        train = draw(item, "train", args.train_shots)
        routes = fit(train)
        policies.append(routes)
        train_samples.append(train.metadata)
        del train
        data = draw(item, "validation", args.validation_shots)
        validation_rows.extend(compare(data, routes, args.repeats, 0, index))
    validation = summarize(validation_rows)
    choices = select(validation)
    finalists = sorted(set(choices.values()) - set(BASELINES))
    frozen = {"validation_choices_by_runtime_multiple": choices, "adaptive_finalists": finalists,
              "scenarios": [{"train_sample": meta, "policies": {n: p.to_dict() for n, p in routes.items()}}
                            for meta, routes in zip(train_samples, policies, strict=True)]}
    save_json(args.out / "validation.json", {"summary": validation, "measurements": validation_rows})
    save_csv(args.out / "validation.csv", validation_rows)
    save_json(args.out / "frozen-policies.json", frozen)
    print(f"Frozen validation selections: {choices}", flush=True)

    # Reload the frozen artifact before drawing any test samples.
    frozen = json.loads((args.out / "frozen-policies.json").read_text(encoding="utf-8"))
    test_rows = []
    for index, item in enumerate(inputs):
        data = draw(item, "final_test", args.test_shots)
        routes = {name: RoutingPolicy.from_dict(frozen["scenarios"][index]["policies"][name])
                  for name in frozen["adaptive_finalists"]}
        test_rows.extend(compare(data, routes, args.repeats, args.latency_shots, index))
    result = summarize(test_rows)
    save_json(args.out / "test.json", {"summary": result, "measurements": test_rows})
    save_csv(args.out / "test.csv", test_rows)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--circuit", type=Path, help="organizer-supplied .stim file, used as-is")
    parser.add_argument("--seed", type=int, default=731923)
    parser.add_argument("--train-shots", type=int, default=30000)
    parser.add_argument("--validation-shots", type=int, default=10000)
    parser.add_argument("--test-shots", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--latency-shots", type=int, default=128)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
