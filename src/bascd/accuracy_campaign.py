"""Select low-error decoders by distance and test a frozen policy on fresh Stim shots."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from .accuracy_decoders import compile_candidate, load_frozen_decoder
from .bath_prior import bath_prior_from_profile
from .circuits import memory_circuit
from .ensemble import FAST_ENSEMBLE, FIVE_MEMBER_ENSEMBLE
from .hypergraph import HypergraphSpec, compile_hypergraph
from .measure import measure_decoder, sample_circuit
from .records import manifest, save_csv, save_json, scenarios
from .stats import paired_ler_difference, shot_failures, wilson95

REFERENCES = ("native", "bath_clock", "belief")
# Stable tie order uses the simpler prediction paths first.
CANDIDATES = ("exact_small", "native", "bath_clock", "ensemble3", "ensemble5", "belief", "bath_belief",
              "search_native", "search_belief", "dispute", "belief_pair")


def recipe_from_profile(raw_profile):
    profile = json.loads(raw_profile.decode("utf-8-sig"))
    priors = {c: bath_prior_from_profile(profile, channel=c).ratios for c in ("clock", "selected")}
    return dict(bath_xyz=priors, ensemble3=FAST_ENSEMBLE.to_dict(), ensemble5=FIVE_MEMBER_ENSEMBLE.to_dict(),
                belief=dict(iterations=12, threads=4), search=asdict(HypergraphSpec()))


def draw(case, root_seed, split, shots):
    params = dict(**asdict(case), rounds=case.distance)
    circuit = memory_circuit(**params)
    batch = sample_circuit(lambda: circuit, case.seed(root_seed, split), split, shots,
                           dict(**params, noise_model="simple_uniform", noise_status="provisional"))
    return circuit, params, batch


def validate_case(case, root_seed, shots, recipe):
    circuit, params, data = draw(case, root_seed, "validation", shots)
    predictions, components, diagnostics = {}, {}, {}
    primitives = [*REFERENCES, "ensemble3", "ensemble5", "bath_belief"]
    if circuit.num_detectors <= 24:
        primitives.append("exact_small")
    for name in primitives:
        start = perf_counter()
        decoder = compile_candidate(circuit, params, recipe, name)
        ready = perf_counter()
        predictions[name] = decoder.decode_batch(data.detectors)
        components[name] = dict(setup_s=ready-start, validation_batch_s=perf_counter()-ready)
        if name == "exact_small":
            diagnostics[name] = dict(estimated_optimal_ler=decoder.estimated_optimal_ler,
                                     numerical_excess_ler_bound=decoder.excess_ler_bound)
    start = perf_counter()
    search = compile_hypergraph(data.model, HypergraphSpec(**recipe["search"]))
    ready = perf_counter()
    raw, exhausted = search.predictions_and_low_confidence(data.detectors)
    components["bounded_search"] = dict(setup_s=ready-start, validation_batch_s=perf_counter()-ready)
    diagnostics["exhausted_searches"] = int(exhausted.sum())
    predictions["search_native"] = np.where(exhausted[:, None], predictions["native"], raw)
    predictions["search_belief"] = np.where(exhausted[:, None], predictions["belief"], raw)
    for candidate, second in (("dispute", "ensemble5"), ("belief_pair", "bath_belief")):
        disputed = predictions[second] != predictions["belief"]
        predictions[candidate] = np.where(disputed, predictions["search_belief"], predictions["belief"])
        diagnostics[candidate + "_disputed_shots"] = int(disputed.any(axis=1).sum())
    failures = {}
    for name, pred in predictions.items():
        n = int(shot_failures(pred, data.truth).sum())
        failures[name] = dict(failures=n, ler=n/shots, wilson95=list(wilson95(n, shots)))
    return dict(metadata=data.metadata, failures=failures, component_timings=components, diagnostics=diagnostics)


def select(validation):
    choices, totals = {}, {}
    for d in sorted({row["metadata"]["distance"] for row in validation}):
        rows = [r for r in validation if r["metadata"]["distance"] == d]
        complete = [name for name in CANDIDATES if all(name in r["failures"] for r in rows)]
        counts = {name: sum(r["failures"][name]["failures"] for r in rows) for name in complete}
        choices[str(d)] = min(complete, key=counts.__getitem__)
        totals[str(d)] = counts
    return choices, totals


def test_case(case, root_seed, shots, repeats, latency, frozen, policy_path, position):
    circuit, params, data = draw(case, root_seed, "final_test", shots)
    chosen = frozen["by_distance"][str(case.distance)]
    order = list(dict.fromkeys([*REFERENCES, chosen]))
    offset = position % len(order)
    rows, masks = [], {}
    for name in order[offset:] + order[:offset]:
        def build():
            if name == chosen:
                return load_frozen_decoder(policy_path, circuit)
            return compile_candidate(circuit, params, frozen["recipe"], name)
        row, masks[name] = measure_decoder(data, build, name, {"candidate": name}, repeats, latency)
        rows.append(row)
    pairs = []
    for ref in REFERENCES:
        losses, incumbent = masks[chosen], masks[ref]
        pairs.append(dict(**data.metadata, candidate=chosen, reference=ref,
                          recovered=int((incumbent & ~losses).sum()), lost=int((losses & ~incumbent).sum()),
                          both_correct=int((~incumbent & ~losses).sum()), both_failed=int((incumbent & losses).sum())))
    return rows, pairs


def summarize(rows, pairs, choices):
    result = {}
    for name in (*REFERENCES, "frozen"):
        group = [r for r in rows if r["decoder"] == (choices[str(r["distance"])] if name == "frozen" else name)]
        n = sum(r["shots"] for r in group)
        failures = sum(r["failures"] for r in group)
        result[name] = dict(shots=n, failures=failures, ler=failures/n,
                            batch_us_per_shot=1e6*sum(r["median_batch_s"] for r in group)/n)
    for ref in REFERENCES:
        group = [r for r in pairs if r["reference"] == ref]
        result["frozen"][f"paired_vs_{ref}"] = paired_ler_difference(
            (r["shots"] for r in group), (r["recovered"] for r in group), (r["lost"] for r in group))
    return result


def run(args):
    if (min(args.validation_shots, args.test_shots) < 2 or args.repeats < 1
            or not 0 <= args.latency_shots <= args.test_shots or not 0 <= args.seed < 2**64):
        raise ValueError("invalid seed, shot count, latency sample size, or timing repetitions")
    cases = scenarios(args.distances, args.probabilities, args.bases)
    if len(set(cases)) != len(cases):
        raise ValueError("each scenario must be unique")
    raw = args.profile.read_bytes()
    recipe = recipe_from_profile(raw)
    protocol = manifest(["python", "-m", "bascd.accuracy_campaign", *sys.argv[1:]],
                        args.seed, args.test_shots, args.repeats)
    protocol.update(validation_shots=args.validation_shots, latency_shots=args.latency_shots,
                    scenarios=[asdict(c) for c in cases], recipe=recipe, search_backend=HypergraphSpec().to_dict(),
                    belief_method="product_sum", backend_version=version("stabilizer-qec"),
                    profile_sha256=hashlib.sha256(raw).hexdigest(), profile=str(args.profile),
                    selection="minimum validation failures pooled by distance; ties use candidate order",
                    candidate_order=CANDIDATES, final_references=REFERENCES,
                    paired_intervals="95% stratified paired normal approximation across fixed scenarios")
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "manifest.json", protocol)
    validation, circuits = [], []
    for i, case in enumerate(cases, 1):
        row = validate_case(case, args.seed, args.validation_shots, recipe)
        validation.append(row)
        circuits.append(dict(sha256=row["metadata"]["circuit_sha256"],
                             scenario=dict(**asdict(case), rounds=case.distance)))
        save_json(args.out / "validation.json", validation)
        print(f"Validation {i}/{len(cases)}  d={case.distance}  p={case.p}  {case.basis}", flush=True)
    choices, totals = select(validation)
    policy_path = args.out / "frozen-policy.json"
    save_json(policy_path, dict(format_version=1, by_distance=choices, validation_failures=totals,
                                recipe=recipe, circuits=circuits, backend_version=version("stabilizer-qec")))
    frozen = json.loads(policy_path.read_text(encoding="utf-8"))
    print(f"Frozen selections {choices}", flush=True)
    rows, pairs = [], []
    for i, case in enumerate(cases):
        measurements, comparisons = test_case(case, args.seed, args.test_shots, args.repeats,
                                               args.latency_shots, frozen, policy_path, i)
        rows.extend(measurements)
        pairs.extend(comparisons)
        save_csv(args.out / "test.csv", rows)
        save_csv(args.out / "paired.csv", pairs)
        print(f"Test {i+1}/{len(cases)}  d={case.distance}  p={case.p}  {case.basis}", flush=True)
    result = summarize(rows, pairs, choices)
    save_json(args.out / "summary.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44712030)
    parser.add_argument("--validation-shots", type=int, default=10000)
    parser.add_argument("--test-shots", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--latency-shots", type=int, default=32)
    parser.add_argument("--distances", type=int, choices=(3,5,7), nargs="+", default=[3,5,7])
    parser.add_argument("--probabilities", type=float, nargs="+", default=[.001,.003,.005,.01])
    parser.add_argument("--bases", choices=("x","z"), nargs="+", default=["x","z"])
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
