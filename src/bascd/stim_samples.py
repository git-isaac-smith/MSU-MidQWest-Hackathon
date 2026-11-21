"""Generate a replayable archive of Stim shots and score the frozen decoder."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import stim

from .accuracy_decoders import compile_candidate, load_frozen_decoder
from .circuits import memory_circuit
from .measure import PROBABILITIES
from .records import manifest, save_csv, save_json, scenarios
from .stats import paired_ler_difference, shot_failures, wilson95

REFERENCES = ("native", "bath_clock")
METHODS = (*REFERENCES, "frozen")


def collect_case(folder, case, policy_path, policy, shots, position):
    folder.mkdir(parents=True, exist_ok=False)
    params = {k: case[k] for k in ("basis", "distance", "rounds", "p")}
    physical = memory_circuit(**params)
    physical.to_file(folder / "circuit.stim")
    # Sample the serialized physical circuit exactly as another team member would.
    circuit = stim.Circuit.from_file(folder / "circuit.stim")
    sampler = circuit.compile_detector_sampler(seed=case["seed"])
    start = perf_counter()
    packed_events, packed_truth = sampler.sample(shots, separate_observables=True, bit_packed=True)
    sampling_s = perf_counter() - start
    packed_events.tofile(folder / "detectors.b8")
    packed_truth.tofile(folder / "observables.b8")
    events = np.unpackbits(packed_events, axis=1, count=circuit.num_detectors, bitorder="little").view(bool)
    events.flags.writeable = False
    common = dict(**case, shots=shots, num_detectors=circuit.num_detectors,
                  detector_events=int(events.sum()), shots_with_events=int(events.any(axis=1).sum()),
                  sampling_s=sampling_s)
    predictions, timings = {}, {}
    order = METHODS[position % len(METHODS):] + METHODS[:position % len(METHODS)]
    for name in order:
        start = perf_counter()
        decoder = (load_frozen_decoder(policy_path, circuit) if name == "frozen"
                   else compile_candidate(circuit, params, policy["recipe"], name))
        decoder.decode_batch(np.zeros((1, circuit.num_detectors), dtype=bool))
        ready = perf_counter()
        predictions[name] = decoder.decode_batch(events)
        elapsed = perf_counter() - ready
        pred = predictions[name]
        if pred.shape != (shots, circuit.num_observables) or not np.isin(pred, [0, 1]).all():
            raise ValueError(f"invalid logical predictions from {name}")
        np.packbits(pred, axis=1, bitorder="little").tofile(folder / (name + ".b8"))
        timings[name] = dict(setup_s=ready - start, inference_s=elapsed, batch_us_per_shot=1e6 * elapsed / shots)
    # Only the evaluator compares actual observable flips with the predictions.
    truth = np.unpackbits(packed_truth, axis=1, count=circuit.num_observables, bitorder="little")
    common["observable_flips"] = int(truth.sum())
    common["mean_events_per_shot"] = common["detector_events"] / shots
    masks = {name: shot_failures(pred, truth) for name, pred in predictions.items()}
    rows, pairs = [], []
    for name in METHODS:
        n = int(masks[name].sum())
        low, high = wilson95(n, shots)
        rows.append(dict(**common, decoder=name, failures=n, ler=n / shots, wilson95_low=low,
                         wilson95_high=high, **timings[name]))
    for ref in REFERENCES:
        old, new = masks[ref], masks["frozen"]
        pairs.append(dict(case=case["case"], shots=shots, reference=ref,
                          recovered=int((old & ~new).sum()), lost=int((new & ~old).sum()),
                          both_failed=int((old & new).sum()), both_correct=int((~old & ~new).sum())))
    hashes = {}
    for f in sorted(folder.iterdir()):
        with f.open("rb") as fh:
            hashes[f.name] = hashlib.file_digest(fh, "sha256").hexdigest()
    return rows, pairs, hashes


def summarize(rows, pairs):
    cases = [r for r in rows if r["decoder"] == "frozen"]
    total = sum(r["shots"] for r in cases)
    result = dict(shots=total, scenario_count=len(cases),
                  detector_events=sum(r["detector_events"] for r in cases),
                  shots_with_events=sum(r["shots_with_events"] for r in cases),
                  observable_flips=sum(r["observable_flips"] for r in cases), methods={}, paired={})
    for name in METHODS:
        data = [r for r in rows if r["decoder"] == name]
        failures = sum(r["failures"] for r in data)
        result["methods"][name] = dict(failures=failures, ler=failures/total,
                                      batch_us_per_shot=1e6*sum(r["inference_s"] for r in data)/total)
    for ref in REFERENCES:
        group = [r for r in pairs if r["reference"] == ref]
        result["paired"][ref] = paired_ler_difference(
            (r["shots"] for r in group), (r["recovered"] for r in group), (r["lost"] for r in group))
    return result


def run(out, policy_path, cases, shots, seed, command):
    cases = list(cases)
    if shots < 2 or not cases or len(set(cases)) != len(cases):
        raise ValueError("supply at least two shots and unique, nonempty scenarios")
    raw = policy_path.read_bytes()
    policy = json.loads(raw)
    plan = [dict(case=f"d{c.distance}-{c.basis}-p{repr(c.p)}", **asdict(c), rounds=c.distance,
                 seed=c.seed(seed, "final_test")) for c in cases]
    supported = [c["scenario"] for c in policy["circuits"]]
    if any(dict(**asdict(c), rounds=c.distance) not in supported for c in cases):
        raise ValueError("a requested circuit is outside the supplied frozen policy")
    out.mkdir(parents=True, exist_ok=False)
    saved_policy = out / "frozen-policy.json"
    saved_policy.write_bytes(raw)
    protocol = manifest(command, seed, shots, 1)
    protocol.update(cases=plan, methods=METHODS, split="final_test",
                    policy_sha256=hashlib.sha256(raw).hexdigest(), backend_version=version("stabilizer-qec"),
                    sampling="Stim compile_detector_sampler, one bit-packed sample call per case, separate observables",
                    format="Stim b8, little-endian bits, each shot starts on a byte boundary",
                    timing="one warm full batch per decoder, order rotated by scenario; setup includes warm-up")
    save_json(out / "manifest.json", protocol)
    rows, pairs, hashes = [], [], {}
    for i, case in enumerate(plan):
        batch, comparison, hashes[case["case"]] = collect_case(
            out / "cases" / case["case"], case, saved_policy, policy, shots, i)
        rows.extend(batch)
        pairs.extend(comparison)
        print(f"{i+1}/{len(plan)} {case['case']}  frozen {batch[-1]['failures']}/{shots}", flush=True)
    save_csv(out / "stats.csv", rows)
    save_csv(out / "paired.csv", pairs)
    save_json(out / "sha256.json", hashes)
    result = summarize(rows, pairs)
    save_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new archive directory")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=10000, help="shots per scenario")
    parser.add_argument("--seed", type=int, default=954130)
    parser.add_argument("--distances", choices=(3,5,7), type=int, nargs="+", default=[3,5,7])
    parser.add_argument("--probabilities", type=float, nargs="+", default=PROBABILITIES)
    parser.add_argument("--bases", choices=("x","z"), nargs="+", default=["x","z"])
    args = parser.parse_args()
    try:
        result = run(args.out, args.policy, scenarios(args.distances, args.probabilities, args.bases),
                     args.shots, args.seed, ["python", "-m", "bascd.stim_samples", *sys.argv[1:]])
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
