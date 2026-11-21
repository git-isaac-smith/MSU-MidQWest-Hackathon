"""Recompute archived logical error rates and replay their samples and predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from math import isclose, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import stim

from bascd.accuracy_decoders import circuit_hash, compile_candidate, load_frozen_decoder
from bascd.circuits import memory_circuit
from bascd.measure import Scenario
from bascd.records import save_json
from verify_stim_samples import BIT_COUNTS, DECODERS, REFERENCES, read_shots, read_table


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def check(actual, expected, label):
    matches = (
        isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)
        if isinstance(expected, float)
        else actual == expected
    )
    if not matches:
        raise ValueError(f"{label}: found {actual!r}, expected {expected!r}")


def check_row(row, expected, label):
    for key, value in expected.items():
        check(type(value)(row[key]), value, f"{label} {key}")


def wilson_interval(failures, shots):
    z = NormalDist().inv_cdf(0.975)
    scale = shots + z**2
    center = (failures + z**2 / 2) / scale
    half_width = z * sqrt(failures * (1 - failures / shots) + z**2 / 4) / scale
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def audit_case(folder, case, shots, policy_path, policy, hashes, replay_decoders):
    name = case["case"]
    files = {file.name: sha256(file) for file in folder.iterdir() if file.is_file()}
    check(files, hashes, f"{name} archive hashes")
    params = {key: case[key] for key in ("distance", "p", "basis", "rounds")}
    circuit = stim.Circuit.from_file(folder / "circuit.stim")
    check(circuit == memory_circuit(**params), True, f"{name} generated circuit")
    calibrations = [item["scenario"] for item in policy["circuits"]
                    if item["sha256"] == circuit_hash(circuit)]
    check(calibrations, [params], f"{name} frozen circuit calibration")

    detectors = read_shots(folder / "detectors.b8", shots=shots,
                           bits=circuit.num_detectors, is_detector=True)
    regenerated = circuit.compile_detector_sampler(seed=case["seed"]).sample(
        shots, separate_observables=True, bit_packed=True
    )
    check(np.array_equal(detectors, regenerated[0]), True, f"{name} seeded detector replay")
    truth = read_shots(folder / "observables.b8", shots=shots, bits=circuit.num_observables)
    check(np.array_equal(truth, regenerated[1]), True, f"{name} seeded observable replay")
    events = np.unpackbits(detectors, axis=1, count=circuit.num_detectors, bitorder="little").view(bool)
    events.flags.writeable = False
    result = dict(
        **case, shots=shots, num_detectors=circuit.num_detectors,
        detector_events=int(BIT_COUNTS[detectors].sum()),
        shots_with_events=int(np.any(detectors, axis=1).sum()),
        observable_flips=int(BIT_COUNTS[truth].sum()),
        frozen_decoder=policy["by_distance"][str(case["distance"])],
        methods={}, paired={},
    )
    failed = {}
    for method in DECODERS:
        packed = read_shots(folder / f"{method}.b8", shots=shots, bits=circuit.num_observables)
        if replay_decoders:
            decoder = (load_frozen_decoder(policy_path, circuit) if method == "frozen"
                       else compile_candidate(circuit, params, policy["recipe"], method))
            decoder.decode_batch(np.zeros((1, circuit.num_detectors), dtype=bool))
            predictions = decoder.decode_batch(events)
            check(predictions.shape, (shots, circuit.num_observables), f"{name} {method} shape")
            check(np.isin(predictions, (0, 1)).all(), True, f"{name} {method} binary predictions")
            replay = np.packbits(predictions, axis=1, bitorder="little")
            check(np.array_equal(replay, packed), True, f"{name} {method} prediction replay")
        failed[method] = np.any(packed != truth, axis=1)
        count = int(failed[method].sum())
        result["methods"][method] = dict(failures=count, ler=count / shots,
                                         wilson95=wilson_interval(count, shots))

    candidate = failed["frozen"]
    for method in REFERENCES:
        reference = failed[method]
        result["paired"][method] = dict(
            recovered=int((reference & ~candidate).sum()), lost=int((candidate & ~reference).sum()),
            both_failed=int((candidate & reference).sum()),
            both_correct=int((~candidate & ~reference).sum()),
        )
    return result


def summarize(scenarios, summary):
    fields = ("shots", "detector_events", "shots_with_events", "observable_flips")
    pooled = {key: sum(case[key] for case in scenarios) for key in fields}
    pooled["scenario_count"] = len(scenarios)
    for key, value in pooled.items():
        check(summary[key], value, f"summary.json {key}")
    total = pooled["shots"]
    pooled.update(methods={}, paired={})
    for method in DECODERS:
        count = sum(case["methods"][method]["failures"] for case in scenarios)
        row = dict(failures=count, ler=count / total)
        check_row(summary["methods"][method], row, f"summary.json {method}")
        pooled["methods"][method] = row
    for method in REFERENCES:
        reference = [case["paired"][method] for case in scenarios]
        row = {key: sum(pair[key] for pair in reference) for key in reference[0]}
        recovered, lost = row["recovered"], row["lost"]
        delta = (lost - recovered) / total
        squared_errors = 0.0
        for case, pair in zip(scenarios, reference, strict=True):
            n, r, l = case["shots"], pair["recovered"], pair["lost"]
            center = (l - r) / n
            sum_squared_residuals = r * (-1 - center)**2 + l * (1 - center)**2 + (n - r - l) * center**2
            squared_errors += sum_squared_residuals * n / (n - 1)
        margin = 1.96 * sqrt(squared_errors) / total
        interval = [delta - margin, delta + margin] if recovered + lost >= 30 else None
        check_row(summary["paired"][method], dict(recovered=recovered, lost=lost, ler_difference=delta),
                  f"summary.json paired vs {method}")
        archived = summary["paired"][method]["approximate_95_ci"]
        if interval is None:
            check(archived, None, f"summary.json paired vs {method} interval")
        else:
            for actual, expected in zip(archived, interval, strict=True):
                check(actual, expected, f"summary.json paired vs {method} interval")
        pooled["paired"][method] = dict(**row, ler_difference=delta, approximate_95_ci=interval)
    return pooled


def audit(archive, *, replay_decoders):
    manifest = load_json(archive / "manifest.json")
    hashes = load_json(archive / "sha256.json")
    summary = load_json(archive / "summary.json")
    policy_path = archive / "frozen-policy.json"
    policy = load_json(policy_path)
    check(manifest["format_version"], 1, "manifest format")
    check(policy["format_version"], 1, "policy format")
    check(sha256(policy_path), manifest["policy_sha256"], "frozen policy hash")
    check(policy["backend_version"], manifest["backend_version"], "policy backend version")
    if replay_decoders:
        check(version("stabilizer-qec"), manifest["backend_version"], "installed backend version")
    check(tuple(manifest["methods"]), DECODERS, "manifest methods")
    check(manifest["split"], "final_test", "manifest split")

    stats = read_table(archive / "stats.csv", ("case", "decoder"))
    paired = read_table(archive / "paired.csv", ("case", "reference"))
    cases = manifest["cases"]
    names = {case["case"] for case in cases}
    shots = manifest["shots_per_scenario"]
    check(len(names), len(cases), "manifest unique scenarios")
    check(shots >= 2 and bool(cases), True, "archive has scenarios with at least two shots each")
    check(set(hashes), names, "case hash keys")
    check({p.name for p in (archive / "cases").iterdir() if p.is_dir()}, names, "case directories")

    results = []
    for position, case in enumerate(cases, start=1):
        name = case["case"]
        scenario = Scenario(case["distance"], case["p"], case["basis"])
        check(case["seed"], scenario.seed(manifest["root_seed"], manifest["split"]), f"{name} scenario seed")
        counted = audit_case(archive / "cases" / name, case, shots, policy_path, policy,
                             hashes[name], replay_decoders)
        common = {key: counted[key] for key in ("distance", "p", "basis", "rounds", "seed", "shots",
                  "num_detectors", "detector_events", "shots_with_events", "observable_flips")}
        for method, row in counted["methods"].items():
            low, high = row["wilson95"]
            check_row(stats.pop((name, method)), dict(**common, failures=row["failures"], ler=row["ler"],
                      wilson95_low=low, wilson95_high=high, mean_events_per_shot=common["detector_events"] / shots),
                      f"stats.csv {name} {method}")
        for method, row in counted["paired"].items():
            check_row(paired.pop((name, method)), dict(shots=shots, **row), f"paired.csv {name} {method}")
        results.append(counted)
        print(f"Verified {position}/{len(cases)} {name}", file=sys.stderr, flush=True)
    check(len(stats), 0, "extra stats.csv rows")
    check(len(paired), 0, "extra paired.csv rows")

    packages = ("stim", "numpy", "pymatching", "stabilizer-qec") if replay_decoders else ("stim", "numpy")
    return dict(
        format_version=1, root_seed=manifest["root_seed"], split=manifest["split"],
        policy_sha256=manifest["policy_sha256"],
        replay_environment=dict(python=platform.python_version(), platform=platform.platform(),
                                versions={package: version(package) for package in packages}),
        verification=dict(case_file_hashes_checked=sum(map(len, hashes.values())),
                          seeded_samples_matched=len(cases),
                          decoder_prediction_files_matched=len(cases) * len(DECODERS) if replay_decoders else None,
                          scoring_rows_matched=len(cases) * len(DECODERS),
                          paired_rows_matched=len(cases) * len(REFERENCES),
                          summary_scoring_matched=True),
        pooled=summarize(results, summary), scenarios=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, help="Write the audited counts and replay record to JSON")
    parser.add_argument("--skip-decoder-replay", action="store_true",
                        help="Score the saved predictions without rerunning the decoders")
    args = parser.parse_args()
    try:
        result = audit(args.archive, replay_decoders=not args.skip_decoder_replay)
        if args.out is not None:
            save_json(args.out, result)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
