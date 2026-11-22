"""Compare the controlled five-round decoders on identical detector shots."""

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROTOTYPE = ROOT / "new-approach"
sys.path.insert(0, str(PROTOTYPE))

from bascd.stats import paired_counts, wilson95
from distance7_study import MARGIN_LIMIT, overridden
from frequency import Passband, features
from full_map_decoder import FullMapDecoder
from hybrid_cnn import HybridCNN
from hybrid_policy import HybridDecoder
from hybrid_study import decisions_for_all
from list_matching import ListMatchingPosterior
from localizer import Localizer
from logical_reference import build_reference
from matching_confidence import MatchingConfidence

DISTANCES = (5, 9, 17, 33)
PHYSICAL_NOISE = (0.008, 0.04)
BATCH = 64
RESULTS = PROTOTYPE / "results"
SITE = RESULTS / "screen-run-2026112102"
HYBRID = RESULTS / "hybrid-01"
LIST = RESULTS / "list-matching-01"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workload(distance):
    return 4096 if distance == 5 else 1024


def run_case(distance, output, shots):
    physical = (distance, *PHYSICAL_NOISE)
    reference = build_reference(*physical)
    confidence = MatchingConfidence(*physical)
    site_spec = read_json(SITE / "metrics.json")
    band = Passband(**site_spec["passband"])
    site = {mode: Localizer.load(SITE / f"{mode}-cnn.npz") for mode in ("raw", "fft")}
    list_policy = read_json(LIST / "policy.json")
    engine_spec = list_policy["list_matching"]
    lists = ListMatchingPosterior(*physical, members=engine_spec["members"], seed=engine_spec["seed"])
    hybrid = HybridDecoder.load(HYBRID) if distance == 5 else None
    raw_hybrid = HybridCNN.load(HYBRID / "raw-cnn.npz") if hybrid else None
    map_decoder = FullMapDecoder(physical, backend="native") if hybrid else None

    seed = 92701100 + distance
    sampler = reference.circuit.compile_detector_sampler(seed=seed)
    truth_parts, records_parts, route_parts = [], [], []
    prediction_parts = {}
    for start in range(0, shots, BATCH):
        detectors, truth = sampler.sample(min(BATCH, shots - start), separate_observables=True)
        records_parts.append(np.packbits(detectors, axis=1, bitorder="little"))
        truth_parts.append(truth[:, 0])
        observed, terminal = reference.public_checks(detectors)
        mwpm = reference.mwpm_parity(detectors)
        predicted = {"mwpm": mwpm}
        for mode, model in site.items():
            inputs = features(observed, reference.check_mask, band) if mode == "fft" else features(
                observed, reference.check_mask, None
            )
            selected = model.predict(inputs) >= site_spec["thresholds"][mode]
            predicted[f"{mode}_site_parity"] = reference.observable_parity(selected)
            predicted[f"{mode}_site_final"] = reference.observable_parity(
                reference.complete_fft(selected, terminal)
            )

        costs = confidence.features(detectors)
        route = abs(costs[:, 1] - costs[:, 0]) <= MARGIN_LIMIT
        route_parts.append(route)
        probability = lists.predict_proba(detectors[route])
        predicted["list_matching"] = overridden(mwpm, route, probability, list_policy["cutoff"])

        if hybrid is not None:
            hybrid_predictions, _ = decisions_for_all(hybrid, raw_hybrid, detectors)
            predicted.update((f"hybrid_{key}", value) for key, value in hybrid_predictions.items() if key != "mwpm")
            predicted["full_posterior"] = map_decoder.predict(detectors)
        for method, bits in predicted.items():
            prediction_parts.setdefault(method, []).append(bits)

    truth = np.concatenate(truth_parts)
    predictions = {key: np.concatenate(parts) for key, parts in prediction_parts.items()}
    baseline_failures = predictions["mwpm"] != truth
    routed = int(np.concatenate(route_parts).sum())
    rows = []
    for method, bits in predictions.items():
        failures = bits != truth
        errors = int(failures.sum())
        counts = paired_counts(baseline_failures, failures)
        low, high = wilson95(errors, shots)
        rows.append(dict(distance=distance, data_qubits=distance**2, detectors=reference.circuit.num_detectors,
                         method=method, shots=shots, failures=errors, logical_error_percent=100 * errors / shots,
                         wilson95_low_percent=100 * low, wilson95_high_percent=100 * high,
                         recovered=counts["mwpm_only_failed"], regressed=counts["correlated_only_failed"],
                         list_routed=routed if method == "list_matching" else ""))
    circuit_file = output / f"d{distance}.stim"
    reference.circuit.to_file(circuit_file)
    np.savez_compressed(output / f"d{distance}.npz", detectors=np.concatenate(records_parts), truth=truth,
                        list_route=np.concatenate(route_parts), **predictions)
    return rows, dict(distance=distance, shots=shots, seed=seed, circuit_sha256=digest(circuit_file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shots", type=int, help="Override the per-distance shot counts")
    args = parser.parse_args()
    if args.shots is not None and args.shots < 1:
        parser.error("--shots must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    rows, cases = [], []
    for distance in DISTANCES:
        case_rows, case = run_case(distance, args.out, args.shots or workload(distance))
        rows.extend(case_rows)
        cases.append(case)
        print(f"d{distance}: {case['shots']} shots completed", flush=True)
    with (args.out / "accuracy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifacts = [SITE / name for name in ("metrics.json", "raw-cnn.npz", "fft-cnn.npz")]
    artifacts += [HYBRID / name for name in ("policy.json", "site-cnn.npz", "logical-cnn.npz", "raw-cnn.npz")]
    artifacts += [LIST / "policy.json", ROOT / "build" / "native_posterior.dll"]
    write_json(args.out / "run.json", {
        "noise": "Five-round rotated Z memory with independent per-round data X and check readout faults "
                 "and ideal terminal data readout",
        "p_data": PHYSICAL_NOISE[0], "p_readout": PHYSICAL_NOISE[1],
        "rounds": 5, "batch_size": BATCH, "cases": cases,
        "versions": {package: version(package) for package in ("numpy", "stim", "pymatching", "scipy")},
        "source_sha256": digest(Path(__file__)),
        "artifact_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in artifacts},
        "list_margin_limit": MARGIN_LIMIT, "timing": "No runtime timing collected",
        "detector_archive": "Little-endian packed detector bytes. Truth and named predictions are Boolean vectors.",
    })


if __name__ == "__main__":
    main()
