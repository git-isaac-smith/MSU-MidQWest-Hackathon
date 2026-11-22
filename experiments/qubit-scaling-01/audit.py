"""Audit saved scaling records and compare d31 decoders on common shot prefixes."""

import argparse
import csv
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
import stim

from bascd.stats import wilson95


def require(condition, context):
    if not condition:
        raise ValueError(context)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_samples(folder, case, shots):
    circuit = stim.Circuit.from_file(folder / "circuit.stim")
    require(sha256(str(circuit).encode()).hexdigest() == case["circuit_sha256"], f"{folder}: circuit hash")
    require(circuit.num_detectors == case["detectors"] and circuit.num_observables == 1,
            f"{folder}: detector and observable counts")
    with np.load(folder / "records.npz", allow_pickle=False) as data:
        detectors, truth = data["detectors"], data["truth"]
    require(detectors.shape == (shots, (circuit.num_detectors + 7) // 8) and detectors.dtype == np.uint8,
            f"{folder}: packed detectors")
    require(truth.shape == (shots, 1) and truth.dtype == np.uint8 and np.all(truth <= 1),
            f"{folder}: packed observables")
    hashed = sha256()
    hashed.update(detectors)
    hashed.update(truth)
    require(hashed.hexdigest() == case["sample_sha256"], f"{folder}: sample hash")
    again = circuit.compile_detector_sampler(seed=case["seed"]).sample(
        shots, separate_observables=True, bit_packed=True
    )
    require(np.array_equal(detectors, again[0]) and np.array_equal(truth, again[1]),
            f"{folder}: seeded replay differs")
    return truth[:, 0]


def verify_predictions(folder, name, status, truth):
    path = folder / f"{name}.npy"
    if status["status"] not in ("ok", "pilot"):
        require(not path.exists(), f"{path}: predictions exist for unsuccessful method")
        return None
    prediction = np.load(path, allow_pickle=False)
    shots = status["shots"]
    require(0 < shots <= len(truth) and prediction.shape == (shots,) and prediction.dtype.kind in "bui"
            and np.all((prediction == 0) | (prediction == 1)), f"{path}: invalid predictions")
    require(sha256(prediction).hexdigest() == status["prediction_sha256"], f"{path}: prediction hash")
    return prediction != truth[:shots]


def score(failed, ref):
    shots = len(failed)
    require(len(ref) >= shots, "ordinary MWPM prefix is shorter than the recorded metric")
    count = int(failed.sum())
    low, high = wilson95(count, shots)
    return dict(shots=shots, failures=count, ler_pct=100 * count / shots,
                ci_low_pct=100 * low, ci_high_pct=100 * high,
                recovered=int((ref[:shots] & ~failed).sum()), lost=int((~ref[:shots] & failed).sum()))


def compare(method, baseline, failed, ref, case):
    n = min(len(failed), len(ref))
    failed, ref = failed[:n], ref[:n]
    recovered = int((ref & ~failed).sum())
    lost = int((~ref & failed).sum())
    discordant = recovered + lost
    # The risk difference is P(discordance) times the conditional signed difference.
    # Two exact 97.5% binomial intervals give a conservative joint 95% rectangle.
    frequency = binomtest(discordant, n).proportion_ci(confidence_level=.975, method="exact")
    direction = binomtest(lost, discordant).proportion_ci(confidence_level=.975, method="exact") if discordant else (0, 1)
    limits = [p * (2 * q - 1) for p in frequency for q in direction]
    return dict(case=case, method=method, baseline=baseline, n=n, method_failures=int(failed.sum()),
                baseline_failures=int(ref.sum()), discordant=discordant, recovered=recovered, lost=lost,
                net_recovered=recovered - lost, difference_pp=100 * (lost - recovered) / n,
                paired95_low_pp=100 * min(limits), paired95_high_pp=100 * max(limits),
                mcnemar_exact_p=binomtest(lost, discordant).pvalue if discordant else 1.0)


def audit(run, selected, partial):
    protocol = read_json(run / "protocol.json")
    require(stim.__version__ == protocol["versions"]["stim"], "Stim version differs from the recorded protocol")
    cases = {case["folder"]: case for case in protocol["cases"] if not selected or case["folder"] in selected}
    require(not selected or set(selected) == set(cases), "Unknown --cases value")
    metrics = run / "metrics.csv"
    recorded = {}
    if metrics.exists() and not partial:
        with metrics.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                key = (f'd{row["distance"]}-{row["basis"]}', row["method"])
                if key[0] in cases:
                    require(key not in recorded, f"Duplicate metrics row at {key}")
                    recorded[key] = row
    else:
        require(partial, f"Missing {metrics}; use --partial for a run in progress")

    comparisons, verified, seen = [], 0, set()
    for label, case in cases.items():
        folder = run / label
        require(read_json(folder / "case.json") == case, f"{folder}: case metadata differs from protocol")
        truth = verify_samples(folder, case, protocol["shots_target"])
        masks = {}
        for method in protocol["method_labels"]:
            status_path = folder / f"{method}.json"
            if not status_path.is_file():
                require(partial, f"Missing {status_path}")
                continue
            status = read_json(status_path)
            source = folder
            probe = folder / "short-probes" / f"{method}.json"
            if probe.is_file():
                supplement = read_json(probe)
                require(status["status"] == "resource_limit" and supplement["original_attempt"] == status
                        and supplement["shots"] == 16, f"{probe}: invalid follow-up provenance")
                source, status = probe.parent, supplement
            key = label, method
            seen.add(key)
            if not partial:
                require(key in recorded, f"Missing published metrics row for {key}")
                require(recorded[key]["status"] == status["status"], f"Published status differs for {key}")
            failed = verify_predictions(source, method, status, truth)
            if failed is not None:
                masks[method] = failed
                verified += 1
        if not partial:
            for method, failed in masks.items():
                require("mwpm" in masks, f"{folder}: missing baseline")
                for field, value in score(failed, masks["mwpm"]).items():
                    require(isclose(float(recorded[label, method][field]), value, rel_tol=1e-10, abs_tol=1e-10),
                            f"{folder}/{method}: incorrect {field}")
        if case["distance"] == 31:
            for name, failed in masks.items():
                for baseline in ("mwpm", "correlated"):
                    if baseline != name and baseline in masks:
                        comparisons.append(compare(name, baseline, failed, masks[baseline], label))
    require(partial or seen == set(recorded), "Unexpected published metric rows")
    print(f"Regenerated samples for {len(cases)} cases and verified {verified} prediction arrays.")
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cases", nargs="+", help="Optional case folders, for example d3-z d5-z")
    parser.add_argument("--partial", action="store_true", help="Audit completed method files without checking metrics.csv")
    parser.add_argument("--paired-out", type=Path, help="New CSV for d31 common-prefix comparisons")
    args = parser.parse_args()
    comparisons = audit(args.run, args.cases, args.partial)
    if args.paired_out:
        require(comparisons and not args.partial, "Paired CSV requires completed d31 comparisons")
        with args.paired_out.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        print(f"Wrote {len(comparisons)} distance-31 common-prefix comparisons.")


if __name__ == "__main__":
    main()

