"""Audit bath-transfer outputs and plot the fixed-clock correlated contrast."""

import argparse
import csv
import json
from collections import defaultdict
from math import isclose
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bascd.measure import Scenario
from bascd.stats import paired_ler_difference


PRIORS = ("native", "expanded_uniform", "clock", "selected")
DECODERS = ("mwpm", "correlated_mwpm")
METADATA = ("shots", "batch_size", "split", "seed", "rounds", "circuit_sha256", "sample_sha256",
            "num_detectors", "num_observables", "noise_model", "noise_status")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def scenario(row):
    return Scenario(int(row["distance"]), float(row["p"]), row["basis"])


def paired_stats(records):
    shots, recovered, lost = zip(*records, strict=True)
    return paired_ler_difference(shots, recovered, lost)


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8-sig"))
    listed_cases = [scenario(row) for row in manifest["scenarios"]]
    cases = set(listed_cases)
    require(cases and len(cases) == len(listed_cases), f"{run}: invalid manifest scenarios")
    shots = int(manifest["shots_per_scenario"])
    root = int(manifest["root_seed"])
    split = manifest["split"]
    benchmark, fingerprints = {}, {}
    for row in read_csv(run / "benchmark.csv"):
        case = scenario(row)
        key = case, row["decoder"], row["prior"]
        require(key not in benchmark, f"{run}: duplicate benchmark row {key}")
        require(case in cases and row["split"] == split and int(row["seed"]) == case.seed(root, split),
                f"{run}: unexpected sample seed or scenario {key}")
        require(int(row["shots"]) == int(row["batch_size"]) == shots and int(row["rounds"]) == case.distance,
                f"{run}: inconsistent scenario size or rounds {key}")
        require(all(len(bytes.fromhex(row[field])) == 32 for field in ("sample_sha256", "circuit_sha256")),
                f"{run}: invalid fingerprint {key}")
        metadata = tuple(row[field] for field in METADATA)
        require(fingerprints.setdefault(case, metadata) == metadata, f"{run}: samples differ within {case}")
        failures = int(row["failures"])
        require(0 <= failures <= shots and isclose(float(row["ler"]), failures / shots),
                f"{run}: invalid benchmark failures or rate for {key}")
        benchmark[key] = failures
    expected = {(case, decoder, prior) for case in cases for decoder in DECODERS for prior in PRIORS}
    require(set(benchmark) == expected, f"{run}: incomplete benchmark design")
    pair_rows = {}
    for row in read_csv(run / "paired.csv"):
        case, decoder, prior, reference = scenario(row), row["decoder"], row["prior"], row["reference"]
        key = case, decoder, prior, reference
        require(key not in pair_rows, f"{run}: duplicate pair row {key}")
        require(fingerprints.get(case) == tuple(row[field] for field in METADATA),
                f"{run}: pair fingerprints or scenario metadata differ for {key}")
        correct, failed, recovered, lost = (int(row[field]) for field in
                                            ("both_correct", "both_failed", "recovered", "lost"))
        require(min(correct, failed, recovered, lost) >= 0 and correct + failed + recovered + lost == shots,
                f"{run}: invalid paired counts for {key}")
        require(failed + recovered == benchmark.get((case, decoder, reference)) and
                failed + lost == benchmark.get((case, decoder, prior)), f"{run}: pair totals disagree for {key}")
        pair_rows[key] = shots, recovered, lost
    references = {"expanded_uniform": ("native",), "clock": ("native", "expanded_uniform"),
                  "selected": ("native", "expanded_uniform"), "native": ()}
    require(set(pair_rows) == {(*key, ref) for key in expected for ref in references[key[2]]},
            f"{run}: incomplete or unexpected paired design")
    require(len(summary) == len(DECODERS) * len(PRIORS) and
            {(row["decoder"], row["prior"]) for row in summary} ==
            {(decoder, prior) for decoder in DECODERS for prior in PRIORS}, f"{run}: invalid summary design")
    for row in summary:
        decoder, prior = row["decoder"], row["prior"]
        failures = sum(benchmark[case, decoder, prior] for case in cases)
        require(row["shots"] == len(cases) * shots and row["failures"] == failures and
                isclose(row["ler"], failures / row["shots"]), f"{run}: inconsistent summary totals for {decoder}/{prior}")
        for reference in references[prior]:
            actual = paired_stats(pair_rows[case, decoder, prior, reference] for case in listed_cases)
            saved = row[f"paired_vs_{reference}"]
            require(actual["recovered"] == saved["recovered"] and actual["lost"] == saved["lost"] and
                    isclose(actual["ler_difference"], saved["ler_difference"]),
                    f"{run}: inconsistent summary pairs for {decoder}/{prior}/{reference}")
            intervals = actual["approximate_95_ci"], saved["approximate_95_ci"]
            require(intervals == (None, None) or all(intervals) and
                    all(isclose(a, b, abs_tol=1e-15) for a, b in zip(*intervals, strict=True)),
                    f"{run}: inconsistent summary interval for {decoder}/{prior}/{reference}")
    return manifest, fingerprints, [
        (case.distance, benchmark[case, "correlated_mwpm", "clock"],
         benchmark[case, "correlated_mwpm", "expanded_uniform"],
         pair_rows[case, "correlated_mwpm", "clock", "expanded_uniform"])
        for case in listed_cases
    ]


def plot(groups, figure):
    fig, (rates, differences) = plt.subplots(
        ncols=2, figsize=(10, 3.65), sharey=True, gridspec_kw={"width_ratios": [1, 1.35]})
    fig.suptitle("Fixed-clock bath priors in correlated matching", x=.08, ha="left", fontweight="bold")
    for y, (name, records) in enumerate(groups.items()):
        pairs = [record[3] for record in records]
        shots = sum(pair[0] for pair in pairs)
        bath_rate = 100 * sum(record[1] for record in records) / shots
        uniform_rate = 100 * sum(record[2] for record in records) / shots
        stats = paired_stats(pairs)
        delta = 100 * stats["ler_difference"]
        interval = stats["approximate_95_ci"]
        rates.text(.38, y, f"{bath_rate:.4f}%", ha="center", va="center")
        rates.text(.79, y, f"{uniform_rate:.4f}%", ha="center", va="center")
        color = "#0c6472" if name == "Pooled" else "#385664"
        if interval is None:
            differences.plot(delta, y, "o", color=color)
            ci_text = "CI unavailable"
        else:
            low, high = [100 * bound for bound in interval]
            differences.errorbar(delta, y, xerr=[[delta - low], [high - delta]],
                                 fmt="o", color=color, capsize=4)
            ci_text = f"95% CI [{low:+.4f}, {high:+.4f}] pp"
        print(f"{name}: n={shots:,}, bath={bath_rate:.4f}%, expanded uniform={uniform_rate:.4f}%, "
              f"recovered={stats['recovered']:,}, lost={stats['lost']:,}, delta={delta:+.4f} pp, {ci_text}")
    rates.set_xlim(0, 1)
    rates.set_xticks([])
    rates.set_yticks(range(len(groups)), groups)
    rates.tick_params(axis="y", length=0)
    rates.set_ylim(len(groups) - .5, -.5)
    rates.set_title("Logical error rate", fontsize=11, loc="left", pad=16)
    rates.text(.38, 1.01, "Bath clock", transform=rates.transAxes, ha="center", fontsize=9)
    rates.text(.79, 1.01, "Expanded uniform", transform=rates.transAxes, ha="center", fontsize=9)
    rates.spines[:].set_visible(False)
    differences.axvline(0, color="#7a8284", linestyle="--", linewidth=1)
    differences.tick_params(axis="y", left=False, labelleft=False)
    differences.spines[["top", "left", "right"]].set_visible(False)
    differences.set_title("Paired difference and 95% confidence interval", fontsize=11, pad=16)
    differences.set_xlabel("Bath clock − expanded uniform (percentage points)\nNegative values favor the bath prior.")
    differences.grid(axis="x", color="#e6eaeb")
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True, help="one or more bath_transfer output directories")
    parser.add_argument("--figure", type=Path, required=True, help="output PNG path")
    args = parser.parse_args()
    records, roots, samples = [], set(), set()
    for run in args.run:
        manifest, fingerprints, data = audit(run)
        root = manifest["root_seed"]
        hashes = {metadata[METADATA.index("sample_sha256")] for metadata in fingerprints.values()}
        require(root not in roots and len(hashes) == len(fingerprints) and not hashes & samples,
                f"{run}: repeated seeds or sample fingerprints")
        roots.add(root)
        samples.update(hashes)
        records.extend(data)
    by_distance = defaultdict(list)
    for record in records:
        by_distance[record[0]].append(record)
    groups = {f"Distance {distance}": group for distance, group in sorted(by_distance.items())}
    groups["Pooled"] = records
    plot(groups, args.figure)


if __name__ == "__main__":
    main()
