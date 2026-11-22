"""Screen belief-matching budgets, then confirm one frozen specialist."""

import argparse
import csv
from functools import partial
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import pymatching
import stim
from scipy.stats import binomtest

from .circuits import memory_circuit
from .css_routed import observable_detector_mask
from .stats import paired_ler_difference, wilson95


_PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)
_DEFAULT = "product-12"
_BUDGETS = {
    "product-4": ("product_sum", 4, 1.0),
    "product-8": ("product_sum", 8, 1.0),
    _DEFAULT: ("product_sum", 12, 1.0),
    "minimum-4-08": ("minimum_sum", 4, .8),
    "minimum-8-08": ("minimum_sum", 8, .8),
    "minimum-12-08": ("minimum_sum", 12, .8),
    "minimum-8-10": ("minimum_sum", 8, 1.0),
}


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def prepare(folder):
    """Record the grid, selection rule, and sampler seeds before scoring."""
    folder.mkdir(parents=True, exist_ok=False)
    circuit = memory_circuit("z", 7, 7, .005)
    circuit_path = folder / "circuit.stim"
    circuit.to_file(circuit_path)
    _json(folder / "protocol.json", dict(
        model="Provisional simple_uniform circuit noise, rotated Z memory, d=7, rounds=7, p=.005",
        source_sha256=_hash(Path(__file__)),
        circuit_sha256=_hash(circuit_path),
        host=platform.platform(),
        packages={p: version(p) for p in ("stim", "pymatching", "numpy", "scipy", "stabilizer-qec")},
        budgets={k: dict(bp_method=v[0], max_bp_iters=v[1], ms_scaling_factor=v[2]) for k, v in _BUDGETS.items()},
        default=_DEFAULT,
        development=dict(seeds=[92400100, 92400101], shots_per_seed=30000, timing_seed=92400200,
                         batch_size=2048, timing_repetitions=7),
        confirmation=dict(seeds=[92500100, 92500101, 92500102], shots_per_seed=60000, timing_seed=92500200,
                          batch_size=8192, timing_repetitions=9),
        selection_rule="Among budgets with positive net recovery versus correlated and at least half the "
                       "default hybrid's net recovery, choose the fastest median development batch time. "
                       "Break ties by fewer failures then name. If none qualify, choose by fewest failures, "
                       "then lowest median batch time, then name.",
        primary="Selected budget versus full correlated matching, exact two-sided McNemar at .05. "
                "All other paired comparisons and timing are descriptive.",
        specialist="One logical component ordinary vote and full correlated vote. Only disagreements receive "
                   "a single-thread belief-matching vote, accepted whether or not BP converges.",
    ))


class _Votes:
    def __init__(self, model):
        ordinary = pymatching.Matching.from_detector_error_model(model)
        correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
        self.ordinary = partial(ordinary.decode_batch, **_PACKED)
        self.correlated = partial(correlated.decode_batch, enable_correlations=True, **_PACKED)
        self.mask = observable_detector_mask(ordinary, model.num_detectors)

    def packed_disputes(self, records, correlated):
        return (self.ordinary(records & self.mask) != correlated)[:, 0]


def _compile_belief(model, settings):
    from stabilizer_qec import BeliefMatching

    belief = BeliefMatching(model, **settings)
    return partial(belief.decode_batch, threads=1, **_PACKED)


def _factory(model, method, config):
    votes = _Votes(model)
    if method == "ordinary":
        return votes.ordinary
    if method == "correlated":
        return votes.correlated
    belief = _compile_belief(model, config["budgets"][method])

    def decode(records):
        bits = votes.correlated(records)
        disputed = votes.packed_disputes(records, bits)
        if np.any(disputed):
            bits[disputed] = belief(records[disputed])
        return bits
    return decode


def _score(circuit, model, config, stage, methods):
    settings = config[stage]
    votes = _Votes(model)
    specialists = {m: _compile_belief(model, config["budgets"][m]) for m in methods[2:]}
    deploy = {m: _factory(model, m, config) for m in methods}
    results = []
    for seed in settings["seeds"]:
        packed, truth = circuit.compile_detector_sampler(seed=seed).sample(
            settings["shots_per_seed"], bit_packed=True, separate_observables=True)
        ordinary = votes.ordinary(packed)
        correlated = votes.correlated(packed)
        disputed = votes.packed_disputes(packed, correlated)
        hard_records = packed[disputed]
        predictions = {"ordinary": ordinary, "correlated": correlated}
        for method, belief in specialists.items():
            prediction = correlated.copy()
            if len(hard_records):
                prediction[disputed] = belief(hard_records)
            predictions[method] = prediction
        corr_errors = correlated != truth
        default_errors = predictions[config["default"]] != truth
        for method, pred in predictions.items():
            if not np.array_equal(deploy[method](packed[:256]), pred[:256]):
                raise ValueError(f"Deployed decoder differs from score assembly for {method} at {seed}")
            errors = pred != truth
            results.append(dict(
                seed=seed, method=method, shots=len(truth), failures=int(errors.sum()),
                belief_calls=int(disputed.sum()) if method in specialists else 0,
                recovered_vs_correlated=int((corr_errors & ~errors).sum()),
                introduced_vs_correlated=int((~corr_errors & errors).sum()),
                recovered_vs_default=int((default_errors & ~errors).sum()),
                introduced_vs_default=int((~default_errors & errors).sum()),
            ))
        print(f"{stage} seed {seed} scored", flush=True)
    return results


def _time(circuit, model, config, stage, methods):
    settings = config[stage]
    sampler = circuit.compile_detector_sampler(seed=settings["timing_seed"])
    rng = np.random.default_rng(settings["timing_seed"])
    functions = {m: _factory(model, m, config) for m in methods}
    warm = sampler.sample(1024, bit_packed=True)
    for fn in functions.values():
        fn(warm)
    rows = []
    for repetition in range(settings["timing_repetitions"]):
        records = sampler.sample(settings["batch_size"], bit_packed=True)
        for method in rng.permutation(methods):
            start = perf_counter()
            functions[method](records)
            elapsed = perf_counter() - start
            rows.append(dict(repetition=repetition, method=method, shots=len(records),
                             microseconds_per_shot=1e6 * elapsed / len(records)))
    return rows


def _aggregate(scoring, timing, methods):
    rows = []
    elapsed = {}
    for row in timing:
        elapsed.setdefault(row["method"], {})[row["repetition"]] = row["microseconds_per_shot"]
    for method in methods:
        matching = [row for row in scoring if row["method"] == method]
        n = [row["shots"] for row in matching]
        shots = sum(n)
        failures = sum(row["failures"] for row in matching)
        ci = wilson95(failures, shots)
        row = dict(method=method, shots=shots, failures=failures, error_percent=100*failures/shots,
                   wilson95_low_percent=100*ci[0], wilson95_high_percent=100*ci[1],
                   belief_calls=sum(r["belief_calls"] for r in matching))
        for baseline in ("correlated", "default"):
            won = [r[f"recovered_vs_{baseline}"] for r in matching]
            lost = [r[f"introduced_vs_{baseline}"] for r in matching]
            paired = paired_ler_difference(n, won, lost)
            lo, hi = paired["approximate_95_ci"] or (None, None)
            row.update({
                f"recovered_vs_{baseline}": sum(won),
                f"introduced_vs_{baseline}": sum(lost),
                f"change_pp_vs_{baseline}": 100 * paired["ler_difference"],
                f"ci_low_pp_vs_{baseline}": 100 * lo if lo is not None else None,
                f"ci_high_pp_vs_{baseline}": 100 * hi if hi is not None else None,
                f"paired_p_vs_{baseline}": binomtest(sum(won), sum(won)+sum(lost)).pvalue if any(won+lost) else 1.,
            })
        paired_times = [elapsed["correlated"][rep] / cost for rep, cost in elapsed[method].items()]
        default_times = [elapsed[_DEFAULT][rep] / cost for rep, cost in elapsed[method].items()]
        row["median_us_per_shot"] = median(elapsed[method].values())
        row["median_speedup_vs_correlated"] = median(paired_times)
        row["median_speedup_vs_default"] = median(default_times)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["failures"], row["median_us_per_shot"], row["method"]))


def _select(rows, config):
    budgets = [row for row in rows if row["method"] in config["budgets"]]
    default = next(row for row in budgets if row["method"] == config["default"])
    min_recovery = max(1, (default["recovered_vs_correlated"] - default["introduced_vs_correlated"]) / 2)
    eligible = [row for row in budgets if row["recovered_vs_correlated"] - row["introduced_vs_correlated"] >= min_recovery]
    if eligible:
        return min(eligible, key=lambda row: (row["median_us_per_shot"], row["failures"], row["method"]))
    return min(budgets, key=lambda row: (row["failures"], row["median_us_per_shot"], row["method"]))


def _report(folder, config, selected, rows, per_seed):
    settings = config["confirmation"]
    lines = [
        "# Specialist iteration budget",
        "",
        config["model"] + ".",
        "",
        f"Development selected **{selected}** under the frozen rule. Confirmation used "
        f"{len(settings['seeds']) * settings['shots_per_seed']:,} fresh paired shots. Warm serial timing "
        f"used {settings['timing_repetitions']} paired batches of {settings['batch_size']:,} shots. "
        "Logical error is measured per complete memory shot.",
        "",
        "| Method | Failures | Error, % | Recovered vs correlated | Introduced vs correlated | Paired p | Median batch µs/shot |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['failures']:,} | {row['error_percent']:.4f} | "
                     f"{row['recovered_vs_correlated']} | {row['introduced_vs_correlated']} | "
                     f"{row['paired_p_vs_correlated']:.4g} | {row['median_us_per_shot']:.3f} |")
    result = next(row for row in rows if row["method"] == selected)
    lower = result["change_pp_vs_correlated"] < 0
    passes = lower and result["paired_p_vs_correlated"] <= .05
    statement = ("The selected budget reduced failures under the prespecified paired criterion."
                 if passes else "The paired result did not establish lower error for the selected budget.")
    lines += [
        "",
        statement,
        "Other accuracy comparisons and all timing results are descriptive.",
        "",
        "## Paired comparison with the original hybrid budget",
        "",
        "The original hybrid uses up to 12 product-sum iterations on disagreements. Serial timing includes "
        "both matching votes, selection and all belief calls.",
        "",
        "| Test seed | Original failures | Selected failures | Recovered vs original | Introduced vs original |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    original_by_seed = {row["seed"]: row for row in per_seed if row["method"] == config["default"]}
    for row in per_seed:
        if row["method"] == selected:
            original = original_by_seed[row["seed"]]
            lines.append(f"| {row['seed']} | {original['failures']:,} | {row['failures']:,} | "
                         f"{row['recovered_vs_default']} | {row['introduced_vs_default']} |")
    lines += [
        "",
        f"The original hybrid took {result['median_speedup_vs_default']:.3f}× the selected budget’s time "
        "at the median paired batch ratio.",
        "",
        "[Protocol](protocol.json) · [Development ranking](development.csv) · "
        "[Frozen selection](frozen.json) · [Confirmation accuracy](confirmation.csv) · "
        "[Per-seed accuracy](confirmation-per-seed.csv) · [Confirmation timing repetitions](confirmation-timing.csv)",
        "",
    ]
    with (folder / "results.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def run(folder, stage):
    protocol = folder / "protocol.json"
    circuit_path = folder / "circuit.stim"
    config = json.loads(protocol.read_text(encoding="utf-8"))
    if _hash(Path(__file__)) != config["source_sha256"] or _hash(circuit_path) != config["circuit_sha256"]:
        raise ValueError("Use the source and circuit that were frozen in the protocol")
    if any(version(package) != expected for package, expected in config["packages"].items()):
        raise ValueError("Use the package versions that were frozen in the protocol")
    frozen = folder / "frozen.json"
    outputs = [folder / f"{stage}{ending}.csv" for ending in ("", "-per-seed", "-timing")]
    additional = frozen if stage == "development" else folder / "results.md"
    if any(path.exists() for path in (*outputs, additional)):
        raise FileExistsError("Choose a run folder without saved outputs for this stage")
    if stage == "development":
        methods = ["ordinary", "correlated", *config["budgets"]]
    else:
        prior = json.loads(frozen.read_text(encoding="utf-8"))
        if prior["protocol_sha256"] != _hash(protocol):
            raise ValueError("Protocol changed after selection")
        if prior["development_sha256"] != _hash(folder / "development.csv"):
            raise ValueError("Saved development ranking changed after selection")
        selected = prior["selected"]
        methods = ["ordinary", "correlated", *dict.fromkeys((config["default"], selected))]

    circuit = stim.Circuit.from_file(circuit_path)
    model = circuit.detector_error_model(decompose_errors=True)
    scoring = _score(circuit, model, config, stage, methods)
    timing = _time(circuit, model, config, stage, methods)
    aggregate = _aggregate(scoring, timing, methods)
    for path, rows in zip(outputs, (aggregate, scoring, timing), strict=True):
        _csv(path, rows)
    if stage == "development":
        selected = _select(aggregate, config)["method"]
        _json(frozen, dict(selected=selected, protocol_sha256=_hash(protocol),
                           development_sha256=_hash(folder / "development.csv")))
        print(f"Frozen specialist {selected}", flush=True)
    else:
        _report(folder, config, selected, aggregate, scoring)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "development", "confirmation"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    else:
        run(args.out, args.stage)


if __name__ == "__main__":
    main()
